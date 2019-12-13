import pickle
from tqdm import tqdm

import numpy as np
import pandas as pd

from nltk.util import ngrams
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.lm.preprocessing import padded_everygram_pipeline, pad_both_ends
from nltk.lm import NgramCounter, Vocabulary, MLE, Lidstone, WittenBellInterpolated, KneserNeyInterpolated
from nltk.lm.models import InterpolatedLanguageModel
from nltk.lm.smoothing import WittenBell
from nltk.lm.api import Smoothing

from scipy.special import softmax
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from symspellpy.symspellpy import SymSpell, Verbosity
from scipy.stats import poisson
import itertools

from hyperopt import fmin, tpe, hp
from jiwer import wer


# modifications on NLTK

class MLidstone(Lidstone):
    """Provides (modified from NLTK) Lidstone-smoothed scores."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vocab_len = len(self.vocab)
    
    
    def unmasked_score(self, word, context=None):
        """Modified to use vocab_len to store length of vocabulary
        Results in much faster implementation
        """
        counts = self.context_counts(context)
        word_count = counts[word]
        norm_count = counts.N()
        return (word_count + self.gamma) / (norm_count + self.vocab_len * self.gamma)


def _count_non_zero_vals(dictionary):
    return sum(1.0 for c in dictionary.values() if c > 0)


class MKneserNey(Smoothing):
    """Kneser-Ney Smoothing."""

    def __init__(self, vocabulary, counter, discount=0.1, **kwargs):
        super().__init__(vocabulary, counter, **kwargs)
        self.discount = discount
        self.vocab_len = len(self.vocab)
    
    
    def unigram_score(self, word):
        return 1.0 / self.vocab_len
    
    
    def alpha_gamma(self, word, context):
        prefix_counts = self.counts[context]
        prefix_total_ngrams = prefix_counts.N()
        if prefix_total_ngrams:
            alpha = max(prefix_counts[word] - self.discount, 0.0) / prefix_total_ngrams
            gamma = (
                self.discount * _count_non_zero_vals(prefix_counts) / prefix_total_ngrams
            )
        else:
            alpha, gamma = 0, 1
        return alpha, gamma


class MKneserNeyInterpolated(InterpolatedLanguageModel):
    """(modified from NLTK) Interpolated version of Kneser-Ney smoothing."""
    
    def __init__(self, order, discount=0.1, **kwargs):
        super().__init__(MKneserNey, order, params={"discount": discount}, **kwargs)


class MWittenBell(WittenBell):
    """(modified from NLTK) Witten-Bell smoothing."""
    
    def __init__(self, vocabulary, counter, **kwargs):
        super().__init__(vocabulary, counter, **kwargs)
        self.countsdb = {}
        for i in range(10):
            self.countsdb[i] = self.counts[i].N()
    
    def gamma(self, context):
        n_plus = _count_non_zero_vals(self.counts[context])
        return n_plus / (n_plus + self.countsdb[len(context) + 1])


class MWittenBellInterpolated(InterpolatedLanguageModel):
    """(modified from NLTK) Interpolated version of Witten-Bell smoothing."""
    
    def __init__(self, order, **kwargs):
        super().__init__(MWittenBell, order, **kwargs)


# Helper function for training a ngram

def count_ngrams_and_vocab(corpus, n=3, unk_cutoff=10):
    tokenized_text = [list(map(str.lower, word_tokenize(sent))) for sent in corpus]
    training_ngrams, padded_sents = padded_everygram_pipeline(n, tokenized_text)
    return NgramCounter(training_ngrams), Vocabulary(padded_sents, unk_cutoff=unk_cutoff)


def train_ngram_lm(corpus, models, n=3, a=0.0015, unk_cutoff=10, discount=0.1):
    tokenized_text = [list(map(str.lower, word_tokenize(sent))) for sent in corpus]
    training_ngrams, padded_sents = padded_everygram_pipeline(n, tokenized_text)
    vocab = Vocabulary(padded_sents, unk_cutoff=unk_cutoff)
    lms = []
    for model in models:
        training_ngrams, padded_sents = padded_everygram_pipeline(n, tokenized_text)
        if model == 'Kneser Ney':
            lm = MKneserNeyInterpolated(order=n, discount=discount, vocabulary=vocab)
        elif model == 'WBI':
            lm = MWittenBellInterpolated(order=n, vocabulary=vocab)
        elif model == 'Lidstone':
            lm = MLidstone(gamma=a, order=n, vocabulary=vocab)
        lm.fit(training_ngrams)
        lms += [lm]
    return lms

def train_ngram_lm(tokenized_text, models, n=3, a=0.0015, unk_cutoff=10, discount=0.1):
    training_ngrams, padded_sents = padded_everygram_pipeline(n, tokenized_text)
    vocab = Vocabulary(padded_sents, unk_cutoff=unk_cutoff)
    lms = []
    for model in models:
        training_ngrams, padded_sents = padded_everygram_pipeline(n, tokenized_text)
        if model == 'Kneser Ney':
            lm = MKneserNeyInterpolated(order=n, discount=discount, vocabulary=vocab)
        elif model == 'WBI':
            lm = MWittenBellInterpolated(order=n, vocabulary=vocab)
        elif model == 'Lidstone':
            lm = MLidstone(gamma=a, order=n, vocabulary=vocab)
        lm.fit(training_ngrams)
        lms += [lm]
    return lms


# ngram Tokenizer

class ngramTokenizer():
    
    def __init__(self, lm):
        self.bos_token = '<s>'
        self.eos_token = '</s>'
        self.unk_token = lm.vocab.unk_label
        self.order = lm.order
    
    
    def encode(self, sentence):
        return tuple(pad_both_ends(tuple(map(str.lower, word_tokenize(sentence))), self.order))
    
    
    def decode(self, sentence):
        detokenize = TreebankWordDetokenizer().detokenize
        content = []
        for token in sentence:
            if token == self.bos_token:
                continue
            if token == self.eos_token:
                break
            content.append(token)
        return detokenize(content)


# Noisy Channel Model: Beam Search, Viterbi, Poisson Channel Model, inversely proportional to edit distances channel model

class NoisyChannelModel():
    
    def __init__(self, lm, max_ed=4, prefix_length=7, l=1, channel_method_poisson=True, channel_prob_param=0.02):
        self.show_progress = False
        self.lm = lm
        self.l = l
        self.channel_method_poisson = channel_method_poisson
        self.channel_prob_param = channel_prob_param
        
        self.sym_spell = SymSpell(max_ed, prefix_length)
        
        if isinstance(self.lm, GPT2LMHeadModel):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.lm_sent_logscore = self.gpt2_sent_logscore
            self.beam_init = self.beam_GPT_init
            self.skipstart = 1
            self.skipend = -1
            self.update_sentence_history = self.updateGPT2history
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            for subword in range(self.tokenizer.vocab_size):
                self.sym_spell.create_dictionary_entry(key=self.tokenizer.decode(subword), count=1)
        else:
            self.lm_sent_logscore = self.ngram_sent_logscore
            self.beam_init = self.beam_ngram_init
            self.skipstart = self.lm.order-1
            self.skipend = None
            self.update_sentence_history = self.updatengramhistory
            self.tokenizer = ngramTokenizer(self.lm)
            for word in lm.vocab:
                self.sym_spell.create_dictionary_entry(key=word, count=self.lm.counts[word])
    
    
    def GPTrun(self, indexed_tokens, past=None):
        tokens_tensor = torch.tensor([indexed_tokens])
        tokens_tensor = tokens_tensor.to(self.device)
        with torch.no_grad():
            return self.lm(tokens_tensor, past=past, labels=tokens_tensor)
    
    
    def gpt2_sent_logscore(self, sentence):
        loss, next_loss = self.sentence_history[sentence[:self.pos]]
        return loss + next_loss[sentence[self.pos]]
    
    def gpt2_nohist_sent_logscore(self, sentence):
        loss, prediction_scores, past = self.GPTrun(sentence)
        return np.array(-(loss.cpu()))/np.log(2)
    
    
    def updateGPT2history(self):
        if self.pos > 1:
            for sentence in tuple(self.suggestion_sentences):
                formed_sentence = sentence[:self.pos]
                loss, prediction_scores, past = self.GPTrun(formed_sentence)
                next_loss = prediction_scores[0, -1].cpu().detach().numpy()
                self.sentence_history[formed_sentence] = (np.array(-(loss.cpu()))/np.log(2), np.log2(softmax(next_loss)))
        else:
            formed_sentence = torch.tensor([self.tokenizer.bos_token_id]).to(self.device)
            prediction_scores, past = self.lm(formed_sentence)
            formed_sentence = tuple([formed_sentence.item()])
            next_loss = prediction_scores[0].cpu().detach().numpy()
            loss = np.array(0)
            self.sentence_history[formed_sentence] = (loss, np.log2(softmax(next_loss)))
    
    
    def ngram_sent_logscore(self, sentence):
        qs = []
        for ngram in ngrams(sentence, self.lm.order):
            q = (ngram[-1], ngram[:-1])
            if q not in self.logscoredb:
                self.logscoredb[q] = self.lm.logscore(*q)
            qs += [q]
        return np.array([self.logscoredb[q] for q in qs]).sum()
    
    
    def updatengramhistory(self):
        return None
    
    
    def channel_probabilities(self):
        eds = np.array([candidate.distance for candidate in self.candidates])
        logprobs = self.poisson_channel_model(eds) if self.channel_method_poisson else self.inv_prop_channel_model(eds)
        self.channel_logprobs = {candidate.term: logprob for candidate, logprob in zip(self.candidates, logprobs)}
    
    
    def poisson_channel_model(self, eds):
        for ed in eds:
            if ed not in self.poisson_probsdb:
                self.poisson_probsdb[ed] = np.log2(poisson.pmf(k=ed, mu=self.channel_prob_param))
        return np.array([self.poisson_probsdb[ed] for ed in eds])
    
    
    def inv_prop_channel_model(self, eds):
        inv_eds = np.reciprocal(eds.astype(float), where=eds!=0)
        inv_eds[inv_eds < 1e-100] = 0.
        probs = (1-self.channel_prob_param)/inv_eds.sum() * inv_eds
        return np.log2(np.where(probs == 0., self.channel_prob_param, probs))
    
    
    def generate_suggestion_sentences(self):
        new_suggestion_sentences = {}
        self.update_sentence_history()
        for changed_word in tuple(self.channel_logprobs):
            if self.channel_logprobs[changed_word] != 0:
                for sentence in tuple(self.suggestion_sentences):
                    new_sentence = list(sentence)
                    new_sentence[self.pos] = changed_word
                    new_sentence = tuple(new_sentence)
                    new_suggestion_sentences[new_sentence] = self.lm_sent_logscore(new_sentence) * self.l + self.channel_logprobs[changed_word]
        self.suggestion_sentences.update(new_suggestion_sentences)
    
    
    def beam_all_init(self, input_sentence):
        self.logscoredb = {}
        self.poisson_probsdb = {}
        self.channel_logprobs = None
        self.suggestion_sentences = None
        self.candidates = None
        self.pos = 0
        if self.channel_method_poisson:
            chan_prob = np.log2(poisson.pmf(k=0, mu=self.channel_prob_param))
        else:
            chan_prob = np.log2(self.channel_prob_param)
        return self.beam_init(input_sentence, chan_prob)
    
    
    def beam_GPT_init(self, input_sentence, chan_prob):
        self.sentence_history = {}
        observed_sentence = tuple(self.tokenizer.encode(self.tokenizer.bos_token + input_sentence + self.tokenizer.eos_token))
        self.suggestion_sentences = {observed_sentence: self.gpt2_nohist_sent_logscore(observed_sentence) * self.l + chan_prob}
        return observed_sentence
    
    
    def beam_ngram_init(self, input_sentence, chan_prob):
        observed_sentence = self.tokenizer.encode(input_sentence)
        self.suggestion_sentences = {observed_sentence: self.lm_sent_logscore(observed_sentence) * self.l + chan_prob}
        return observed_sentence
    
    
    def beam_search(self, input_sentence, beam_width=10, max_ed=3, candidates_cutoff=50):
        observed_sentence = self.beam_all_init(input_sentence)
        for e, observed_word in enumerate(observed_sentence[self.skipstart:self.skipend]):
            self.pos = e + self.skipstart
            lookup_word = self.tokenizer.decode(observed_word) if isinstance(self.lm, GPT2LMHeadModel) else observed_word
            if lookup_word == ' ':
                continue
            self.candidates = self.sym_spell.lookup(lookup_word, Verbosity.ALL, max_ed)[:candidates_cutoff]
            if isinstance(self.lm, GPT2LMHeadModel):
                for candidate in self.candidates:
                    candidate.term = self.tokenizer.encode(candidate.term)[0]
            self.channel_probabilities()
            self.generate_suggestion_sentences()
            self.suggestion_sentences = dict(sorted(self.suggestion_sentences.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)[:beam_width])
        if isinstance(self.lm, GPT2LMHeadModel):
            return {self.tokenizer.decode(sentence)[13:-13]: np.power(2, self.suggestion_sentences[sentence]) for sentence in self.suggestion_sentences}
        else:
            return {self.tokenizer.decode(sentence): np.power(2, self.suggestion_sentences[sentence]) for sentence in self.suggestion_sentences}
    
    
    def beam_search_sentences(self, sentences):
        iterate = tqdm(sentences) if self.show_progress else sentences
        df = pd.DataFrame()
        for sent in iterate:
            corrections = self.beam_search(sent)
            df_sents = pd.DataFrame(corrections.keys())
            df_probs = pd.DataFrame(corrections.values())
            df_sents = df_sents.append(df_probs, ignore_index=True).transpose()
            df = df.append(df_sents, ignore_index=True)
        return df
    
    
    def viterbi(self, input_sentence, max_ed=3, candidates_cutoff=10):
        observed_sentence = self.tokenizer.encode(input_sentence)
        V = {(1, ('<s>','<s>')): 0}
        backpointer = {}
        candidate_words = [['<s>'], ['<s>']]
        gengram = lambda ngram, w: tuple([w] + list(ngram[:-1]))
        self.poisson_probsdb = {}
        
        for e, observed_word in enumerate(observed_sentence[self.skipstart:]):
            t = e + self.skipstart
            self.candidates = self.sym_spell.lookup(observed_word, Verbosity.ALL, max_ed, transfer_casing=True)[:candidates_cutoff]
            self.channel_probabilities()
            candidate_words += [[candidate.term for candidate in self.candidates]]
            for ngram in itertools.product(*candidate_words[t-self.lm.order+2:t+1]):
                options = [V[t-1, gengram(ngram, w)] + self.lm.logscore(ngram[-1], gengram(ngram, w)) + self.channel_logprobs[ngram[-1]] for w in candidate_words[t-self.lm.order+1]]
                best_option = np.argmax(options)
                V[t, ngram] = options[best_option]
                backpointer[t] = candidate_words[t-self.lm.order+1][best_option]
        return self.tokenizer.decode([token for _, token in backpointer.items()])
