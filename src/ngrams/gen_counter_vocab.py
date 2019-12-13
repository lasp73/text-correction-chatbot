from nltk.lm import NgramCounter, Vocabulary
from nltk.lm.preprocessing import padded_everygram_pipeline
import pickle

model_dir = '../../data/ngrams/'

with open(f'{model_dir}tokenized_text.pickle', 'rb') as file:
    tokenized_text = pickle.load(file)
    
training_ngrams, padded_sents = padded_everygram_pipeline(3, tokenized_text)
counter = NgramCounter(training_ngrams)
vocabulary = Vocabulary(padded_sents, unk_cutoff=10)

with open(f'{model_dir}counter.pickle', 'wb') as file:
    pickle.dump(counter, file)

with open(f'{model_dir}vocabulary.pickle', 'wb') as file:
    pickle.dump(vocabulary, file)
