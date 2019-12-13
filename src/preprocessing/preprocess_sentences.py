from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
import pandas as pd

detokenize = TreebankWordDetokenizer().detokenize
preprocess_sentence = lambda sentence: detokenize(word_tokenize(sentence.lower()))

corpus_dir = '../../data/corpus/'

train = pd.read_csv(f'{corpus_dir}train.csv')
perpl = pd.read_csv(f'{corpus_dir}perpl.csv')
noisy = pd.read_csv(f'{corpus_dir}noisy.csv')
test = pd.read_csv(f'{corpus_dir}test.csv')

for df in [train, perpl, noisy, test]:
    df.cor = df.cor.apply(preprocess_sentence)
    df.err = df.err.apply(preprocess_sentence)

train.to_csv(f'{corpus_dir}train.csv', index=False)
perpl.to_csv(f'{corpus_dir}perpl.csv', index=False)
noisy.to_csv(f'{corpus_dir}noisy.csv', index=False)
test.to_csv(f'{corpus_dir}test.csv', index=False)
