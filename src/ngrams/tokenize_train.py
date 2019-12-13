from tqdm import tqdm
import pickle
import pandas as pd
from nltk import word_tokenize


corpus_dir = '../../data/corpus/'
model_dir = '../../data/ngrams/'

train = pd.read_csv(f'{corpus_dir}train.csv')

tokenized_text = [list(word_tokenize(sent)) for sent in tqdm(train.cor)]

with open(f'{model_dir}tokenized_text.pickle', 'wb') as file:
    pickle.dump(tokenized_text, file)
