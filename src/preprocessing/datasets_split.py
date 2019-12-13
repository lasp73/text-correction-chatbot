import pandas as pd
from sklearn.model_selection import train_test_split

corpus_dir = '../../data/corpus/'

wiked = pd.read_csv(f'{corpus_dir}wiked.csv')
wi = pd.read_csv(f'{corpus_dir}wi.csv')
wi = wi.dropna()
wi_cor = wi[wi.cor == wi.err].sample(frac=0.1)
wi_err = wi[wi.cor != wi.err]
wi_balanced = wi_err.append(wi_cor)

heldout1, train = train_test_split(wiked, test_size=0.10)
heldout2, test = train_test_split(wi_balanced, test_size=2500)
heldout3, noisy = train_test_split(heldout2, test_size=50)
heldout4, perpl = train_test_split(heldout1, test_size=2573)

train.to_csv(f'{corpus_dir}train.csv', index=False)
perpl.to_csv(f'{corpus_dir}perpl.csv', index=False)
noisy.to_csv(f'{corpus_dir}noisy.csv', index=False)
test.to_csv(f'{corpus_dir}test.csv', index=False)
