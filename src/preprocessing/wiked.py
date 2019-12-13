import pandas as pd

corpus_dir = '../../data/corpus/'

err = pd.read_table(f'{corpus_dir}wiked.tok.err', header=None)
cor = pd.read_table(f'{corpus_dir}wiked.tok.cor', header=None)
data = pd.concat([err, cor], axis=1)
data.columns = ["err", "cor"]
data.to_csv(f'{corpus_dir}wiked.csv', index=False)
