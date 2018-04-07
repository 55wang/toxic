import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

print(train.head(1))
print(train.shape)

for label in ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']:
    print(label, "{:.3f}".format(train[label].sum()/float(train.shape[0])))


def sum_label(row):
    # print(row)
    return row['toxic'] + row['severe_toxic'] + row['obscene'] + row['threat'] + row['insult'] + row['identity_hate']


train['sum_label'] = train.apply(sum_label, axis=1)
print(train.head())

# train.to_csv('../output/output.csv', index=False)

sum_label = train.groupby(['sum_label']).size()
print(sum_label)


train['len_text'] = train.apply(lambda row: len(row['comment_text']), axis=1)
print(train.head())


# plot sum, and len distribution
plt.hist(train['sum_label'], bins=np.arange(train['sum_label'].min(), train['sum_label'].max()+1))
plt.yscale('log')
plt.show()

plt.hist(train['len_text'], bins='auto')
plt.yscale('log')
plt.show()











