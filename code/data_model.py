# encoding=utf8
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, precision_score, recall_score
import sys
from data_preprocess import pre_process
import nltk
from nltk.corpus import stopwords
import pickle
import pandas as pd


def test_transform(test, tf_transformer, model):
    pkl_file = open('../data/lowTF_words.pkl', 'rb')
    lowTF_words = pickle.load(pkl_file)

    porter = nltk.PorterStemmer()
    stops = set(stopwords.words('english'))
    stops.add('rt')
    test['comment_text'] = test['comment_text'].apply(lambda x: x.replace('\n', ' '))

    tweets_new = []
    for index, tweet in test.iterrows():
        words = tweet['comment_text'].split(' ')
        new = []
        for w in words:
            if w not in lowTF_words:
                new.append(w)
        new_tweet = ' '.join(new)
        tweets_new.append(new_tweet)
    test_feats = tf_transformer.transform(tweets_new)
    print(test_feats)
    test_predicts = model.predict(test_feats)
    print(test_predicts)


def main():
    data_dir = '../data'

    print("Loading data...")
    with open(os.path.join(data_dir, 'original_samples_processed.txt'), 'r') as f:
        x = f.readlines()
    with open(os.path.join(data_dir, 'labels.csv'), 'r') as f:
        y = np.array(f.readlines())

    for i in range(len(x)):
        x[i] = x[i][:-1].strip()

    for i in range(len(y)):
        y[i] = y[i][:-1]

    print("Extract features...")
    tf_transformer = TfidfVectorizer().fit(x)
    x_feats = tf_transformer.transform(x)

    print("Start training and predict...")

    model = MultinomialNB().fit(x_feats, y)
    predicts = model.predict(x_feats)

    print(predicts)
    print('testing')
    test_filename = 'mytest.csv'
    test = pd.read_csv(os.path.join(data_dir, test_filename))
    test_transform(test, tf_transformer, model)


if __name__ == '__main__':
    main()