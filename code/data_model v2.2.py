# encoding=utf8
import os
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_score, recall_score
# from data_preprocess import pre_process
# import nltk
# from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def main():
    data_dir = '../data'

    print("Loading data...")
    with open(os.path.join(data_dir, 'original_samples_processed.txt'), 'r') as f:
        x = f.readlines()
    with open(os.path.join(data_dir, 'labels.csv'), 'r') as f:
        y = np.array(f.readlines())

    #    labels = pd.read_csv(os.path.join(data_dir, 'labels.csv'))
    #    y = labels.iloc[:,0]

    # pkl_file = open('../data/features.pkl', 'rb')
    # features = pickle.load(pkl_file)

    for i in range(len(x)):
        x[i] = x[i][:-1].strip()

    #    for i in range(len(y)):
    #        y[i] = y[i][:-1]

    print("Extract features...")
    tf_transformer = TfidfVectorizer().fit(x)
    x_feats = tf_transformer.transform(x)
    print(x_feats.shape)
    #    x_feats_df = pd.DataFrame(x_feats)

    print("Split train and test data")
    # msk = np.random.rand(len(y)) < 0.8
    x_train, x_test, y_train, y_test = train_test_split(x_feats, y, test_size=0.2, random_state=2018)

    # x_train = x_feats[msk]
    # y_train = y[msk]
    #
    # x_test = x_feats[~msk]
    # y_test = y[~msk]

    print("Start training and predict...")
    # split k folds
    kf = KFold(n_splits=10)

    # Model selection using 10-fold CV
    # NB classifier
    avg_p = 0
    avg_r = 0
    for trainn, testn in kf.split(x_train):
        #       print(trainn)
        model = MultinomialNB().fit(x_train[trainn], y_train[trainn])
        predicts = model.predict(x_train[testn])
        # print(classification_report(y_train[testn],predicts))
        avg_p += precision_score(y_train[testn], predicts, average='macro')
        avg_r += recall_score(y_train[testn], predicts, average='macro')

    print('Average Precision is %f.' % (avg_p / 10.0))
    print('Average Recall is %f.' % (avg_r / 10.0))

    # RF classifier
    avg_p = 0
    avg_r = 0
    for trainn, testn in kf.split(x_train):
        #       print(trainn)
        model = RandomForestClassifier().fit(x_train[trainn], y_train[trainn])
        predicts = model.predict(x_train[testn])
        # print(classification_report(y_train[testn],predicts))
        avg_p += precision_score(y_train[testn], predicts, average='macro')
        avg_r += recall_score(y_train[testn], predicts, average='macro')

    print('Average Precision is %f.' % (avg_p / 10.0))
    print('Average Recall is %f.' % (avg_r / 10.0))

    # KNN classifier
    avg_p = 0
    avg_r = 0
    for trainn, testn in kf.split(x_train):
        #       print(trainn)
        model = KNeighborsClassifier().fit(x_train[trainn], y_train[trainn])
        predicts = model.predict(x_train[testn])
        # print(classification_report(y_train[testn],predicts))
        avg_p += precision_score(y_train[testn], predicts, average='macro')
        avg_r += recall_score(y_train[testn], predicts, average='macro')

    print('Average Precision is %f.' % (avg_p / 10.0))
    print('Average Recall is %f.' % (avg_r / 10.0))

    # SVM classifier
    avg_p = 0
    avg_r = 0
    for trainn, testn in kf.split(x_train):
        #       print(trainn)
        model = SVC(kernel='linear', C=1, random_state=2018).fit(x_train[trainn], y_train[trainn])
        predicts = model.predict(x_train[testn])
        # print(classification_report(y_train[testn],predicts))
        avg_p += precision_score(y_train[testn], predicts, average='macro')
        avg_r += recall_score(y_train[testn], predicts, average='macro')

    print('Average Precision is %f.' % (avg_p / 10.0))
    print('Average Recall is %f.' % (avg_r / 10.0))

    # Use model for final prediction
    avg_p = 0
    avg_r = 0
    model = MultinomialNB().fit(x_train, y_train)
    predicts = model.predict(x_test)
    print(classification_report(y_test, predicts))
    avg_p = precision_score(y_test, predicts, average='macro')
    avg_r = recall_score(y_test, predicts, average='macro')

    print('Test Precision is %f.' % (avg_p))
    print('Test Recall is %f.' % (avg_r))

    output = open(os.path.join(data_dir, 'model_transformer.pkl'), 'wb')
    pickle.dump([tf_transformer, model], output)
    output.close()


if __name__ == '__main__':
    main()
