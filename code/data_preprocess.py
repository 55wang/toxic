# encoding=utf8
import sys

reload(sys)
sys.setdefaultencoding('utf8')

import os
import re
import nltk
# nltk.download()
from nltk.corpus import stopwords
import simplejson as json
import pickle
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
import pickle


def rm_html_tags(str):
    html_prog = re.compile(r'<[^>]+>', re.S)
    return html_prog.sub('', str)


def rm_html_escape_characters(str):
    pattern_str = r'&quot;|&amp;|&lt;|&gt;|&nbsp;|&#34;|&#38;|&#60;|&#62;|&#160;|&#20284;|&#30524;|&#26684|&#43;|&#20540|&#23612;'
    escape_characters_prog = re.compile(pattern_str, re.S)
    return escape_characters_prog.sub('', str)


def rm_at_user(str):
    return re.sub(r'@[a-zA-Z_0-9]*', '', str)


def rm_url(str):
    return re.sub(r'http[s]?:[/+]?[a-zA-Z0-9_\.\/]*', '', str)


def rm_repeat_chars(str):
    return re.sub(r'(.)(\1){2,}', r'\1\1', str)


def rm_hashtag_symbol(str):
    return re.sub(r'#', '', str)


def replace_emoticon(emoticon_dict, str):
    for k, v in emoticon_dict.items():
        str = str.replace(k, v)
    return str


def rm_time(str):
    return re.sub(r'[0-9][0-9]:[0-9][0-9]', '', str)


def rm_punctuation(current_tweet):
    return re.sub(r'[^\w\s]', '', current_tweet)


def pre_process(str, porter):
    # do not change the preprocessing order only if you know what you're doing
    str = str.lower()
    str = rm_url(str)
    str = rm_at_user(str)
    str = rm_repeat_chars(str)
    str = rm_hashtag_symbol(str)
    str = rm_time(str)
    str = rm_punctuation(str)

    try:
        str = nltk.tokenize.word_tokenize(str)
        try:
            str = [porter.stem(t) for t in str]
        except:
            print(str)
            pass
    except:
        print(str)
        pass

    return str


if __name__ == "__main__":
    start = time.time()
    data_dir = '../data'  ##Setting your own file path here.

    x_filename = 'train.csv'

    porter = nltk.PorterStemmer()
    stops = set(stopwords.words('english'))
    stops.add('rt')

    ##load and process samples
    print('start loading and process samples...')
    words_stat = {}  # record statistics of the df and tf for each word; Form: {word:[tf, df, tweet index]}
    tweets = []
    cnt = 0

    train = pd.read_csv(os.path.join(data_dir, x_filename))

    # remove some 0 label
    def sum_label(row):
        # print(row)
        return row['toxic'] + row['severe_toxic'] + row['obscene'] + row['threat'] + row['insult'] + row[
            'identity_hate']


    train['len_text'] = train.apply(lambda row: len(row['comment_text']), axis=1)
    train = train[train['len_text'] > 100]

    train['sum_label'] = train.apply(sum_label, axis=1)
    train_label_0 = train[train['sum_label'] == 0]
    train_label_1 = train[train['sum_label'] == 1]
    train_label_2 = train[train['sum_label'] == 2]
    train_label_3 = train[train['sum_label'] == 3]
    train_label_4 = train[train['sum_label'] == 4]
    train_label_5 = train[train['sum_label'] == 5]
    train_label_6 = train[train['sum_label'] == 6]

    train_label_0 = train_label_0.sample(n=2000, random_state=2018)
    train_label_1 = train_label_1.sample(n=2000, random_state=2018)
    train_label_2 = train_label_2.sample(n=1872, random_state=2018)
    print(train_label_0.shape, train_label_1.shape, train_label_2.shape, train_label_3.shape, train_label_4.shape, train_label_5.shape, train_label_6.shape)

    train = pd.concat([train_label_0, train_label_1, train_label_2, train_label_3, train_label_4, train_label_5, train_label_6])
    print(train.shape)

    # save label
    train[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].to_csv('../data/labels.csv', index=False, header=False)

    train['comment_text'] = train['comment_text'].apply(lambda x: x.replace('\n', ' '))

    for index, tweet in tqdm(train.iterrows()):
        postprocess_tweet = []
        words = pre_process(tweet['comment_text'], porter)

        for word in words:
            if word not in stops:
                postprocess_tweet.append(word)
                if word in words_stat.keys():
                    words_stat[word][0] += 1
                    if index != words_stat[word][2]:
                        words_stat[word][1] += 1
                        words_stat[word][2] = index
                else:
                    words_stat[word] = [1, 1, index]
        tweets.append(' '.join(postprocess_tweet))

    ##saving the statistics of tf and df for each words into file
    print("The number of unique words in data set is %i." % len(words_stat.keys()))

    lowTF_words = set()
    with open(os.path.join(data_dir, 'original_words_statistics.txt'), 'w') as f:
        f.write('TF\tDF\tWORD\n')
        for word, stat in sorted(words_stat.items(), key=lambda i: i[1], reverse=True):
            f.write('\t'.join([str(m) for m in stat[0:2]]) + '\t' + word + '\n')
            if stat[0] < 2:
                lowTF_words.add(word)
    print("The number of low frequency words is %d." % len(lowTF_words))
    # print(stops)
    output = open(os.path.join(data_dir, 'lowTF_words.pkl'), 'wb')
    pickle.dump(lowTF_words, output)
    output.close()


    ###Re-process samples, filter low frequency words...
    fout = open(os.path.join(data_dir, 'original_samples_processed.txt'), 'w')
    tweets_new = []
    for tweet in tweets:
        words = tweet.split(' ')
        new = []
        for w in words:
            if w not in lowTF_words:
                new.append(w)
        new_tweet = ' '.join(new)
        tweets_new.append(new_tweet)
        fout.write('%s\n' % new_tweet)
    fout.close()

    print("Preprocessing is completed")
    end = time.time()
    print('{:.3f}'.format((end - start)/60))
