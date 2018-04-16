# encoding=utf8
# import sys

# reload(sys)
# sys.setdefaultencoding('utf8')

import os
import re
import nltk
# nltk.download()
from nltk.corpus import stopwords
from nltk import bigrams
import string
import simplejson as json
import pickle
import numpy as np
import pandas as pd
import time
from sklearn.feature_extraction.text import CountVectorizer
from string import ascii_letters, punctuation
from tqdm import tqdm


def rm_html_tags(str1):
    html_prog = re.compile(r'<[^>]+>', re.S)
    return html_prog.sub('', str1)


def rm_html_escape_characters(str1):
    pattern_str = r'&quot;|&amp;|&lt;|&gt;|&nbsp;|&#34;|&#38;|&#60;|&#62;|&#160;|&#20284;|&#30524;|&#26684|&#43;|&#20540|&#23612;'
    escape_characters_prog = re.compile(pattern_str, re.S)
    return escape_characters_prog.sub('', str1)


def rm_at_user(str1):
    return re.sub(r'@[a-zA-Z_0-9]*', '', str1)


def rm_url(str1):
    return re.sub(r'http[s]?:[/+]?[a-zA-Z0-9_\.\/]*', '', str1)


def rm_repeat_chars(str1):
    return re.sub(r'(.)(\1){2,}', r'\1\1', str1)


def rm_hashtag_symbol(str1):
    return re.sub(r'#', '', str1)


def replace_emoticon(emoticon_dict, str1):
    for k, v in emoticon_dict.items():
        str1 = str1.replace(k, v)
    return str1


def rm_time(str1):
    return re.sub(r'[0-9][0-9]:[0-9][0-9]', '', str1)


def rm_punctuation(str1):
    return re.sub(r'[^\w\s]', '', str1)


def rm_ip(str1):
    return re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", "", str1)


def rm_username(str1):
    return re.sub("\[\[.*\]", "", str1)


def pre_process(str1, porter):
    str1 = str1.lower()
    str1 = rm_url(str1)
    str1 = rm_ip(str1)
    str1 = rm_username(str1)
    str1 = rm_at_user(str1)
    str1 = rm_repeat_chars(str1)
    str1 = rm_hashtag_symbol(str1)
    str1 = rm_time(str1)
    str1 = rm_punctuation(str1)

    try:
        str1 = nltk.tokenize.word_tokenize(str1)
        try:
            str1 = [porter.stem(t) for t in str1]
        except:
            print(str1)
            pass
    except:
        print(str1)
        pass

    return str1


# Feature Engineering
def features(df):
    df['count_sent'] = df["comment_text"].apply(lambda x: len(re.findall("\n", str(x))) + 1)

    # Word count in each comment:
    df['count_word'] = df["comment_text"].apply(lambda x: len(str(x).split()))

    # Unique word count
    df['count_unique_word'] = df["comment_text"].apply(lambda x: len(set(str(x).split())))

    # Letter count
    df['count_letters'] = df["comment_text"].apply(lambda x: len(str(x)))

    # punctuation count
    df["count_punctuations"] = df["comment_text"].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))

    # upper case words count
    df["count_words_upper"] = df["comment_text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))

    # title case words count
    df["count_words_title"] = df["comment_text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))

    # Number of stopwords
    eng_stopwords = set(stopwords.words("english"))
    df["count_stopwords"] = df["comment_text"].apply(
        lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))

    # Average length of the words
    df["mean_word_len"] = df["comment_text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
    return df


if __name__ == "__main__":
    start = time.time()
    data_dir = './data/'  ##Setting your own file path here.

    x_filename = 'train.csv'
    test_filename = 'test.csv'

    porter = nltk.PorterStemmer()
    stops = set(stopwords.words('english'))
    stops.add('rt')

    ##load and process samples
    print('start loading and process samples...')
    words_stat = {}  # record statistics of the df and tf for each word; Form: {word:[tf, df, tweet index]}
    tweets = []
    cnt = 0

    train = pd.read_csv(os.path.join(data_dir, x_filename))


    # train.describe()

    # remove some 0 label
    def sum_label(row):
        # print(row)
        return row['toxic'] + row['severe_toxic'] + row['obscene'] + row['threat'] + row['insult'] + row[
            'identity_hate']


    # remove text shorter than x characters (do we really need? some toxic are short)
    train['len_text'] = train.apply(lambda row: len(row['comment_text']), axis=1)
    train = train[train['len_text'] > 100]

    # undersample to correct skewed class
    train['sum_label'] = train.apply(sum_label, axis=1)
    train_clean = train[train['sum_label'] == 0].sample(n=8766, random_state=2018)
    train_tox = train[train['toxic'] == 1]
    #    train_clean = train[train['sum_label'] == 0].sample(n=14167, random_state=2018)
    #    train_tox = train[train['toxic'] == 1]

    df_train = pd.concat([train_clean, train_tox])
    df_train = np.take(df_train, np.random.permutation(df_train.shape[0]), axis=0)
    print(df_train.shape)

    #    # feature engineering
    #    df_train = features(df_train)
    #
    #    output = open(os.path.join(data_dir, 'features.pkl'), 'wb')
    #    pickle.dump(df_train.iloc[:, 10:20], output)
    #    output.close()

    # save label
    #    df_train[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].to_csv('./data/labels.csv', index=False, header=True)
    df_train[['toxic']].to_csv('./data/labels.csv', index=False, header=False)
    df_train['comment_text'] = train['comment_text'].apply(lambda x: x.replace('\n', ' '))

    for index, tweet in tqdm(df_train.iterrows()):
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
    with open(os.path.join(data_dir, 'original_words_statistics.txt'), 'w', encoding='utf8') as f:
        f.write('TF\tDF\tWORD\n')
        for word, stat in sorted(words_stat.items(), key=lambda i: i[1], reverse=True):
            f.write('\t'.join([str(m) for m in stat[0:2]]) + '\t' + word + '\n')
            if stat[0] < 3:
                lowTF_words.add(word)

    print("The number of low frequency words is %d." % len(lowTF_words))
    # print(stops)

    ###Re-process samples, filter low frequency words...
    fout = open(os.path.join(data_dir, 'original_samples_processed.txt'), 'w', encoding='utf8')
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
    print('{:.3f}'.format((end - start) / 60))
