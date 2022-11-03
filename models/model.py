from Utils.Datasets import Emotion_Dataset, Emotion_Testset

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.feature_extraction.text import CountVectorizer

import numpy as np
import sys
import argparse
import pickle
import json
import re
import gzip
import os

import scipy
from collections import defaultdict
from scipy.stats import pearsonr

from nltk import ngrams


class Vocab(defaultdict):
    def __init__(self, train=True):
        super(Vocab, self).__init__(lambda : len(self))
        self.train = train
        self.UNK = "UNK"
        # set UNK token to 0 index
        self[self.UNK]

    def ws2ids(self, ws):
        """ If train, you can use the default dict to add tokens
            to the vocabulary, given these will be updated during
            training. Otherwise, we replace them with UNK.
        """
        if self.train:
            return [self[w] for w in ws]
        else:
            return [self[w] if w in self else 0 for w in ws]

    def ids2sent(self, ids):
        idx2w = dict([(i, w) for w, i in self.items()])
        return [idx2w[int(i)] if i in idx2w else "UNK" for i in ids]


def open_nrc_sentiment(file="lexicons/NRC-Hashtag-Sentiment-Lexicon-v0.1/unigrams-pmilexicon.txt.gz"):
    lex = {}
    for line in gzip.open(file, mode="r"):
        line = line.decode()
        word, score, count, _ = line.strip().split("\t")
        lex[word] = float(score)
    return lex

def open_nrc_emotion(file="lexicons/lexicons/NRC-emotion-lexicon-wordlevel-v0.92.txt.gz"):
    lex = {}
    for line in gzip.open(file, mode="r"):
        line = line.decode()
        word, emotion, score = line.strip().split("\t")
        if word not in lex:
            lex[word] = []
            lex[word].append(int(score))
        else:
            lex[word].append(int(score))
    return lex

def open_nrc_hashtag(file="lexicons/lexicons/NRC-Hashtag-Emotion-Lexicon-v0.2.txt.gz"):
    lex = {}
    emotions = {"anticipation":0,
                "fear":1,
                "anger":2,
                "trust":3,
                "surprise":4,
                "sadness":5,
                "joy":6,
                "disgust":7,
                }
    for line in gzip.open(file, mode="r"):
        try:
            line = line.decode()
            emotion, word, score = line.strip().split("\t")
            if word not in lex:
                lex[word] = np.zeros(len(emotions))
                lex[word][emotions[emotion]] = float(score)
            else:
                lex[word][emotions[emotion]] = float(score)
        # for lines at the beginning, just ignore them
        except:
            pass
    return lex

def get_numerical_lexicon_features(split, lexicons=[None]):
    X = []
    for tweet in split:
        # we create a feature for both positive and negative words
        # for all the lexicons we have
        features = np.zeros(2 * len(lexicons))
        for i, lexicon in enumerate(lexicons):
            for w in tweet.split():
                if w in lexicon:
                    if lexicon[w] > 0:
                        features[i*2] += lexicon[w]
                    else:
                        features[(i*2)+1] += lexicon[w]
        X.append(features)
    return np.array(X)

def get_emotion_lexicon_features(split,
                                 lexicons=[None],
                                 num_emotions=10):
    X = []
    for tweet in split:
        # we create a feature vector for all emotion categories
        features = np.zeros(num_emotions)
        for i, lexicon in enumerate(lexicons):
            for w in tweet.split():
                if w in lexicon:
                    features += lexicon[w]
        X.append(features)
    return np.array(X)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-sd', '--src_dataset', default="../dataset/en")
    parser.add_argument('-td', '--trg_dataset', default="../dataset/es")
    parser.add_argument('-emo', '--emotion', default="anger")
    parser.add_argument('-f', '--features', nargs='+', default=["ngrams"])


    args = parser.parse_args()


    # open datasets
    src_dataset = Emotion_Dataset(args.src_dataset,
                                  emotion=args.emotion)
    print('src_dataset done')
    trg_dataset = Emotion_Testset(args.trg_dataset,
                                  emotion=args.emotion)
    print('trg_dataset done')

    if "ngrams" in args.features:
        # get ngram (1 to 4-grams) representations
        ngrams_vectorizer = CountVectorizer(ngram_range=(1,4))
        ngrams_vectorizer.fit(src_dataset.data["train"]["text"])

        Xtrain_ngrams = ngrams_vectorizer.transform(src_dataset.data["train"]["text"])
        Xdev_ngrams = ngrams_vectorizer.transform(src_dataset.data["dev"]["text"])
        Xstest_ngrams = ngrams_vectorizer.transform(src_dataset.data["test"]["text"])
        Xtest_ngrams = ngrams_vectorizer.transform(trg_dataset.data["test"]["text"])

    if "char_ngrams" in args.features:
        # get character ngrams (3-5)
        char_vectorizer = CountVectorizer(analyzer="char_wb", ngram_range=(3,5))
        char_vectorizer = char_vectorizer.fit(src_dataset.data["train"]["text"])
        Xtrain_char = char_vectorizer.transform(src_dataset.data["train"]["text"])
        Xdev_char = char_vectorizer.transform(src_dataset.data["dev"]["text"])
        Xstest_char = char_vectorizer.transform(src_dataset.data["test"]["text"])
        Xtest_char = char_vectorizer.transform(trg_dataset.data["test"]["text"])


    # get lexicon information
    ###################################################################
    nrc_sent_unigrams = open_nrc_sentiment()
    nrc_emotion = open_nrc_emotion()
    nrc_hashtag = open_nrc_hashtag()

    if "NRC_sent" in args.features:
        # get sentiment features
        Xtrain_lex = get_numerical_lexicon_features(src_dataset.data["train"]["text"], lexicons=[nrc_sent_unigrams])

        Xdev_lex = get_numerical_lexicon_features(src_dataset.data["dev"]["text"], lexicons=[nrc_sent_unigrams])

        Xstest_lex = get_numerical_lexicon_features(src_dataset.data["test"]["text"], lexicons=[nrc_sent_unigrams])

        Xtest_lex = get_numerical_lexicon_features(trg_dataset.data["test"]["text"], lexicons=[nrc_sent_unigrams])

    if "NRC_emo" in args.features:
        # get emotion features
        Xtrain_emo_lex = get_emotion_lexicon_features(src_dataset.data["train"]["text"], lexicons=[nrc_emotion])

        Xdev_emo_lex = get_emotion_lexicon_features(src_dataset.data["dev"]["text"], lexicons=[nrc_emotion])

        Xstest_emo_lex = get_emotion_lexicon_features(src_dataset.data["test"]["text"], lexicons=[nrc_emotion])

        Xtest_emo_lex = get_emotion_lexicon_features(trg_dataset.data["test"]["text"], lexicons=[nrc_emotion])

    if "NRC_hash" in args.features:
        # get hashtag emotion features
        Xtrain_emo_hash = get_emotion_lexicon_features(src_dataset.data["train"]["text"], lexicons=[nrc_hashtag], num_emotions=8)

        Xdev_emo_hash = get_emotion_lexicon_features(src_dataset.data["dev"]["text"], lexicons=[nrc_hashtag], num_emotions=8)

        Xstest_emo_hash = get_emotion_lexicon_features(src_dataset.data["test"]["text"], lexicons=[nrc_hashtag], num_emotions=8)

        Xtest_emo_hash = get_emotion_lexicon_features(trg_dataset.data["test"]["text"], lexicons=[nrc_hashtag], num_emotions=8)

    train_feats = []
    if "ngrams" in args.features:
        train_feats.append(Xtrain_ngrams)
    if "char_ngrams" in args.features:
        train_feats.append(Xtrain_char)
    if "NRC_sent" in args.features:
        train_feats.append(Xtrain_lex)
    if "NRC_emo" in args.features:
        train_feats.append(Xtrain_emo_lex)
    if "NRC_hash" in args.features:
        train_feats.append(Xtrain_emo_hash)

    dev_feats = []
    if "ngrams" in args.features:
        dev_feats.append(Xdev_ngrams)
    if "char_ngrams" in args.features:
        dev_feats.append(Xdev_char)
    if "NRC_sent" in args.features:
        dev_feats.append(Xdev_lex)
    if "NRC_emo" in args.features:
        dev_feats.append(Xdev_emo_lex)
    if "NRC_hash" in args.features:
        dev_feats.append(Xdev_emo_hash)

    stest_feats = []
    if "ngrams" in args.features:
        stest_feats.append(Xstest_ngrams)
    if "char_ngrams" in args.features:
        stest_feats.append(Xstest_char)
    if "NRC_sent" in args.features:
        stest_feats.append(Xstest_lex)
    if "NRC_emo" in args.features:
        stest_feats.append(Xstest_emo_lex)
    if "NRC_hash" in args.features:
        stest_feats.append(Xstest_emo_hash)

    test_feats = []
    if "ngrams" in args.features:
        test_feats.append(Xtest_ngrams)
    if "char_ngrams" in args.features:
        test_feats.append(Xtest_char)
    if "NRC_sent" in args.features:
        test_feats.append(Xtest_lex)
    if "NRC_emo" in args.features:
        test_feats.append(Xtest_emo_lex)
    if "NRC_hash" in args.features:
        test_feats.append(Xtest_emo_hash)

    # concatenate all the vector representations
    Xtrain = scipy.sparse.hstack(train_feats)
    Xdev = scipy.sparse.hstack(dev_feats)
    Xstest = scipy.sparse.hstack(stest_feats)
    Xtest = scipy.sparse.hstack(test_feats)

    # train Support Vector Regression on source
    print('Training SVR...')
    clf = SVR(C=100, kernel="linear")
    history = clf.fit(Xtrain, src_dataset.data["train"]["labels"])

    # test on src devset and trg devset
    src_pred = clf.predict(Xstest)
    score, p = pearsonr(src_dataset.data["test"]["labels"], src_pred)
    print("SRC-SRC: {0:.2f} ({1:.2f})".format(score, p))

    trg_pred = clf.predict(Xtest)
    score, p = pearsonr(trg_dataset.data["test"]["labels"], trg_pred)
    print("SRC-TRG: {0:.2f} ({1:.2f})".format(score, p))
