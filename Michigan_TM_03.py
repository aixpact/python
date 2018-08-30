

import pandas as pd
import numpy as np

spam_data = pd.read_csv('spam.csv')

spam_data['target'] = np.where(spam_data['target']=='spam',1,0)
spam_data.head(10)

from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(spam_data['text'],
                                                    spam_data['target'],
                                                    random_state=0)

def answer_one():
    return spam_data['target'].mean() * 100


from sklearn.feature_extraction.text import CountVectorizer

def answer_two():

    vect = CountVectorizer().fit(X_train)

    names = sorted(vect.get_feature_names(), key=lambda x :len(x), reverse=True)
    # sort by length, return longest token


    return names[0]

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score

def answer_three():

    vect = CountVectorizer().fit(X_train)
    X_train_vectorized = vect.transform(X_train)
    X_test_vectorized = vect.transform(X_test)

    nb = MultinomialNB(alpha=0.1).fit(X_train_vectorized, y_train)
    y_predict = nb.predict(X_test_vectorized)
    AUC = roc_auc_score(y_test, y_predict)

    return AUC

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer


def answer_four():
    vect = TfidfVectorizer().fit(X_train)
    X_train_vectorized = vect.transform(X_train)
    feature_names = np.array(vect.get_feature_names())
    tfidf_values = X_train_vectorized.max(0).toarray()[0]

    df = pd.DataFrame([feature_names, tfidf_values], index=['feature', 'tf-idf']).T
    smallest_tfidfs = df.sort_values(by=['tf-idf', 'feature'], ascending=[1, 1]).set_index('tf-idf').head(20)
    largest_tfidfs = df.sort_values(by=['tf-idf', 'feature'], ascending=[0, 1]).set_index('tf-idf').head(20)

    return smallest_tfidfs, largest_tfidfs


def answer_five():
    vect = TfidfVectorizer(min_df=3).fit(X_train)
    X_train_vectorized = vect.transform(X_train)

    X_test_vectorized = vect.transform(X_test)

    nb = MultinomialNB(alpha=0.1).fit(X_train_vectorized, y_train)
    y_predict = nb.predict(X_test_vectorized)
    AUC = roc_auc_score(y_test, y_predict)

    return AUC


def answer_six():
    spam_index = spam_data['target'] == 1
    spam_data['len'] = [len(x) for x in spam_data['text']]

    avg_len_ham = spam_data.loc[~spam_index, 'len'].mean()
    avg_len_spam = spam_data.loc[spam_index, 'len'].mean()

    return avg_len_ham, avg_len_spam


def add_feature(X, feature_to_add):
    """
    Returns sparse feature matrix with added feature.
    feature_to_add can also be a list of features.
    """
    from scipy.sparse import csr_matrix, hstack
    return hstack([X, csr_matrix(feature_to_add).T], 'csr')


from sklearn.svm import SVC


def answer_seven():
    vect = TfidfVectorizer(min_df=6).fit(X_train)
    X_train_vectorized = vect.transform(X_train)
    X_test_vectorized = vect.transform(X_test)
    doc_len_train = [len(x) for x in X_train]
    doc_len_test = [len(x) for x in X_test]

    X_train_vectorized = add_feature(X_train_vectorized, doc_len_train)
    X_test_vectorized = add_feature(X_test_vectorized, doc_len_test)

    fit = SVC(C=10000).fit(X_train_vectorized, y_train)
    y_pred = fit.predict(X_test_vectorized)
    AUC = roc_auc_score(y_test, y_pred)

    return AUC


def answer_eight():
    import re
    spam_index = spam_data['target'] == 1
    spam_data['len'] = [len(''.join(re.findall('\d+', x))) for x in spam_data['text']]

    avg_len_ham = spam_data.loc[~spam_index, 'len'].mean()
    avg_len_spam = spam_data.loc[spam_index, 'len'].mean()

    return avg_len_ham, avg_len_spam


from sklearn.linear_model import LogisticRegression


def answer_nine():
    import re

    vect = TfidfVectorizer(min_df=6, ngram_range=(1, 3)).fit(X_train)
    X_train_vectorized = vect.transform(X_train)
    X_test_vectorized = vect.transform(X_test)

    doc_len_train = [len(x) for x in X_train]
    doc_len_test = [len(x) for x in X_test]

    dig_len_train = [len(''.join(re.findall('\d+', x))) for x in X_train]
    dig_len_test = [len(''.join(re.findall('\d+', x))) for x in X_test]

    X_train_vectorized = add_feature(X_train_vectorized, doc_len_train)
    X_test_vectorized = add_feature(X_test_vectorized, doc_len_test)

    X_train_vectorized = add_feature(X_train_vectorized, dig_len_train)
    X_test_vectorized = add_feature(X_test_vectorized, dig_len_test)

    lr = LogisticRegression(C=100).fit(X_train_vectorized, y_train)
    y_pred = lr.predict(X_test_vectorized)

    AUC = roc_auc_score(y_test, y_pred)

    return AUC


def answer_ten():
    import re

    spam_index = spam_data['target'] == 1
    spam_data['len_'] = [len(''.join(re.findall('\W+', x))) for x in spam_data['text']]

    avg_len_ham = spam_data.loc[~spam_index, 'len_'].mean()
    avg_len_spam = spam_data.loc[spam_index, 'len_'].mean()

    return avg_len_ham, avg_len_spam


def answer_eleven():
    import re

    vect = CountVectorizer(min_df=6, ngram_range=(2, 5), analyzer='char_wb').fit(X_train)
    X_train_vectorized = vect.transform(X_train)
    X_test_vectorized = vect.transform(X_test)

    # Train
    ftrs = pd.DataFrame([(len(x), len(''.join(re.findall('\d+', x))), len(''.join(re.findall('\W+', x))))
                         for x in X_train], columns=['length_of_doc', 'digit_count', 'non_word_char_count'])

    X_train_vectorized = add_feature(X_train_vectorized, [ftrs.loc[:, 'length_of_doc'],
                                                          ftrs.loc[:, 'digit_count'],
                                                          ftrs.loc[:, 'non_word_char_count']])

    # Test
    ftrs = pd.DataFrame([(len(x), len(''.join(re.findall('\d+', x))), len(''.join(re.findall('\W+', x))))
                         for x in X_test], columns=['length_of_doc', 'digit_count', 'non_word_char_count'])

    X_test_vectorized = add_feature(X_test_vectorized, [ftrs.loc[:, 'length_of_doc'],
                                                        ftrs.loc[:, 'digit_count'],
                                                        ftrs.loc[:, 'non_word_char_count']])

    # Fit
    lr = LogisticRegression(C=100).fit(X_train_vectorized, y_train)
    y_predict = lr.predict(X_test_vectorized)
    AUC = roc_auc_score(y_test, y_predict)

    #
    feature_adds = ['length_of_doc', 'digit_count', 'non_word_char_count']
    feature_names = np.array(vect.get_feature_names())
    feature_names = np.append(feature_names, feature_adds)
    sorted_coef_index = lr.coef_[0].argsort()

    smallest_coefs = feature_names[sorted_coef_index[:10]].tolist()
    largest_coefs = feature_names[sorted_coef_index[:-11:-1]].tolist()

    return AUC, smallest_coefs, largest_coefs