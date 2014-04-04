""" Classification

The objective of this task is to build a classifier that can tell us whether a new, unseen deal 
requires a coupon code or not. 

We would like to see a couple of steps:

1. You should use bad_deals.txt and good_deals.txt as training data
2. You should use test_deals.txt to test your code
3. Each time you tune your code, please commit that change so we can see your tuning over time

Also, provide comments on:
    - How general is your classifier?
    - How did you test your classifier?

"""

from sklearn.datasets.base import Bunch
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.cross_validation import cross_val_score
from numpy import array
from sklearn import svm, preprocessing


def load_data(f, subset, target=None, df=None):
    """
    Custom data loader for good/bad/test deals

    :param f: Input data file
    :param subset: Either train/test.For test, target is set to None
    :param target: target for the provided input, 0/1/None. Default None
    :param df: Existing Bunch() to append new data. Default None
    :return: Bunch()
    """
    if subset == 'train' and target is None:
        raise ValueError("target not specified for train")
    docs = [w.strip().lower() for w in open(f, 'r').readlines()]
    if subset == 'test':
        labels = None
    else:
        labels = [target] * len(docs)
    if df is not None:
        df.data.extend(docs)
        df.target.extend(labels)
    else:
        df = Bunch(data=docs, target=labels)
    return df


def feature_extractor(data, scale=False):
    """
    Tf-Idf based feature extraction. Stopwords are removed before transformation.

    :param data: dict() with keys: train/test
    :param scale: Boolean whether to normalize/scale feature matrix. Default False
    :return: np.darray, np.darray. np.darray
    """
    vectorizer = TfidfVectorizer(stop_words=set(stopwords.words('english')))
    X_train = array(vectorizer.fit_transform(data['train'].data).toarray())
    y_train = array(data['train'].target)
    X_test = array(vectorizer.transform(data['test'].data).toarray())
    if scale:
        X_train = preprocessing.scale(X_train)
        X_test = preprocessing.scale(X_test)
    return X_train, y_train, X_test


def train(X_train, y_train, verbose=True):
    clf = svm.LinearSVC(C=0.65)
    cv = 10
    scores = cross_val_score(clf, X_train, y_train, cv=cv)
    if verbose:
        print "Sample size: {0}".format(len(X_train))
        print "Cross-validation: {0} folds".format(cv)
        print "Accuracy: {0} (+/- {1})".format(scores.mean(), scores.std() * 2)
    return clf


def test(classifier, X_test):
    pass


if __name__ == "__main__":
    good_deals_file = '../data/good_deals.txt'
    train_data = load_data(target=1, f=good_deals_file, subset='train')
    assert len(train_data.data) == 30

    bad_deals_file = '../data/bad_deals.txt'
    train_data = load_data(df=train_data, target=0, f=bad_deals_file, subset='train')
    assert len(train_data.data) == 60

    test_file = '../data/test_deals.txt'
    test_data = load_data(f=test_file, subset='test')
    assert len(test_data.data) == 58

    cache = {'train': train_data, 'test': test_data}
    X_train, y_train, X_test = feature_extractor(cache, scale=False)

    classifier = train(X_train, y_train)
    test(classifier, X_test)