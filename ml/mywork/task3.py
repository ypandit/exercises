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
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.cross_validation import cross_val_score, StratifiedKFold
from numpy import array, linspace
from sklearn import svm, preprocessing
import pylab as pl
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import roc_curve, auc
from scipy import interp


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


def train(classifier, X_train, y_train, verbose=True):
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


def classification_perf(X, y):
    classifiers = {'L1 LogReg': LogisticRegression(C=1.0, penalty='l1'),
                   'L2 LogReg': LogisticRegression(C=1.0, penalty='l2'),
                   'LinearSVC': svm.SVC(kernel='linear', C=0.65, probability=True, random_state=0),
                   # 'GridSearchCV': grid_search.GridSearchCV(svm.SVC(), [{'kernel':('linear', 'rbf'), 'C':[1, 10]}])}
                   'DTree': DecisionTreeClassifier(max_depth=None, min_samples_split=1, random_state=0),
                   'RForest': RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=1,
                                                     random_state=0),
                   # 'SGD': SGDClassifier(),
                   'ExtraTree': ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=1,
                                                     random_state=0)}

    n_classifiers = len(classifiers)

    pl.figure(figsize=(3 * 2, n_classifiers * 2))
    pl.subplots_adjust(bottom=.2, top=.95)

    for index, (name, classifier) in enumerate(classifiers.iteritems()):
        folds = 3
        cv = StratifiedKFold(y_train, n_folds=folds)
        cl_mean_tpr = 0.0
        cl_mean_fpr = linspace(0, 1, 100)
        cl_all_tpr = []
        for i, (train, test) in enumerate(cv):
            probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
            fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
            cl_mean_tpr += interp(cl_mean_fpr, fpr, tpr)
            cl_mean_tpr[0] = 0.0
        cl_mean_tpr /= len(cv)
        cl_mean_tpr[-1] = 1.0
        mean_auc = auc(cl_mean_fpr, cl_mean_tpr)
        pl.plot(cl_mean_fpr, cl_mean_tpr, lw=2, label='%s - Mean ROC (%d) (area = %0.2f)' % (name, folds, mean_auc))

    pl.xlim([-0.05, 1.05])
    pl.ylim([-0.05, 1.05])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title('ROC for deals\' classifier')
    pl.legend(loc="lower right")
    pl.show()


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

    classification_perf(X_train, y_train)