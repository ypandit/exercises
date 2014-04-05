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
from sklearn.cross_validation import StratifiedKFold

from sklearn.datasets.base import Bunch
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
# from sklearn.cross_validation import cross_val_score
from numpy import array, linspace, interp
from sklearn import svm, preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, accuracy_score


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


def feature_extractor(data, scale=False, n_best=50):
    """
    Tf-Idf based feature extraction. Stopwords are removed before transformation.

    :param n_best: int for # best features to be selected. Default 50
    :param data: dict() with keys: train/test
    :param scale: Boolean whether to normalize/scale feature matrix. Default False
    :return: np.darray, np.darray. np.darray
    """
    vectorizer = TfidfVectorizer(stop_words=set(stopwords.words('english')))
    X_train = array(vectorizer.fit_transform(data['train'].data).toarray())
    y_train = array(data['train'].target)
    X_test = array(vectorizer.transform(data['test'].data).toarray())

    ch2 = SelectKBest(chi2, k=n_best)
    X_train = ch2.fit_transform(X_train, y_train)
    X_test = ch2.transform(X_test)
    if scale:
        X_train = preprocessing.scale(X_train)
        X_test = preprocessing.scale(X_test)
    return X_train, y_train, X_test


def train(X_train, y_train):
    """
    Selecting the best classifier from the classification performance test

    :param X_train: np.array feature set
    :param y_train: np.darray labels
    :return: Object of trained classifier
    """
    # clf = svm.LinearSVC(C=0.65)
    clf = LogisticRegression(C=1.0, penalty='l2')

    # cv = 10
    # score = cross_val_score(clf, X_train, y_train, cv=cv)
    # print "Accuracy: {0} (+/- {1})".format(score.mean(), score.std() * 2)

    clf.fit(X_train, y_train)
    return clf


def test(classifier, X_test):
    y_pred = classifier.predict(X_test)
    return y_pred


def classification_perf(X, y):
    """
    Function to compare different classifier. Runs k-fold classification and plots mean (AUC) ROC
    The higher the AUC the higher the true-positive rate for the classifier

    To test the performance of a supervised classifier there should be a gold-standard dataset
    where the samples are labelled. Since we do not have that in this case, we will used cross-fold validation
    on the train data to test the performance of the classifier.

    We are looking at the performance of 6 classifiers with 10-fold cross validation.
    The best one so far has been LogisticRegression(C=1.0, penalty='l2'). 
    Please take a look at the ROC plots

    :param X: np.darray object for feature set
    :param y: np.darray object for target labels
    """
    import pylab as pl

    classifiers = {'L1 LogReg': LogisticRegression(C=1.0, penalty='l1'),
                   'L2 LogReg': LogisticRegression(C=1.0, penalty='l2'),
                   'LinearSVC': svm.SVC(kernel='linear', C=0.65, probability=True, random_state=0),
                   # 'GridSearchCV': grid_search.GridSearchCV(svm.SVC(), [{'kernel': ('linear', 'rbf'), 'C': [1, 10]}]),
                   'DTree': DecisionTreeClassifier(max_depth=None, min_samples_split=1, random_state=0),
                   'RForest': RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=1,
                                                     random_state=0),
                   'ExtraTree': ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=1,
                                                     random_state=0)}

    pl.figure(figsize=(12, 12))
    pl.subplots_adjust(bottom=.2, top=.95)

    folds = 10
    cv = StratifiedKFold(y_train, n_folds=folds)
    for index, (name, classifier) in enumerate(classifiers.iteritems()):
        cl_mean_tpr = 0.0
        cl_mean_fpr = linspace(0, 1, 100)
        cl_accuracy = 0
        for i, (train, test) in enumerate(cv):
            classifier.fit(X[train], y[train])
            probas_ = classifier.predict_proba(X[test])
            y_pred = classifier.predict(X[test])
            cl_accuracy += accuracy_score(y[test], y_pred)
            fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
            cl_mean_tpr += interp(cl_mean_fpr, fpr, tpr)
            cl_mean_tpr[0] = 0.0
        cl_mean_tpr /= len(cv)
        cl_accuracy /= len(cv)
        cl_mean_tpr[-1] = 1.0
        mean_auc = auc(cl_mean_fpr, cl_mean_tpr)
        pl.plot(cl_mean_fpr, cl_mean_tpr, lw=2,
                label='%s - Mean ROC (auc=%0.2f, acc=%0.2f)' % (name, mean_auc, cl_accuracy))

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
    X_train, y_train, X_test = feature_extractor(cache, scale=True, n_best=50)

    # classification_perf(X_train, y_train)
    classifier = train(X_train, y_train)
    y_pred = test(classifier, X_test)
    print y_pred