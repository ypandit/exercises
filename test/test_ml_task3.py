import numpy as np

from ml.mywork import task3


def test_load_data():
    test_file = "test/test_ml_task3_data.txt"
    dat_mat = task3.load_data(target=1, f=test_file, subset='train')
    assert len(dat_mat) == 2, "dimensions should be 2"
    assert type(dat_mat.data[0]) == str, "object should be string"
    assert len(dat_mat.target) == 58, "#samples should be 58"
    dat_mat = task3.load_data(target=1, f=test_file, df=dat_mat, subset='train')
    assert len(dat_mat.target) == 116, "#samples should be 116"
    dat_mat = task3.load_data(target=0, f=test_file, subset='test')
    assert dat_mat.target == None, "target should be None for test data"


def test_feature_extractor():
    test_file = "test/test_ml_task3_data.txt"
    dat_mat_train = task3.load_data(target=1, f=test_file, subset='train')
    dat_mat_test = task3.load_data(target=0, f=test_file, subset='test')
    cache = {'train': dat_mat_train, 'test': dat_mat_test}
    X_train, y_train, X_test = task3.feature_extractor(cache)
    assert type(y_train) == np.ndarray, "target object should np.darray"
    assert len(X_train) == len(y_train), "n_samples should be equal to n_labels"
    assert X_train.shape[1] == X_test.shape[1], "n_features should be same on train & test"
    assert X_train.shape[1] == len(X_train[0]), "n_features should be equal to len() of featureset"