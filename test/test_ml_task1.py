import collections

from ml.mywork import task1

# TODO: set_up() for loading data (fd, sentences)

def test_process():
    test_file = "test/test_ml_task1_data.txt"
    fd, sentences = task1.process(f=test_file)
    assert len(sentences) == 10, "correct len() of processed data"
    assert fd != None, "FreqDist should not be None"
    assert ['and', 'for', 'it'] not in fd.keys(), "stopwords should be removed"


def test_get_popular_term():
    test_file = "test/test_ml_task1_data.txt"
    fd, sentences = task1.process(f=test_file)
    # TODO: test to check if exception is raised for no argument

    mpt = task1.get_most_popular_term(fd)
    assert type(mpt) == tuple, "should be a tuple"
    lpt = task1.get_least_popular_term(fd)
    assert type(lpt) == tuple, "should be a tuple"


def test_get_types_of_guitar():
    test_file = "test/test_ml_task1_data.txt"
    fd, sentences = task1.process(f=test_file)

    C = task1.get_types_of_guitar(sentences)
    assert type(C) == collections.Counter, "should be a tuple"
    if C is not None:
        assert 'guitar' in C.items()[0][0], "should be a guitar"