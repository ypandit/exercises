""" Features

The objective of this task is to explore the corpus, deals.txt. 

The deals.txt file is a collection of deal descriptions, separated by a new line, from which 
we want to glean the following insights:

1. What is the most popular term across all the deals?
2. What is the least popular term across all the deals?
3. How many types of guitars are mentioned across all the deals?

"""

import string
import collections

from nltk.tokenize.punkt import PunktWordTokenizer
from nltk.corpus import stopwords
from nltk import FreqDist


def process(f):
    """
    Function to process deals data.
    Splits text into sentences. FreqDist is incremented from tokenization.
    Using PunktWordTokenizer, since it is a decent regexp-based tokenizer.
    Deals are also about domain names. Not intending to split it up

    :rtype : FreqDist, list() of str
    :param f: Input file with a deal per line
    """
    fd = FreqDist()
    sentences = [line.strip() for line in open(f, 'r').readlines()]
    for line in sentences:
        for word in PunktWordTokenizer().tokenize(line.lower()):
            if word not in set(stopwords.words('english')) and word not in set(string.punctuation):
                fd.inc(word)
    return fd, sentences


def get_most_popular_term(freq_dist):
    """
    Returns the term,freq tuple with highest frequency from FreqDist

    :param freq_dist: FreqDist object with token frequencies
    :return: tuple
    """
    return freq_dist.items()[0]


def get_least_popular_term(freq_dist):
    """
    Returns the term,freq tuple with lowest frequency from FreqDist

    :param freq_dist: FreqDist object with token frequencies
    :return: tuple
    """
    return freq_dist.items()[-1]


def get_types_of_guitar(sentences):
    """
    Return a type,freq list of tuples from different types of guitars.

    The question can be dealt with in two ways:
        1. Deals on guitars itself
        2. Deals on guitar lessons
    For this task just counting the mentions of types of guitar, irrespective of the type of deal

    Also a lot of the mentions here are for ways to play the guitar
    e.g. gypsy, fingerstyle, flatpick
    Using this as a type of guitar will be noise (IMO)

    Since the types of guitars is a finite list, for proper/correct identification we are using a predefined list.
    Another approach would be to extract dependencies on the NP-guitar. However, in this case the casing on the
    text is improper and that will make pos_tag error prone.
    For properly structured sentence, and grammar parser like {<JJ><NN>} can work well

    Types of guitars taken from Wikipedia

    :param sentences: list() of sentences
    :rtype : collections.Counter object
    """
    g_types = ['acoustic', 'steel-string', 'classical', 'electric', 'twelve-string', 'archtop', 'resonator', 'bass',
               'double-neck']
    found = []
    term = 'guitar'
    for sentence in sentences:
        sentence = sentence.lower()
        for gt in g_types:
            if gt in sentence and term in sentence:
                found.append("{0} {1}".format(gt, term))
    return collections.Counter(found)


if __name__ == "__main__":
    deals_file = '../data/deals.txt'
    fd, sentences = process(deals_file)

    mpt = get_most_popular_term(fd)
    print "Most popular term: {0}, freq={1}".format(mpt[0], mpt[1])

    lpt = get_least_popular_term(fd)
    print "Least popular term: {0}, freq={1}".format(lpt[0], lpt[1])

    C = get_types_of_guitar(sentences)
    print "Types of guitar: {0}".format(len(C))
    print C