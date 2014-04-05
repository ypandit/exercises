""" Groups and Topics

The objective of this task is to explore the structure of the deals.txt file. 

Building on task 1, we now want to start to understand the relationships to help us understand:

1. What groups exist within the deals?
2. What topics exist within the deals?

"""

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from ml.mywork import task3


def get_groups():
    pass


def get_topics(data, n_topics=10, n_words=10, min_df=0.02, max_df=0.85, n_features=5000):
    """
    A "topic" consists of a cluster of words that frequently occur together.
    Here we are performing a latent semantic analysis (LSA) based on tf-idf.
    Not considering the most frequent terms (max_df) since the correlation between them would be
    high and they will show up in most of the topics.

    :param min_df: Threshold for minimum term frequency. Default 0.02
    :param max_df: Threshold for maximum term frequency. Default 0.85
    :param n_features: Number of features to be considered. Default 5000
    :param n_topics: Number of topics to decompose data to
    :param n_words: Number of words per topics
    :param data: Bunch() object for data; Bunch.target is optional
    """

    vectorizer = TfidfVectorizer(stop_words='english', min_df=min_df, max_df=max_df, max_features=n_features,
                                 binary=True, use_idf=False, smooth_idf=False)

    X = vectorizer.fit_transform(data.data).toarray()
    # X = preprocessing.scale(X)
    features = vectorizer.get_feature_names()

    tm = TruncatedSVD(n_components=n_topics, algorithm='randomized', n_iterations=10, random_state=1).fit(X)

    topics = []
    for topic_idx, topic in enumerate(tm.components_):
        words = [features[i] for i in topic.argsort()[:-n_words - 1:-1]]
        topics.append(words)
    return topics


if __name__ == "__main__":
    deals_file = '../data/deals.txt'
    data = task3.load_data(f=deals_file, subset='test')

    topics = get_topics(data, n_words=10, n_features=20000, min_df=0.01, max_df=0.80)
    for i in range(0, len(topics)):
        print "Topic #{0} - {1}".format(i, ", ".join(topics[i]))
        # print "Topic #{0} - {1}".format(i, ", ".join(sorted(topics[i])))

