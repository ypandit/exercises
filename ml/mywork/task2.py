""" Groups and Topics

The objective of this task is to explore the structure of the deals.txt file. 

Building on task 1, we now want to start to understand the relationships to help us understand:

1. What groups exist within the deals?
2. What topics exist within the deals?

"""

from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer
from numpy import bincount

from ml.mywork import task3


# TODO: Elbow method to select the best number of clusters

def get_groups(data, n_cluster=5, n_topics=30, min_df=0.01, max_df=0.98):
    """
    Returns an object of trained KMeans classifier. The number of cluster is
    given by the argument n_cluster
    The best number of clusters is selected by the Elbow method

    :param data: Bunch() object for data. Bunch.target is optional
    :param n_cluster: Number of clusters
    :param n_topics: Number of topics/components for decomposition
    :param min_df: Threshold for minimum term frequency. Default 0.02
    :param max_df: Threshold for maximum term frequency. Default 0.85
    :return: 
    """
    vectorizer = TfidfVectorizer(stop_words='english', min_df=min_df, max_df=max_df, binary=True, use_idf=True,
                                 smooth_idf=True)

    X = vectorizer.fit_transform(data.data).toarray()
    tsvd = TruncatedSVD(n_components=n_topics, algorithm='randomized', n_iterations=10, random_state=1)
    X = tsvd.fit_transform(X)
    X = Normalizer(copy=False).fit_transform(X)

    kmeans = KMeans(n_clusters=n_cluster, n_init=1, init='random')
    kmeans.fit(X)
    return kmeans


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

    topics = get_topics(data, n_topics=10, n_words=10, n_features=10000, min_df=0.01, max_df=0.80)
    print "Following {0} topics exist in the data:".format(len(topics))
    for i in range(0, len(topics)):
        print "Topic #{0} - {1}".format(i, ", ".join(topics[i]))
        # print "Topic #{0} - {1}".format(i, ", ".join(sorted(topics[i])))

    best_k = []
    for i in range(3, 10):
        kmeans = get_groups(data, n_cluster=i, n_topics=15, min_df=0.02, max_df=0.80)
        print bincount(kmeans.labels_)
        best_k.append(dict(n=i, df=kmeans.inertia_))
    sorted_best_k = sorted(best_k, key=lambda k: k['df'])
    print sorted_best_k[0]
    print  sorted_best_k[-1]