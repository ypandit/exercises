""" Groups and Topics

The objective of this task is to explore the structure of the deals.txt file. 

Building on task 1, we now want to start to understand the relationships to help us understand:

1. What groups exist within the deals?
2. What topics exist within the deals?

"""

import os

from gensim.corpora import Dictionary, MmCorpus
from gensim.models import LdaModel
from scipy.cluster.vq import kmeans
from scipy.spatial.distance import cdist, pdist
from numpy import min, sum, arange
from sklearn.decomposition import TruncatedSVD, RandomizedPCA
from sklearn.feature_extraction.text import TfidfVectorizer

from ml.mywork import task3, task1


def feature_extractor(data, n_samples=None, min_df=0.02, max_df=0.98):
    if n_samples is None:
        n_samples = len(data.data) / 2
    vectorizer = TfidfVectorizer(stop_words='english', min_df=min_df, max_df=max_df)
    X = vectorizer.fit_transform(data.data[:n_samples]).toarray()
    return X, vectorizer.get_feature_names()


def find_best_k(X):
    """
    Draws a plot based on the Elbow method. Calculates variance for 1..20 clusters.
    Calcuates the percentage of variance from between-cluster sum of squares and
    the total sum of squares.
    k where the change is variance doesn't change much for k+1 is the best for given
    set of data

    Elbow plot for this dataset with 20000 samples is saved to task2_kmeans_v1.png

    :param X: Feature set
    """
    from matplotlib import pyplot as plt

    pca = RandomizedPCA(n_components=2).fit(X)
    X = pca.transform(X)

    k_max = 20
    k_range = range(1, k_max + 1)
    km = [kmeans(X, k) for k in k_range]
    centroids = [centroid for (centroid, var) in km]
    d_c = [cdist(X, centroid, 'euclidean') for centroid in centroids]
    dist = [min(dist, axis=1) for dist in d_c]

    within_ss2 = [sum(d ** 2) for d in dist]  # Total within-cluster sum of squares
    total_ss2 = sum(pdist(X) ** 2) / X.shape[0]  # The total sum of squares
    between_ss2 = total_ss2 - within_ss2  # Between cluster sum of squares

    k = 4
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.plot(k_range, between_ss2 / total_ss2 * 100, 'b*-')
    ax.plot(k_range[k], between_ss2[k] / total_ss2 * 100, marker='o', markersize=12, markeredgewidth=2,
            markeredgecolor='r', markerfacecolor='None')
    ax.set_ylim((0, 100))
    ax.xaxis.set_ticks(arange(1, k_max, 1))

    plt.grid(True)
    plt.xlabel('Number of clusters')
    plt.ylabel('Percentage of variance explained (%)')
    plt.title('Elbow for KMeans clustering')
    plt.savefig('task2_kmeans_v1.png')


def get_groups(X, k=5, n_components=2):
    """
    Returns an object of trained KMeans classifier. The number of cluster is
    given by the argument k
    The best number of clusters is selected by the Elbow method (find_best_k)

    :param X: Feature set
    :param n_components: Size for decomposition of features. Default 2
    :param k: Number of clusters. Default 5
    :return: dict() of kmeans centroids and variance
    """
    # tsvd = TruncatedSVD(n_components=n_topics, algorithm='randomized', n_iterations=10, random_state=1)
    # X = tsvd.fit_transform(X)
    # X = Normalizer(copy=False).fit_transform(X)
    # kmeans = KMeans(n_clusters=n_cluster, n_init=1, init='random')
    # kmeans.fit(X)
    # return kmeans

    pca = RandomizedPCA(n_components=n_components).fit(X)
    X = pca.transform(X)
    km = kmeans(X, k)
    return km


def get_topics_lsa(X, features, n_topics=10, n_words=10):
    """
    A "topic" consists of a cluster of words that frequently occur together.
    Here we are performing a latent semantic analysis (LSA) based on tf-idf.
    Not considering the most frequent terms (max_df) since the correlation between them would be
    high and they will show up in most of the topics.

    :param X: Feature set
    :param features: Features names for X
    :param n_topics: Number of topics to decompose data to
    :param n_words: Number of words per topics
    """

    tm = TruncatedSVD(n_components=n_topics, algorithm='randomized', n_iterations=10, random_state=1).fit(X)
    topics = []
    for topic_idx, topic in enumerate(tm.components_):
        words = [features[i] for i in topic.argsort()[:-n_words - 1:-1]]
        topics.append(words)
    return topics


def get_topics_lda(tokens, n_topics=10):
    """

    :param tokens:
    :param n_topics:
    :return:
    """
    dict_file = 'resources/deals.dict'
    if not os.path.isfile(dict_file):
        print "Dictionary file does not exist. Creating one"
        dictionary = Dictionary(tokens)
        freq1 = [id for id, freq in dictionary.dfs.iteritems() if freq == 1]
        dictionary.filter_tokens(freq1)
        dictionary.compactify()
        dictionary.save(dict_file)
    dictionary = Dictionary.load(dict_file)
    # print dictionary

    corpus_file = 'resources/deals.mm'
    if not os.path.isfile(corpus_file):
        print "Corpus file does not exist. Creating one"
        corpus = [dictionary.doc2bow(token) for token in tokens]
        MmCorpus.serialize(corpus_file, corpus)
    mm = MmCorpus(corpus_file)
    # print mm
    # tfidf = TfidfModel(mm)
    # corpus_tfidf = tfidf[mm]

    lda = LdaModel(corpus=mm, id2word=dictionary, num_topics=n_topics, update_every=1, chunksize=1000,
                   passes=1)
    topics = []
    for i in range(0, n_topics):
        words = lda.print_topic(i).split('+')
        topic = []
        for word in words:
            score, w = word.split('*')
            topic.append((w, score))
        topics.append(topic)
    return topics


if __name__ == "__main__":
    deals_file = '../data/deals.txt'
    data = task3.load_data(f=deals_file, subset='test')

    X, features = feature_extractor(data, min_df=0.02, max_df=0.98, n_samples=len(data.data))
    topics = get_topics_lsa(X, features, n_topics=10, n_words=10)
    print "Following {0} topics (LSA) exist in the data:".format(len(topics))
    for i in range(0, len(topics)):
        print "Topic #{0} - {1}".format(i, ", ".join(topics[i]))
    #   print "Topic #{0} - {1}".format(i, ", ".join(sorted(topics[i])))
    print "----------"

    fd, sentences, tokens = task1.process(deals_file, return_freqdist=False)
    lda = get_topics_lda(tokens)
    print "Following {0} topics (LDA) exist in the data:".format(len(lda))
    for i in range(0, len(lda)):
        print "Topic #{0} - {1}".format(i, lda[i])
    print "----------"

    n_samples = 20000
    X, features = feature_extractor(data, min_df=0.02, max_df=0.98, n_samples=n_samples)
    # find_best_k(data)
    print "----------"

    k = 5  # From find_best_k plot
    kmeans = get_groups(X, k)
    print "There are {0} groups in the data (based on samples={1})".format(len(kmeans[0]), n_samples)
    print "Centroids: \n{0}".format(kmeans[0])
    print "Variance (distortion): {0}".format(kmeans[1])