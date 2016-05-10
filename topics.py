#from __future__ import print_function
from contextlib import closing

from time import time
from os import path
import cPickle
from sklearn.feature_extraction import text
from sklearn import decomposition

n_samples = 0
n_features = 10000
n_topics = 20
n_top_words = 20

# Load the 20 newsgroups dataset and vectorize it using the most common word
# frequency with TF-IDF weighting (without top 5% stop words)
for ds in ['full', 'against', 'for', 'neutral']:
    t0 = time()
    print("Loading dataset and extracting TF-IDF features...")
    with closing(open(path.join(__file__, '..', 'data', 'corpus_%s.pkl' % ds), 'rb')) as pkl:
        corpus = cPickle.load(pkl)

    n_samples = len(corpus)
    dataset = []
    for row in corpus:
        dataset.append(unicode(row, "ISO-8859-1"))

    vectorizer = text.CountVectorizer(max_df=0.95,
                                      stop_words='english', token_pattern='#\w+')

    counts = vectorizer.fit_transform(dataset)
    tfidf = text.TfidfTransformer().fit_transform(counts)
    print "done in %0.3fs." % (time() - t0)

    # Fit the NMF model
    print "Fitting the NMF model with n_samples=%d and n_features=%d..." % (n_samples, n_features)
    nmf = decomposition.NMF(n_components=n_topics).fit(tfidf)
    print "done in %0.3fs." % (time() - t0)

    feature_names = vectorizer.get_feature_names()

    # # Inverse the vectorizer vocabulary to be able

    with closing(open(path.join(__file__, '..', 'data', 'hashtag_topics_%s.txt' % ds), 'wb')) as out:
        for topic_idx, topic in enumerate(nmf.components_):
            out.write("Topic #%d:\r\n" % topic_idx)
            _topics = '%s\r\n' % " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
            out.write(_topics.encode('ascii', 'xmlcharrefreplace'))
            out.write('\r\n')