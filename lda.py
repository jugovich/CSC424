import logging, cPickle, sys, hashlib, pickle
from imp import find_module, load_module
from contextlib import closing
from itertools import izip
from os import path, makedirs
from time import time
from datetime import datetime
from numpy import *
from pprint import pprint
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn import svm, metrics
from sklearn.lda import LDA
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC

class TrainError(Exception):
    pass

parameters = {
    'vect__max_df': (0.5, 0.75, 1.0),
    'vect__max_features': (None, 5000, 50000),
    'vect__ngram_range': ((1, 2), (1, 3)),  # unigrams or bigrams
    'tfidf__use_idf': (True,), #(True, False),
    'tfidf__norm': ('l2',),#('l1', 'l2'),
    #'clf__alpha': (0.00001, 0.000001),
    #'clf__penalty': ('l2', 'elasticnet'),
    #'clf__n_iter': (10, 50, 80),
}

p_opt = {'model'}
preprocess_method = 'position_lt3_gt3_e3_LDA'

p_opt_hash = hashlib.md5(( '|'.join(["%s_%s" % (str(k), str(v)) for k, v in parameters.iteritems()])+preprocess_method)).hexdigest()

opt = {'out_path': r'D:\workspace\CSC424\report\%s' % p_opt_hash,
       'clf': 'sgd',
       'clf_model': SGDClassifier(), 
       }
if not path.isdir(opt['out_path']): makedirs(opt['out_path'])

with closing(open(path.join(__file__, '..', 'data', 'X_train.pkl'), 'rb')) as pkl:
    data = cPickle.load(pkl)

with closing(open(path.join(__file__, '..', 'data', 'y_train.pkl'), 'rb')) as pkl:
    target = cPickle.load(pkl)

vectorizer = CountVectorizer(max_df=.75, min_df = 2, ngram_range=(1,2))

X=vectorizer.fit_transform(data, array(target))
transformer = TfidfTransformer()
X = transformer.fit_transform(X, array(target))

#hv = HashingVectorizer()
clf = LDA()
clf.fit_transform(X.toarray(), array(target), store_covariance=True)

for name,obj in [('saclings.txt', clf.scalings_), ('coef.txt', clf.coef_),
                 ('covariance.txt', clf.covariance_),('xbar.txt', clf.xbar_),
                 ('means.txt', clf.means_)]:
    with closing(open(path.join(opt['out_path'], name), 'wb')) as out:
        print 'saving %s' % name
        for row in obj:
            out.write(str(row)+'\r\n')

print 'priors'
print clf.priors

del X
del data
del target

with closing(open(path.join(__file__, '..', 'data', 'X_test.pkl'), 'rb')) as pkl:
    data = cPickle.load(pkl)
with closing(open(path.join(__file__, '..', 'data', 'y_test.pkl'), 'rb')) as pkl:
    target= cPickle.load(pkl)
y=array(target)
X=vectorizer.fit_transform(data, array(target))

transformer = TfidfTransformer()
tfidf = transformer.fit_transform(X, array(target))
X = tfidf.toarray()

for name,obj in [('saclings.txt', clf.scalings_), ('coef.txt', clf.coef_),
                 ('covariance.txt', clf.covariance_),('xbar.txt', clf.xbar_),
                 ('means.txt', clf.means_)]:
    with closing(open(path.join(opt['out_path'], name), 'wb')) as out:
        print 'saving %s' % name
        for row in obj:
            out.write(str(row)+'\r\n')

print 'priors'
print clf.priors


first = True
with closing(open(path.join(opt['out_path'], 'full_report.csv'), 'wb')) as out:
    for line in classification_report(y, clf.predict(X)).split('\n'):
        if not line: continue
        line = ' '.join(line.split()).lower()
        line = line.replace('avg / total', 'avg/total')
        line = line.split(' ')
        if first:  
            out.write(','.join(['class'] + line + ['\n']))
            first = False
        else:
            out.write(','.join(line) + '\n')

_scores = []
_scores.append(("score", clf.score(X, y)))
#_scores.append(("accuracy", metrics.accuracy_score(y, X)))
# _scores.extend([(k,v) for k,v in izip(("precision", "recall", "f1score", "support"), metrics.precision_recall_fscore_support(dv_train, dv_predict_train, beta=1))])

with closing(open(path.join(opt['out_path'], 'score.txt'), 'wb')) as out:
    for _score in _scores:
        out.write("%s: %s\n" % (_score[0], str(_score[1])))
#pipeline.fit(data_train, array(target_train))
