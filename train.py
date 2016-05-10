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
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
import pylab as pl
from pickle import dumps
class TrainError(Exception):
    pass

parameters = {
    'vect__max_df': (0.5, 0.75, .95, 1.0),
    'vect__max_features': (None, 5000, 50000),
    'vect__ngram_range': ((1, 2), (1, 3)),  # unigrams or bigrams
    'vect__token_pattern':['#\w+'],
    'tfidf__use_idf': (True,), #(True, False),
    'tfidf__norm': (None, 'l1', 'l2',),#('l1', 'l2'),
    'clf__alpha': (0.00001, 0.000001),
    #'clf__penalty': ('l2', 'elasticnet'),
    #'clf__n_iter': (10, 50, 80),
    #'reduce_dim__n_components':[2, 5, 10],
}

p_opt = {'model'}
preprocess_method = 'position_lt3_gt3_e3'

p_opt_hash = hashlib.md5(( '|'.join(["%s_%s" % (str(k), str(v)) for k, v in parameters.iteritems()])+preprocess_method)).hexdigest()
     
opt = {'out_path': r'D:\workspace\CSC424\report\%s' % p_opt_hash,
       'clf': 'sgd',
       'clf_model': SGDClassifier(),
       }

#svm.SVC()
            
pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', opt['clf_model']),
])

with closing(open(path.join(__file__, '..', 'data', 'X_train.pkl'), 'rb')) as pkl:
    data_train = cPickle.load(pkl)
with closing(open(path.join(__file__, '..', 'data', 'X_test.pkl'), 'rb')) as pkl:
    data_test = cPickle.load(pkl)
with closing(open(path.join(__file__, '..', 'data', 'y_train.pkl'), 'rb')) as pkl:
    target_train = cPickle.load(pkl)
with closing(open(path.join(__file__, '..', 'data', 'y_test.pkl'), 'rb')) as pkl:
    target_test = cPickle.load(pkl)



#vectorizer = CountVectorizer(min_df=1)
#X = vectorizer.fit_transform(data_train)
#clf = LDA()
#print data_train
#print target_train
#clf.fit(array(data_train), array(target_train))

if __name__ == "__main__":
    # multiprocessing requires the fork to happen in a __main__ protected
    # block

    # find the best parameters for both the feature extraction and the
    # classifier
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, cv=3)

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()
    
    grid_search.fit(data_train, array(target_train))
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
    
    _bp = grid_search.best_estimator_.get_params()
    
    if not path.isdir(opt['out_path']): makedirs(opt['out_path'])
    
    with closing(open(path.join(opt['out_path'], 'parameters.txt'), 'wb')) as out:
        for k,v in _bp.iteritems():
            out.write('|'.join([str(k),str(v)])+ '\n')
        out.write(preprocess_method + '\n')
        out.write('length train\t%i' % len(data_train) + '\n')
        out.write('length test\t%i' % len(data_test))
        
    if opt['clf'] == 'svc':
        clf = svm.SVC(C=_bp['clf__C'],
                      kernel=_bp['clf__kernel'],
                      class_weight=_bp['clf__class_weight'],
                      probability=True)
        
    elif opt['clf'] == 'linear_svc':
        clf = svm.LinearSVC(C=_bp['clf__C'], 
                            class_weight=_bp['clf__class_weight'])
        
    elif opt['clf'] == 'sgd':
        clf = SGDClassifier(alpha=_bp['clf__alpha'],
                            penalty=_bp['clf__penalty'],
                            n_iter=_bp['clf__n_iter'],
                            loss=_bp['clf__loss'],
                            l1_ratio=_bp['clf__l1_ratio'])
    else:
        raise TrainError("Unsupported algorithm: %s" % opt['clf'])
        
    
    
    clf = Pipeline([('vect', CountVectorizer(#tokenizer=lambda s: s.split(),
                                             token_pattern='#\w+',
                                             max_df=_bp['vect__max_df'],
                                             ngram_range=_bp['vect__ngram_range'],
                                             min_df=_bp['vect__min_df'],
                                             max_features=_bp['vect__max_features'])), 
                    ('tfidf', TfidfTransformer(norm= _bp['tfidf__norm'])), 
                    ('clf', clf)])
    
    clf.fit(data_train, target_train)
    
    test_vect = CountVectorizer(tokenizer=lambda s: s.split(),
                                             max_df=_bp['vect__max_df'],
                                             ngram_range=_bp['vect__ngram_range'],
                                             min_df=_bp['vect__min_df'],
                                             max_features=_bp['vect__max_features'])


    #test_vect.fit(data_test[:100])
    #print 'TESTING TOKENIZER: FEATURE NAMES = %s' % test_vect.get_feature_names()
    
    #TODO move this into report module
    first = True
    y_pred = clf.predict(data_test)
    with closing(open(path.join(opt['out_path'], 'full_report.csv'), 'wb')) as out:
        for line in classification_report(target_test, y_pred).split('\n'):
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
    _scores.append(("score", clf.score(data_test, target_test)))
    _scores.append(("accuracy", metrics.accuracy_score(target_test, data_test)))
   # _scores.extend([(k,v) for k,v in izip(("precision", "recall", "f1score", "support"), metrics.precision_recall_fscore_support(dv_train, dv_predict_train, beta=1))])
    
    with closing(open(path.join(opt['out_path'], 'score.txt'), 'wb')) as out:
        for _score in _scores:
            out.write("%s: %s\n" % (_score[0], str(_score[1])))


    cm = confusion_matrix(target_test, y_pred)
    print(cm)
    pl.matshow(cm)
    pl.title('Confusion matrix')
    pl.colorbar()
    pl.ylabel('True label')
    pl.xlabel('Predicted label')
    pl.show()
        #only save model if it has a better score than the currently saved model
#    _old_score = 0
#    _path = r'D:\workspace\CSC424\data'
#    try:
#        with closing(open(path.join(_path, 'clf.pkl'), 'rb')) as file:
#            _old_clf = cPickle.load(file)
#            _old_score = _old_clf.score(data_test, target_test)
#    except IOError, EOFError:
#        _old_score = 0 #if no model exists set to 0
#    
#    if _score > _old_score:
#        with closing(open(path.join(_path, 'clf.pkl'), 'wb')) as file:
#            pickle.dump(clf, file)