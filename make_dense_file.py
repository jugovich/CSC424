from itertools import izip
from contextlib import closing
import cPickle, os, re
from numpy import *
from sklearn.cross_validation import StratifiedShuffleSplit
import csv

"""pre-process"""
ds = csv.reader(open(r'/data/CTUStrike_analysis_file_public_release.csv', 'rb'))

_row_count = 1
corpus = []
labels = []

#iterate over data
_re_str = re.compile('\s')
for row in ds:
    if _row_count == 1:
        var_names = [x for x in row]
        _row_count += 1
        continue
    else:
        d_row = dict([(x,y) for x, y in izip(var_names, row)])

    #only use coded tweet text
    codes = set()
    for x in range(1,6):
        codes.add(d_row['coder_%i' % x])

    if d_row['coder_1'] != '' and float(d_row['relevance_mean']) > 3:
        '''Do not include position = 3
        '''
#        tweet = []
#        for ngram in [n for n in wordpunct_tokenize(d_row['text'].lower().strip()) if n != "."]:
#            if ngram != None:
#                tweet.append(ngram)

        #tweet = 'tis is a test ' + str(int(random()*100))
        tweet = d_row['text'].lower()
        try:

            corpus.append(tweet)

            if float(d_row['position_mean']) < 3:
                #add the tweet text to the corpus
                labels.append(1)

            elif float(d_row['position_mean']) > 3:
                labels.append(2)

            elif float(d_row['position_mean']) == 3:
                labels.append(3)

            else:
                labels.append(9)
#            if float(d_row['position_mean']) < 3:
#                #add the tweet text to the corpus
#                if float(d_row['relevance_mean']) < 3:
#                    labels.append(1)
#                elif float(d_row['relevance_mean']) > 3:
#                    labels.append(2)
#                elif float(d_row['relevance_mean']) == 3:
#                    labels.append(3)
#
#            elif float(d_row['position_mean']) > 3:
#                if float(d_row['relevance_mean']) < 3:
#                    labels.append(4)
#                elif float(d_row['relevance_mean']) > 3:
#                    labels.append(5)
#                elif float(d_row['relevance_mean']) == 3:
#                    labels.append(6)
#
#            elif float(d_row['position_mean']) == 3:
#                if float(d_row['relevance_mean']) < 3:
#                    labels.append(7)
#                elif float(d_row['relevance_mean']) > 3:
#                    labels.append(8)
#                elif float(d_row['relevance_mean']) == 3:
#                    labels.append(9)
        except:
            print '1'
            print d_row['position_mean']



print len(labels)
strat_split = StratifiedShuffleSplit(labels, n_iter=1, test_size=0.3, random_state=37)
X_train, y_train, X_test, y_test = [],[],[],[]
for train_index, test_index in strat_split:
    for _train_i in train_index:
        X_train.append(unicode(corpus[_train_i], "ISO-8859-1"))
        y_train.append(labels[_train_i])
    for _test_i in test_index:
        X_test.append(unicode(corpus[_test_i], "ISO-8859-1"))
        y_test.append(labels[_test_i])


for _ds, name in [(X_train, 'X_train'), (X_test, 'X_test'), (y_train, 'y_train'), (y_test, 'y_test')]:
    with closing(open(os.path.join(__file__, '..', 'data', '%s.pkl' % name), 'wb')) as pkl:
        cPickle.dump(_ds, pkl)

