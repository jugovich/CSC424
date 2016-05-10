from contextlib import closing
from os import path
import cPickle
from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer()
with closing(open(path.join(__file__, '..', 'data', 'corpus.pkl'), 'rb')) as pkl:
    corpus = cPickle.load(pkl)
new_corpus = []
for row in corpus:
    new_corpus.append(unicode(row, "ISO-8859-1"))

vect.fit(new_corpus)

with closing(open(path.join(__file__, '..', 'data', 'corpus_vect.txt'), 'wb')) as out:
    s = '%s\r\n' % ','.join(vect.get_feature_names())
    out.write(s.encode('ascii','xmlcharrefreplace'))

for row in new_corpus:
    with closing(open(path.join(__file__, '..', 'data', 'corpus_vect.txt'), 'a')) as out:
        out.write('%s\r\n' % ','.join([str(x) for x in vect.transform(['this is the second of the next three']).toarray()[0]]))