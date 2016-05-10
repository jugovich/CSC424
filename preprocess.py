import cardsharp as cs
from contextlib import closing
import csv
user_id = {}
user_id_c = 1
coder_id = {}
coder_id_c = 1
query_id = {}
query_id_c = 1

#print 'loading'
#ds = cs.load(source=r'd:\workspace\CSC424\data\CTUStrike_analysis_file.csv', format='text', delimiter=',', encoding='utf-32')
#cs.wait()
#print 'done'
out = open(r'/data/CTUStrike_analysis_file_new.csv', 'wb')
file = csv.reader(open(r'/data/CTUStrike_analysis_file.csv', 'rb'))
worker_file = csv.reader(open(r'/data/tweets_main_workers.csv', 'rb'))
worker_out = open(r'/data/tweets_main_workers_public_file.csv', 'wb')
x = 0
for row in file:
    x+=1
    if x == 1:
        
        print row[25]
        row.extend(['from_user_id', 'to_user_id', 'coder_1', 'coder_2', 'coder_3', 'coder_4', 'coder_5', 'coder_6', 'query_recode', '\r\n'])
        print row
        out.write(','.join(row))
    else:
        spamwriter = csv.writer(out)
        #from and to user
        for v in [12, 15]:
            if row[v] and row[v] != u'0':
                
                if row[v] in user_id:
                    row.append(str(user_id[row[v]]))
                else:
                    user_id[row[v]] = user_id_c
                    row.append(str(user_id_c))
                    user_id_c += 1
            else:
                row.append('')
        
        #coder
        for v in [25,26,27,28,29,30]:
            
            if row[v] and row[v] != '':
                if row[v] in coder_id:
                    row.append(str(coder_id[row[v]]))
                else:
                    coder_id[row[v]] = coder_id_c
                    row.append(str(coder_id_c))
                    coder_id_c += 1
            else:
                row.append('')
        
        #query
        if row[2] in query_id:
            row.append(str(query_id[row[2]]))
        else:
            query_id[row[2]] = query_id_c
            row.append(str(query_id_c))
            query_id_c += 1
                
        #print len(row)
        #print row
        row.append('\r\n')
        spamwriter.writerow(row)
x = 1
#new coder_ids are present in worker file but not in main analysis file
coder_id_c += 7809
for row in worker_file:
    if x == 1:
        row.extend(['worker_id', '\r\n'])
        print row
        worker_out.write(','.join(row))
        x+=1
    else:
        
        writer = csv.writer(worker_out)
        try:
            row.append(coder_id[row[1]])
        except KeyError:
            print row
            print coder_id_c
            coder_id[row[1]] = coder_id_c
            row.append(str(coder_id_c))
            coder_id_c += 1
            
        row.append('\r\n')
        writer.writerow(row)
        
out.close()   