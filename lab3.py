from collections import defaultdict
from datetime import datetime
import re

__author__ = 'jugovich-michael'

file1 = open(r'C:\Users\jugovich-michael\Documents\csc424\UCI ADL Binary Dataset\OrdonezA_ADLs_new.txt', 'rb')
file2 = open(r'C:\Users\jugovich-michael\Documents\csc424\UCI ADL Binary Dataset\OrdonezB_ADLs.txt', 'rb')
files = [file1, file2]
sensors = open(r'C:\Users\jugovich-michael\Documents\csc424\UCI ADL Binary Dataset\OrdonezB_Sensors.txt', 'rb')
out = open(r'C:\Users\jugovich-michael\Documents\csc424\UCI ADL Binary Dataset\new_Ordonez_ADLs_features.txt', 'wb')
x = 1
y = 1
z=0
d = {}
for file in files:
    z +=1
    for row in file:
        if x in [1,2]:
            row = ['td', 'activity']
            if x == 1: out.write('%s\r\n' % ','.join(row))
            x+= 1

        else:
            row = row.replace('\t\t', '\t')
            row = row.replace('\r\n', '')
            row = [r.strip() for r in row.split('\t')]
            nrow = []
            for r in row:
                if r != '':
                    nrow.append(r)
            td = datetime.strptime(nrow[1], '%Y-%m-%d %H:%M:%S')-datetime.strptime(nrow[0], '%Y-%m-%d %H:%M:%S')
            if td.seconds/60/60 > 8:
                print nrow
            if nrow[2] == '':
                print nrow
                print '---------'
            if nrow[2] not in d:
                print nrow[2] + ':' + str(y)
                d[nrow[2]] = y
                y+=1
            new_row = [td.seconds]
            for x in range(1,11):
                if x == d[nrow[2]]:
                    new_row.append(1)
                else:
                    new_row.append(0)

            new_row.append(z)
            out.write('%s\r\n' % ','.join([str(n) for n in new_row]))

    x = 1
out.close()
file1.close()
file2.close()
sensors.close()
