import sys
import os
from os.path import isfile, join
import csv
import numpy as np
import pandas as pd

# combine the stock features in one file
'''
concept = pd.read_csv('SP500_concepts.csv')
datapath = join(sys.path[0], 'dataset/')

stocknames = [f for f in os.listdir(datapath) if isfile(join(datapath, f))]
stocknames.sort()
concept = concept.sort_values(by=['Symbol']).reset_index(drop=True)

first = True
filew = open(join(sys.path[0],'justcom_all.csv'), 'w', newline='')
writer = csv.writer(filew)

for i in range(concept.shape[0]):
    name = concept['Symbol'][i]
    if os.path.exists('./dataset/'+name+'.csv'):
        full_text = []
        stockfile = open('./dataset/'+name+'.csv', 'r')
        reader = csv.reader(stockfile)
        for line in reader:
            full_text.append(line)
        nums = len(full_text)

        for j in range(nums):
            writer.writerow(full_text[j])
    break
filew.close()
'''

# combine the market value
'''
# for market value
datapath = join(sys.path[0], 'market/')

stocknames = [f for f in os.listdir(datapath) if isfile(join(datapath, f))]
stocknames.sort()
# print(stocknames)

first = True
filew = open(join(sys.path[0],'justcomMarket.csv'), 'w')
writer = csv.writer(filew)

for i in range(1,len(stocknames)):
    full_text = []
    stockfile = open(join(datapath,stocknames[i]), 'r')
    reader = csv.reader(stockfile)
    for line in reader:
        full_text.append(line)
    nums = len(full_text)

    if first:
        first = False
        writer.writerow(full_text[0])
    for j in range(1,nums):
        writer.writerow(full_text[j])

filew.close()
'''

# rearrange the feature
# '''
stockfea = open(join(sys.path[0],'comine_all_data.csv'), 'r')
reader = csv.reader(stockfea)
full_text = []
for line in reader:
    full_text.append(line)
nums = len(full_text)

filew = open(join(sys.path[0],'new_feature_all.csv'), 'w', newline='')
writer = csv.writer(filew)
writer.writerow(full_text[0])

for i in range(1,3748):
    for j in range(int((nums-1)/3747)):
        writer.writerow(full_text[i + j*3747])

filew.close()
# '''

# rearrange the market value
'''
stockfea = open(join(sys.path[0],'comine_all_label.csv'), 'r')
reader = csv.reader(stockfea)
full_text = []
for line in reader:
    full_text.append(line)
nums = len(full_text)

filew = open(join(sys.path[0],'new_marketVal_all.csv'), 'w', newline='')
writer = csv.writer(filew)
writer.writerow(full_text[0])

for i in range(1,3748):
    for j in range(int((nums-1)/3747)):
        writer.writerow(full_text[i + j*3747])

filew.close()
'''