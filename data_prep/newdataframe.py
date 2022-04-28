import sys
import os
from os.path import isfile, join
import csv
import numpy as np
import pandas as pd


filename = 'sp500\dataset'
template = 'sp500\\feature.csv'
l = 339

mypath = '.\sp500\dataset'
onlyfiles = [f for f in os.listdir(mypath) if isfile(join(mypath, f))]
# onlyfiles.sort()
# print(onlyfiles)
newpath = join(sys.path[0], 'sp500\dataframe')
newpath2 = join(sys.path[0], 'sp500\market')\

feature = []
read_fea = open(join(sys.path[0],template), 'r')
reader2 = csv.reader(read_fea)
for line in reader2:
    feature.append(line)
    break

filew = open('./sp500/comine_all_data.csv', 'w', newline='')
filew2 = open('./sp500/comine_all_label.csv', 'w', newline='')
# print("part1")
writer = csv.writer(filew)
writer2 = csv.writer(filew2)
# print("part2")
# print(feature[0])
writer.writerow(feature[0])
writer2.writerow(['datetime', 'instrument', 'label'])
# print("part3")

concept = pd.read_csv('./sp500/SP500_concepts.csv')
concept = concept.sort_values(by=['Symbol']).reset_index(drop=True)

sum = 0
for i in range(concept.shape[0]):
    name_stock = concept['Symbol'][i]
    if os.path.exists('./sp500/dataset/'+name_stock+'.csv'):
        sum+=1
        print(i, name_stock)
        df = pd.read_csv("./sp500/dataset/"+name_stock+'.csv')

        nums,_ = df.shape

        close = []
        open = []
        high = []
        low = []
        VWAP = []
        volume = []
        # print("initial")
        d = 0
        while d < 60:
            close.append(df.iloc[d][5])
            open.append(df.iloc[d][2])
            high.append(df.iloc[d][4])
            low.append(df.iloc[d][1])
            VWAP.append(0)
            volume.append(df.iloc[d][3])
            d+=1
        # print("start wirting")
        # print(d)
        list1 = [df.iloc[d-1][0],name_stock]
        list2 = [df.iloc[d][0],name_stock, df.iloc[d][5]]
        # print(list1)
        # print(list2)
        writer2.writerow(list2)
        for name in close, open, high, low, VWAP, volume:
            for j in range(60):
                list1.append(name[j])
        writer.writerow(list1)
        # d+=1
        # print(list1)

        while d < nums-1:
            close.pop(0)
            open.pop(0)
            high.pop(0)
            low.pop(0)
            volume.pop(0)

            close.append(df.iloc[d][5])
            open.append(df.iloc[d][2])
            high.append(df.iloc[d][4])
            low.append(df.iloc[d][1])
            volume.append(df.iloc[d][3])
            
            newlist2 = [df.iloc[d+1][0],name_stock, df.iloc[d+1][5]]
            writer2.writerow(newlist2)
            newlist = [df.iloc[d][0],name_stock]
            for name in close, open, high, low, VWAP, volume:
                for j in range(60):
                    newlist.append(name[j])
            writer.writerow(newlist)
            d+=1
            # print(newlist2)
            # print(newlist)

filew.close()
filew2.close()
print("sum: ", sum)