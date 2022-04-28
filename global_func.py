"""
Authors: Zeqian Cai, Yurui Chang, Jinhuan Ke, Yichen Li, Suyuan Wang
Date: 2022/03/05
Version: V 1.0
Function: Gather all the global functions
"""

import numpy as np
import pandas as pd
import torch

repeat = 3  #10
epoch = 200
num_companies = 308

train_batch_size = 800  #200
test_batch_size = 500   #200
inference_batch_size = 100

train_start_date = 0
train_end_date = 1500
valid_start_date = 1501
valid_end_date = 2000
test_start_date = 2001
test_end_date = 3000

min_num = 90
max_num = 100

num_layers=2
K=5
Adamlr=0.002
smooth_steps = 5
early_stop = 30

# model = "OptHIST"
model = "HIST"

class DataLoader:
    def __init__(self):
        if model == "OptHIST":
            self.stock2concept = np.load("./data/relation.npy", allow_pickle=True)
        else:
            self.stock2concept = np.load("./data/concept.npy", allow_pickle=True)
        self.stock_price = pd.read_csv("./data/new_feature_all.csv")
        self.market_value = pd.read_csv("./data/new_marketVal_all.csv")
        # self.parsing()

    # for test only
    def parsing(self):
        self.stock2concept = self.stock2concept[0:10]
        self.market_value = self.market_value


def cal_cos_similarity(x, y):  # the 2nd dimension of x and y are the same
    xy = x.mm(torch.t(y))
    x_norm = torch.sqrt(torch.sum(x * x, dim=1)).reshape(-1, 1)
    y_norm = torch.sqrt(torch.sum(y * y, dim=1)).reshape(-1, 1)
    cos_similarity = xy / x_norm.mm(torch.t(y_norm))
    cos_similarity[cos_similarity != cos_similarity] = 0
    return cos_similarity
