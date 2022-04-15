"""
Authors: Zeqian Cai, Yurui Chang, Jinhuan Ke, Yichen Li, Suyuan Wang
Date: 2022/03/05
Version: V 1.0
Function: Gather all the global functions
"""

import numpy as np
import pandas as pd
import torch


class DataLoader:
    def __init__(self):
        self.stock_index = np.load("./data/csi300_stock_index.npy", allow_pickle=True).item()
        self.stock2concept = np.load("./data/relation.npy", allow_pickle=True)
        self.stock_price = pd.read_csv("./data/new_feature.csv")
        self.market_value = pd.read_csv("./data/new_marketVal.csv")
        self.parsing()

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
