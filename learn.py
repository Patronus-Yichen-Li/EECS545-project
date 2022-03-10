"""
Authors: Zeqian Cai, Yurui Chang, Jinhuan Ke, Yichen Li, Suyuan Wang
Date: 2022/03/06
Version: V 1.0
Function: To build the model class
"""

import copy
import json
import collections
import argparse  # for Linux implementation
import numpy as np
import pandas as pd
# pytorch
import torch
import torch.nn as nn
import torch.nn.init as init
# self edit files
import global_func
from model import OptHIST
from model_org import HIST
from global_func import DataLoader


if __name__ == "__main__":
    # load data
    dataset = DataLoader()
    # data modification (basically from df & array to tensor)
    data_matrix = dataset.stock_price.drop(["datetime", "instrument"], axis=1)
    data_matrix = torch.from_numpy(data_matrix.to_numpy()).float()
    marketValue = dataset.market_value.drop(["datetime", "instrument"], axis=1)
    marketValue = torch.from_numpy(marketValue.to_numpy()).float()
    stock2concept = torch.from_numpy(dataset.stock2concept).float()

    # parameters
    (num_stocks, num_attributes) = dataset.stock2concept.shape
    size_train = 192
    num_companies = 10
    # variables declaration
    seed = np.random.randint(1000000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    data_matrix_train = data_matrix[0:size_train * num_stocks]   # generate training data set

    # initialize model
    model = OptHIST()
    # model_org = HIST()
    # data matrix modification
    data_matrix_train = data_matrix_train.reshape(model.size_train, num_companies, -1)

    """ training """
    for t in range(5):
        # encode feature
        x0_t = model.encode_feature(data_matrix_train[t])   # # hidden_size attributes remained (360 > 64)
        marketValue_t = marketValue[t * num_companies: (t + 1) * num_companies]
        # predefined module
        p_sharedInfo_back_t, p_sharedIno_fore_t = model.predefined_concept(x0_t, stock2concept, marketValue_t)
        # hidden module
        x1_t = x0_t - p_sharedInfo_back_t
        h_sharedInfo_back_t, h_sharedIno_fore_t = model.hidden_concept(x1_t)
        # individual module
        x2_t = x1_t - h_sharedInfo_back_t
        model.individual_concept(x2_t)
        # predict all
        pred_all = model.predict()
        # pred_all_org = model_org.forward(data_matrix_train[t], stock2concept, marketValue_t)
        print("Day " + str(t) + ":")
        print(pred_all)
        # print(pred_all_org)
