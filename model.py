"""
Authors: Zeqian Cai, Yurui Chang, Jinhuan Ke, Yichen Li, Suyuan Wang
Date: 2022/03/05
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


class OptHIST(nn.Module):
    # further declaration
    x0: torch.tensor
    p_sharedInfo_back: torch.tensor
    h_sharedInfo_back: torch.tensor
    p_sharedInfo_fore: torch.tensor
    h_sharedInfo_fore: torch.tensor
    individualInfo: torch.tensor
    pred: torch.tensor

    def __init__(self, input_size=6, hidden_size=64, num_layers=2, dropout=0.0, K=3):
        """
        :param input_size: # of attributes
        :param hidden_size: # of features encoded after GRU
        :param num_layers: for GRU
        :param dropout: for GRU
        :param K: for hidden concepts
        """

        # super class inheritance
        super().__init__()
        # global parameters
        self.size_train = 192
        # input parameters' record
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = 0.0
        self.K = K

        # feature encoder
        self.encoder = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        # predefined concept module
        self.fc_ps = nn.Linear(hidden_size, hidden_size)
        init.xavier_uniform_(self.fc_ps.weight)
        self.fc_ps_fore = nn.Linear(hidden_size, hidden_size)
        init.xavier_uniform_(self.fc_ps_fore.weight)
        self.fc_ps_back = nn.Linear(hidden_size, hidden_size)
        init.xavier_uniform_(self.fc_ps_back.weight)
        self.fc_out_ps = nn.Linear(hidden_size, 1)  # y0
        # hidden concept module
        self.fc_hs = nn.Linear(hidden_size, hidden_size)
        init.xavier_uniform_(self.fc_hs.weight)
        self.fc_hs_fore = nn.Linear(hidden_size, hidden_size)
        init.xavier_uniform_(self.fc_hs_fore.weight)
        self.fc_hs_back = nn.Linear(hidden_size, hidden_size)
        init.xavier_uniform_(self.fc_hs_back.weight)
        self.fc_out_hs = nn.Linear(hidden_size, 1)  # y1
        # individual concept module
        self.fc_individual = nn.Linear(hidden_size, hidden_size)
        init.xavier_uniform_(self.fc_individual.weight)
        self.fc_out_individual = nn.Linear(hidden_size, 1)  # y2
        # concept
        self.leaky_relu = nn.LeakyReLU()
        self.softmax_s2t = torch.nn.Softmax(dim=0)
        self.softmax_t2s = torch.nn.Softmax(dim=1)
        # final output
        self.fc_out = nn.Linear(hidden_size, 1)

    def encode_feature(self, x):
        """
        :param x: input raw data matrix S, which should be populated with Double
        :return: GRU-extracted feature X0
        """
        encoded_feature = x.reshape(len(x), self.input_size, -1)  # [N, F, T]
        # print(encoded_feature.size())
        encoded_feature = encoded_feature.permute(0, 2, 1)  # [N, T, F]
        # print(encoded_feature.size())
        encoded_feature, _ = self.encoder(encoded_feature)

        # assign to class
        self.x0 = encoded_feature[:, -1, :]
        return encoded_feature[:, -1, :]

    def predefined_concept(self, x0, concept_matrix: torch.tensor, market_value: torch.tensor):
        """
        Calculates only ONE day's predefined result
        :param x0: encoded feature after GRU (num_stocks * hidden_size)
        :param concept_matrix: matrix recording the concepts for each stock (num_stocks * 4635)
        :param market_value: matrix recording the market value for each stock for (num_stocks)
        :return: forecast output output_ps, backcast output sharedInfo_back
        """
        # variables definition
        device = torch.device("cpu")  # probably for potential GPU choice
        (num_stocks, num_attributes) = concept_matrix.shape
        # build the weight from each stock to the predefined concept
        marketValue_matrix = market_value.reshape(num_stocks, 1).repeat(1, num_attributes)
        stock2concept_matrix = concept_matrix * marketValue_matrix  # c in [4], market capitalization
        stock2concept_sum = torch.sum(stock2concept_matrix, 0).reshape(1, -1).repeat(num_stocks, 1)
        stock2concept_sum = concept_matrix * stock2concept_sum  # same as M1.mul(M2)
        stock2concept_sum += torch.ones(num_stocks, num_attributes)  # make sum legal to be denominate
        # weight from stock (alpha0)
        stock2concept_origin = stock2concept_matrix / stock2concept_sum  # alpha0 in [4], representing the weight of size 10 * 4635
        # initial representation (e0),
        initial_rep = torch.t(stock2concept_origin).mm(x0)  # [5]
        initial_rep = initial_rep[initial_rep.sum(1) != 0]  # [5] keep only relative concepts for saving computation
        # weight from stock (alpha1)
        cosSimilarity_initial = global_func.cal_cos_similarity(x0, initial_rep)  # [6] similarity
        stock2concept_update = self.softmax_s2t(cosSimilarity_initial)  # [6] softmax normalization
        # update representation (e1)
        update_rep = self.fc_ps(torch.t(stock2concept_update).mm(x0))  # [7]
        # weight from concept (beta)
        cosSimilarity_update = global_func.cal_cos_similarity(x0, update_rep)  # [10] similarity
        concept2stock = self.softmax_t2s(cosSimilarity_update)  # [10] softmax normalization
        # shared information (s0)
        sharedInfo = self.fc_ps(concept2stock.mm(update_rep))   # [11]
        # outputs
        sharedInfo_back = self.leaky_relu(self.fc_ps_back(sharedInfo))  # [12] x0_hat
        sharedInfo_fore = self.leaky_relu(self.fc_ps_fore(sharedInfo))  # [12] y0
        output_ps = self.fc_out_ps(sharedInfo_fore).squeeze()

        # assign to class
        self.p_sharedInfo_back = sharedInfo_back
        self.p_sharedInfo_fore = sharedInfo_fore
        return sharedInfo_back, output_ps

    def predefined_concept_novelty(self, x0, concept_matrix: torch.tensor):
        # Implementation in TGC
        (num_stocks, num_relations) = concept_matrix.shape[1:3] # concept_matrix as relation_matrix(N,N,K)
        (num_stocks, hidden_size) = x0.shape
        # print("x0:", x0.shape)
        # print(concept_matrix.size())
        sharedInfo = np.zeros((num_stocks, hidden_size))
        for i in range(num_stocks):
            for j in range(num_stocks):
                dj = 0
                # print("predefined: ",i,j)
                if j == i or torch.sum(concept_matrix[i,j,:]==0): 
                    continue
                else:
                    dj += 1  # num of j satisfy
                    similarity = torch.t(x0[i]).mm(x0[j])
                    relation_importance = self.fc_ps(concept_matrix[i,j,:])
                    relation_strength = similarity + relation_importance   #Explicit Modeling
                    sharedInfo[i, :] += relation_strength * x0[j] / dj
        
        
        # shared information (s0)
        # sharedInfo = self.fc_ps(concept2stock.mm(update_rep))   # [11]

        # outputs
        sharedInfo = torch.from_numpy(sharedInfo).float()
        sharedInfo_back = self.leaky_relu(self.fc_ps_back(sharedInfo))  # [12] x0_hat
        sharedInfo_fore = self.leaky_relu(self.fc_ps_fore(sharedInfo))  # [12] y0
        output_ps = self.fc_out_ps(sharedInfo_fore).squeeze()

        # assign to class
        self.p_sharedInfo_back = sharedInfo_back
        self.p_sharedInfo_fore = sharedInfo_fore
        return sharedInfo_back, output_ps

    def hidden_concept(self, x1):
        """
        Calculates only ONE day's hidden result
        :param x1: encoded feature after excluding shared info (num_stocks * hidden_size)
        :return: forecast output output_ps, backcast output sharedInfo_back
        """
        # variables definition
        device = torch.device("cpu")  # probably for potential GPU choice
        stock2concept = global_func.cal_cos_similarity(x1, x1)  ### ???????????????
        dim = stock2concept.shape[0]
        diag = stock2concept.diagonal(0)
        stock2concept = stock2concept * (torch.ones(dim, dim) - torch.eye(dim)).to(device)
        # for each row and column
        row = torch.linspace(0, dim - 1, dim).reshape([-1, 1]).repeat(1, self.K).reshape(1, -1).long().to(device)
        col = torch.topk(stock2concept, self.K, dim=1)[1].reshape(1, -1)
        mask = torch.zeros([stock2concept.shape[0], stock2concept.shape[1]], device=stock2concept.device)
        mask[row, col] = 1
        stock2concept = stock2concept * mask
        stock2concept += torch.diag_embed((stock2concept.sum(0) != 0).float() * diag)
        # build hidden concept (u0)
        hiddenConcept = torch.t(x1).mm(stock2concept).t()
        hiddenConcept = hiddenConcept[hiddenConcept.sum(1) != 0]
        # weight from concept (gamma)
        cosSimilarity = global_func.cal_cos_similarity(x1, hiddenConcept)  # [10] similarity
        concept2stock = self.softmax_t2s(cosSimilarity)  # [10] softmax normalization
        # shared information (s1)
        sharedInfo = self.fc_ps(concept2stock.mm(hiddenConcept))
        # outputs
        sharedInfo_back = self.leaky_relu(self.fc_hs_back(sharedInfo))  # [12] x1_hat
        sharedInfo_fore = self.leaky_relu(self.fc_ps_fore(sharedInfo))  # [12] y1
        output_hs = self.fc_out_ps(sharedInfo_fore).squeeze()

        # assign to class
        self.h_sharedInfo_back = sharedInfo_back
        self.h_sharedInfo_fore = sharedInfo_fore
        return sharedInfo_back, output_hs

    def individual_concept(self, x2):
        """
        Calculates only ONE day's individual result
        :param x2: encoded feature after excluding shared info (num_stocks * hidden_size)
        :return: forecast output output_in
        """
        individualInfo = self.leaky_relu(self.fc_individual(x2))  # [13]
        output_individual = self.fc_out_individual(individualInfo).squeeze()

        # assign to class
        self.individualInfo = individualInfo
        return output_individual

    def predict(self):
        """
        Calculates only ONE day's prediction
        :return: prediction result for current day
        """
        allInfo = self.p_sharedInfo_fore + self.h_sharedInfo_fore + self.individualInfo
        pred_all = self.fc_out(allInfo).squeeze()

        # assign to class
        self.pred = pred_all
        return pred_all

    def forward(self,x,concept_matrix:torch.tensor,market_value:torch.tensor):
        '''encode_feature'''
        x0 = self.encode_feature(x)

        '''predefined_concept'''
        self.predefined_concept_novelty(x0,concept_matrix)

        '''hidden_concept'''
        x1 = x0 - self.p_sharedInfo_back
        self.hidden_concept(x1)

        '''individual_concept'''
        x2 = x1 - self.h_sharedInfo_back
        self.individual_concept(x2)

        '''predict'''
        return self.predict()