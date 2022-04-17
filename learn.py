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
from tqdm import tqdm
import random
from zmq import device
# pytorch
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
# self edit files
import global_func
from model import OptHIST
from model_org import HIST
from global_func import DataLoader

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def average_params(params_list):
    assert isinstance(params_list, (tuple, list, collections.deque))
    n = len(params_list)
    if n == 1:
        return params_list[0]
    new_params = collections.OrderedDict()
    keys = None
    for i, params in enumerate(params_list):
        if keys is None:
            keys = params.keys()
        for k, v in params.items():
            if k not in keys:
                raise ValueError('the %d-th model has different params'%i)
            if k not in new_params:
                new_params[k] = v / n
            else:
                new_params[k] += v / n
    return new_params

def mse(pred, label):
    loss = (pred - label)**2
    return torch.mean(loss)

def loss_fn(pred, label):
    mask = ~torch.isnan(label)
    return mse(pred[mask], label[mask])

def metric_fn(preds):
    preds = preds[~np.isnan(preds['label'])]
    precision = {}
    recall = {}
    temp = preds.groupby(level='datetime').apply(lambda x: x.sort_values(by='score', ascending=False))
    if len(temp.index[0]) > 2:
        temp = temp.reset_index(level =0).drop('datetime', axis = 1)
        
    for k in [1, 3, 5, 10, 20, 30, 50, 100]:
        precision[k] = temp.groupby(level='datetime').apply(lambda x:(x.label[:k]>0).sum()/k).mean()
        recall[k] = temp.groupby(level='datetime').apply(lambda x:(x.label[:k]>0).sum()/(x.label>0).sum()).mean()

    ic = preds.groupby(level='datetime').apply(lambda x: x.label.corr(x.score)).mean()
    rank_ic = preds.groupby(level='datetime').apply(lambda x: x.label.corr(x.score, method='spearman')).mean()

    return precision, recall, ic, rank_ic

global_step = -1
def train_epoch(epoch, model, optimizer, train_loader, stock2concept_matrix = None, marketValue = None):

    global global_step

    model.train()

    batch_size = 10
    num_companies = 308
    dates = 1000
    """关于时间的循环应该加在哪里？"""
    for i in range(batch_size):
        global_step += 1
        r = random.randrange(int(num_companies/2-5),int(num_companies/2+5))
        stock_index = torch.from_numpy(np.array(random.sample(range(num_companies), r))).long()
        # print(stock_index)
        # x0_t = model.encode_feature(data_matrix_train[t])   # # hidden_size attributes remained (360 > 64)
        # marketValue_t = marketValue[t * num_companies: (t + 1) * num_companies]
        # feature, label, market_value , stock_index, _ = train_loader.get(slc)
        random_date = random.randrange(0,dates)
        # print(random_date)
        repeat_date_companies = torch.ones(r)*random_date*num_companies
        # print(repeat_date_companies)
        idx = stock_index+repeat_date_companies

        feature = train_loader[idx.long()]

        # pred = model(feature, stock2concept_matrix[stock_index], marketValue[idx.long()])
        # print(stock2concept_matrix.size())
        pred = model(feature, stock2concept_matrix, marketValue[idx.long()])
        repeat_date_companies_label = torch.ones(r)*(random_date+1)*num_companies
        idx_label = stock_index + repeat_date_companies_label
        label = marketValue[idx_label.long()].squeeze()
        loss = loss_fn(pred, label)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 3.)
        optimizer.step()


def test_epoch(epoch, model, test_loader, stock2concept_matrix=None, marketValue = None, prefix='Test'):

    model.eval()

    losses = []
    preds = []
    if prefix=='Valid':
        dates_start = 1001
        dates_end = 1500
        # batch_size = 10
    elif prefix == 'Test':
        dates_start = 1501
        dates_end = 2000
        # batch_size = 10
    else:
        dates_start = 0
        dates_end = 1000
        # batch_size = 10
    num_companies = 308
    print(prefix)

    # for i, slc in tqdm(test_loader.iter_daily(), desc=prefix, total=test_loader.daily_length):
    # for i in range(dates_start, dates_end):
    for i in range(100):
        print("test_epoch: ", i)
        r = num_companies
        stock_index = torch.arange(0, r).long()
        # random_date = i
        random_date = random.randrange(dates_start,dates_end)
        repeat_date_companies = torch.ones(r)*random_date*num_companies
        idx = stock_index+repeat_date_companies

        feature = test_loader[idx.long()]

        # pred = model(feature, stock2concept_matrix[stock_index], marketValue[idx.long()])
        pred = model(feature, stock2concept_matrix, marketValue[idx.long()])
        repeat_date_companies_label = torch.ones(r)*(random_date+1)*num_companies
        idx_label = stock_index + repeat_date_companies_label
        label = marketValue[idx_label.long()].squeeze()

        arrays = [(np.ones(r)*(random_date+1)), stock_index.numpy()]
        # print(arrays)
        index = pd.MultiIndex.from_arrays(arrays, names=('datetime', 'stock'))
        # print(index)

        with torch.no_grad():
            pred = model(feature, stock2concept_matrix[stock_index], marketValue[idx.long()])
            loss = loss_fn(pred, label)
            preds.append(pd.DataFrame({ 'score': pred.cpu().numpy(), 'label': label.cpu().numpy(), }, index=index))

        losses.append(loss.item())
    #evaluate
    # print(preds)
    preds = pd.concat(preds, axis=0)
    precision, recall, ic, rank_ic = metric_fn(preds)
    scores = ic

    # writer.add_scalar(prefix+'/Loss', np.mean(losses), epoch)
    # writer.add_scalar(prefix+'/std(Loss)', np.std(losses), epoch)
    # writer.add_scalar(prefix+'/'+args.metric, np.mean(scores), epoch)
    # writer.add_scalar(prefix+'/std('+args.metric+')', np.std(scores), epoch)

    return np.mean(losses), scores, precision, recall, ic, rank_ic

def inference(model, data_loader, stock2concept_matrix=None, prefix='test'):

    model.eval()

    preds = []
    if prefix=='valid':
        dates_start = 1001
        dates_end = 1500
        # batch_size = 10
    elif prefix == 'test':
        dates_start = 1501
        dates_end = 2000
        # batch_size = 10
    else:
        dates_start = 0
        dates_end = 1000
        # batch_size = 10
    num_companies = 308
    print(prefix)

    # for i, slc in tqdm(test_loader.iter_daily(), desc=prefix, total=test_loader.daily_length):
    # for i in range(dates_start, dates_end):
    for i in range(50):
        print("inference: ", i)
        r = num_companies
        stock_index = torch.arange(0, r).long()
        # random_date = i
        random_date = random.randrange(dates_start,dates_end)
        repeat_date_companies = torch.ones(r)*random_date*num_companies
        idx = stock_index+repeat_date_companies

        feature = data_loader[idx.long()]

        pred = model(feature, stock2concept_matrix[stock_index], marketValue[idx.long()])
        pred = model(feature, stock2concept_matrix, marketValue[idx.long()])
        repeat_date_companies_label = torch.ones(r)*(random_date+1)*num_companies
        idx_label = stock_index + repeat_date_companies_label
        label = marketValue[idx_label.long()].squeeze()

        arrays = [(np.ones(r)*(random_date+1)), stock_index.numpy()]
        index = pd.MultiIndex.from_arrays(arrays, names=('datetime', 'stock'))
        # print(index)

        with torch.no_grad():
            pred = model(feature, stock2concept_matrix[stock_index], marketValue[idx.long()])
            loss = loss_fn(pred, label)
            preds.append(pd.DataFrame({ 'score': pred.cpu().numpy(), 'label': label.cpu().numpy(), }, index=index))

    preds = pd.concat(preds, axis=0)
    return preds


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
    # (num_stocks, num_attributes) = dataset.stock2concept.shape
    dates = 1000
    valid_dates = 500
    test_dates = 500
    num_companies = 308
    size_train = dates * num_companies
    end_valid = (dates+valid_dates) * num_companies
    end_test = (dates+valid_dates+test_dates) * num_companies
    # variables declaration
    seed = np.random.randint(1000000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    data_matrix_train = data_matrix[0:size_train]   # generate training data set
    # print(data_matrix_train.size())
    data_matrix_valid = data_matrix[0:end_valid]
    data_matrix_test = data_matrix[0:end_test]
    # print(data_matrix_train.size())

    # initialize model
    model = OptHIST()
    # model_org = HIST()
    # data matrix modification
    # data_matrix_train = data_matrix_train.reshape(dates, num_companies, -1)
    # print(data_matrix_train.size())

    """ training """
    # for t in range(5):
    #     # encode feature
    #     x0_t = model.encode_feature(data_matrix_train[t])   # # hidden_size attributes remained (360 > 64)
    #     marketValue_t = marketValue[t * num_companies: (t + 1) * num_companies]
    #     # predefined module
    #     p_sharedInfo_back_t, p_sharedIno_fore_t = model.predefined_concept(x0_t, stock2concept, marketValue_t)
    #     # hidden module
    #     x1_t = x0_t - p_sharedInfo_back_t
    #     h_sharedInfo_back_t, h_sharedIno_fore_t = model.hidden_concept(x1_t)
    #     # individual module
    #     x2_t = x1_t - h_sharedInfo_back_t
    #     model.individual_concept(x2_t)
    #     # predict all
    #     pred_all = model.predict()
    #     # pred_all_org = model_org.forward(data_matrix_train[t], stock2concept, marketValue_t)
    #     print("Day " + str(t) + ":")
    #     print(pred_all)
    #     # print(pred_all_org)


    all_precision = []
    all_recall = []
    all_ic = []
    all_rank_ic = []
    for times in range(20):
        print('create model...')
        model = OptHIST(input_size=6, num_layers=2, K=3)
        
        # model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=0.002)
        best_score = -np.inf
        best_epoch = 0
        stop_round = 0
        best_param = copy.deepcopy(model.state_dict())
        params_list = collections.deque(maxlen=2)
        for epoch in range(3):
            print('Running', times,'Epoch:', epoch)

            print('training...')
            train_epoch(epoch, model, optimizer, data_matrix_train, stock2concept, marketValue)

            # torch.save(model.state_dict(), output_path+'/model.bin.e'+str(epoch))
            # torch.save(optimizer.state_dict(), output_path+'/optimizer.bin.e'+str(epoch))

            params_ckpt = copy.deepcopy(model.state_dict())
            params_list.append(params_ckpt)
            avg_params = average_params(params_list)
            model.load_state_dict(avg_params)

            print('evaluating...')
            train_loss, train_score, train_precision, train_recall, train_ic, train_rank_ic = test_epoch(epoch, model, data_matrix_train, stock2concept, marketValue, prefix='Train')
            val_loss, val_score, val_precision, val_recall, val_ic, val_rank_ic = test_epoch(epoch, model, data_matrix_valid, stock2concept, marketValue, prefix='Valid')
            test_loss, test_score, test_precision, test_recall, test_ic, test_rank_ic = test_epoch(epoch, model, data_matrix_test, stock2concept, marketValue, prefix='Test')

            print('train_loss %.6f, valid_loss %.6f, test_loss %.6f'%(train_loss, val_loss, test_loss))
            print('train_score %.6f, valid_score %.6f, test_score %.6f'%(train_score, val_score, test_score))
            # pprint('train_mse %.6f, valid_mse %.6f, test_mse %.6f'%(train_mse, val_mse, test_mse))
            # pprint('train_mae %.6f, valid_mae %.6f, test_mae %.6f'%(train_mae, val_mae, test_mae))
            print('train_ic %.6f, valid_ic %.6f, test_ic %.6f'%(train_ic, val_ic, test_ic))
            print('train_rank_ic %.6f, valid_rank_ic %.6f, test_rank_ic %.6f'%(train_rank_ic, val_rank_ic, test_rank_ic))
            print('Train Precision: ', train_precision)
            print('Valid Precision: ', val_precision)
            print('Test Precision: ', test_precision)
            print('Train Recall: ', train_recall)
            print('Valid Recall: ', val_recall)
            print('Test Recall: ', test_recall)
            model.load_state_dict(params_ckpt)

            if val_score > best_score:
                best_score = val_score
                stop_round = 0
                best_epoch = epoch
                best_param = copy.deepcopy(avg_params)
            else:
                stop_round += 1
                # if stop_round >= args.early_stop:
                if stop_round >= 3:
                    print('early stop')
                    break

        print('best score:', best_score, '@', best_epoch)
        model.load_state_dict(best_param)
        # torch.save(best_param, output_path+'/model.bin')

        print('inference...')
        res = dict()
        for name in ['train', 'valid', 'test']:

            pred= inference(model, eval('data_matrix_'+name), stock2concept,name)
            # pred.to_pickle(output_path+'/pred.pkl.'+name+str(times))

            precision, recall, ic, rank_ic = metric_fn(pred)
            # print("FLOAT?????????????????")
            # print(ic, rank_ic)
            # if len(ic) == 1:
            print(('%s: IC %.6f Rank IC %.6f')%(
                    name, ic, rank_ic))
            # else:
            #     print(('%s: IC %.6f Rank IC %.6f')%(
            #                 name, ic.mean(), rank_ic.mean()))
            print(name, ': Precision ', precision)
            print(name, ': Recall ', recall)
            res[name+'-IC'] = ic
            # res[name+'-ICIR'] = ic.mean() / ic.std()
            res[name+'-RankIC'] = rank_ic
            # res[name+'-RankICIR'] = rank_ic.mean() / rank_ic.std()
        
        all_precision.append(list(precision.values()))
        all_recall.append(list(recall.values()))
        all_ic.append(ic)
        all_rank_ic.append(rank_ic)

        print('save info...')
        # writer.add_hparams(
        #     vars(args),
        #     {
        #         'hparam/'+key: value
        #         for key, value in res.items()
        #     }
        # )

        # info = dict(
        #     config=vars(args),
        #     best_epoch=best_epoch,
        #     best_score=res,
        # )

    print(('IC: %.4f (%.4f), Rank IC: %.4f (%.4f)')%(np.array(all_ic).mean(), np.array(all_ic).std(), np.array(all_rank_ic).mean(), np.array(all_rank_ic).std()))
    precision_mean = np.array(all_precision).mean(axis= 0)
    precision_std = np.array(all_precision).std(axis= 0)
    N = [1, 3, 5, 10, 20, 30, 50, 100]
    for k in range(len(N)):
        print (('Precision@%d: %.4f (%.4f)')%(N[k], precision_mean[k], precision_std[k]))

    print('finished.')
