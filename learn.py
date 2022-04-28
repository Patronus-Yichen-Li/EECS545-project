"""
Authors: Zeqian Cai, Yurui Chang, Jinhuan Ke, Yichen Li, Suyuan Wang
Date: 2022/03/06
Version: V 1.0
Function: To build the model class
"""

import copy
import collections
import numpy as np
import pandas as pd
import random
# pytorch
import torch
import torch.optim as optim
# self edit files
import global_func
from model import OptHIST
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
def train_epoch(device, model, optimizer, train_loader, stock2concept_matrix = None, closePrice = None):

    global global_step

    model.train()

    num_companies = global_func.num_companies
    for i in range(global_func.train_batch_size):
        global_step += 1
        # randomly select r companies
        r = random.randrange(global_func.min_num,global_func.max_num)
        stock_index = torch.from_numpy(np.array(random.sample(range(num_companies), r))).long()
        # randomly select a date
        random_date = random.randrange(global_func.train_start_date,global_func.train_end_date)
        repeat_date_companies = torch.ones(r)*random_date*num_companies
        # find the related row numbers in train_loader
        idx = stock_index+repeat_date_companies
        feature = train_loader[idx.long()]

        if global_func.model == "OptHIST":
            pred = model(feature.to(device), stock2concept_matrix.to(device),stock_index.to(device),closePrice[idx.long()].to(device))
        else:
            pred = model(feature.to(device), stock2concept_matrix[stock_index].to(device),stock_index.to(device),closePrice[idx.long()].to(device))
        
        # calculate the trend as the model's label
        repeat_date_companies_label = torch.ones(r)*(random_date+1)*num_companies
        idx_label = stock_index + repeat_date_companies_label
        close_price = closePrice[idx_label.long()].squeeze()
        prev_close_price = train_loader[idx.long(),59]
        label = (close_price-prev_close_price) / prev_close_price
        
        loss = loss_fn(pred, label.to(device))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 3.)
        optimizer.step()


def test_epoch(device, model, test_loader, stock2concept_matrix=None, closePrice = None, prefix='Test'):

    model.eval()

    losses = []
    preds = []
    if prefix=='Valid':
        dates_start = global_func.valid_start_date
        dates_end = global_func.valid_end_date
    elif prefix == 'Test':
        dates_start = global_func.test_start_date
        dates_end = global_func.test_end_date
    else:
        dates_start = global_func.train_start_date
        dates_end = global_func.train_end_date
    num_companies = global_func.num_companies
    print(prefix+"_test_epoch_function")

    for i in range(global_func.test_batch_size):
        # r = num_companies
        # stock_index = torch.arange(0, r).long()
        r = random.randrange(global_func.min_num,global_func.max_num)
        stock_index = torch.from_numpy(np.array(random.sample(range(num_companies), r))).long()
        random_date = random.randrange(dates_start,dates_end)
        repeat_date_companies = torch.ones(r)*random_date*num_companies
        idx = stock_index+repeat_date_companies

        feature = test_loader[idx.long()]

        repeat_date_companies_label = torch.ones(r)*(random_date+1)*num_companies
        idx_label = stock_index + repeat_date_companies_label
        close_price = closePrice[idx_label.long()].squeeze()
        prev_close_price = test_loader[idx.long(),59]
        label = (close_price-prev_close_price) / prev_close_price

        arrays = [(np.ones(r)*(random_date+1)), stock_index.numpy()]
        index = pd.MultiIndex.from_arrays(arrays, names=('datetime', 'stock'))

        with torch.no_grad():
            if global_func.model == "OptHIST":
                pred = model(feature.to(device), stock2concept_matrix.to(device),stock_index.to(device),closePrice[idx.long()].to(device))
            else:
                pred = model(feature.to(device), stock2concept_matrix[stock_index].to(device),stock_index.to(device),closePrice[idx.long()].to(device))
            loss = loss_fn(pred, label.to(device))
            preds.append(pd.DataFrame({ 'score': pred.cpu().numpy(), 'label': label.cpu().numpy(), }, index=index))

        losses.append(loss.item())
    #evaluate
    preds = pd.concat(preds, axis=0)
    precision, recall, ic, rank_ic = metric_fn(preds)
    scores = ic

    return np.mean(losses), scores, precision, recall, ic, rank_ic

def inference(device, model, data_loader, stock2concept_matrix=None, closePrice = None, prefix='Test'):

    model.eval()

    preds = []
    if prefix=='Valid':
        dates_start = global_func.valid_start_date
        dates_end = global_func.valid_end_date
    elif prefix == 'Test':
        dates_start = global_func.test_start_date
        dates_end = global_func.test_end_date
    else:
        dates_start = global_func.train_start_date
        dates_end = global_func.train_end_date
    num_companies = global_func.num_companies
    print(prefix+"_inference_function")

    for i in range(global_func.inference_batch_size):
        # r = num_companies
        # stock_index = torch.arange(0, r).long()
        r = random.randrange(global_func.min_num,global_func.max_num)
        stock_index = torch.from_numpy(np.array(random.sample(range(num_companies), r))).long()
        random_date = random.randrange(dates_start,dates_end)
        repeat_date_companies = torch.ones(r)*random_date*num_companies
        idx = stock_index+repeat_date_companies

        feature = data_loader[idx.long()].to(device)

        repeat_date_companies_label = torch.ones(r)*(random_date+1)*num_companies
        idx_label = stock_index + repeat_date_companies_label
        close_price = closePrice[idx_label.long()].squeeze()
        prev_close_price = data_loader[idx.long(),59]
        label = (close_price-prev_close_price) / prev_close_price

        arrays = [(np.ones(r)*(random_date+1)), stock_index.numpy()]
        index = pd.MultiIndex.from_arrays(arrays, names=('datetime', 'stock'))

        with torch.no_grad():
            if global_func.model == "OptHIST":
                pred = model(feature.to(device), stock2concept_matrix.to(device),stock_index.to(device),closePrice[idx.long()].to(device))
            else:
                pred = model(feature.to(device), stock2concept_matrix[stock_index].to(device),stock_index.to(device),closePrice[idx.long()].to(device))
            preds.append(pd.DataFrame({ 'score': pred.cpu().numpy(), 'label': label.cpu().numpy(), }, index=index))

    preds = pd.concat(preds, axis=0)
    return preds


if __name__ == "__main__":
    # load data
    dataset = DataLoader()
    # data modification (basically from df & array to tensor)
    data_matrix = dataset.stock_price.drop(["datetime", "instrument"], axis=1)
    data_matrix = torch.from_numpy(data_matrix.to_numpy()).float()
    closePrice = dataset.market_value.drop(["datetime", "instrument"], axis=1)
    closePrice = torch.from_numpy(closePrice.to_numpy()).float()
    stock2concept = torch.from_numpy(dataset.stock2concept).float()

    # parameters
    num_companies = global_func.num_companies
    end_train = global_func.train_end_date * num_companies
    end_valid = global_func.valid_end_date * num_companies
    end_test = global_func.test_end_date * num_companies
    # variables declaration
    seed = np.random.randint(1000000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    data_matrix_train = data_matrix[0:end_train]   # generate training data set
    data_matrix_valid = data_matrix[0:end_valid]
    data_matrix_test = data_matrix[0:end_test]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print(device)

    all_precision = []
    all_recall = []
    all_ic = []
    all_rank_ic = []
    for times in range(global_func.repeat):
        print('create model...')
        model = OptHIST(input_size=6, num_layers=global_func.num_layers, K=global_func.K,device=device)
        
        model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=global_func.Adamlr)
        best_score = -np.inf
        best_epoch = 0
        stop_round = 0
        best_param = copy.deepcopy(model.state_dict())
        params_list = collections.deque(maxlen=global_func.smooth_steps)
        for epoch in range(global_func.epoch):
            print('Running', times,'Epoch:', epoch)

            print('training...')
            train_epoch(device, model, optimizer, data_matrix_train, stock2concept, closePrice)

            params_ckpt = copy.deepcopy(model.state_dict())
            params_list.append(params_ckpt)
            avg_params = average_params(params_list)
            model.load_state_dict(avg_params)

            print('evaluating...')
            train_loss, train_score, train_precision, train_recall, train_ic, train_rank_ic = test_epoch(device, model, data_matrix_train, stock2concept, closePrice, prefix='Train')
            val_loss, val_score, val_precision, val_recall, val_ic, val_rank_ic = test_epoch(device, model, data_matrix_valid, stock2concept, closePrice, prefix='Valid')
            test_loss, test_score, test_precision, test_recall, test_ic, test_rank_ic = test_epoch(device, model, data_matrix_test, stock2concept, closePrice, prefix='Test')

            print('train_loss %.6f, valid_loss %.6f, test_loss %.6f'%(train_loss, val_loss, test_loss))
            print('train_score %.6f, valid_score %.6f, test_score %.6f'%(train_score, val_score, test_score))
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
                if stop_round >= global_func.early_stop:
                    print('early stop')
                    break

        print('best score:', best_score, '@', best_epoch)
        model.load_state_dict(best_param)

        print('inference...')
        res = dict()
        for name in ['train', 'valid', 'test']:

            pred= inference(device,model, eval('data_matrix_'+name), stock2concept, closePrice ,name)

            precision, recall, ic, rank_ic = metric_fn(pred)
            print(('%s: IC %.6f Rank IC %.6f')%(name, ic, rank_ic))
            print(name, ': Precision ', precision)
            print(name, ': Recall ', recall)
            res[name+'-IC'] = ic
            res[name+'-RankIC'] = rank_ic
        
        all_precision.append(list(precision.values()))
        all_recall.append(list(recall.values()))
        all_ic.append(ic)
        all_rank_ic.append(rank_ic)


    print(('IC: %.4f (%.4f), Rank IC: %.4f (%.4f)')%(np.array(all_ic).mean(), np.array(all_ic).std(), np.array(all_rank_ic).mean(), np.array(all_rank_ic).std()))
    precision_mean = np.array(all_precision).mean(axis= 0)
    precision_std = np.array(all_precision).std(axis= 0)
    N = [1, 3, 5, 10, 20, 30, 50, 100]
    for k in range(len(N)):
        print (('Precision@%d: %.4f (%.4f)')%(N[k], precision_mean[k], precision_std[k]))

    # import os
    os.makedirs('output', exist_ok=True)
    df1 = pd.DataFrame(np.array(all_ic).T, columns=["all_ic"])
    df1.to_csv('./output/result_'+global_func.model+'_all_ic.csv')
    df2 = pd.DataFrame(np.array(all_rank_ic).T, columns=["all_rank_ic"])
    df2.to_csv('./output/result_'+global_func.model+'_all_rank_ic.csv')
    df3 = pd.DataFrame(np.array(all_precision), columns=["1","3","5","10","20","30","50","100"])
    df3.to_csv('./output/result_'+global_func.model+'_all_precision.csv')

    print('finished.')
