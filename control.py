"""
Authors: Zeqian Cai, Yurui Chang, Jinhuan Ke, Yichen Li, Suyuan Wang
Date: 2022/03/06
Version: V 1.0
Function: To build the model class
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.layers import Dense, LSTM, GRU
from keras.models import Sequential
from scipy.stats import pearsonr, spearmanr

# parameters
num_stocks = 308
lens = 60
model_name = "GRU"


def load_data(filename: str):
    return pd.read_csv(filename)


def metrix_fn(pred_list, test_list, mse_list):
    # parameters declaration
    k_list = [1, 3, 5, 10, 20, 30, 50, 100]
    ic_all, rank_ic_all = [], []
    precision, recall = {}, {}

    # ic and rank_ic
    for t in range(pred_list.shape[1]):
        # ic
        ic_all.append(pearsonr(pred_list[:, t], test_list[:, t]))
        # rank_ic
        pred = np.sort(pred_list[:, t])[::-1]
        test = np.sort(test_list[:, t])[::-1]
        rank_ic_all.append(spearmanr(pred, test))
    ic = np.mean(ic_all)
    rank_ic = np.mean(rank_ic_all)
    # precision
    score = mse_list.argsort()
    for k in k_list:
        precision[k] = np.sum([np.sum(pred_list[index, :] > 0) for index in score[:k]]) / (k * pred_list.shape[1])
        recall[k] = np.sum([np.sum(pred_list[index, :] > 0) for index in score[:k]]) / np.sum([np.sum(pred_list[:, :] > 0)])

    return ic, rank_ic, precision, recall


def main():
    # data loading
    feature = load_data("./data/new_feature_all.csv")
    # modification
    feature = feature.iloc[:, 2:62].values

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(feature).reshape((-1, 308, 60)) + 10e-8

    # parameters definition
    num_train = 1500
    num_days = scaled_data.shape[0]
    prediction_list, test_list, mse_list = [], [], []

    # training
    x_train_all = scaled_data[:num_train, :, :]
    y_train_all = scaled_data[1:num_train+1, :, -1]
    x_test_all = scaled_data[num_train:, :, :]
    y_test_all = scaled_data[num_train:, :, -1]

    # build model
    if model_name == "LSTM":
        model = Sequential()
        model.add(LSTM(units=32, return_sequences=True, input_shape=(lens, 1)))
        model.add(LSTM(units=16))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='mean_squared_error', optimizer='adam')
    elif model_name == "GRU":
        model = Sequential()
        model.add(GRU(units=32, return_sequences=True))
        model.add(GRU(units=16))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='mean_squared_error', optimizer='adam')

    # for each company
    for stock_index in range(num_stocks):
        # prepare data
        x_train = x_train_all[:, stock_index, :]
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        y_train = y_train_all[:, stock_index]
        x_test = x_test_all[:, stock_index, :]
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
        y_test = y_test_all[:, stock_index]

        # train model
        model.fit(x_train, y_train, batch_size=500, epochs=1)

        # predict
        prediction = model.predict(x_test).squeeze()
        prediction_list.append(np.insert(np.diff(prediction), 0, 10e-8, axis=0) / prediction)
        test_list.append(np.insert(np.diff(y_test), 0, 10e-8, axis=0) / y_test)

        # error score
        mse_list.append(mean_squared_error(prediction, y_test))

    # get metrics
    ic, rank_ic, precision, recall = metrix_fn(np.array(prediction_list), np.array(test_list), np.array(mse_list))
    print("ic:" + str(ic))
    print("rank_ic:" + str(rank_ic))
    print("precision:" + str(precision))
    print("recall:" + str(recall))


if __name__ == "__main__":
    main()

