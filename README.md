# EECS545-project

# Stock Price Prediction based on Concept-Oriented Shared Information
#### Author: Zeqian Cai(czq@umich.edu), Yurui Chang(yuruic@umich.edu), Jinhuan Ke(jhgoblue@umich.edu), Yichen Li(liyichen@umich.edu), Suyuan Wang(wasuyuan@umich.edu)

## Introduction
Stock trend forecasting plays an essential role in investment, many methods such as tree methods and neural networks has been used for analysis. 

One important feature of stocks is that some of them have shared information, thus their corresponding stock prices are highly related. Some research overlooks their hidden connections, while a part of analyses are based on the assumptions that their mutual relationships are stationary. However, stock correlations are dynamic as many factors like the policy, market, and technology are changing.

## Files
./global_func.py: Implement global functions, announce global parameters, like similarity calculation  

./learn.py: (where __main__ lies) training, testing, and inferencing functions

./model.py: Implementation of OptHIST model class, with modified Predefined Module and Hidden Module

./model_org.py: The copy of original HIST model, for details, please refer to https://github.com/wentao-xu/hist  

./control.py: Implement LSTM and GRU algorithm with same amount of layers and similar model complexity to be the control group of prediction using OptHIST

./data_prep/: directory storing the script building data we need

## Data
Since the raw data files are too large to be uploaded to Github, please refer to this link (Google Drive) to acquire data

Google Drive: https://drive.google.com/file/d/1cI2XH-OxVe9qG4ri46hKdPyqKoyrthvn/view?usp=sharing

## Project introduction & Result Demostration
We have concluded our methods and novelties into a single poster, please refer to this link for reading or downloading 

Google Drive: https://drive.google.com/file/d/1hvWgrEZmiiulEB87ACTnc-_egBz8U1nW/view?usp=sharing

## Versions
03/16: Version 1.0: implemented the model structure, received the same output format

04/14: Version 2.0: updated model.py to the format of super class nn.Module in order to do further training and testing. Rearranged the concepts functions. Implemented and tunned the training/test epoch processes, the program can run smoothly

04/17: Version 3.0: modified learn.py; rearranging global functions and global parameters;

04/18: Version 4.0: Added control.py
