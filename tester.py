#coding: utf-8
import os
import sys
sys.path.append(os.pardir)
from tools.dao import DataAccessObj

import threading
from multiprocessing import Pool
import time
import numpy as np
import datetime as dt
import copy
import csv

import chainer
import chainer.functions as F
from chainer import serializers, optimizers


from conv_board import loader
from tools.models import *
from predictor import PredictModel


from pymongo import MongoClient

mongo_client = MongoClient('localhost', 27017)
db = mongo_client['bitflyer(FX_BTC_JPY)-2017-05-23-cache']

def classify_ltp_change(before, after):
    diff = after - before
    res = None
    if diff < 0:
        res = [0]
    elif 0 < diff:
        res = [1]
    elif diff == 0:
        res = [2]
    return np.array(res).astype(np.int32)

def main(load='./', mini_batch_size=1000, future=30, epoch_len=3600):
    db_itr = db['step3'].find().batch_size(100)
    bigdata_list = []
    for data in db_itr:
        data_list = []
        data_list.append(data['ltp'])
        data_list.extend(data['conved_board'])
        data_list.extend(data['ticker'])
        bigdata_list.append(data_list)

    bigdata_size = len(bigdata_list) - future
    for i in range(bigdata_size):
        label = classify_ltp_change(bigdata_list[i][0],bigdata_list[i+future][0])
        # 0番目にラベルデータ入力
        bigdata_list[i][0] = label

    model = PredictModel(input_num=73)
    if load is not None:
        serializers.load_npz(os.path.join(load, "model.npz"), model)

    accurate_count = 0
    num_of_enable_data = 0

    for i in range(1, bigdata_size):
        x_list = [bigdata_list[i][1:]]
        x = chainer.Variable(np.array(x_list).astype(np.float32)).reshape(-1,73)
        y_list = [bigdata_list[i][0]]
        y = chainer.Variable(np.array(y_list).astype(np.int32)).reshape(1).data
        h = model(x)

        if np.argmax(h.data) != 2 and y[0] != 2:
            print 'not 2'
            accurate_count += F.accuracy(h, y).data
            num_of_enable_data += 1
            accuracy = accurate_count / num_of_enable_data

        print 'label:%s predict:%s accuracy:%s' % (y, np.argmax(h.data), accuracy)

def args_parse():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--data', type=int, default=100)
    parser.add_argument('--batch', type=int, default=100)
    parser.add_argument('--bigdata', type=int, default=10000)
    parser.add_argument('--epoch_len', type=int, default=3600)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = args_parse()
    main()
