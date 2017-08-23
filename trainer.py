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

class Trainer(object):
    """docstring for Trainer."""
    def __init__(self, model, optimizer, mini_batch_size=100, auto_encode=True, train_count=1, save_dir='./'):
        super(Trainer, self).__init__()
        self.model = model
        self.opt = optimizer
        self.opt.setup(model)
        self.save_dir = save_dir
        self.mini_batch_size = mini_batch_size
        self.auto_encode = auto_encode
        self.bigdata = []
        self.dao = DataAccessObj()
        self.global_t = 0

    def train(self):
        pass

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

def main(load=None, mini_batch_size=1000, future=30, epoch_len=3600):
    db_itr = db['step3'].find().batch_size(100)
    bigdata_list = []
    for data in db_itr:
        data_list = []
        data_list.append(data['ltp'])
        data_list.extend(data['conved_board'])
        data_list.extend(data['ticker'])
        bigdata_list.append(data_list)
        # print data['conv_board']
        # print data['ticker']
        # print data_list
        # print 'length:', len(data_list)

    bigdata_size = len(bigdata_list) - future
    for i in range(bigdata_size):
        label = classify_ltp_change(bigdata_list[i][0],bigdata_list[i+future][0])
        bigdata_list[i][0] = label

    model = PredictModel(input_num=73)
    if load is not None:
        serializers.load_npz(os.path.join(load, 'model.npz'), model)
        print 'loaded .npz file.'
    elif os.path.isfile('./model.npz'):
        serializers.load_npz(os.path.join('./model.npz'), model)
        print 'loaded ./model.npz'


    optimizer = optimizers.Adam()
    optimizer.setup(model)

    global_t = 1
    while True:
        # lstmをリセット
        model.lstm1.reset_state()
        # bigdata_list-future個の中からmini_batch_size分データを選択
        indexes = np.random.randint(0,bigdata_size-future,mini_batch_size)

        # 選択したデータを+1しながらループ
        t = 1
        while True:
            x_list = [bigdata_list[i][1:] for i in indexes]
            x = chainer.Variable(np.array(x_list).astype(np.float32)).reshape(-1,73)
            # print 'x_shape:', x.shape
            y_list = [bigdata_list[i][0] for i in indexes]
            y = chainer.Variable(np.array(y_list).astype(np.int32)).reshape(mini_batch_size)
            # print 'y_shape:', y.shape
            h = model(x)
            loss = F.softmax_cross_entropy(h, y)
            model.zerograds()
            loss.backward()
            optimizer.update()
            # print y
            # print h
            print '%s loss: %s' % (global_t, loss.data)
            global_t += 1
            t += 1
            # 選択したデータのリストの最大値がbigdata_list-futureを超えたらbreak
            indexes+=1
            if bigdata_size <= np.max(indexes) or t%epoch_len==0:
                print 'break'
                break

        serializers.save_npz(os.path.join("./", "model.npz"), model)
        print 'model saved'


def args_parse():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--batch', type=int, default=100)
    parser.add_argument('--future', type=int, default=30)
    parser.add_argument('--epoch_len', type=int, default=3600)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = args_parse()
    main(load=args.load, mini_batch_size=args.batch, future=args.future, epoch_len=args.epoch_len)
