#coding: utf-8

import os
import sys
sys.path.append(os.pardir)
import numpy as np
from tools.models import *

# Chainer
import chainer
from chainer import serializers
from chainer import functions as F

from pymongo import MongoClient


mongo_client = MongoClient('localhost', 27017)
db = mongo_client['bitflyer(FX_BTC_JPY)-2017-05-23-cache']

def get_ltp_centered_board(book, ltp, pad=1024):
    ltp = int(ltp)
    mid_price = int(book.mid_price)
    ltp_centered_board = book.board[ltp-pad:ltp+pad+1]
    return ltp_centered_board

def step1(num=864000):
    # データを読み込み
    bigdata = []
    ticker_itr = db['tickers'].find().batch_size(100)
    book_itr = db['books'].find().batch_size(100)

    for i in range(0,num):
        print '%s件目のdata読み込み' % i
        data = {}
        ticker = ticker_itr.next()
        if i%10 == 0:
            book_json = book_itr.next()
            book = Book(zip_json=book_json)
            ltp_centered_board = get_ltp_centered_board(book, ticker['ltp'])

        print book_json['j_timestamp']
        print ticker['j_timestamp']

        data['ticker'] = ticker
        data['ltp_centered_board'] = ltp_centered_board
        bigdata.append(data)

    print 'finished loading'

    save_bigdata('step1',bigdata)

    return bigdata

def standardization(self, batch):
    std_batch = (batch - batch.mean()) / batch.std()
    return std_batch

def ticker2list(ticker):
    res_list = []
    res_list.append(ticker['best_bid'])
    res_list.append(ticker['best_bid_size'])
    res_list.append(ticker['best_ask'])
    res_list.append(ticker['best_ask_size'])
    res_list.append(ticker['volume'])
    res_list.append(ticker['ltp'])
    res_list.append(ticker['volume_by_product'])
    res_list.append(ticker['total_ask_depth'])
    res_list.append(ticker['total_bid_depth'])
    return res_list

# 標準化と正規化
def step2(bigdata=None):
    if bigdata is None:
        bigdata = load_bigdata('step1')

    ticker_list = []
    ticker_mean = None
    ticker_std = None
    ltp_centered_board_list = []
    ltp_centered_board_mean = None
    ltp_centered_board_std  = None

    # 標準化と正常化
    bigdata_list = []
    for data in bigdata:
        data['j_timestamp'] = data['ticker']['j_timestamp']
        data['ltp'] = data['ticker']['ltp']
        data['ticker'] = ticker2list(data['ticker'])
        ticker_list.append(data['ticker'])
        ltp_centered_board_list.append(data['ltp_centered_board'])
        bigdata_list.append(data)

    # ltp_centered_boardの平均と標準偏差
    ltp_centered_board_list = np.array(ltp_centered_board_list)
    ltp_centered_board_mean = ltp_centered_board_list.mean()
    ltp_centered_board_std  = ltp_centered_board_list.std()

    # tickerの平均と標準偏差
    ticker_list = np.array(ticker_list)
    ticker_mean = ticker_list.mean(axis=0)
    ticker_std  = ticker_list.std(axis=0)

    print 'ltp_centered_board_mean:', ltp_centered_board_mean
    print 'ltp_centered_board_std:', ltp_centered_board_std
    print 'ticker_mean:', ticker_mean
    print 'ticker_std:', ticker_std

    res_bigdata = []
    for data in bigdata_list:
        res_data = {}
        res_data['j_timestamp'] = data['j_timestamp']
        res_data['ltp'] = data['ltp']
        # 標準化(standardization)
        res_data['ltp_centered_board'] = \
            (data['ltp_centered_board'] - ltp_centered_board_mean)/ltp_centered_board_std
        res_data['ticker'] = (data['ticker'] - ticker_mean)/ticker_std

        # 正規化(normarization)
        res_data['ltp_centered_board'] = F.tanh(res_data['ltp_centered_board']).data.tolist()
        res_data['ticker'] = F.tanh(res_data['ticker']).data.tolist()

        res_bigdata.append(res_data)

    save_bigdata('step2', res_bigdata)
    return res_bigdata

# boardデータを畳み込み
def step3(bigdata=None):
    if bigdata is None:
        bigdata = load_bigdata('step2')

    from conv_board import ConvolutionBoard
    conv_board = ConvolutionBoard(level=64)
    if True is not None:
        serializers.load_npz(os.path.join('../conv_board/', 'model.npz'), conv_board)
        print 'loaded model'

    res_bigdata = []
    for data in bigdata:
        res_data = {}
        res_data['j_timestamp'] = data['j_timestamp']
        res_data['ltp'] = data['ltp']
        np_array = np.array(data['ltp_centered_board'])
        x = chainer.Variable(np_array.astype(np.float32)).reshape(1,1,2049)
        res_data['conved_board'] = conv_board(x).data.reshape(64).tolist()
        res_data['ticker'] = data['ticker']
        res_bigdata.append(res_data)

    save_bigdata('step3', res_bigdata)
    return res_bigdata

def load_bigdata(co_name, span=10):
    bigdata = []
    # データの読み込み
    if db[co_name].count() == 0:
        print 'collection \'%s\' does not exist.' % co_name
        sys.exit()

    db_itr = db[co_name].find().batch_size(100)
    for i, data in enumerate(db_itr):
        if i%span ==0:
            bigdata.append(data)
            print '%s件目のdata読み込み完了' % i
    return bigdata

def save_bigdata(co_name, bigdata):
    print 'collection=>%s' % co_name
    print 'saving bigdata...'

    db[co_name].drop()
    for i, data in enumerate(bigdata):
        # print '%s件目のdata保存完了' % i
        db[co_name].save(data)

    print 'saved bigdata'

def main(step=1, bigdata_size=864000):
    bigdata = None
    if step <= 1:
        print 'step1開始'
        bigdata = step1(bigdata_size)
    if step <= 2:
        print 'step2開始'
        bigdata = step2(bigdata)
    if step <= 3:
        print 'step3開始'
        bigdata = step3(bigdata)

def args_parse():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--bigdata_size', type=int, default=864000)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = args_parse()
    main(step=args.step, bigdata_size=args.bigdata_size)
