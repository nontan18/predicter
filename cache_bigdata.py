#coding: utf-8
import os
import sys
sys.path.append(os.pardir)

# データの読み込み関連
from tools.dao_v2 import DataAccessObj
from pymongo import MongoClient
import datetime as dt
import time

# マルチプロセッシング関連
from multiprocessing import Pool
import multiprocessing as multi
from tqdm import tqdm

# mongo_client = None
mongo_client = MongoClient('localhost', 27017)
dao = DataAccessObj()
product_code = 'FX_BTC_JPY'
start_dt = None

def cache_bitflyer_tickers(tickers, start_dt, end_dt, span=1):
    tickers = tickers.sort('timestamp')
    ticker = tickers.next()
    ticker_dt = ticker_dt_fix(dt.datetime.strptime(ticker['timestamp'][:-1][:25], '%Y-%m-%dT%H:%M:%S.%f'))

    i = 0
    while start_dt < end_dt:

        while ticker_dt < start_dt:
            pre_ticker = ticker
            try:
                ticker = tickers.next()
            except StopIteration as e:
                break
            pre_ticker_dt = ticker_dt_fix(dt.datetime.strptime(pre_ticker['timestamp'][:-1][:25], '%Y-%m-%dT%H:%M:%S.%f'))
            ticker_dt = ticker_dt_fix(dt.datetime.strptime(ticker['timestamp'][:-1][:25], '%Y-%m-%dT%H:%M:%S.%f'))
            # print '%s < %s' % (ticker_dt, (start_dt + dt.timedelta(seconds=1)).isoformat())

        print "%i:%s saved" % (i, pre_ticker_dt)
        cache_bitflyer_ticker(pre_ticker, start_dt)
        i +=1

        start_dt += dt.timedelta(seconds=span)

def ticker_dt_fix(ticker_dt):
    return ticker_dt + dt.timedelta(hours=9)

def cache_bitflyer_ticker(ticker, date):
    del ticker['_id']
    ticker['_id'] = date.isoformat()
    ticker['j_timestamp'] = date.isoformat()
    db_name = get_db_name(date, product_code)
    mongo_client[db_name]['tickers'].save(ticker)

def cache_bitflyer_books(books, start_dt, end_dt):
    books = books.sort('timestamp')
    pre_book = books.next()
    book = books.next()
    print 'pre_book:', pre_book['timestamp']
    print 'book:', book['timestamp']
    i = 0
    while start_dt < end_dt:
        book_dt = dt.datetime.strptime(book['timestamp'], '%Y-%m-%dT%H:%M:%S.%f+09:00')
        pre_book_dt = dt.datetime.strptime(pre_book['timestamp'], '%Y-%m-%dT%H:%M:%S.%f+09:00')

        print "%i:%s saved" % (i, pre_book['timestamp'])
        cache_bitflyer_book(pre_book, start_dt)

        i +=1
        if book_dt < start_dt + dt.timedelta(seconds=1):
            pre_book = book
            try:
                book = books.next()
            except StopIteration as e:
                pass

        start_dt += dt.timedelta(seconds=1)

def cache_bitflyer_book(book, date):
    del book['_id']
    book['_id'] = date.isoformat()
    book['j_timestamp'] = date.isoformat()
    db_name = get_db_name(date, product_code)
    mongo_client[db_name]['books'].save(book)


def cache_bitflyer_execution(execution):
    del execution['_id']
    execution['j_timestamp'] = execution['timestamp']
    del execution['timestamp']
    date = dt.datetime.strptime(execution['j_timestamp'], '%Y-%m-%dT%H:%M:%S+09:00')
    db_name = get_db_name(date, product_code)
    mongo_client[db_name]['executions'].save(execution)


def get_db_name(date, product_code):
    return 'bitflyer(%s)-%s-cache' % (product_code, date.date())

def main():

    # とりあえず、データがどこまであるのか確認
    db = mongo_client['bitflyer(FX_BTC_JPY)-2017-05-23']
    books = db['books'].find()
    print 'books↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓'
    print 'start_dt:', books.sort('timestamp')[0]['timestamp']
    print 'end_dt:', books.sort('timestamp', -1)[0]['timestamp']
    print 'len:', books.count()

    tickers = db['tickers'].find()
    print 'tickers↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓'
    print 'start_dt:', tickers.sort('timestamp')[0]['timestamp']
    print 'end_dt:', tickers.sort('timestamp', -1)[0]['timestamp']
    print 'len:', tickers.count()

    executions = db['executions'].find()
    print 'executions↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓'
    print 'start_dt:', executions.sort('timestamp')[0]['timestamp']
    print 'end_dt:', executions.sort('timestamp', -1)[0]['timestamp']
    print 'len:', executions.count()

    # executionsのキャッシュ(移行)
    for execution in executions:
        cache_bitflyer_execution(execution)

    print 'executionsの移行完了'

    global start_dt
    start_dt = dt.datetime.strptime('2017-05-23T00:00:00+09:00', '%Y-%m-%dT%H:%M:%S+09:00')
    # start_dt = dt.datetime.strptime('2017-05-23T23:59:59+09:00', '%Y-%m-%dT%H:%M:%S+09:00')
    end_dt = dt.datetime.strptime('2017-05-24T00:00:00+09:00', '%Y-%m-%dT%H:%M:%S+09:00')
    # end_dt = dt.datetime.strptime('2017-05-23T00:01:00+09:00', '%Y-%m-%dT%H:%M:%S+09:00')

    cache_bitflyer_books(books, start_dt, end_dt)
    cache_bitflyer_tickers(tickers, start_dt, end_dt, span=0.1)

    sys.exit()
    print 'cache終了'


if __name__ == '__main__':
    main()
