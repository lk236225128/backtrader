from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import datetime  # For datetime objects
import os.path  # To manage paths
import sys  # To find out the script name (in argv[0])

# Import the backtrader platform
import backtrader as bt
import tushare as ts
import pandas as pd
import numpy as np
from MyTT import *
from pandas.api.indexers import FixedForwardWindowIndexer


# Create a Stratey
class TestStrategy(bt.Strategy):
    params = (
        ('maperiod', 15),
        ('printlog', False),
    )

    def log(self, txt, dt=None, doprint=False):
        ''' Logging function fot this strategy'''
        if self.params.printlog or doprint:
            dt = dt or self.datas[0].datetime.date(0)
            print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close
        self.dataopen = self.datas[0].open
        self.datahigh = self.datas[0].high
        self.datalow = self.datas[0].low
        self.datavolume = self.datas[0].volume
        self.datadate = self.datas[0].datetime

        df_dict = {
            'datetime': self.datas[0].datetime.array,
            'open': self.datas[0].open.array,
            'high': self.datas[0].high.array,
            'low': self.datas[0].low.array,
            'close': self.datas[0].close.array,
            'volume': self.datas[0].volume.array
        }
        self.df = pd.DataFrame(df_dict)

        print(f'__init__ '
              f' \ndataclose:{list(self.dataclose)} ,'
              f' \ndataopen:{list(self.dataopen)} ,'
              f' \ndatahigh:{list(self.datahigh)} ,'
              f' \ndatalow:{list(self.datalow)},'
              f' \ndatavolume :{list(self.datavolume)}')

        # To keep track of pending orders and buy price/commission
        self.order = None
        self.buyprice = None
        self.buycomm = None

        # Add a MovingAverageSimple indicator
        self.sma = bt.indicators.SimpleMovingAverage(self.datas[0], period=self.params.maperiod)

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    'BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                    (order.executed.price,
                     order.executed.value,
                     order.executed.comm), doprint=True)

                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:  # Sell
                self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                         (order.executed.price,
                          order.executed.value,
                          order.executed.comm), doprint=True)

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        # Write down: no pending order
        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' %
                 (trade.pnl, trade.pnlcomm), doprint=True)

    def get_var9(self, value):
        VAR1 = (self.df['high'] + self.df['low'] + self.df['open'] + 2 * self.df['close']) / 5

        # print(f'VAR1:\n{VAR1}')
        VAR2 = REF(VAR1, 1)
        # print(f'VAR2:\n{VAR2}')
        print(f'MAX(VAR1 - VAR2, 0):\n{MAX(VAR1 - VAR2, 0)}')
        print(f'SMA(MAX(VAR1 - VAR2, 0), 10, 1):\n{SMA(MAX(VAR1 - VAR2, 0), 10, 1)}')

        VAR8 = SMA(MAX(VAR1 - VAR2, 0), 10, 1) / SMA(ABS(VAR1 - VAR2), 10, 1) * 100

        condition1 = COUNT(VAR8 < 20, 5) >= 1
        condition2 = COUNT(VAR1 == LLV(VAR1, 10), 10) >= 1
        condition3 = self.df['close'] >= self.df['open'] * 1.038
        condition4 = self.df['volume'] > MA(self.df['volume'], 5) * 1.2

        self.df["JJ9"] = condition1.astype('int64') + condition2.astype('int64') + condition3.astype(
            'int64') + condition4.astype('int64')

        self.df["max30"] = HHV(self.df['high'], 30)
        self.df["min30"] = LLV(self.df['low'], 30)

        # print(f'df : {self.df}')
        b_value = self.df.loc[self.df['datetime'] == value, 'JJ9'].iloc[0]
        if b_value >= 4:
            print(f'{value} : b_value:{b_value}')
        return b_value

    def next(self):
        # Simply log the closing price of the series from the reference
        self.log('Close, %.2f' % self.dataclose[0])

        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return

        # Check if we are in the market
        if not self.position:
            if self.get_var9(self.datadate[0]) >= 4:
                # BUY, BUY, BUY!!!
                self.log('BUY CREATE, %.2f' % self.dataclose[0], doprint=True)

                # Keep track of the created order to avoid a 2nd order
                self.order = self.buy()

        else:
            # Already in the market ... we might sell
            if len(self) >= (self.bar_executed + 5):
                # SELL, SELL, SELL!!! (with all possible default parameters)
                self.log('SELL CREATE, %.2f' % self.dataclose[0])

                # Keep track of the created order to avoid a 2nd order
                self.order = self.sell()

    def stop(self):
        self.log('(MA Period %2d) Ending Value %.2f' %
                 (self.params.maperiod, self.broker.getvalue()), doprint=True)


if __name__ == '__main__':
    # Create a cerebro entity
    cerebro = bt.Cerebro()

    # Add a strategy
    cerebro.addstrategy(TestStrategy)


    # Create a Data Feed, 使用tushare旧版接口获取数据
    def get_data(code, start='2023-03-31', end='2023-06-24'):
        df = ts.get_hist_data(code, ktype='D', start=start, end=end)
        df['openinterest'] = 0
        df = df[['open', 'high', 'low', 'close', 'volume', 'openinterest']]
        df.index = pd.to_datetime(df.index)  # 将索引转换为datetime类型
        df = df.sort_index()  # 按索引排序
        print(f'df:{df}')
        return df


    dataframe = get_data('000568')  # 合盛硅业

    # 加载数据
    data = bt.feeds.PandasData(dataname=dataframe)

    # Add the Data Feed to Cerebro
    cerebro.adddata(data)

    # Set our desired cash start
    cerebro.broker.setcash(1000.0)

    # Add a FixedSize sizer according to the stake
    cerebro.addsizer(bt.sizers.FixedSize, stake=10)

    # Set the commission
    cerebro.broker.setcommission(commission=0.0)

    # Run over everything
    cerebro.run(maxcpus=1)
