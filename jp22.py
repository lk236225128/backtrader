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

    def next(self):
        # Simply log the closing price of the series from the reference
        self.log('Close, %.2f' % self.dataclose[0])

        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return

        # Check if we are in the market
        if not self.position:
            if self.data0.lines.JJ9[0] >= 4:
                print(f'JJ9 :{self.data0.lines.JJ9[0]}')
                # print(f'self.datas:{self.datas[0].JJ9}')
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


class PandasData_more(bt.feeds.PandasData):
    lines = ('JJ9',)  # 要添加的线
    # 设置 line 在数据源上的列位置
    params = (
        ('JJ9', -1),  # -1表示自动按列明匹配数据，也可以设置为线在数据源中列的位置索引 (('pe',6),('pb',7),)
    )


if __name__ == '__main__':
    # 设置显示选项以显示完整的数据内容
    pd.set_option('display.max_rows', None)  # 显示所有行
    pd.set_option('display.max_columns', None)  # 显示所有列
    pd.set_option('display.width', None)  # 自动调整输出宽度

    # Create a cerebro entity
    cerebro = bt.Cerebro()

    # Add a strategy
    cerebro.addstrategy(TestStrategy)


    def add_JJ9(df):
        # 新增字段
        VAR1 = (df['high'] + df['low'] + df['open'] + 2 * df['close']) / 5
        VAR2 = REF(VAR1, 1)
        VAR8 = SMA(MAX(VAR1 - VAR2, 0), 10, 1) / SMA(ABS(VAR1 - VAR2), 10, 1) * 100

        condition1 = COUNT(VAR8 < 20, 5) >= 1
        condition2 = COUNT(VAR1 == LLV(VAR1, 10), 10) >= 1
        condition3 = df['close'] >= df['open'] * 1.038
        condition4 = df['volume'] > MA(df['volume'], 5) * 1.2

        df["JJ9"] = condition1.astype('int64') + \
                    condition2.astype('int64') + \
                    condition3.astype('int64') + \
                    condition4.astype('int64')
        # df = df[['open', 'high', 'low', 'close', 'volume', 'openinterest', 'JJ9']]
        df.index = pd.to_datetime(df.index)  # 将索引转换为datetime类型
        print(f'df:{df}')
        return df


    # Create a Data Feed, 使用tushare旧版接口获取数据
    def get_data(code, start='2023-03-31', end='2023-06-24'):
        df = ts.get_hist_data(code, ktype='D', start=start, end=end)
        df['openinterest'] = 0
        df = df.sort_index()  # 按索引排序
        df = add_JJ9(df)
        return df


    dataframe = get_data('000568')  # 合盛硅业

    # 加载数据
    # data = bt.feeds.PandasData(dataname=dataframe)

    # 这里使用上述定义的新类PandasData_more（继承了bt.feeds.PandasData）
    datafeed1 = PandasData_more(dataname=dataframe)
    cerebro.adddata(datafeed1)

    # Add the Data Feed to Cerebro
    # cerebro.adddata(data)

    # Set our desired cash start
    cerebro.broker.setcash(1000.0)

    # Add a FixedSize sizer according to the stake
    cerebro.addsizer(bt.sizers.FixedSize, stake=10)

    # Set the commission
    cerebro.broker.setcommission(commission=0.0)

    # Run over everything
    cerebro.run(maxcpus=1)
