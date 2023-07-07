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
        self.sma = bt.indicators.SimpleMovingAverage(self.datas[0], period=self.params.maperiod, plot=False)

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
            if self.data0.lines.JJ9[0] >= 4 \
                    or self.data0.lines.JJ6[0] >= 4 \
                    or (self.data0.lines.ZJDC[-1] >= 100 and self.data0.lines.ZJDC[0] < 100):
                self.log(
                    f'BUY CREATE: {self.dataclose[0]} , '
                    f'JJ9: {self.data0.lines.JJ9[0]} , '
                    f'JJ6: {self.data0.lines.JJ6[0]} , '
                    f'ZJDC : {self.data0.lines.ZJDC[-1]} -> {self.data0.lines.ZJDC[0]}',
                    doprint=True)

                self.order = self.buy()  # Keep track of the created order to avoid a 2nd order
        else:
            # Already in the market ... we might sell
            if len(self) >= (self.bar_executed + 5):
                # SELL, SELL, SELL!!! (with all possible default parameters)
                self.log('SELL CREATE, %.2f' % self.dataclose[0])
                self.order = self.sell()


class MyIndicators:
    def add_JJ9(self, df):
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
        return df

    def add_JJ6(self, df):
        VAR1 = df['volume'] >= 1.3 * MA(df['volume'], 5)

        VAR4 = (df['close'] - REF(df['open'], 29)) / REF(df['open'], 29) * 100

        VAR5 = EMA(0.667 * REF(VAR4, 1) + 0.333 * VAR4, 5)

        condition1 = VAR1.astype('bool')

        condition2 = COUNT(VAR4 < -17, 3).astype('bool')

        condition3 = COUNT(VAR4 >= VAR5, 3).astype('bool')

        condition4 = (REF(df['low'], 1) == LLV(df['low'], 120))

        df["JJ6"] = condition1.astype('int64') + \
                    condition2.astype('int64') + \
                    condition3.astype('int64') + \
                    condition4.astype('int64')
        return df

    def add_ZJDC(self, df):
        VAR2 = REF(df['low'], 1)

        VAR3 = SMA(ABS(df['low'] - VAR2), 3, 1) / SMA(MAX(df['low'] - VAR2, 0), 3, 1) * 100

        VAR4 = EMA(IF(df['close'] * 1.3, VAR3 * 10, VAR3 / 10), 3)

        VAR5 = LLV(df['low'], 30)

        VAR6 = HHV(VAR4, 30)

        VAR7 = IF(MA(df['close'], 58), 1, 0)

        VAR8 = EMA(IF(df['low'] <= VAR5, (VAR4 + VAR6 * 2) / 2, 0), 3) / 618 * VAR7

        df["ZJDC"] = RD(VAR8, 2)

        return df


class PandasData_more(bt.feeds.PandasData):
    lines = ('JJ9', 'JJ6', 'ZJDC')  # 要添加的线
    # 设置 line 在数据源上的列位置
    params = (
        ('JJ6', -1),
        ('JJ9', -1),  # -1表示自动按列明匹配数据，也可以设置为线在数据源中列的位置索引 (('pe',6),('pb',7),)
        ('ZJDC', -1),
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

    mi = MyIndicators()


    # Create a Data Feed, 使用tushare旧版接口获取数据
    def get_data(code, start='2022-11-01', end='2023-07-07'):
        df = ts.get_hist_data(code, ktype='D', start=start, end=end)
        df['openinterest'] = 0
        df = df.sort_index()  # 按索引排序
        df.index = pd.to_datetime(df.index)  # 将索引转换为datetime类型
        df = mi.add_JJ9(df)
        df = mi.add_JJ6(df)
        df = mi.add_ZJDC(df)
        print(f'df:{df}')
        return df


    dataframe = get_data('601012')  # 合盛硅业

    # 新定义的PandasData_more（继承自bt.feeds.PandasData）
    data = PandasData_more(dataname=dataframe)
    cerebro.adddata(data)

    # 加载数据
    # data = bt.feeds.PandasData(dataname=dataframe)
    # Add the Data Feed to Cerebro
    # cerebro.adddata(data)

    # Set our desired cash start
    cerebro.broker.setcash(100000.0)

    # Add a FixedSize sizer according to the stake
    cerebro.addsizer(bt.sizers.FixedSize, stake=10)

    # Set the commission
    cerebro.broker.setcommission(commission=0.0)

    # Run over everything
    cerebro.run(maxcpus=1)

    # Plot the result
    # cerebro.plot(style='candle')
