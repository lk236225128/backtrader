from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import datetime  # For datetime objects
import os.path  # To manage paths
import sys  # To find out the script name (in argv[0])

# Import the backtrader platform
import backtrader as bt
# import tushare as ts
import pandas as pd
import numpy as np
from pandas.api.indexers import FixedForwardWindowIndexer

from hb_hq_api import *  # 数字货币行情库
from MyTT import *


# Create a Stratey
class TestStrategy(bt.Strategy):
    params = (
        ('maperiod', 15),
    )

    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close

        # To keep track of pending orders and buy price/commission
        self.order = None
        self.buyprice = None
        self.buycomm = None

        # Add a MovingAverageSimple indicator
        self.sma = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.params.maperiod)
        self.var1 = (self.datas[0].high + self.datas[0].low + self.datas[0].open + 2 * self.datas[0].close) / 5
        self.log(f'self.var1:\n{self.var1}')
        self.var2 = self.var1(-1)
        self.log(f'self.var2:\n{self.var1(-1)}')

        self.max_value2 = abs(self.var1 - self.var2)
        self.log(f'abs(self.var1 - self.var2):\n{self.max_value2}')

        # self.var8 = bt.indicators.SMA(self.max_value, period=10) / bt.indicators.SMA(self.max_value2, period=10) * 100

        # Indicators for the plotting show
        bt.indicators.ExponentialMovingAverage(self.datas[0], period=25)
        bt.indicators.WeightedMovingAverage(self.datas[0], period=25,
                                            subplot=True)
        bt.indicators.StochasticSlow(self.datas[0], plot=False)
        bt.indicators.MACDHisto(self.datas[0], plot=False)
        rsi = bt.indicators.RSI(self.datas[0], plot=False)
        bt.indicators.SmoothedMovingAverage(rsi, period=10, plot=False)
        bt.indicators.ATR(self.datas[0], plot=False)

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
                     order.executed.comm))

                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:  # Sell
                self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                         (order.executed.price,
                          order.executed.value,
                          order.executed.comm))

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        # Write down: no pending order
        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' %
                 (trade.pnl, trade.pnlcomm))

    def next(self):
        # Simply log the closing price of the series from the reference
        self.log('==================================================')
        self.log('Close, %.2f' % self.dataclose[0])
        self.log(f'self.var1:{self.var1[0]}')
        self.log(f'self.var2:{self.var2[0]}')

        self.max_value = self.var1[0] - self.var2[0]
        self.log(f'self.max_value:{self.max_value}')
        self.log(f'abs(self.max_value):{abs(self.max_value)}')
        numbers = self.data.lines.close.get(ago=0, size=10)
        self.log(f'numbers:{numbers}')

        self.var8 = bt.indicators.SMA(numbers, period=10)
        self.log(f'self.var8:{self.var8}')

        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return

        # Check if we are in the market
        if not self.position:

            # Not yet ... we MIGHT BUY if ...
            if self.dataclose[0] > self.sma[0]:
                # BUY, BUY, BUY!!! (with all possible default parameters)
                self.log('BUY CREATE, %.2f' % self.dataclose[0])

                # Keep track of the created order to avoid a 2nd order
                self.order = self.buy()

        else:

            if self.dataclose[0] < self.sma[0]:
                # SELL, SELL, SELL!!! (with all possible default parameters)
                self.log('SELL CREATE, %.2f' % self.dataclose[0])

                # Keep track of the created order to avoid a 2nd order
                self.order = self.sell()


class TDXIndex():

    def HHV(self, Series, N):
        return pd.Series(Series).rolling(N).max()

    def back_HHV(self, Series, N):
        return Series.rolling(FixedForwardWindowIndexer(window_size=N)).max()

    def back_LLV(self, Series, N):
        return Series.rolling(FixedForwardWindowIndexer(window_size=N)).min()

    def LLV(self, Series, N):
        return pd.Series(Series).rolling(N).min()

    def SMA(self, Series, N, M=1):
        ret = []
        i = 1
        length = len(Series)
        # 跳过X中前面几个 nan 值
        while i < length:
            if np.isnan(Series.iloc[i]):
                i += 1
            else:
                break
        preY = Series.iloc[i]  # Y'
        ret.append(preY)
        while i < length:
            Y = (M * Series.iloc[i] + (N - M) * preY) / float(N)
            ret.append(Y)
            preY = Y
            i += 1
        return pd.Series(ret, index=Series.tail(len(ret)).index)

    def EMA2(self, Series, N):
        return pd.Series.ewm(Series, span=N, min_periods=N - 1, adjust=True).mean()

    def EMA(self, Series, N):
        var = pd.Series.ewm(Series, span=N, min_periods=N - 1, adjust=True).mean()
        if N > 0:
            var[0] = 0
            # y=0
            a = 2.00000000 / (N + 1)
            for i in range(1, N):
                y = pd.Series.ewm(Series, span=i, min_periods=i - 1, adjust=True).mean()
                y1 = a * Series[i] + (1 - a) * y[i - 1]
                var[i] = y1
        return var

    def MA(self, Series, N):
        return pd.Series.rolling(Series, N).mean()

    def REF(self, Series, N, sign=0):
        # sign=1表示保留数据,并延长序列
        if sign == 1:
            for i in range(N):
                Series = Series.append(pd.Series([0], index=[len(Series) + 1]))
        return Series.shift(N)

    def IF(self, COND, V1, V2):
        var = np.where(COND, V1, V2)
        # return pd.Series(var, index=COND.index)
        return pd.Series(var, index=COND.index)

    def COUNT(self, COND, N):
        return pd.Series(np.where(COND, 1, 0), index=COND.index).rolling(N).sum()

    def MAX(self, A, B):
        var = self.IF(A > B, A, B)
        return pd.Series(var, name='maxs')

    def MIN(self, A, B):
        var = self.IF(A < B, A, B)
        return var

    def ABS(self, Series):
        return abs(Series)

    def CROSS(self, A, B):
        A2 = np.array(A)
        var = np.where(A2 < B, 1, 0)
        # dd=(pd.Series(var, index=A.index).diff()<0).apply(int)
        return (pd.Series(var).diff() < 0).apply(int)

    def SINGLE_CROSS(self, A, B):
        if A.iloc[-2] < B.iloc[-2] and A.iloc[-1] > B.iloc[-1]:
            return True
        else:
            return False


if __name__ == '__main__':
    # Create a cerebro entity
    cerebro = bt.Cerebro()

    # Add a strategy
    cerebro.addstrategy(TestStrategy)

    # Datas are in a subfolder of the samples. Need to find where the script is
    # because it could have been called from anywhere
    modpath = os.path.dirname(os.path.abspath(sys.argv[0]))
    datapath = os.path.join(modpath, 'datas/orcl-1995-2014.txt')

    # Create a Data Feed
    data = bt.feeds.YahooFinanceCSVData(
        dataname=datapath,
        # Do not pass values before this date
        fromdate=datetime.datetime(2000, 1, 1),
        # Do not pass values before this date
        todate=datetime.datetime(2000, 12, 31),
        # Do not pass values after this date
        reverse=False)

    # Add the Data Feed to Cerebro
    cerebro.adddata(data)

    # Set our desired cash start
    cerebro.broker.setcash(1000.0)

    # Add a FixedSize sizer according to the stake
    cerebro.addsizer(bt.sizers.FixedSize, stake=10)

    # Set the commission
    cerebro.broker.setcommission(commission=0.0)

    # Print out the starting conditions
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # Run over everything
    cerebro.run()

    # Print out the final result
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # Plot the result
    # cerebro.plot(style='candle')
