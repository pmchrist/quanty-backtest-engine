# Mandatory Imports
from hashlib import shake_128
from pandas import DataFrame
from backtesting.strategy import Strategy
import talib.abstract as ta
from modules.setup.config import qtpylib_methods as qtpylib
from modules.public.hyperopt_parameter import integer_parameter, float_parameter
from modules.public.trading_stats import TradingStats
import pandas as pd

import numpy as np
import kmeans1d
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn import metrics

#Artemis
class Artemis (Strategy):

    MIN_CANDLES = 1000

    # Tecnical Indicators
    direction_ma = integer_parameter(default=360, low=48, high=336, step=12)
    direction_mfi = integer_parameter(default=4, low=4, high=72, step=4)
    mfi_low_border = integer_parameter(default=20, low=0, high=70, step=5)
    mfi_high_border = integer_parameter(default=80, low=30, high=100, step=5)
    #direction_rsi = integer_parameter(default=4, low=4, high=48, step=2)

    # Support Resistance
    # Refresh Period (How often we S/R should be updated)
    support_refresh_rate = integer_parameter(default=120, low=120, high=120, step=24)    # Fix in some value that makes sense and doesn't kill PC
    # Lookback Period (How far back should we look for S/R calculation)
    support_refresh_rate_lookback_multiplier = integer_parameter(default=10, low=2, high=18, step=2)    # Should be far enough, but not too far
    
    # How many levels there should be in S/R
    support_ammount_diviser = integer_parameter(default=3, low=1, high=9, step=1)   # Lower value, more S/R levels. Depends mostly on the Timeframe
    support_atr = integer_parameter(default=8, low=4, high=60, step=2)              # ATR value used for S/R calculations. Try some value that makes sense
    support_distance_coeff = float_parameter(default=1.1, low=0.2, high=2.0, step=0.2)  # ATR coeff range in which price should be to trigger S/R

    # TP/SL values
    tp_coeff_support = integer_parameter(default=2, low=1, high=6, step=1)
    sl_coeff_support = integer_parameter(default=2, low=1, high=6, step=1)
    tp_coeff_atr = float_parameter(default=1.0, low=0.0, high=5.0, step=0.5)
    sl_coeff_atr = float_parameter(default=1.0, low=0.0, high=5.0, step=0.5)


    def generate_indicators(self, dataframe: DataFrame) -> DataFrame:

        # Normal Prices
        #dataframe['hlc3'] = ( dataframe['high'] + dataframe['low'] + dataframe['close'] ) / 3
        #dataframe['ohlc4'] = ( dataframe['open'] + dataframe['high'] + dataframe['low'] + dataframe['close'] ) / 4
        dataframe['hlcc4'] = ( dataframe['high'] + dataframe['low'] + dataframe['close'] + dataframe['close'] ) / 4        

        # Direction Calculation
        dataframe['direction_ma'] = qtpylib.wma(dataframe['hlcc4'], window=self.direction_ma)
        dataframe['direction_mfi'] = ta.MFI(dataframe, timeperiod=self.direction_mfi)
        #dataframe['direction_rsi'] = ta.RSI(dataframe, timeperiod=self.direction_rsi)

        # S/R Levels
        dataframe['support_atr'] = ta.ATR(dataframe, timeperiod=self.support_atr)
        dataframe['support_lines'] = ""
        
        # Calculating S/R
        centroids = []
        for n in range(1, len(dataframe)):
            # Finding all fractals in current trading range
            if (n % self.support_refresh_rate == 0 and n >= self.support_refresh_rate*self.support_refresh_rate_lookback_multiplier):
                fractals = []
                # Top Fractals
                for j in range(n - self.support_refresh_rate*self.support_refresh_rate_lookback_multiplier, n):
                    if ((dataframe['close'].iloc[j-4] < dataframe['close'].iloc[j-3])
                    and (dataframe['close'].iloc[j-3] < dataframe['close'].iloc[j-2])
                    and (dataframe['close'].iloc[j-2] > dataframe['close'].iloc[j-1])
                    and (dataframe['close'].iloc[j-1] > dataframe['close'].iloc[j])) :
                        fractals.append(dataframe['hlcc4'].iloc[j-2])
                # Down Fractals
                for j in range(n - self.support_refresh_rate*self.support_refresh_rate_lookback_multiplier, n):
                    if ((dataframe['close'].iloc[j-4] > dataframe['close'].iloc[j-3])
                    and (dataframe['close'].iloc[j-3] > dataframe['close'].iloc[j-2])
                    and (dataframe['close'].iloc[j-2] < dataframe['close'].iloc[j-1])
                    and (dataframe['close'].iloc[j-1] < dataframe['close'].iloc[j])) :
                        fractals.append(dataframe['hlcc4'].iloc[j-2])
                
                # Find Results of Fractals for Clustering
                fractals.sort(reverse=True)
                highest_price = fractals[0]
                lowest_price = fractals[len(fractals)-1]

                # Decrease Multiplier to add more levels
                atrThreeshold = dataframe['support_atr'].iloc[n]*self.support_ammount_diviser
                # Cluster with Kmeans
                x = np.array(fractals)
                k = int((highest_price-lowest_price)/atrThreeshold)
                # if number of clusters is too low, set it to 3
                if (k < 3): k=3
                # Cluster centroids are approximate of S/R zones
                clusters, centroids = kmeans1d.cluster(x, k)
            # Save S/R levels into dataframe
            dataframe.iat[n, dataframe.columns.get_loc('support_lines')] = centroids.copy()

        return dataframe

    # Searches for the closest value in the array
    def find_nearest(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    def buy_signal(self, dataframe: DataFrame) -> DataFrame:
        
        # Initialize all booleans with 0
        dataframe['tp_target'] = 0
        dataframe['sl_target'] = 0
        dataframe['in_trade'] = 0

        for n in range(1, len(dataframe)):
            # If there are any S/R levels
            if dataframe['support_lines'].iloc[n]:
                # Update rolling values
                if (dataframe['in_trade'].iloc[n-1]):
                    dataframe['tp_target'].iloc[n]=dataframe['tp_target'].iloc[n-1]
                    dataframe['sl_target'].iloc[n]=dataframe['sl_target'].iloc[n-1]
                    dataframe['in_trade'].iloc[n]=1

                # TP
                if (dataframe['close'].iloc[n] > dataframe['tp_target'].iloc[n]) and dataframe['in_trade'].iloc[n]:
                    dataframe['sell'].iloc[n]=1
                    dataframe['in_trade'].iloc[n]=0
                    dataframe['tp_target'].iloc[n]=0
                    dataframe['sl_target'].iloc[n]=0
                # SL
                elif (dataframe['close'].iloc[n] < dataframe['sl_target'].iloc[n]) and dataframe['in_trade'].iloc[n]:
                    dataframe['sell'].iloc[n]=1
                    dataframe['in_trade'].iloc[n]=0
                    dataframe['tp_target'].iloc[n]=0
                    dataframe['sl_target'].iloc[n]=0
                
                # if we are not already in trade
                if not dataframe['in_trade'].iloc[n]:
                    # Find Closest level 
                    closest_level_id = self.find_nearest(dataframe['support_lines'].iloc[n], dataframe['close'].iloc[n])
                    # Distance between price and S/R is small enough
                    if abs(dataframe['hlcc4'].iloc[n] - dataframe['support_lines'].iloc[n][closest_level_id]) < dataframe['support_atr'].iloc[n] * self.support_distance_coeff:
                        # Price goes with the trend
                        if (dataframe['close'].iloc[n] > dataframe['direction_ma'].iloc[n]):
                            # Oscillator helps to filter signals where price is piercing throw S/R or price bounces for the first time. Only in correct range
                            if (dataframe['direction_mfi'].iloc[n] > self.mfi_low_border and dataframe['direction_mfi'].iloc[n] < self.mfi_high_border):
                                
                                # Set booleans to generate buy/sell signals with fixed tp/sl
                                dataframe['buy'].iloc[n]=1
                                dataframe['in_trade'].iloc[n]=1

                                # Check if it is worth in general to use S/R as TP/SL, or better to use ATR
                                if len(dataframe['support_lines'].iloc[n]) > closest_level_id + self.tp_coeff_support:
                                    dataframe['tp_target'].iloc[n] = dataframe['support_lines'].iloc[n][closest_level_id + self.tp_coeff_support]
                                else:
                                    dataframe['tp_target'].iloc[n] = dataframe['hlcc4'].iloc[n] + dataframe['support_atr'].iloc[n] * self.tp_coeff_atr

                                if closest_level_id >= self.sl_coeff_support:
                                    dataframe['sl_target'].iloc[n] = dataframe['support_lines'].iloc[n][closest_level_id - self.sl_coeff_support]
                                else:
                                    dataframe['sl_target'].iloc[n] = dataframe['hlcc4'].iloc[n] - dataframe['support_atr'].iloc[n] * self.sl_coeff_atr

        return dataframe

    def sell_signal(self, dataframe: DataFrame) -> DataFrame:

        dataframe.to_csv("_frame.csv")

        return dataframe

    def loss_function(self, stats: TradingStats):
        
        trade_returns = list(map(lambda trade: trade.profit_dollar, stats.trades))
        trade_opens = list(map(lambda trade: trade.opened_at, stats.trades))
        trade_closes = list(map(lambda trade: trade.closed_at, stats.trades))
        amount_of_trades = len(stats.trades)

        # Finding Average DD
        start_amount = 1000
        current_equity = start_amount

        current_high = start_amount
        current_low = start_amount
        in_red = False

        drawdowns_percents = []
        drawdowns_durations = []
        current_low = current_equity
        current_high = current_equity

        # Because not closed trades are placed in front of closed trade list, yeah
        trade_closes_offset = 0
        for n in trade_closes:
            if not n: trade_closes_offset = trade_closes_offset + 1
            if n: break

        trade_id = trade_closes_offset - 1
        for trade in trade_returns[trade_closes_offset:]:
            trade_id = trade_id + 1

            current_equity += trade
            if trade < 0:
                if current_equity < current_low:
                    current_low = current_equity
                    if not in_red:
                        # Update DD Time
                        DD_trade_id_start = trade_id
                        in_red = True
            if trade > 0:
                if current_equity > current_high:
                    if (in_red):
                        # Update DD Time
                        DD_trade_id_end = trade_id
                        drawdowns_durations.append((trade_closes[DD_trade_id_end]-trade_opens[DD_trade_id_start]).days)
                        # Update DD List
                        drawdowns_percents.append(((current_low-current_high)/current_high))
                        in_red = False
                    current_low = current_equity
                    current_high = current_equity
        if (in_red):
            DD_trade_id_end = trade_id
            drawdowns_durations.append((trade_closes[DD_trade_id_end]-trade_opens[DD_trade_id_start]).days)
            # Update DD List
            drawdowns_percents.append(((current_low-current_high)/current_high))
            in_red = False

        # Sample is statistically sufficient
        if amount_of_trades < 2000:
            return 1.

        # DD coeff
        profitable_days_cutoff = 0.4
        trading_duration = ((trade_closes[len(trade_closes)-1]-trade_closes[trade_closes_offset]).days)
        drawdown_duration = sum(drawdowns_durations)
        profitable_days_coefficent = (trading_duration-drawdown_duration)/trading_duration
        if profitable_days_coefficent < profitable_days_cutoff:
            return 1.
        print("Profitable Days:", profitable_days_coefficent)

        # DD highest percentile
        drawdown_percentile = 0.15
        drawdown_extreme_cutoff = -0.25
        if drawdowns_percents:
            biggest_drawdown_percentile = int(len(drawdowns_percents)*drawdown_percentile)
            if biggest_drawdown_percentile < 1 :
                biggest_drawdown_percentile = 1
        drawdowns_percents.sort()
        biggest_drawdown = drawdowns_percents[0:biggest_drawdown_percentile]
        if sum(biggest_drawdown)/len(biggest_drawdown) < drawdown_extreme_cutoff:
            return 1.
        print("DD " + drawdown_percentile + " percentile:", biggest_drawdown, "Average:", sum(biggest_drawdown)/len(biggest_drawdown))

        trading_months = 24
        overall_return = sum(trade_returns)
        CAGR = (((overall_return + start_amount) / start_amount) ** (1/trading_months) - 1) * 100
        mean_drawdown = -1 * (sum(drawdowns_percents)/len(drawdowns_percents)*100)
        print("CAGR/DD:", CAGR/mean_drawdown, "CAGR:", CAGR, "Mean DD:", mean_drawdown)

        return -1 * CAGR/mean_drawdown