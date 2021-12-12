# create date time features of dataset
from pandas import read_csv
from pandas import DataFrame
series = read_csv('daily-minimum-temperatures.csv', header=0, index_col=0,parse_dates=True, squeeze=True)
dataframe = DataFrame()
dataframe['month'] = [series.index[i].month for i in range(len(series))]
dataframe['day']  = [series.index[i].day for i in range(len(series))]
dataframe['temperature'] = [series[i] for i in range(len(series))]
print(dataframe.head(10))

#creating a lag feature
from pandas import concat
temps = DataFrame(series.values)
dataframe_lag = concat([temps.shift(1),temps],axis=1)
dataframe_lag.columns = ['t', 't+1']
print(dataframe_lag.head(10))

#creating statistics feature
temps = DataFrame(series.values)
shifted = temps.shift(2)
window = shifted.rolling(3)
# means = window.mean()
dataframe_stat = concat([window.min(),window.mean(),window.max(),temps],axis=1)
dataframe_stat.columns = ['min','mean','max','t+1']
print(dataframe_stat.head(10))

#creating expanding features
window = temps.expanding()
dataframe_exp = concat([window.min(),window.mean(),window.max(),temps.shift(-1)],axis=1)
dataframe_exp.columns = ['min','mean','max','t+1']
print(dataframe_exp.head(10))


