#downsample to quarterly intervals
from pandas import read_csv
from datetime import datetime
from matplotlib import pyplot

def parser(x):
    return datetime.strptime('190'+x, '%Y-%m')

series = read_csv('shampoo-sales.csv',header=0, index_col=0, parse_dates=True,squeeze=True, date_parser=parser)
quarterly_mean_sales = series.resample('Q').mean()
print(quarterly_mean_sales.head())
quarterly_mean_sales.plot()
pyplot.show()

#downsample to yearly intervals
resample = series.resample('A')
yearly_mean_sales = resample.sum()
print(yearly_mean_sales.head())
yearly_mean_sales.plot()
pyplot.show()


