#create a line plot
from pandas import read_csv
from pandas import DataFrame
from pandas import Grouper
from matplotlib import pyplot
from pandas import concat

series = read_csv('daily-minimum-temperatures.csv',header=0, index_col=0,parse_dates=True,squeeze=True)
series.plot(style='k.')
pyplot.show()


#create a stacked line plot
groups = series.groupby(Grouper(freq='A'))
years = DataFrame()
for name,group in groups:
    years[name.year] = group.values

years.plot(subplots=True, legend=False)
pyplot.show()

#create a histogram
series.hist()
pyplot.show()

#create a density plot
series.plot(kind='kde')
pyplot.show()

#creating box and whisker plot
years.boxplot()
pyplot.show()

#creating monthly boxplot
one_year = series['1990']
groups = one_year.groupby(Grouper(freq='M'))
months =  concat([DataFrame(x[1].values) for x in groups],axis=1)
months = DataFrame(months)
months.columns = range(1,13)
months.boxplot()
pyplot.show()

#creating yearly heatmaps
years = years.T
pyplot.matshow(years, interpolation=None, aspect='auto')
pyplot.show()

#creating monthly heatmaps
pyplot.matshow(months, interpolation=None, aspect='auto')
pyplot.show()

#creating a lag scatter plot
from pandas.plotting import lag_plot
lag_plot(series)
pyplot.show()

#create multiple scatter plot
values = DataFrame(series.values)
lags = 7
columns = [values]
for i in range(1,(lags+1)):
    columns.append(values.shift(i))

dataframe = concat(columns,axis=1)
columns = ['t']
for i in range(1,(lags+1)):
    columns.append('t-'+str(i))

dataframe.columns = columns
pyplot.figure(1)
for i in range(1,(lags + 1)):
    ax = pyplot.subplot(240 + i)
    ax.set_title('t vs t-' + str(i))
    pyplot.scatter(x=dataframe['t'].values, y=dataframe['t-'+str(i)].values)
pyplot.show()

#plotting autocorrelation plot
from pandas.plotting import autocorrelation_plot
autocorrelation_plot(series)
pyplot.show()

