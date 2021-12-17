#upsample to daily intervals
from pandas import read_csv
from datetime import datetime

def parser(x):
    return datetime.strptime('190'+x, '%Y-%m')

series = read_csv('shampoo-sales.csv',header=0, index_col=0, parse_dates=True,squeeze=True, date_parser=parser)
upsamples = series.resample('D').mean()
print(upsamples.head(32))

#upsample to daily intervals with linear interpolation
from matplotlib import pyplot
interpolated = upsamples.interpolate(method='linear')
print(interpolated.head(32))
interpolated.plot()
pyplot.show()

#spline interpolation
interpolated_spline = upsamples.interpolate(method='spline',order=2)
print(interpolated_spline.head(32))
interpolated_spline.plot()
pyplot.show()