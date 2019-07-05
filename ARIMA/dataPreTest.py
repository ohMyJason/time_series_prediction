from pandas import datetime
from matplotlib import pyplot
from SVR.svmprediction import read_csv
def set_ch():
	from pylab import mpl
	mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
	mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
def parser(x):
    return datetime.strptime(x, '%Y/%m/%d')
set_ch()

series,value = read_csv('../data/PRSA_data_ff.csv')
pyplot.plot(series,value)
pyplot.xlabel('time')
pyplot.ylabel('PM2.5')
pyplot.show()
'''output
data
2018-01-18    33
2018-01-19    31
2018-01-20    22
2018-01-21    16
2018-01-22     4
Name: value, dtype: int64

'''