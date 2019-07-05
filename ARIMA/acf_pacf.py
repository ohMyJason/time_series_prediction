import statsmodels.api as sm
from pandas import datetime
from pandas import read_csv
import matplotlib.pyplot as plt
from sklearn import preprocessing
from DataPre_util.readCsv import read_csv
def parser(x):
    return datetime.strptime(x, '%Y/%m/%d')



'''计算相关系数和偏自相关系数'''
# train=read_csv('../data/PRSA_data_ff.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)
index,train = read_csv('../data/PRSA_data_ff.csv')

fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(train, lags=16, ax=ax1)
ax1.xaxis.set_ticks_position('bottom')
fig.tight_layout()

ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(train, lags=16, ax=ax2)
ax2.xaxis.set_ticks_position('bottom')
fig.tight_layout()
plt.show()