# _*_ coding: UTF-8 _*_
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from pandas import DataFrame
import statsmodels.api as sm
from sklearn import preprocessing
import warnings

def set_ch():
	from pylab import mpl
	mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
	mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
set_ch()

def parser(x):
    return datetime.strptime(x, '%Y/%m/%d')

series = read_csv('../data/testData.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
X = series.values
X=preprocessing.scale(X)
size = 30  #取前30个数据作训练
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()

for t in range(len(test)):
    model = ARIMA(history, order=(2, 0, 0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))

error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)


# plot
pyplot.figure(figsize=(12,6))
pyplot.plot(test,color='blue',label="实际值")
pyplot.plot(predictions, color='red',label="预测值")
pyplot.xlabel('天数')
pyplot.ylabel('日销量（件）')
pyplot.legend()
pyplot.show()







'''显示残差图'''
residuals = DataFrame(model_fit.resid)
residuals.plot()
pyplot.show()
residuals.plot(kind='kde')
pyplot.show()
print(residuals.describe())

'''对残差序列进行ADF检验'''
t=sm.tsa.stattools.adfuller(model_fit.resid)
print(model_fit.resid)

# '''保存残差数据'''
# type(model_fit.resid)
#
# output=DataFrame(index=['Test Statistic Value', "p-value", "Lags Used", "Number of Observations Used","Critical Value(1%)","Critical Value(5%)","Critical Value(10%)"],columns=['value'])
# output['value']['Test Statistic Value'] = t[0]
# output['value']['p-value'] = t[1]
# output['value']['Lags Used'] = t[2]
# output['value']['Number of Observations Used'] = t[3]
# output['value']['Critical Value(1%)'] = t[4]['1%']
# output['value']['Critical Value(5%)'] = t[4]['5%']
# output['value']['Critical Value(10%)'] = t[4]['10%']
# print(output)


