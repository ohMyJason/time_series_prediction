from pandas import read_csv
from ARIMA import buildArima
from pandas import datetime
from sklearn import preprocessing

'''定义时间格式'''
def parser(x):
    return datetime.strptime(x, '%Y/%m/%d')


'''比较三个候选参数'''
series = read_csv('../data/testData.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
X=preprocessing.scale(series.values)
mse  = buildArima.evaluate_arima_model(X,(1,0,0))
print("p=1,d=0,q=0  mse= %.3f" %mse)
mse  = buildArima.evaluate_arima_model(X,(2,0,0))
print("p=2,d=0,q=0  mse= %.3f" %mse)
mse  = buildArima.evaluate_arima_model(X,(6,0,0))
print("p=6,d=0,q=0  mse= %.3f" %mse)
