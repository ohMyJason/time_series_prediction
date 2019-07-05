from pandas import read_csv
from ARIMA.buildArima import evaluate_arima_model
from pandas import datetime
from sklearn import preprocessing
import warnings

# '''定义时间格式'''
# def parser(x):
#     return datetime.strptime(x, '%Y/%m/%d')


'''比较三个候选参数'''
series = read_csv('../data/PRSA_data_ff.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)
X=preprocessing.scale(series.values)

try:
    mse  = evaluate_arima_model(X,(1,0,0))
    print("p=1,d=0,q=0  mse= %.3f" % mse)
except:
    print("第一组参数无法计算")

# try:
#     mse  = evaluate_arima_model(X,(2,0,0))
#     print("p=2,d=0,q=0  mse= %.3f" %mse)
# except:
#     print("第二组参数无法计算")
#
# try:
#     mse  = evaluate_arima_model(X,(7,0,5))
#     print("p=7,d=0,q=5  mse= %.3f" %mse)
# except:
#     print("第三组参数无法计算")