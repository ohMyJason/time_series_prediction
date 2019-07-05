
import numpy as np
from matplotlib import pyplot as plt
from sklearn.svm import SVR
from sklearn import preprocessing
import pandas as pd
from sklearn.metrics import mean_squared_error

from DataPre_util.readCsv import read_csv

from DataPre_util import  ch

ch.set_ch()



def svm_timeseries_prediction(data,value,c_parameter,gamma_paramenter):
    X_data = data
    Y_data = value
    # print(len(X_data))
    # 整个数据的长度
    long = len(X_data)
    # 取前多少个X_data预测下一个数据
    X_long = 200
    error = []   #用于存储错误数据
    svr_rbf = SVR(kernel='rbf', C=c_parameter, gamma=gamma_paramenter)
    # svr_rbf = SVR(kernel='rbf', C=1e5, gamma=1e1)
    # svr_rbf = SVR(kernel='linear',C=1e5)
    # svr_rbf = SVR(kernel='poly',C=1e2, degree=1)
    X = []
    Y = []
    for k in range(len(X_data) - X_long - 1):
        t = k + X_long
        X.append(Y_data[k:t])
        Y.append(Y_data[t])
    # print("---X[:-long_predict]----")
    # print(X[:-long_predict])
    # print("----Y[:-long_predict]---")
    # print(Y[:-long_predict])
    # print("-------")
    svr_rbf=svr_rbf.fit(X[:], Y[:])
    y_rbf = svr_rbf.predict(X[3500:len(X)])
    # print(y_rbf)
    #X[:-long_predict] ：取从开始到 len_predict的值。
    # print("---")
    # print("-y.size--")
    # print(len(Y))
    # print("-x.size--")
    # print(len(X))
    # print("-y_rbf.size--")
    # print(y_rbf.size)
    for e in range(len(y_rbf)):
        i=X_long + e
        error.append(Y_data[i] - y_rbf[e])
        real = []
        predict = []
        real.append(Y_data[i])
        predict.append(y_rbf[e])

    mse = mean_squared_error(predict, real)

    # print('Test MSE: %.3f' % mse)
    # print('gamma_paramenter= %d'%gamma_paramenter)
    # print('C= %d'%c_parameter)

    return X_data,Y_data,X_data[X_long+1:],y_rbf,error,mse


# data,value = read_csv('../data/PRSA_data_ff.csv')
# '''样本归一化处理'''
# value=preprocessing.scale(value)
#
# X_data,Y_data,X_prediction,y_prediction,error,mse = svm_timeseries_prediction(data,value,9,9)
# print("mse : %.3f"%mse)
# figure = plt.figure()
# tick_plot = figure.add_subplot(1, 1, 1)
# tick_plot.plot(X_data, Y_data, label='真实值', color='green', linestyle='-')
# # tick_plot.axvline(x=X_data[-long_predict], alpha=0.2, color='gray')
# tick_plot.plot(X_prediction[3500:len(X_data)], y_prediction, label='拟合值', color='red', linestyle='--')
# plt.legend()
# # tick_plot = figure.add_subplot(2, 1, 2)
# # tick_plot.plot(X_prediction,error)
# plt.show()
