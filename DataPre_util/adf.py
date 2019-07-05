import numpy as np
import statsmodels.tsa.stattools as ts
from matplotlib import pyplot
from pandas import datetime
from pandas import read_csv
import pandas as pd

def parser(x):
    return datetime.strptime(x, '%Y/%m/%d')


'''对时间序列ADF检验'''
train=read_csv('../data/PRSA_data_ff.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)
result = ts.adfuller(train, 1)
print(result)
