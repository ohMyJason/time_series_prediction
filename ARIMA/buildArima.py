from pandas import read_csv
from pandas import datetime
from statsmodels.tsa.arima_model import ARIMA
# from matplotlib import pyplot
# import statsmodels.api
# import warnings
from sklearn.metrics import mean_squared_error


'''评估模型MSE'''
# evaluate an ARIMA model for a given order (p,d,q)
def evaluate_arima_model(X, arima_order):
    # prepare training dataset
    train_size = int(len(X) * 0.66)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    # make predictions
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit(disp=0)
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
    # calculate out of sample error
    error = mean_squared_error(test, predictions)
    return error

'''寻找最优参'''
# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
    dataset = dataset.astype('float32')
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)
                try:
                    mse = evaluate_arima_model(dataset, order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                    print('ARIMA%s MSE=%.3f' % (order, mse))
                except:
                    continue
    print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))
    return best_cfg


def parser(x):
    return datetime.strptime(x, '%Y/%m/%d')

series = read_csv('../data/testData.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)

'''通过AIC准则寻找最优参'''
def findC(series):
    temp = 1000000
    ansp = 0
    ansq = 0
    ansd = 0
    for p in range(0, 8):
        for q in range(0, 8):
            # if p+q!=0:
            try:
                testModel = ARIMA(series, order=(p, 0, q))
                testModel_fit = testModel.fit(disp=0)
                aic = testModel_fit.aic
                if aic < temp:
                    temp = aic
                    ansp = p
                    ansq = q
                    ansd = 0
            except:
                continue
    return ansp,ansd,ansq

# fit model
'''寻找最优参'''
# p_values = [0, 1, 2, 3 , 4, 5,6]
# d_values = range(0,1)
# q_values = range(0, 6)
#

# X = series.values
# bestOrder=evaluate_models(X, p_values, d_values, q_values)
# model = ARIMA(series, order=bestOrder)

p,d,q=findC(series.values)

print(p,d,q)
mse  = evaluate_arima_model(series.values,(p,d,q))
print("mse = %.3f"%mse)
model = ARIMA(series, order=(p,d,q))
# model = ARIMA(series, order=(2,0,0))
model_fit = model.fit(disp=0)  # disp=0关#闭对训练信息的打印


'''打印模型信息'''
print(model_fit.summary())

# plot residual errors
# residuals = DataFrame(model_fit.resid)
# residuals.plot()
# pyplot.show()
# residuals.plot(kind='kde')
# pyplot.show()
# print(residuals.describe())
