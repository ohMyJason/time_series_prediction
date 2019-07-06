
from SVR.svmprediction import svm_timeseries_prediction
from DataPre_util.readCsv import read_csv
from sklearn import preprocessing
data,value = read_csv("../data/PRSA_data_ff.csv")

value=preprocessing.scale(value)
temp_mse = 10000  #mse初始值 默认无限大
c_weight = range(1,50)  #c的取值范围
gamma_weight = range(1,50) #gamma的取值范围


for c_paramenter in c_weight:
    for gamma_paramenter in gamma_weight:
        X_data,Y_data,X_prediction,y_prediction,error,mse = svm_timeseries_prediction(data,value,c_paramenter,gamma_paramenter)
        if(mse<temp_mse):
            temp_mse = mse
            temp_c = c_paramenter
            temp_gamma = gamma_paramenter

print("best mse:")
print(temp_mse)
print("best c:")
print(c_paramenter)
print("best gamma:")
print(gamma_paramenter)