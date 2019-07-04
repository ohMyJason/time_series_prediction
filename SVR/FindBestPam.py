
import SVR.svmprediction as sv

data,value = sv.read_csv("../data/testData.csv")

temp_mse = 10000
for c_paramenter in range(1,10):
    for gamma_paramenter in range(1,10):
        X_data,Y_data,X_prediction,y_prediction,error,mse = sv.svm_timeseries_prediction(data,value,c_paramenter,gamma_paramenter)
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