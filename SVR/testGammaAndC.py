import SVR.svmprediction as sv
from matplotlib import pyplot as plt

data,value = sv.read_csv("../data/testData.csv")

def testC(gamma):
    print("when Gamma=%d "%gamma)
    cs= []
    mses = []
    for c in range(1,10):
        X_data, Y_data, X_prediction, y_prediction, error, mse = sv.svm_timeseries_prediction(data, value, gamma, c)
        print("C= %.3f" %c)
        cs.append(c)
        print("mse = %.3f" %mse)
        mses.append(mse)
    plt.plot(cs,mses)
    # plt.axis([0,9,10,30])
    plt.xlabel('C')
    plt.ylabel('MSE')
    plt.show()

def testGamma(c):
    print("when c=%d "%c)
    gammas = []
    mses = []
    for gamma in range(1,10):
        X_data, Y_data, X_prediction, y_prediction, error, mse = sv.svm_timeseries_prediction(data, value, gamma, c)
        print("Gamma= %d" %gamma)
        print("mse = %.3f" %mse)
        gammas.append(gamma)
        mses.append(mse)

    plt.plot(gammas, mses)
    # plt.axis([0, 9, 0, 30])
    plt.xlabel('Gamma')
    plt.ylabel('MSE')
    plt.show()

# testC(10)
testGamma(100)