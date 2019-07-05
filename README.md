# 关于时间序列预测中ARIMA模型的比较与探究

## How to run


-testModel.py

启动ARIMA部分


- svmprediction.py

启动SVR部分


## Code explain

### ARIMA部分

- acf_pacf.py
用于计算自相关系数与偏自相关系数
- buildArima.py
用于探究最佳模型参数
- dataPreTest.py
用于做数据的查看，画出数据散点图
- testModel.py
用于预测后40数据，并利用真实的数据与预测数据计算mse值
-CompareParm.py
用于比较三组候选参数的军方误差

### SVR部分
- FindBestPam.py
找出最佳的参数
- svmprediction.py
建立模型并预测预测后40数据，并利用真实的数据与预测数据计算mse值
- testGammaAndC.py
用于SVR中gamma参数与C参数的探究

### DataPre_util部分
- adf.py
用于数据的adf平稳性检验
- ch.py
用于解决画图中的中文乱码问题
- readCsv.py
用于读取数据

