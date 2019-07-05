import pandas as pd
def read_csv(path):
    csv_data = pd.read_csv(path)  # 读取训练数据
    data=[]
    value = []
    for i in range(0,csv_data.data.size):
        data.append(i)
        value.append(csv_data.value[i])
    return data,value