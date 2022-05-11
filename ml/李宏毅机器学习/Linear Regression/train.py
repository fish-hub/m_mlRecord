import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
"""
根据9个小时的空气各分子浓度预测第10小时PM2.5浓度
"""

# 读取训练数据集
def get_data(data_path):
    datasets = pd.read_csv(data_path)
    # 前两列对训练没有用
    datasets = datasets.iloc[:, 3:].copy()
    # 数据集中为NR的位置填0
    datasets[datasets == "NR"] = 0
    raw_datasets = datasets.to_numpy()
    print(raw_datasets.shape)
    return raw_datasets

# 将数据格式转换为训练需要的
def data_process(raw_datasets):
    # 4320x24 --> Dict(0:18x480,2:18x480,...12:18x480)
    month_data = {}
    # 每个月的18个特征的20天24小时数据
    for month in range(12):
        sample = np.empty([18, 480])
        # 每一天的18个特征的24小时数据
        for day in range(20):
            sample[:, day*24:(day+1)*24] = raw_datasets[(month * 20 + day) * 18:((month * 20 + (day + 1)) * 18), :]
        month_data[month] = sample

    # 数据分批，10小时为1批，9data，1label (480-10+1=471组数据)
    x = np.empty([12 * 471, 18 * 9], dtype=float)
    y = np.empty([12 * 471, 1], dtype=float)
    for month in range(12):
        for day in range(20):
            for hour in range(24):
                if day == 19 and hour > 14:
                    continue
                x[month * 471 + day * 24 + hour, :] = month_data[month][:, day * 24 + hour: day * 24 + hour + 9].reshape(1, -1)
                y[month * 471 + day * 24 + hour, 0] = month_data[month][9, day * 24 + hour + 9]
    print("x", x.shape)
    print("y", y.shape)
    # 数据归一化
    mean_x = np.mean(x, axis=0)  # 18 * 9
    std_x = np.std(x, axis=0)  # 18 * 9
    for i in range(len(x)):  # 12 * 471
        for j in range(len(x[0])):  # 18 * 9
            if std_x[j] != 0:
                x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]
    np.save("meanVar.npy",np.array(([mean_x, std_x]), dtype=float))
    return x, y

def fit_function(input, target):
    # 线性回归常数项加1
    dim = 18 * 9 + 1
    w = np.zeros([dim, 1])
    # 添加常数项 12*471,18*9 --> 12*471,18*9+1
    input = np.concatenate((np.ones([12 * 471, 1]), input), axis=1).astype(float)
    print("input shape",input.shape)
    # 初始学习率
    learning_rate = 100
    # 迭代次数
    iter_time = 1000
    adagrad = np.zeros([dim, 1])
    # 防止分母为0
    eps = 0.0000000001
    loss_list = []
    for t in range(iter_time):
        # 一次性求出所有数据loss和取平均
        loss = np.sqrt(np.sum(np.power(np.dot(input, w) - target, 2)) / 471 / 12)
        loss_list.append(loss)
        if (t % 100 == 0):
            print(str(t) + ":" + str(loss))
        # np.dot(input, w) - target -->12*471,1  gradient -> 18*9+1,1 18种分子每个小时有一个对应的梯度
        gradient = 2 * np.dot(input.transpose(), np.dot(input, w) - target)  # dim*1
        # adagrad 累加以往所有时刻梯度值平方做分母限制学习率
        adagrad += gradient ** 2
        lr = learning_rate / np.sqrt(adagrad + eps)
        w = w - lr * gradient
    np.save('weight.npy', w)

    plt.figure()
    plt.title("loss-epoch")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(range(iter_time), loss_list, "r-", label="loss-epoch")
    plt.legend()
    plt.show()



if __name__ == '__main__':
    train_data_path = "./ml2020spring-hw1/train.csv"
    raw_datasets = get_data(train_data_path)
    input, target = data_process(raw_datasets)
    fit_function(input, target)
