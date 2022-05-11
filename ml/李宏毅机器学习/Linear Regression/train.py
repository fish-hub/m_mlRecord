import pandas as pd
import numpy as np


def get_data(data_path):
    datasets = pd.read_csv(data_path)
    # 前两列对训练没有用
    datasets = datasets.iloc[:, 3:].copy()
    # 数据集中为NR的位置填0
    datasets[datasets == "NR"] = 0
    raw_datasets = datasets.to_numpy()




if __name__ == '__main__':
    train_data_path = "./ml2020spring-hw1/train.csv"
    get_data(train_data_path)