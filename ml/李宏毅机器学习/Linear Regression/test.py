import pandas
import numpy as np

def get_data(data_path):
    test_data = pandas.read_csv(data_path, header=None, encoding='big5')
    test_data = test_data.iloc[:, 2:].copy()
    test_data[test_data == 'NR'] = 0
    test_data = test_data.to_numpy()
    # 240组数据，240,18*9
    test_x = np.empty([240, 18 * 9], dtype=float)
    for i in range(240):
        test_x[i, :] = test_data[18 * i: 18 * (i + 1), :].reshape(1, -1)
    mean_x, std_x = np.load("./meanVar.npy")
    for i in range(len(test_x)):
        for j in range(len(test_x[0])):
            if std_x[j] != 0:
                test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]
    test_x = np.concatenate((np.ones([240, 1]), test_x), axis = 1).astype(float)
    return test_x

def predict(datasets):
    w = np.load('weight.npy')
    ans_y = np.dot(datasets, w)
    import csv
    with open('submit.csv', mode='w', newline='') as submit_file:
        csv_writer = csv.writer(submit_file)
        header = ['id', 'value']
        print(header)
        csv_writer.writerow(header)
        for i in range(240):
            row = ['id_' + str(i), ans_y[i][0]]
            csv_writer.writerow(row)
            print(row)

if __name__ == '__main__':
    test_data_path = "./ml2020spring-hw1/test.csv"
    datasets = get_data(test_data_path)
    predict(datasets)