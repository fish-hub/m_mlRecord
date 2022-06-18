import numpy as np
import utils
np.random.seed(0)

X_train_fpath = './data/X_train'
Y_train_fpath = './data/Y_train'

# 训练集
with open(X_train_fpath) as f:
    next(f)
    X_train = np.array([line.strip("\n").split(',')[1:] for line in f], dtype=float)
    print("X_train", X_train)
# 训练集标签
with open(Y_train_fpath) as f:
    next(f)
    Y_train = np.array([line.strip('\n').split(',')[1] for line in f], dtype=float)
    print("Y_train", Y_train)

if __name__ == '__main__':
    # 数据特征归一化
    X_train, X_mean, X_std = _normalize(X_train, train=True)
    X_test, _, _ = _normalize(X_test, train=False, specified_column=None, X_mean=X_mean, X_std=X_std)

    # 划分训练集和验证集
    dev_ratio = 0.1
    X_train, Y_train, X_dev, Y_dev = _train_dev_split(X_train, Y_train, dev_ratio=dev_ratio)

    train_size = X_train.shape[0]
    dev_size = X_dev.shape[0]
    test_size = X_test.shape[0]
    data_dim = X_train.shape[1]
    print('Size of training set: {}'.format(train_size))
    print('Size of development set: {}'.format(dev_size))
    print('Size of testing set: {}'.format(test_size))
    print('Dimension of data: {}'.format(data_dim))

    # w->数据特征维度，b->一层只用一个偏执
    w = np.zeros((data_dim,))
    b = np.zeros((1,))
    # epoch, batchsize, lr
    max_iter = 10
    batch_size = 8
    learning_rate = 0.2

    train_loss = []
    dev_loss = []
    train_acc = []
    dev_acc = []
    step = 1
    for epoch in range(max_iter):
        # 打乱训练数据和标签
        X_train, Y_train = _shuffle(X_train, Y_train)

        # 每个epoch迭代次数
        for idx in range(int(np.floor(train_size / batch_size))):
            # 取出当前轮次的训练数据与标签
            X = X_train[idx * batch_size:(idx + 1) * batch_size]
            Y = Y_train[idx * batch_size:(idx + 1) * batch_size]
            # 计算梯度值
            w_grad, b_grad = _gradient(X, Y, w, b)
            # 更新参数
            w = w - learning_rate / np.sqrt(step) * w_grad
            b = b - learning_rate / np.sqrt(step) * b_grad

            step = step + 1

        # Compute loss and accuracy of training set and development set
        y_train_pred = _f(X_train, w, b)
        Y_train_pred = np.round(y_train_pred)
        train_acc.append(_accuracy(Y_train_pred, Y_train))
        train_loss.append(_cross_entropy_loss(y_train_pred, Y_train) / train_size)

        y_dev_pred = _f(X_dev, w, b)
        Y_dev_pred = np.round(y_dev_pred)
        dev_acc.append(_accuracy(Y_dev_pred, Y_dev))
        dev_loss.append(_cross_entropy_loss(y_dev_pred, Y_dev) / dev_size)

    print('Training loss: {}'.format(train_loss[-1]))
    print('Development loss: {}'.format(dev_loss[-1]))
    print('Training accuracy: {}'.format(train_acc[-1]))
    print('Development accuracy: {}'.format(dev_acc[-1]))

    np.save('weight.npy', w)
    np.save('bias.npy', b)

import matplotlib.pyplot as plt

# Loss curve
plt.plot(train_loss)
plt.plot(dev_loss)
plt.title('Loss')
plt.legend(['train', 'dev'])
plt.savefig('loss.png')
plt.show()

# Accuracy curve
plt.plot(train_acc)
plt.plot(dev_acc)
plt.title('Accuracy')
plt.legend(['train', 'dev'])
plt.savefig('acc.png')
plt.show()