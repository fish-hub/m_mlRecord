import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd
from torch.utils.data import DataLoader
import time
from network import m_model
from utils import ImageDatasets


train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
])
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])

trainDataset = ImageDatasets("./food-11/food-11/training", train=True, transform=train_transform)
trainDataLoader = DataLoader(
    trainDataset,
    batch_size=128,
    shuffle=True,
)
valDataset = ImageDatasets("./food-11/food-11/validation", train=False, transform=test_transform)
valDataLoader = DataLoader(
    valDataset,
    batch_size=128,
    shuffle=False
)

model = m_model()
lossFn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epoch = 30

for epoch in range(num_epoch):
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0
    model.train()
    for index, data in enumerate(trainDataLoader):
        images, label = data
        optimizer.zero_grad()
        output = model(images)
        loss = lossFn(output, label)
        loss.backward()
        optimizer.step()
        train_acc += np.sum(np.argmax(output.data.numpy(), axis=1) == label.numpy())
        train_loss += loss.item()
        print("iter:{}, train_acc:{}, train_loss{}".format(index, train_acc/((len(images))*(index+1)), train_loss/(index+1)))

    model.eval()
    with torch.no_grad():
        for index, data in enumerate(valDataLoader):
            images, label = data
            output = model(images)
            loss = lossFn(output, label)
            val_acc += np.sum(np.argmax(output.data.numpy(), axis=1) == label.numpy())
            val_loss += loss.item()
            print("iter:{}, train_acc:{}, train_loss{}".format(index, val_acc / ((len(images))*(index+1)), val_loss / (index + 1)))

    torch.save(model.state_dict(), 'net.pth')
    train_acc = train_acc/len(trainDataset)
    train_loss = train_loss/len(trainDataLoader)
    val_acc = val_acc/len(valDataset)
    val_loss = train_loss / len(valDataLoader)
    print("epoch:{}, average train_acc:{}, train_loss:{},\
     val_acc:{}, val_loss:{}, ".format(epoch, train_acc, train_loss, val_acc, val_loss))


# train val数据拼接起来一起训练网络
# train_val_set = ImageDatasets("./food-11/food-11/training", train=True, transform=train_transform, concatDataDir="./food-11/food-11/validation")
# train_val_loader = DataLoader(train_val_set, batch_size=128, shuffle=True)
# model_best = m_model()
# loss = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model_best.parameters(), lr=0.001)
# num_epoch = 30
#
# for epoch in range(num_epoch):
#     epoch_start_time = time.time()
#     train_acc = 0.0
#     train_loss = 0.0
#
#     model_best.train()
#     for i, data in enumerate(train_val_loader):
#         optimizer.zero_grad()
#         train_pred = model_best(data[0])
#         batch_loss = loss(train_pred, data[1])
#         batch_loss.backward()
#         optimizer.step()
#
#         train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
#         train_loss += batch_loss.item()
#
#     print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f' % \
#       (epoch + 1, num_epoch, time.time()-epoch_start_time, \
#       train_acc/train_val_set.__len__(), train_loss/train_val_set.__len__()))