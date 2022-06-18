import torch
import torch.nn as nn
import cv2
import os
from torch.utils.data import Dataset


class ImageDatasets(Dataset):
    def __init__(self, data_dir, train=True, transform=None, concatDataDir=None):
        super(ImageDatasets, self).__init__()
        if not concatDataDir:
            self.dataDir = data_dir
            self.train = train
            self.imageNameList = os.listdir(self.dataDir)
            self.imageLabelList = []
            self.transform = transform
            self.imagePathList = []
            for imageName in self.imageNameList:
                self.imageLabelList.append(int(imageName.split("_")[0]))
                self.imagePathList.append(os.path.join(self.dataDir, imageName))

        else:
            self.dataDir1 = data_dir
            self.dataDir2 = concatDataDir
            self.train = train
            self.imageNameList1 = os.listdir(self.dataDir1)
            self.imageNameList2 = os.listdir(self.dataDir2)

            self.imageLabelList = []
            self.transform = transform
            self.imagePathList = []
            for imageName in self.imageNameList1:
                self.imageLabelList.append(int(imageName.split("_")[0]))
                self.imagePathList.append(os.path.join(self.dataDir1, imageName))
            for imageName in self.imageNameList2:
                self.imageLabelList.append(int(imageName.split("_")[0]))
                self.imagePathList.append(os.path.join(self.dataDir2, imageName))

    def __getitem__(self, index):
        image_path = self.imagePathList[index]
        image_src = cv2.imread(image_path)
        image_src = cv2.resize(image_src, (128, 128))
        if self.transform:
            image_src = self.transform(image_src)

        image_label = torch.tensor(self.imageLabelList[index])

        if self.train:
            return image_src, image_label
        else:
            return image_src
    def __len__(self):
        return len(self.imageLabelList)