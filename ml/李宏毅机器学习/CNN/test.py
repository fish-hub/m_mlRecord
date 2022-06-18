import torch
from network import m_model
from utils import ImageDatasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])

model = m_model
model_state_dict = torch.load("./net.pth")
model.load_state_dict(model_state_dict, strict=True)
test_set = ImageDatasets("./food-11/food-11/training", transform=test_transform)
test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

model.eval()
prediction = []
with torch.no_grad():
    for i, data in enumerate(test_loader):
        test_pred = model(data)
        test_label = np.argmax(test_pred.data.numpy(), axis=1)
        for y in test_label:
            prediction.append(y)

#將結果寫入 csv 檔
with open("predict.csv", 'w') as f:
    f.write('Id,Category\n')
    for i, y in enumerate(prediction):
        f.write('{},{}\n'.format(i, y))