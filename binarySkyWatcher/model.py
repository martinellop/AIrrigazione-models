import torch
import os
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np


class skyDataset(Dataset):
    def __init__(self, dataset_dim, data_path, label_path):
        super().__init__()
        self.image_w = 125
        self.data = np.memmap(data_path, dtype='float32', mode='r', shape=(dataset_dim, 3, self.image_w, self.image_w))
        self.label = np.memmap(label_path, dtype='uint8', mode='r', shape=(dataset_dim,))
        self.size = dataset_dim

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        image = self.data[idx]
        image = torch.from_numpy(np.array(image))

        label = torch.tensor(self.label[idx], dtype=torch.float32)

        return image, label


class skyModel(nn.Module):
    def __init__(self):
        super(skyModel, self).__init__()

        self.conv1 = nn.Conv2d(3, 3, 5, 4)  # 5x5 con stride 4
        self.conv2 = nn.Conv2d(3, 3, 3, 2)  # 3x3 con stride 2
        self.relu = nn.ReLU()
        self.fc = nn.Linear(675, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = x.reshape(-1,675)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x

