import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np


class skyDataset(Dataset):
    def __init__(self, dataset_dim, data_path, label_path):
        super().__init__()
        self.data = np.memmap(data_path, dtype='float32', mode='r', shape=(dataset_dim, 3, 416, 416))
        self.label = np.memmap(label_path, dtype='uint8', mode='r', shape=(dataset_dim,))
        self.size = dataset_dim

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        image = self.data[idx]
        image = torch.from_numpy(np.array(image))

        label = torch.tensor(self.label[idx], dtype=torch.float32)

        return image, label


train_data = "/Users/infopz/Not_iCloud/ds_bilanciato/train.npm"
train_label = "/Users/infopz/Not_iCloud/ds_bilanciato/train_label.npm"
test_data = "/Users/infopz/Not_iCloud/ds_bilanciato/test.npm"
test_label = "/Users/infopz/Not_iCloud/ds_bilanciato/test_label.npm"

train_dataset = skyDataset(3200, train_data, train_label)
test_dataset = skyDataset(800, test_data, test_label)

dl_train = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
dl_test = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=0)


class skyModel(nn.Module):
    def __init__(self):
        super(skyModel, self).__init__()

        self.conv1 = nn.Conv2d(3, 3, 4, 4)  # 3 out channel, 4x4 con stride 4
        self.conv2 = nn.Conv2d(3, 3, 4, 2)  # idem ma con stride 2
        self.pooling = nn.MaxPool2d(3, 2)  # 3x3 con stride 3
        self.relu = nn.ReLU()
        self.linear = nn.Linear(1875, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pooling(x)

        x = torch.flatten(x, 1)
        x = self.linear(x)
        x = torch.sigmoid(x)
        return x


def eval_acc(model, data_loader):
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y_true in data_loader:

            y_pred = model(x)
            y_pred = torch.squeeze(y_pred)
            y_pred = torch.round(y_pred)

            total += y_true.size(0)
            c = (y_pred == y_true).sum().item()
            correct += c

    return correct / total


learning_rate = 0.01
epoch_num = 10

model = skyModel()
loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

loss_values = []
train_accs = []
test_accs = []

for i in range(epoch_num):
    for j, (x, y_true) in enumerate(dl_train):
        y_pred = model(x)
        y_pred = torch.squeeze(y_pred)

        loss = loss_fn(y_pred, y_true)
        l = loss.item()
        loss_values.append(l)
        print("Epoch:", i, "Batch:", j, "Loss:", l)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if j%50 == 0:
            train_acc = eval_acc(model, dl_train)
            test_acc = eval_acc(model, dl_test)
            print("Epoch:", i, "Batch:", j)
            print("Train acc:", train_acc)
            print("Test acc:", test_acc)
            print("\n\n\n\n\n\n")
            train_accs.append(train_acc)
            test_accs.append(test_accs)


torch.save(model.state_dict(), "skyModel.bin")
print(loss_values)
print(train_accs)
print(test_accs)
