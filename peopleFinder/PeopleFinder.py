import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import time
import os

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

input_size = 224

data_transforms = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

data_dir = os.path.join(".","peopleFinder", "dataset")
print(data_dir)
# Creazione dei dataset di train & validation
train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), data_transforms)
test_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), data_transforms)
# Creazione dei dataloaders di train & validation
dl_train = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
dl_test = DataLoader(test_dataset, batch_size=8, shuffle=True, num_workers=0)


# Prendiamo come modello resnet18
model = models.resnet18(pretrained=True)
# impostiamo i parametri in modo che non calcolino il gradiente, facendo finetuning solo sull'ulitmo fc
for param in model.parameters():
    param.requires_grad = False
# Sostituiamo il vecchio fc con uno nuovo a solo 2 classi
model.fc = nn.Linear(512, 2)

# Indichiamo all'ottimizzatore quali parametri ottimizzzre
params_to_update = []
for param in model.parameters():
    if param.requires_grad:
        params_to_update.append(param)

optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
loss_fn = nn.CrossEntropyLoss()
epoch_num = 40

# Definisciamo l'accuracy
def eval_acc(model, data_loader):
    correct = 0
    total = 0

    with torch.no_grad():
        confusion_matrix = torch.zeros(2, 2)
        for i, (x, y_true) in enumerate(data_loader):
            print("Accuracy eval", i)
            y_pred = model(x)
            y_pred = torch.argmax(y_pred, dim=1)

            total += y_true.size(0)
            correct += (y_pred == y_true).sum().item()

            for t, p in zip(y_true.view(-1), y_pred.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

    return correct / total, confusion_matrix


# Alleniamo e testiamo
for i in range(epoch_num):

    train_acc, train_cm = eval_acc(model, dl_train)
    test_acc, test_cm = eval_acc(model, dl_test)
    print("Epoch:", i)
    print("\nTrain accuracy:", train_acc)
    print("Confusion matrix train")
    print(train_cm)
    print(train_cm.diag() / train_cm.sum(1))
    print("\nTest accuracy:", test_acc)
    print("Confusion matrix test")
    print(test_cm)
    print(test_cm.diag() / test_cm.sum(1))
    print("\n\n\n\n\n\n")

    for j, (x, y_true) in enumerate(dl_train):
        y_pred = model(x)

        y_pred_idx = torch.argmax(y_pred, dim=1)
        o, c = y_pred_idx.unique(return_counts=True)
        print(o,c)

        y_true = y_true.to(torch.long)
        loss = loss_fn(y_pred, y_true)
        print("Epoch:", i, "Batch:", j, "Loss:", loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

torch.save(model.state_dict(), "peopleFinder.bin")
