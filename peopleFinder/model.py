import PIL
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader



class PeopleFinder():
    def __init__(self, path_to_params : str):
        self.model = models.resnet18()
        self.model.fc = nn.Linear(512, 2) #changed last fc
        self.model.load_state_dict(torch.load(path_to_params))
        self.resnet_input_size = 224
        self.data_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.resnet_input_size),
            transforms.CenterCrop(self.resnet_input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def checkIfThereIsPeople(self, original_image):
        """returns 0 if there are not people, 1 if there are people."""
        img = self.data_transforms(original_image)
        img = img[None, :]
        print(img.size())
        y_pred = self.model(img)
        print(y_pred)
        y_pred = torch.argmax(y_pred, dim=1)
        print(y_pred)
        return 1 if y_pred else 0