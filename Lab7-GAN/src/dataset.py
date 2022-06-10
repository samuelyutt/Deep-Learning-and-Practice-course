from torch.utils.data import Dataset
import os
import json
from PIL import Image
from torchvision import transforms
import torch


class CLEVRDataset(Dataset):
    def __init__(self, root='../data'):
        self.root = root
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((64, 64)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        with open(f'{root}/objects.json') as f:
            self.objects = json.load(f)
        with open(f'{root}/train.json') as f:
            labels = json.load(f)
            self.keys = [key for key in labels]
            self.labels = [self.get_one_hot_label(labels[key]) for key in labels]

    def get_one_hot_label(self, label):
        one_hot_label = torch.zeros(len(self.objects))
        for obj in label:
            one_hot_label[self.objects[obj]] = 1
        return one_hot_label

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]
        img = Image.open(f'{self.root}/iclevr/{key}').convert('RGB')
        img = self.transform(img)
        return img, self.labels[index]
