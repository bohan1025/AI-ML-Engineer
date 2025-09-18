import torch
import torchvision.transforms as T
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data: list, label: list, model_name: str = 'resnet'):
        self.data = data
        self.label = label
        self.model_name = model_name
        self.transform = T.Compose([
            T.ToTensor(),
            T.RandomRotation(90),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index]
        label = self.label[index]
        
        img = self.transform(image)
        if self.model_name == 'efficientnet':
            img = torch.cat((img, img, img), 0)
        
        return {
            "images": img,
            "labels": torch.tensor(label)
        }

        