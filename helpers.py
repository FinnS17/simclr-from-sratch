"""Data augmentations, SSL dataset wrapper, and device helpers."""

from torchvision import transforms
import torch
import numpy

def get_transform():
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.3, 1)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.5, saturation=0.4, hue=0.1)], p=0.8),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1,1.0))], p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616)),
])

def to_display_img(img):
    mean = torch.tensor((0.4914, 0.4822, 0.4465)).view(3,1,1)
    std = torch.tensor((0.2470, 0.2435, 0.2616)).view(3,1,1)
    img= img * std + mean
    img = img.clamp(0,1)
    img = img.permute(1,2,0).cpu().numpy()
    return img

class SSLAugmentation(torch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img = self.dataset[idx][0]
        x1 = self.transform(img)
        x2 = self.transform(img)
        return x1, x2
        
        
def get_device():
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    return device

