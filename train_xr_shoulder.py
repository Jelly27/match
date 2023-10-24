import torch
from torch import nn, optim, max, save, manual_seed
from torch.cuda import is_available
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from model import EfficientNet
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from train import Train

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.2581],
                             std=[0.1309])
    ])
    t = Train("XR_SHOULDER", transform, EfficientNet(2, 1))
    t.start()