from torchvision import datasets, transforms
from model import EfficientNet
from train import Train

if __name__ == '__main__':
    transform1 = transforms.Compose([
        # transforms.Grayscale(),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.65791, 0.63881, 0.67597],
                             std=[0.25579, 0.25846, 0.25891])
    ])

    t = Train("blood_cell", transform1, EfficientNet(4, 3), epoches=100, lr=1E-3)
    t.start()
