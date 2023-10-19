import torch
from torch import nn, optim, max, save, manual_seed
from torch.cuda import is_available
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from model import EfficientNet
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def main(opt):
    MODEL = EfficientNet(14)
    BATCH_SIZE = opt.batch_size
    LEARNING_RATE = opt.lr
    EPOCH = opt.epochs
    N_CLASSES = opt.num_classes
    ratio = 0.7
    device = "cuda" if is_available() else "cpu"

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.52],
                             std=[0.23]),
    ])

    dataset = datasets.ImageFolder("cdata", transform)
    data_size = len(dataset)
    split = int(data_size * ratio)
    # manual_seed(42)

    trainSet, testSet = random_split(dataset, [split, data_size - split])

    trainLoader = DataLoader(trainSet, BATCH_SIZE, True)
    testLoader = DataLoader(testSet, BATCH_SIZE, False)

    model = MODEL  # 没有预训练权重，整个网络都需要训练
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    writer = SummaryWriter("logs")

    for epoch in range(EPOCH):
        # 每个epoch进行一轮训练和评估
        # 训练和测试完成后及时回收显存中的数据
        model.train()
        for idx, (images, labels) in enumerate(trainLoader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            print(f"[Epoch: {epoch + 1}, {idx + 1}/{len(trainLoader)}] "
                  f"loss: {round(loss.data.item(), 6)}, "
                  f"lr: {round(optimizer.param_groups[0]['lr'], 6)}")
            loss.backward()
            optimizer.step()
        del images, labels, outputs
        scheduler.step(loss.data.item())
        del loss

        # 评估
        with torch.no_grad():
            model.eval()
            correct = 0
            total = 0
            for images, labels in tqdm(testLoader, f"Epoch {epoch + 1}, test"):
                images = images.to(device)
                outputs = model(images)  # 模型最后一层是全连接层，输出一个数组取最大值为分类标签
                _, predicted = max(outputs.data, 1)  # dim=1 计算每一行的最大值
                total += labels.size(0)  # 统计测试集中标签的个数
                correct += (predicted.cpu() == labels).sum()  # 统计预测正确的个数
            del images, outputs
            writer.add_scalar("acc/test", correct / total, epoch + 1)
            print(f"[Epoch {epoch + 1}] avg acc: {100 * correct / total}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=14)
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('-b', '--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--data-path', type=str, default="./cdata")
    parser.add_argument('--save-path', type=str, default="./save")

    opt = parser.parse_args()

    main(opt)
