import torch, torchvision
from torch import nn, optim, max, save, manual_seed
from torch.cuda import is_available
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from model import EfficientNet
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def main(args):
    MODEL = EfficientNet(10)
    DATA_PATH = args.data_path
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.lr
    EPOCH = args.epochs
    device = "cuda" if is_available() else "cpu"

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3015,))])
    # 下载和加载MNIST数据集
    train_dataset = torchvision.datasets.MNIST(root=DATA_PATH, train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.MNIST(root=DATA_PATH, train=False, transform=transform, download=True)

    # 创建数据加载器
    trainLoader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    testLoader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = MODEL  # 没有预训练权重，整个网络都需要训练
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
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
            writer.add_scalar("loss/train", round(loss.data.item(), 6))
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
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('-b', '--batch-size', type=int, default=768)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--data-path', type=str, default="./MNIST")
    parser.add_argument('--save-path', type=str, default="./save")

    opt = parser.parse_args()

    main(opt)
