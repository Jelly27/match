import random

import torch
from torch import nn, optim, max, save, load, manual_seed
from torch.cuda import is_available
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
from model import EfficientNet
from sklearn.model_selection import KFold

"""
    生成随机种子，随机划分进行K折交叉验证
"""

if __name__ == '__main__':
    TIMES = 10
    K = 5
    DATA_PATH = "blood_cell_train"
    RATIO = 0.7
    BATCH_SIZE = 32
    NET = EfficientNet(4)
    LR = 1E-3
    EPOCHES = 50
    TB = True
    SAVE_PATH = "check_point"
    TRANSFORM = transforms.Compose([
        # transforms.Grayscale(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.65791, 0.63881, 0.67597],
                             std=[0.25579, 0.25846, 0.25891])
    ])
    device = "cuda" if is_available() else "cpu"
    dataset = datasets.ImageFolder(DATA_PATH, TRANSFORM)
    net = NET
    net.to(device)
    for _ in range(TIMES):
        r = random.randint(0, 2 ** 32 - 1)
        kf = KFold(K, shuffle=True, random_state=r)
        for i, (train_index, val_index) in enumerate(kf.split(dataset)):
            # 创建训练集和验证集的子集
            train_dataset = torch.utils.data.Subset(dataset, train_index)
            val_dataset = torch.utils.data.Subset(dataset, val_index)

            # 创建训练集和验证集的 DataLoader
            trainLoader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            testLoader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(net.parameters(), lr=LR)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
            writer = SummaryWriter(comment=f"split_seed{r}")
            best_acc = 0.0
            for epoch in range(EPOCHES):
                # 每个epoch进行一轮训练和评估
                # 训练和测试完成后及时回收显存中的数据
                if optimizer.param_groups[0].get("lr", 0) == 0:
                    print("学习率为0")
                    break
                net.train()
                mean_loss = 0
                for idx, (images, labels) in tqdm(enumerate(trainLoader), f"{i} [Epoch {epoch + 1}] {len(trainLoader)}"):
                    images, labels = images.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = net(images)
                    loss = criterion(outputs, labels)
                    mean_loss += loss
                    loss.backward()
                    optimizer.step()
                del images, labels, outputs
                scheduler.step(loss.data.item())
                del loss
                mean_loss /= len(trainLoader)
                print(f"mean loss:{mean_loss}")
                if TB:
                    writer.add_scalar("mean loss/train", mean_loss, epoch + 1)
                # 评估
                with torch.no_grad():
                    net.eval()
                    correct = 0
                    total = 0
                    for images, labels in tqdm(testLoader, f"{i} Epoch {epoch + 1}, test"):
                        images = images.to(device)
                        outputs = net(images)  # 模型最后一层是全连接层，输出一个数组取最大值为分类标签
                        _, predicted = max(outputs.data, 1)  # dim=1 计算每一行的最大值
                        total += labels.size(0)  # 统计测试集中标签的个数
                        correct += (predicted.cpu() == labels).sum()  # 统计预测正确的个数
                    del images, outputs

                acc = 100 * correct / total
                if acc > best_acc:
                    best_acc = acc
                    if not os.path.exists(SAVE_PATH):
                        os.mkdir(SAVE_PATH)
                    save(net.state_dict(),
                         os.path.join(SAVE_PATH, f"{datetime.now().strftime('%Y%m%d-%H%M%S')}epoch{epoch + 1}.pth"))
                if TB:
                    writer.add_scalar("acc/test", acc, epoch + 1)
                print(f"[Epoch {epoch + 1}] avg acc: {acc}, lr:{optimizer.param_groups[0].get('lr', 0)}")
