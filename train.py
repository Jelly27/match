import torch
from torch import nn, optim, max, save, load, manual_seed
from torch.cuda import is_available
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime

"""
    训练代码的基类
"""


class Train:
    def __init__(self,
                 data_path: str,
                 transform: transforms,
                 net: nn.Module,
                 ratio: float = 0.7,
                 epoches: int = 100,
                 batch_size: int = 32,
                 lr: float = 1E-3,
                 # load: str = "",
                 save_path: str = "./check_point",
                 tensorboard: bool = True):
        """
            前三个没有默认值，需要传入
        :param data_path: 使用ImageFolder格式
        """
        # self.NUM_CLASSES: num_classes
        self.DATA_PATH = data_path
        self.TRANSFORM = transform
        self.NET = net
        self.RATIO = ratio
        self.EPOCHES = epoches
        self.BATCH_SIZE = batch_size
        self.LR = lr
        # self.LOAD = load
        self.SAVE_PATH = save_path
        self.TB = tensorboard

    def start(self):
        device = "cuda" if is_available() else "cpu"
        dataset = datasets.ImageFolder(self.DATA_PATH, self.TRANSFORM)
        data_size = len(dataset)
        split = int(data_size * self.RATIO)
        trainSet, testSet = random_split(dataset, [split, data_size - split])
        trainLoader = DataLoader(trainSet, self.BATCH_SIZE, True)
        testLoader = DataLoader(testSet, self.BATCH_SIZE, False)
        net = self.NET
        net.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=self.LR)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        writer = SummaryWriter()
        best_acc = 0.0
        for epoch in range(self.EPOCHES):
            # 每个epoch进行一轮训练和评估
            # 训练和测试完成后及时回收显存中的数据
            if optimizer.param_groups[0].get("lr", 0) == 0:
                print("学习率为0")
                break
            net.train()
            mean_loss = 0
            for idx, (images, labels) in tqdm(enumerate(trainLoader), f"[Epoch {epoch + 1}] {len(trainLoader)}"):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = net(images)
                loss = criterion(outputs, labels)
                mean_loss += loss
                # print(f"[Epoch: {epoch + 1}, {idx + 1}/{len(trainLoader)}] "
                #       f"loss: {round(loss.data.item(), 6)}, "
                #       f"lr: {round(optimizer.param_groups[0]['lr'], 6)}")
                loss.backward()
                optimizer.step()
            del images, labels, outputs
            scheduler.step(loss.data.item())
            del loss
            mean_loss /= len(trainLoader)
            print(f"mean loss:{mean_loss}")
            if self.TB:
                writer.add_scalar("mean loss/train", mean_loss, epoch + 1)
            # 评估
            with torch.no_grad():
                net.eval()
                correct = 0
                total = 0
                for images, labels in tqdm(testLoader, f"Epoch {epoch + 1}, test"):
                    images = images.to(device)
                    outputs = net(images)  # 模型最后一层是全连接层，输出一个数组取最大值为分类标签
                    _, predicted = max(outputs.data, 1)  # dim=1 计算每一行的最大值
                    total += labels.size(0)  # 统计测试集中标签的个数
                    correct += (predicted.cpu() == labels).sum()  # 统计预测正确的个数
                del images, outputs

            acc = 100 * correct / total
            if acc > best_acc:
                best_acc = acc
                if not os.path.exists(self.SAVE_PATH):
                    os.mkdir(self.SAVE_PATH)
                save(net.state_dict(),
                     os.path.join(self.SAVE_PATH, f"{datetime.now().strftime('%Y%m%d-%H%M%S')}epoch{epoch + 1}.pth"))
            if self.TB:
                writer.add_scalar("acc/test", acc, epoch + 1)
            print(f"[Epoch {epoch + 1}] avg acc: {acc}")
