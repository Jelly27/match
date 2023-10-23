from model import EfficientNet
from torchsummary import summary
def main(opt):
    net = EfficientNet(2, 3)
    print(summary(net, (3, 224, 224)))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--tensorboard', action="store_true")

    opt = parser.parse_args()

    main(opt)
