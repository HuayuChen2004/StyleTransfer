import torch
import matplotlib.pyplot as plt
from torch import nn
import torchvision

#vgg块
def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)) 
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)

#搭建vgg-19网络
conv_arch = ((2, 64), (2, 128), (4, 256), (4, 512), (4, 512))
def vgg(conv_arch, in_channels=1, input_size=(256, 256)):
    conv_blks = []
    input_len = input_size[0]
    #卷积层部分
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(
        *conv_blks, nn.Flatten(),
        #全连接层部分
        nn.Linear(out_channels * input_len // 32 * input_len // 32, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 1000))

net = vgg(conv_arch, in_channels=3, input_size=(256, 256))

X = torch.randn(size=(1, 3, 256, 256))
for blk in net:
    X = blk(X)
    print(blk.__class__.__name__, 'output shape:\t', X.shape)