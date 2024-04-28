import torch
import torchvision
from torch import nn
from device import device
from image import uniform_channels
from image import get_content_imgs, get_style_imgs
from train import train, batch_train

to_pretrained = True
if to_pretrained == True:
    model = torchvision.models.vgg19(pretrained=True).to(device)
elif to_pretrained == False:
    model = torchvision.models.vgg19(pretrained=False).to(device)
    state_dict = torch.load('vgg19_imagenet.pth', map_location=device)
    model.load_state_dict(state_dict)

# 抽取部分层作为风格和内容提取层
style_layers = [0, 5, 10, 19, 28]
content_layers = [25]
net = nn.Sequential(*[model.features[i] for i in range(max(content_layers + style_layers) + 1)]).to(device)


content_imgs = get_content_imgs("/home/stu5/Projects/StyleTransfer/图像")
content_imgs = uniform_channels(content_imgs)
style_imgs, style_imgs_path = get_style_imgs("/home/stu5/Projects/StyleTransfer/图像/油画")
style_imgs = uniform_channels(style_imgs)

"""主函数"""
image_shape = (450, 675)
lr, num_epochs, lr_decay_epoch = 0.3, 500, 50

batch_train(style_imgs, style_imgs_path, content_imgs, lr, num_epochs, lr_decay_epoch, image_shape, content_layers, style_layers, net)

