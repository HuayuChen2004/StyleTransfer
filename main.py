import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from device import device
from style import extract_features, get_contents, get_styles
from model import compute_loss, get_inits
from image import read_image, postprocess
from image import load_images_from_folder
from image import uniform_channels


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

def train(X, contents_Y, styles_Y, lr, num_epochs, lr_decay_epoch):
    X, styles_Y_gram, trainer = get_inits(X, lr, styles_Y)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_decay_epoch, 0.8)
    for epoch in range(num_epochs):
        trainer.zero_grad()
        output = X.forward()
        contents_Y_hat, styles_Y_hat = extract_features(output, content_layers, style_layers, net)
        contents_l, styles_l, tv_l, l = compute_loss(output, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram)
        l.backward()
        trainer.step()
        scheduler.step()
        # 可视化部分
        if (epoch + 1) % 10 == 0:
            pass
    return X

"""主函数"""
# 加载图片和模型(可以选择是否采用自己训练的模型)
images = load_images_from_folder("/home/stu5/Projects/StyleTransfer/图像/建筑", device)
style_img = read_image("/home/stu5/Projects/StyleTransfer/图像/油画/星空.jpg").to(device)
image_shape = (450, 675)
lr, num_epochs, lr_decay_epoch = 0.5, 500, 50
out_imgs = []
images = uniform_channels(images)

num_trials = 5

import os
from image import save_image
for i, img in enumerate(images):
    content_X, contents_Y = get_contents(image_shape, img, content_layers, style_layers, net)
    _, styles_Y = get_styles(image_shape, style_img, content_layers, style_layers, net)
    output = train(content_X, contents_Y, styles_Y, lr, num_epochs, lr_decay_epoch)
    out_imgs.append(output.forward().detach())
    print("one picture finished.")
    if not os.path.exists('output'):
        os.makedirs('output')
    img = postprocess(output.forward().detach())
    path = 'output/output' + str(i+8*num_trials) + '.jpg'
    save_image(img, path)




# images.append(read_image(r"D:\coding\python_work\st\图像\一些私货\又三郎.jpg").to(device))

# from image import save_image
# import os
# # 训练一个检验一下
# content_X, contents_Y = get_contents(image_shape, images[-1], content_layers, style_layers, net)
# _, styles_Y = get_styles(image_shape, style_img, content_layers, style_layers, net)
# output = train(content_X, contents_Y, styles_Y, lr, num_epochs, lr_decay_epoch)
# if not os.path.exists('output'):
#     os.makedirs('output')
# img = postprocess(output.forward().detach())
# # 检查img的类型
# print(type(img))
# path = 'output/output.jpg'
# save_image(img, path)
# raise Exception("Finished")
