from model import get_inits, compute_loss
import torch
import os
from image import save_image, postprocess
from style import get_contents, get_styles
from style import extract_features
def train(X, contents_Y, styles_Y, lr, num_epochs, lr_decay_epoch, content_layers, style_layers, net):
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

def batch_train(style_imgs, style_imgs_path, content_imgs, lr, num_epochs, lr_decay_epoch, image_shape, content_layers, style_layers, net):
    out_imgs = []
    for i, style_img in enumerate(style_imgs):
        if not os.path.exists('output'):
            os.makedirs('output')
        # 获取图片的文件名（不包括扩展名）
        image_name = os.path.splitext(os.path.basename(style_imgs_path[i]))[0]
        # 如果output下已经有这个style image的文件夹，就跳过这个style image
        if os.path.exists('output/' + image_name):
            continue
        os.makedirs('output/' + image_name)
        for j, content_img in enumerate(content_imgs):
            content_X, contents_Y = get_contents(image_shape, content_img, content_layers, style_layers, net)
            _, styles_Y = get_styles(image_shape, style_img, content_layers, style_layers, net)
            output = train(content_X, contents_Y, styles_Y, lr, num_epochs, lr_decay_epoch, content_layers, style_layers, net)
            out_imgs.append(output.forward().detach())
            print("one picture finished.")
            img = postprocess(output.forward().detach())
            path = 'output/' + image_name + '/output' + str(j) + '.jpg'
            save_image(img, path)
    return out_imgs

