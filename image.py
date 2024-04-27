import os
from PIL import Image
import torch
import torchvision
from torchvision import transforms
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from device import device
import torchvision.transforms as T
import PIL
from torchvision.transforms.functional import to_pil_image

"""针对图像的操作"""
rgb_mean = torch.tensor([0.485, 0.456, 0.406], device=device)
rgb_std = torch.tensor([0.229, 0.224, 0.225], device=device)

# 读取图像并转换为张量
def read_image(path):
    img = Image.open(path)
    transform = ToTensor()
    img = transform(img).to(device)
    return img

# 显示
def show_image(img):
    img = img.cpu().numpy()
    plt.imshow(img)
    plt.axis('off')
    plt.show()

import torch
import numpy as np
from PIL import Image

def resize_image(img, size):
    img.squeeze_(0)  # Remove the batch dimension
    pil_image = to_pil_image(img)  # Convert to PIL Image
    resized_pil_image = pil_image.resize(size, resample=Image.BILINEAR)
    transform = T.ToTensor()
    resized_tensor = transform(resized_pil_image)
    return resized_tensor  # Return to the original format CxHxW


# 保存
def save_image(img, path):
    if isinstance(img, torch.Tensor):
        img = img.squeeze(0)
        img = img.permute(1, 2, 0)
        img = img.cpu().numpy()
        img = (img * 255).astype(np.uint8)
        img = torchvision.transforms.ToPILImage()(img)
        smooth_image(img)
        img.save(path)
    else:
        # 处理img是PIL图像对象的情况
        smooth_image(img)
        img.save(path)





def load_images_from_folder(folder, device):
    images = []
    transform = ToTensor()
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder, filename))
        # 如果是异常图片就不下载    
        if img is not None and np.array(img).ndim in [3]:
            if not isinstance(img, torch.Tensor):
                img_tensor = transform(img).unsqueeze(0).to(device)
            else:
                img_tensor = img.unsqueeze(0).to(device)
            images.append(img_tensor)
    # 调整图片大小
    images = [resize_image(img, (450, 675)) for img in images]
    return images


# 预处理
def preprocess(image, image_shape):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(image_shape, antialias=True),
        torchvision.transforms.Normalize(mean=rgb_mean, std=rgb_std).to(device)])  # 转换到[-1,1]之间
    return transforms(image).unsqueeze(0)

# 后处理
def postprocess(image):
    image = image[0].to(rgb_std)
    image = torch.clamp(image.permute(1, 2, 0) * rgb_std + rgb_mean, 0, 1)
    return torchvision.transforms.ToPILImage()(image.permute(2, 0, 1))

def uniform_channels(imgs):
    filtered_imgs = [img.to(device) for img in imgs if img.shape[0] <= 3]
    return filtered_imgs

def smooth_image(img):
    from PIL import ImageFilter
    img = img.filter(ImageFilter.SMOOTH)
    return img