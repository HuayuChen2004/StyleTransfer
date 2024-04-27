
from image import preprocess



def extract_features(X, content_layers, style_layers, net):
    contents = []
    styles = []
    for i in range(len(net)):
        X = net[i](X)
        if i in content_layers:
            contents.append(X)
        if i in style_layers:
            styles.append(X)
    return contents, styles

# 获取初始内容图像和风格图像的特征（后续不再改变）
def get_contents(image_shape, content_img, content_layers, style_layers, net):
    content_X = preprocess(content_img, image_shape)
    contents_Y, _ = extract_features(content_X, content_layers, style_layers, net)
    return content_X, contents_Y

def get_styles(image_shape, style_img, content_layers, style_layers, net):
    style_X = preprocess(style_img, image_shape)
    _, styles_Y = extract_features(style_X, content_layers, style_layers, net)
    return style_X, styles_Y