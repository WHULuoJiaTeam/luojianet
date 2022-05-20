# HRNet-3D网络预测

import os
import argparse
import numpy as np
from luojianet_ms import Model
from luojianet_ms import Tensor
import luojianet_ms.context as context
import luojianet_ms.ops as ops
from luojianet_ms import load_checkpoint, load_param_into_net
from model import HigherHRNet_Binary
from osgeo import gdal
from config import config

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP','.tif'
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

# 输出预测结果影像的函数
def writeTiff(im_data, im_width, im_height, im_bands, path):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape

    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, im_width, im_height, im_bands, datatype)

    for i in range(im_bands):
        dataset.GetRasterBand(i+1).WriteArray(im_data[i])
    del dataset

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_file', type=str, default=None, help='Input image file')
    parser.add_argument('--output_folder', type=str, default=None, help='Output folder')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Saved checkpoint file path')
    parser.add_argument('--device_target', type=str, default="GPU", help='Device target')

    args_opt = parser.parse_args()

    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target)

    # 加载网络模型
    print('Build model ...')
    net = HigherHRNet_Binary(num_classes=config['classnum'], hr_cfg='w18_3d2d_at')
    model_path = args_opt.checkpoint_path
    param_dict = load_checkpoint(model_path)
    load_param_into_net(net, param_dict)
    print('Loaded trainded model.')

    net.set_train(False)
    net.set_grad(False)

    model = Model(net)

    save_path = args_opt.output_folder
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    softmax = ops.Softmax(axis=0)
    argmax = ops.Argmax(axis=0)

    # 读取待预测影像
    image_dataset = gdal.Open(args_opt.input_file, gdal.GA_ReadOnly)
    image = image_dataset.ReadAsArray()
    width = image.shape[2]
    height = image.shape[1]

    # 若需要对影像进行归一化，则逐波段进行标准差归一化
    if(config['normalize']):
        eps = 1e-8
        bands = image.shape[0]
        image_new = np.zeros_like(image)
        for i in range(bands):
            image_i = image[i, :, :]
            mean = np.mean(image_i)
            std = np.std(image_i)
            image_new[i, :, :] = (image_i - mean) / (std + eps)

        image = image_new

    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=0)
    image = Tensor(image)
    
    # 网络预测
    output = model.predict(image)
    output = output.squeeze()
    output = softmax(output)
    output = argmax(output)

    output = output.asnumpy() + 1

    output = output.astype(np.uint8)
    
    # 保存结果
    fname = os.path.basename(args_opt.input_file)
    writeTiff(output, width, height, 1, os.path.join(save_path, fname))

    print('Saved result.')

if __name__ == '__main__':
    main()