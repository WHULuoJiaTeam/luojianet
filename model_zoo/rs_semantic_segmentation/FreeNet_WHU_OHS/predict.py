import os
import argparse
import numpy as np
from luojianet_ms import Model
from luojianet_ms import Tensor
import luojianet_ms.context as context
import luojianet_ms.nn as nn
import luojianet_ms.ops as ops
from luojianet_ms import load_checkpoint, load_param_into_net
from model import FreeNet
from osgeo import gdal
from config import config

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP','.tif'
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

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

class MyWithEvalCell(nn.Module):
    def __init__(self, network):
        super(MyWithEvalCell, self).__init__(auto_prefix=False)
        self.network = network
        self.softmax = ops.Softmax(axis=0)
        self.argmax = ops.Argmax(axis=0)

    def call(self, data):
        output = self.network(data)
        output = output.squeeze()
        output = self.softmax(output)
        output = self.argmax(output)
        return output

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_file', type=str, default=None, help='Input image file')
    parser.add_argument('--output_folder', type=str, default=None, help='Output folder')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Saved checkpoint file path')
    parser.add_argument('--device_target', type=str, default="GPU", help='Device target')

    args_opt = parser.parse_args()

    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target)

    config_net = dict(
            in_channels=config['in_channels'],
            num_classes=config['classnum'],
            block_channels=(96, 128, 192, 256),
            num_blocks=(1, 1, 1, 1),
            inner_dim=128,
            reduction_ratio=1.0,
        )

    # Model
    print('Build model ...')
    net = FreeNet(config=config_net)
    model_path = args_opt.checkpoint_path
    param_dict = load_checkpoint(model_path)
    load_param_into_net(net, param_dict)
    print('Loaded trainded model.')

    net.set_train(False)
    net.set_grad(False)

    model = Model(net)

    eval_net = MyWithEvalCell(net)
    eval_net.set_train(False)
    eval_net.set_grad(False)

    save_path = args_opt.output_folder
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    softmax = ops.Softmax(axis=0)
    argmax = ops.Argmax(axis=0)

    image_dataset = gdal.Open(args_opt.input_file, gdal.GA_ReadOnly)
    image = image_dataset.ReadAsArray().astype(np.float32)
    width = image.shape[2]
    height = image.shape[1]

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
    image = Tensor(image)

    output = model.predict(image)
    output = output.squeeze()
    output = softmax(output)
    output = argmax(output)

    output = output.asnumpy() + 1

    output = output.astype(np.uint8)

    fname = os.path.basename(args_opt.input_file)
    writeTiff(output, width, height, 1, os.path.join(save_path, fname))

    print('Saved result.')

if __name__ == '__main__':
    main()