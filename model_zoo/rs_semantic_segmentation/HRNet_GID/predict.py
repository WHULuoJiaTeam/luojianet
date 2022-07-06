# HRNet网络预测

import os
import luojianet_ms
from hrnetv2 import hrnetv2
from osgeo import gdal, gdalconst
import math
import numpy as np
from luojianet_ms import Model
from luojianet_ms import Tensor
import luojianet_ms.ops as ops
import luojianet_ms.dataset.vision.py_transforms as p_vision
from luojianet_ms.dataset.transforms import py_transforms


def run_spyd(ClassN, input_file, output_folder, checkpoint_path):
    # 加载网络模型
    model = hrnetv2(output_class=ClassN)
    param_dict = luojianet_ms.load_checkpoint(checkpoint_path)
    key_name = list(model.parameters_dict().keys())
    param_dict_new = param_dict.copy()
    for i, name in enumerate(param_dict):
        if not name in key_name:
            param_dict_new[key_name[i]] = param_dict_new[name]
            del param_dict_new[name]

    luojianet_ms.load_param_into_net(model, param_dict_new)

    img_transform = py_transforms.Compose([
        p_vision.ToTensor(),
        p_vision.Normalize([0.3309, 0.3473, 0.3247], [0.2560, 0.2512, 0.2468])
    ])

    # 读取待预测影像
    currentImgdata = gdal.Open(input_file, gdalconst.GA_ReadOnly)
    imgdata_band1 = currentImgdata.GetRasterBand(1)
    imgdata_band2 = currentImgdata.GetRasterBand(2)
    imgdata_band3 = currentImgdata.GetRasterBand(3)
    imagepatch_1 = imgdata_band1.ReadAsArray().astype('uint8')
    imagepatch_2 = imgdata_band2.ReadAsArray().astype('uint8')
    imagepatch_3 = imgdata_band3.ReadAsArray().astype('uint8')
    imagepatch_i = np.stack([imagepatch_1, imagepatch_2, imagepatch_3], axis=2)
    imagepatch_i = img_transform(imagepatch_i)
    imagepatch_i = luojianet_ms.Tensor(imagepatch_i)

    imagepatch_i = imagepatch_i.asnumpy()

    validate_spyd(imagepatch_i, model, os.path.join(output_folder, os.path.split(input_file)[-1]))


# 网络预测
def validate_spyd(image, net, output_folder):
    net.set_train(False)
    net.set_grad(False)
    model = Model(net)

    output_v = predict_whole_image(model, image, grid=512, stride=384)

    # 保存预测结果
    saveResult(output_v, output_folder)

# 滑动窗口预测整景影像
def predict_whole_image(model, image, grid, stride):
    overlap = grid - stride
    invalid_num = int(overlap / 2)

    n, b, r, c = image.shape
    rows = -((grid - r) // (stride + 1e-10)) * stride + grid
    cols = -((grid - c) // (stride + 1e-10)) * stride + grid
    rows = math.ceil(rows)
    cols = math.ceil(cols)
    image_ = np.pad(image, ((0, 0), (0, 0), (0, rows - r), (0, cols - c)), 'symmetric')

    output = np.zeros((rows, cols), dtype=np.uint8)

    softmax = ops.Softmax(axis=0)
    argmax = ops.Argmax(axis=0)

    for i in range(0, rows, stride):
        if (i + grid > rows):
            continue
        print('Current row:', i)
        for j in range(0, cols, stride):
            if (j + grid > cols):
                continue
            patch = image_[0:, 0:, i:i + grid, j:j + grid]
            patch = Tensor(patch)

            pred = model.predict(patch)

            pred = pred.squeeze()
            pred = softmax(pred)
            pred = argmax(pred)
            pred = pred.asnumpy()
            pred = pred.astype(np.uint8)

            output[i + invalid_num:i + grid - invalid_num, j + invalid_num:j + grid - invalid_num] = \
                pred[invalid_num:grid - invalid_num, invalid_num:grid - invalid_num]

    output = output[0:r, 0:c]

    return output

# 保存预测结果
def saveResult(data, output_folder):
    save_bandsize = 3

    if not os.path.exists(output_folder):
        gtif_driver = gdal.GetDriverByName("GTiff")
        out_ds = gtif_driver.Create(output_folder, data.shape[1], data.shape[0], save_bandsize, gdal.GDT_Byte)

        out_ds.FlushCache()
        del out_ds
    else:
        print('file already exist')
    write_ds = gdal.Open(output_folder, gdalconst.GA_Update)

    for j in range(save_bandsize):
        im_data_i = data.copy()
        if j == 0:
            im_data_i[data == 0] = 0
            im_data_i[data == 1] = 255
            im_data_i[data == 2] = 0
            im_data_i[data == 3] = 0
            im_data_i[data == 4] = 255
            im_data_i[data == 5] = 0
        if j == 1:
            im_data_i[data == 0] = 0
            im_data_i[data == 1] = 0
            im_data_i[data == 2] = 255
            im_data_i[data == 3] = 255
            im_data_i[data == 4] = 255
            im_data_i[data == 5] = 0
        if j == 2:
            im_data_i[data == 0] = 0
            im_data_i[data == 1] = 0
            im_data_i[data == 2] = 0
            im_data_i[data == 3] = 255
            im_data_i[data == 4] = 0
            im_data_i[data == 5] = 255

        write_ds.GetRasterBand(j + 1).WriteArray(im_data_i)


if __name__ == '__main__':
    import argparse
    import luojianet_ms.context as context

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', default=r'./GF2_PMS2__L1A0001642620-MSS2.tif')
    parser.add_argument('--output_folder', default=r'./')
    parser.add_argument('--checkpoint_path', default=r'./hrnet_best.ckpt')
    parser.add_argument('--device_target', default='GPU')
    args = parser.parse_args()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    run_spyd(6, args.input_file, args.output_folder, args.checkpoint_path)
