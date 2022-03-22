import os
import argparse
import numpy as np
import cv2
import math
import time
import gdal
import gc
from mindspore import Tensor
import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.deeplab_v3plus import DeepLabV3Plus
# context.set_context(mode=context.GRAPH_MODE, device_target="CPU", save_graphs=False,
#                     device_id=int(os.getenv('DEVICE_ID')))
context.set_context(mode=context.GRAPH_MODE, device_target="GPU", save_graphs=False)

def parse_args():
    parser = argparse.ArgumentParser('LuoJiaNET DeepLabV3+ eval')

    # val data
    parser.add_argument('--data_root', type=str, default='', help='root path of val data')
    parser.add_argument('--data_lst', type=str, default='', help='list of val data')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--crop_size', type=int, default=800, help='crop size')
    parser.add_argument('--image_mean', type=list, default=[103.53, 116.28, 123.675], help='image mean')
    parser.add_argument('--image_std', type=list, default=[57.375, 57.120, 58.395], help='image std')
    parser.add_argument('--scales', type=float, default=[0.5, 1, 1.5, 2], action='append', help='scales of evaluation')
    parser.add_argument('--flip', action='store_true', help='perform left-right flip')
    parser.add_argument('--ignore_label', type=int, default=255, help='ignore label')
    parser.add_argument('--num_classes', type=int, default=6, help='number of classes')

    # model
    parser.add_argument('--model', type=str, default='DeepLabV3plus_s16', help='select model')
    parser.add_argument('--freeze_bn', action='store_true', default=False, help='freeze bn')
    parser.add_argument('--ckpt_path', type=str, default='/home/zgw/deeplabv3plus_gid_tif/result/ckpt/DeepLabV3plus_s16-210_160.ckpt', help='model to evaluate')
    parser.add_argument('--input_img_path', type=str, default='/home/zgw/image_RGB/GF2_PMS1__L1A0000564539-MSS1.tif', help='select model')
    parser.add_argument('--output_img_path', type=str, default='./GF2_PMS1__L1A0000564539-MSS1_result.tif', help='select model')

    args, _ = parser.parse_known_args()
    return args


def cal_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(np.int32) + b[k], minlength=n ** 2).reshape(n, n)


def resize_long(img, long_size=513):
    h, w, _ = img.shape
    if h > w:
        new_h = long_size
        new_w = int(1.0 * long_size * w / h)
    else:
        new_w = long_size
        new_h = int(1.0 * long_size * h / w)
    imo = cv2.resize(img, (new_w, new_h))
    return imo


class BuildEvalNetwork(nn.Cell):
    def __init__(self, network):
        super(BuildEvalNetwork, self).__init__()
        self.network = network
        self.softmax = nn.Softmax(axis=1)

    def construct(self, input_data):
        output = self.network(input_data)
        output = self.softmax(output)
        return output


def pre_process(args, img_, crop_size=513):
    # resize
    img_ = resize_long(img_, crop_size)
    resize_h, resize_w, _ = img_.shape

    # mean, std
    image_mean = np.array(args.image_mean)
    image_std = np.array(args.image_std)
    img_ = (img_ - image_mean) / image_std

    # pad to crop_size
    pad_h = crop_size - img_.shape[0]
    pad_w = crop_size - img_.shape[1]
    if pad_h > 0 or pad_w > 0:
        img_ = cv2.copyMakeBorder(img_, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)

    # hwc to chw
    img_ = img_.transpose((2, 0, 1))
    img_ = img_[np.newaxis,:]
    return img_, resize_h, resize_w


def eval_batch(args, eval_net, img_lst, crop_size=513, flip=True):
    result_lst = []
    batch_size = len(img_lst)
    batch_img = np.zeros((args.batch_size, 3, crop_size, crop_size), dtype=np.float32)
    resize_hw = []
    for l in range(batch_size):
        img_ = img_lst[l]
        img_, resize_h, resize_w = pre_process(args, img_, crop_size)
        batch_img[l] = img_
        resize_hw.append([resize_h, resize_w])

    batch_img = np.ascontiguousarray(batch_img)
    net_out = eval_net(Tensor(batch_img, mstype.float32))
    net_out = net_out.asnumpy()

    if flip:
        batch_img = batch_img[:, :, :, ::-1]
        net_out_flip = eval_net(Tensor(batch_img, mstype.float32))
        net_out += net_out_flip.asnumpy()[:, :, :, ::-1]

    for bs in range(batch_size):
        probs_ = net_out[bs][:, :resize_hw[bs][0], :resize_hw[bs][1]].transpose((1, 2, 0))
        ori_h, ori_w = img_lst[bs].shape[0], img_lst[bs].shape[1]
        probs_ = cv2.resize(probs_, (ori_w, ori_h))
        result_lst.append(probs_)

    return result_lst

def eval_single(args, eval_net, img_, crop_size=513):

    eval_net.set_train(False)
    img_out, resize_h, resize_w = pre_process(args, img_, crop_size)
    net_out = eval_net(Tensor(img_out, mstype.float32))
    net_out = net_out.asnumpy()
    net_out = np.squeeze(net_out, axis=0)
    probs_ = net_out[:, :resize_h, :resize_w].transpose((1, 2, 0))
    ori_h, ori_w = img_.shape[0], img_.shape[1]
    probs_ = cv2.resize(probs_, (ori_w, ori_h))

    return probs_


def eval_batch_scales(args, eval_net, img_lst, scales,
                      base_crop_size=513, flip=True):
    sizes_ = [int((base_crop_size - 1) * sc) + 1 for sc in scales]
    probs_lst = eval_batch(args, eval_net, img_lst, crop_size=sizes_[0], flip=flip)
    print(sizes_)
    for crop_size_ in sizes_[1:]:
        probs_lst_tmp = eval_batch(args, eval_net, img_lst, crop_size=crop_size_, flip=flip)
        for pl, _ in enumerate(probs_lst):
            probs_lst[pl] += probs_lst_tmp[pl]

    result_msk = []
    for i in probs_lst:
        result_msk.append(i.argmax(axis=2))
    return result_msk

def eval_single_scales(args, eval_net, img_, scales,
                      base_crop_size=513):
    eval_net.set_train(False)
    sizes_ = [int((base_crop_size - 1) * sc) + 1 for sc in scales]
    probs_lst = eval_single(args, eval_net, img_, crop_size=sizes_[0])
    # probs_lst = eval_single(args, eval_net, img_, crop_size=base_crop_size)
    print('sizes_',sizes_)
    for crop_size_ in sizes_[1:]:
        probs_lst_tmp = eval_single(args, eval_net, img_, crop_size=crop_size_)
        probs_lst += probs_lst_tmp
    result_msk = probs_lst.argmax(axis=2)

    return result_msk

def computePadSize(src_x_size):
    sub_x = 0
    if src_x_size < 100000:
        if len(str(src_x_size)) == 5:
            sub_x = src_x_size - int(str(src_x_size)[0]) * math.pow(10, len(str(src_x_size)) - 1)  # sub-number
            sub_x = np.ceil(sub_x / 1000) * 1000
            sub_x = sub_x + int(str(src_x_size)[0]) * math.pow(10, len(str(src_x_size)) - 1)
            sub_x = sub_x - src_x_size

        elif len(str(src_x_size)) == 4:
            sub_x = src_x_size - int(str(src_x_size)[0]) * math.pow(10, len(str(src_x_size)) - 1)  # sub-number
            sub_x = np.ceil(sub_x / 100) * 100
            sub_x = sub_x + int(str(src_x_size)[0]) * math.pow(10, len(str(src_x_size)) - 1)
            sub_x = sub_x - src_x_size

        elif len(str(src_x_size)) == 3:
            sub_x = src_x_size - int(str(src_x_size)[0]) * math.pow(10, len(str(src_x_size)) - 1)  # sub-number
            sub_x = np.ceil(sub_x / 10) * 10
            sub_x = sub_x + int(str(src_x_size)[0]) * math.pow(10, len(str(src_x_size)) - 1)
            sub_x = sub_x - src_x_size
        elif len(str(src_x_size)) == 2:
            sub_x = src_x_size - int(str(src_x_size)[0]) * math.pow(10, len(str(src_x_size)) - 1)  # sub-number
            sub_x = np.ceil(sub_x / 1) * 1
            sub_x = sub_x + int(str(src_x_size)[0]) * math.pow(10, len(str(src_x_size)) - 1)
            sub_x = sub_x - src_x_size
    return int(sub_x)

def computePadSizeNew(src_x_size, window_size):

    pad_size = 0
    if src_x_size % window_size == 0:
        pad_size = 0
    else:
        m = np.ceil(src_x_size / window_size) #uppder inter
        # n = src_x_size % window_size
        pad_size = m * window_size - src_x_size
    return int(pad_size)

def bytescaling(data, cmin=None, cmax=None, high=255, low=0):
    """
    Converting the input image to uint8 dtype and scaling
    the range to ``(low, high)`` (default 0-255). If the input image already has
    dtype uint8, no scaling is done.
    :param data: 16-bit image data array
    :param cmin: bias scaling of small values (def: data.min())
    :param cmax: bias scaling of large values (def: data.max())
    :param high: scale max value to high. (def: 255)
    :param low: scale min value to low. (def: 0)
    :return: 8-bit image data array
    """
    if data.dtype == np.uint8:
        return data

    if high > 255:
        high = 255
    if low < 0:
        low = 0
    if high < low:
        raise ValueError("`high` should be greater than or equal to `low`.")

    if cmin is None:
        cmin = data.min()
    if cmax is None:
        cmax = data.max()

    cscale = cmax - cmin
    if cscale == 0:
        cscale = 1

    scale = float(high - low) / cscale
    bytedata = (data - cmin) * scale + low
    return (bytedata.clip(low, high) + 0.5).astype(np.uint8)

def gray2RGB(img):

    H, W = img.shape
    img_rgb = np.random.randint(0, 256, size=[H, W, 3], dtype=np.uint64)
    for i in range(H):
        for j in range(W):
            if img[i][j] == 1:
                img_rgb[i][j] = np.array([255, 0, 0])
            elif img[i][j] == 2:
                img_rgb[i][j] = np.array([0, 255, 0])
            elif img[i][j] == 3:
                img_rgb = np.array([0, 255, 255])
            elif img[i][j] == 4:
                img_rgb[i][j] = np.array([255, 255, 0])
            elif img[i][j] == 5:
                img_rgb[i][j] = np.array([0, 0, 255])
            else:
                print(img[i][j])
                img_rgb[i][j] = np.array([0, 0, 0])
    print('img_rgb shape',img_rgb.shape)
    return img_rgb

def readImage(img_path):
    dataset = gdal.Open(img_path, gdal.GA_ReadOnly)
    print('imgpath',img_path)
    if dataset is None:
        print("Unable to open image file.")
        return None
    else:
        width = dataset.RasterXSize
        height = dataset.RasterYSize
        #proj = dataset.GetProjection()
        #geotrans = dataset.GetGeoTransform()
        #print("Image height:" + dataset.RasterYSize.__str__() + " Image width:" + dataset.RasterXSize.__str__())
        img_data = dataset.ReadAsArray(0, 0 ,width, height) # CHW to HWC
        print('img_data',img_data.shape)
        return img_data


def readWriteAndProcessLargeBlockImageNOSPNOUncertaintySingleOut(input_src_file,  output_img_file, model, args, winx = 512, winy = 512, stride = 512, num_classes=2):

    src_dataset = gdal.Open(input_src_file, gdal.GA_ReadOnly)
    print('src_dataset type', type(src_dataset))
    src_x_size = src_dataset.RasterXSize
    src_y_size = src_dataset.RasterYSize
    im_geotrans = src_dataset.GetGeoTransform()
    im_proj = src_dataset.GetProjection()
    src_bands = src_dataset.RasterCount

    model.set_train(False)

    m_dataType = gdal.GetDataTypeName(src_dataset.GetRasterBand(1).DataType)
    m_strechFlag = False
    if not (m_dataType == 'Byte' or m_dataType == 'Unknown'):
        print('m_dataType,1111111111111')
        m_strechFlag = True
    if m_strechFlag:
         percentBandMinvalue, percentBandMaxvalue, m_histcolor = pre_process.calcImagePercentValue(src_dataset, 5000,
                                                                                                   5000,
                                                                                                   0.25, 0.25)
    dst_x_size = int(src_x_size)
    dst_y_size = int(src_y_size)
    driver = gdal.GetDriverByName("GTiff")
    gdal.SetConfigOption('CHECK_DISK_FREE_SPACE', 'NO')

    output_img_file = output_img_file.replace('\\','/')
    output_img_file_path = output_img_file[0:output_img_file.rindex('/')]
    if not os.path.exists(output_img_file_path):
        os.makedirs(output_img_file_path)

    outDs = driver.Create(output_img_file, dst_x_size, dst_y_size, 1, gdal.GDT_Byte)
    outDs.SetGeoTransform(im_geotrans)
    outDs.SetProjection(im_proj)

    input_winy = winy
    input_winx = winx

    winx_pad = 0
    if winx > src_x_size:
        # winx_pad = winx - src_x_size
        winx_pad = computePadSize(src_x_size)
        winx = src_x_size

    winy_pad = 0
    if winy > src_y_size:
        # winy_pad = winy - src_y_size
        winy_pad = computePadSize(src_y_size)
        winy = src_y_size

    max_stride = max(src_y_size, src_x_size)
    if stride > max_stride:
        stride = max_stride

    overlap_x = winx - stride
    overlap_y = winy - stride
    init_overlap_x = overlap_x
    init_overlap_y = overlap_y

    tt = time.time()
    # count = 0
    for y in range(0, src_y_size, stride):
        for x in range(0, src_x_size, stride):
            #out of window boundary
            winx = input_winx
            winy = input_winy
            overlap_x_new = init_overlap_x
            overlap_y_new = init_overlap_y

            if (y + winy) > src_y_size and y < src_y_size:

                if overlap_y_new + y > src_y_size: #in case perminant overlpping out of boundary
                    overlap_y_new = src_y_size - y

                overlap_y_new = overlap_y_new + y + winy - src_y_size

                print('overlap_y_new, init_overlap_y, y, winy, src_y_size', overlap_y_new, init_overlap_y, y, winy, src_y_size)

                y = src_y_size - winy
                winy_pad = input_winy - winy

                if y < 0:
                    winy_pad = winy - src_y_size #pad sizes
                    winy = winy + y#update winy size, plus negative y
                    y = 0


                # continue
            if (x + winx) > src_x_size and x < src_x_size:

                if overlap_x_new + x > src_x_size: #in case perminant overlpping out of boundary
                    overlap_x_new = src_x_size - x

                overlap_x_new = overlap_x_new + x + winx - src_x_size

                print('overlap_y_new, init_overlap_y, y, winy, src_y_size', overlap_x_new, init_overlap_x, x, winx,
                      src_x_size)

                x = src_x_size - winx
                winx_pad = input_winx - winx

                if x < 0:
                    winx_pad = winx - src_x_size #pad sizes
                    winx = winx + x#update winy size, plus negative y
                    x = 0


            print('x,y', x,y)
            print('winx,winy',winx,winy)
            # print('winx_pad, winy_pad', winx_pad, winy_pad)
            # continue

            if src_bands == 1:
                src_Img = src_dataset.GetRasterBand(1).ReadAsArray(x, y, winx, winy).copy()
                src_Img_gray3 = cv2.cvtColor(np.uint8(src_Img), cv2.COLOR_GRAY2BGR)
                image_B, image_G, image_R = cv2.split(src_Img_gray3)
                print('src_bands 111111')
                # image_B = src_Img
                # image_G = src_Img
                # image_R = src_Img
            else:
                image_B = src_dataset.GetRasterBand(3).ReadAsArray(x, y, winx, winy).copy()
                image_G = src_dataset.GetRasterBand(2).ReadAsArray(x, y, winx, winy).copy()
                image_R = src_dataset.GetRasterBand(1).ReadAsArray(x, y, winx, winy).copy()
                print('src_bands 33333')
                # if m_strechFlag:
                #     image_B = pre_process.percentstrech(image_B, percentBandMinvalue[2], percentBandMaxvalue[2],
                #                                         m_histcolor[2])
                #     image_G = pre_process.percentstrech(image_G, percentBandMinvalue[1], percentBandMaxvalue[1],
                #                                         m_histcolor[1])
                #     image_R = pre_process.percentstrech(image_R, percentBandMinvalue[0], percentBandMaxvalue[0],
                #                                         m_histcolor[0])

            # image_B = src_dataset.GetRasterBand(3).ReadAsArray(x, y, winx, winy).copy()
            # image_G = src_dataset.GetRasterBand(2).ReadAsArray(x, y, winx, winy).copy()
            # image_R = src_dataset.GetRasterBand(1).ReadAsArray(x, y, winx, winy).copy()

            if winx >= src_x_size:
                # winx_pad = winx - src_x_size
                # winx_pad = computePadSize(src_x_size)
                winx_pad = computePadSizeNew(src_x_size, 2)

            if winy >= src_y_size:
                # winy_pad = winy - src_y_size
                # winy_pad = computePadSize(src_y_size)
                winy_pad = computePadSizeNew(src_y_size, 2)



            print('winx_pad, winy_pad', winx_pad, winy_pad)

            if winy_pad > 0 or winx_pad > 0:
                if winy_pad % 2 == 0:
                    pady = (( int(winy_pad/2), int(winy_pad/2)))
                else:
                    pady = ((int(winy_pad / 2), int(winy_pad / 2)+1))

                if winx_pad % 2 == 0:
                    padx = ((int(winx_pad / 2), int(winx_pad / 2)))
                else:
                    padx = ((int(winx_pad / 2), int(winx_pad / 2) + 1))

                more_borders = (pady, padx)
                image_B_pad_new =  np.pad(image_B, pad_width=more_borders, mode='reflect')
                image_G_pad_new =  np.pad(image_G, pad_width=more_borders, mode='reflect')
                image_R_pad_new =  np.pad(image_R, pad_width=more_borders, mode='reflect')
                input_image = np.dstack((image_R_pad_new, image_G_pad_new, image_B_pad_new))
            else:
                input_image = np.dstack((image_R, image_G, image_B))

            #input_image = bytescaling(input_image)
            # input_image = stretch_8bit(input_image)
            # input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
            #input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
            #input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)

            input_image_gray = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
            maskImg = input_image_gray > 0
            maskImg = maskImg.astype(np.uint8)

            # label_pred = predict_img_with_smooth_windowing_no_uncertainty_temp1(
            #     input_image,
            #     window_size=1024,
            #     # window_size=512,
            #     subdivisions=2,  # Minimal amount of overlap for windowing. Must be an even number.
            #     batch_size=3,
            #     num_classes=num_classes,
            #     pred_func=(
            #         lambda img: predict_msprob_wo_loader(model, img, gpu=0)
            #     ),
            # )

            label_pred = eval_single_scales(args, model, input_image, scales=args.scales,
                                             base_crop_size=args.crop_size)
            print('label_pred',label_pred.shape)
            # label_pred = input_image_gray
         
            # label_pred[label_pred>0] = 255
            pred_merge = label_pred

            if winy_pad == 0 and winx_pad != 0:
                pred_merge_unpad = pred_merge[:, padx[0]:-padx[1]]
                maskImg = maskImg[:, padx[0]:-padx[1]]
            elif winx_pad == 0 and winy_pad != 0:
                pred_merge_unpad = pred_merge[pady[0]:-pady[1], :]
                maskImg = maskImg[pady[0]:-pady[1], :]
            elif winy_pad == 0 and  winx_pad == 0:
                pred_merge_unpad = pred_merge[:, :]
                maskImg = maskImg[:, :]
            else:
                pred_merge_unpad = pred_merge[pady[0]:-pady[1], padx[0]:-padx[1]]
                maskImg = maskImg[pady[0]:-pady[1], padx[0]:-padx[1]]

            cur_out_image_gray = pred_merge_unpad
            maskedPred = cv2.bitwise_and(cur_out_image_gray, cur_out_image_gray, mask=maskImg)
            outBand = outDs.GetRasterBand(1)
            outBand.WriteArray(maskedPred, x, y)
            outDs.GetRasterBand(1).FlushCache()
            outDs.FlushCache()
    
            del image_R
            del image_G
            del image_B
            del input_image_gray
            del cur_out_image_gray
            del pred_merge_unpad
            # del output_image
            del label_pred
            del input_image
            gc.collect()
    print('Time used: {} sec'.format(time.time()-tt))
    
    image_gray = readImage(output_img_file)
    image_rgb = gray2RGB(image_gray)
    from PIL import Image
    print('image_out shape',image_rgb)
    im = Image.fromarray(image_rgb)
    im.save("tmprgb.jpg")
    
    print('change tiff2rgb Time used: {} sec'.format(time.time()-tt))
    # outBand = None
    # outDs = None
    # del outDs
    # del outDsGray


def simplePatchMerge(input_src_file,  output_img_file, model, args, winx = 512, winy = 512, stride = 512, num_classes=2):

    src_dataset = gdal.Open(input_src_file, gdal.GA_ReadOnly)
    src_x_size = src_dataset.RasterXSize
    src_y_size = src_dataset.RasterYSize
    im_geotrans = src_dataset.GetGeoTransform()
    im_proj = src_dataset.GetProjection()
    src_bands = src_dataset.RasterCount

    model.set_train(False)


    dst_x_size = int(src_x_size)
    dst_y_size = int(src_y_size)
    driver = gdal.GetDriverByName("GTiff")
    gdal.SetConfigOption('CHECK_DISK_FREE_SPACE', 'NO')

    output_img_file = output_img_file.replace('\\','/')
    output_img_file_path = output_img_file[0:output_img_file.rindex('/')]
    if not os.path.exists(output_img_file_path):
        os.makedirs(output_img_file_path)

    outDs = driver.Create(output_img_file, dst_x_size, dst_y_size, 1, gdal.GDT_Byte)
    outDs.SetGeoTransform(im_geotrans)
    outDs.SetProjection(im_proj)


    tt = time.time()
    # count = 0

    for y in range(0, src_y_size, stride):
        for x in range(0, src_x_size, stride):

            x_act = x
            y_act = y

            if x + stride > src_x_size:
                x_act = src_x_size - stride
            if y + stride > src_y_size:
                y_act = src_y_size - stride

            if src_bands == 1:
                src_Img = src_dataset.GetRasterBand(1).ReadAsArray(x_act, y_act, winx, winy).copy()
                src_Img_gray3 = cv2.cvtColor(np.uint8(src_Img), cv2.COLOR_GRAY2BGR)
                image_B, image_G, image_R = cv2.split(src_Img_gray3)
            else:
                image_B = src_dataset.GetRasterBand(3).ReadAsArray(x_act, y_act, winx, winy).copy()
                image_G = src_dataset.GetRasterBand(2).ReadAsArray(x_act, y_act, winx, winy).copy()
                image_R = src_dataset.GetRasterBand(1).ReadAsArray(x_act, y_act, winx, winy).copy()

            input_image = np.dstack((image_R, image_G, image_B))
            input_image = bytescaling(input_image)
            input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
            input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)

            input_image_gray = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
            maskImg = input_image_gray > 0
            maskImg = maskImg.astype(np.uint8)

            label_pred = eval_single_scales(args, model, input_image, scales=args.scales,
                                             base_crop_size=args.crop_size)
            label_pred[label_pred>0] = 255

            cur_out_image_gray = label_pred

            maskedPred = cv2.bitwise_and(cur_out_image_gray, cur_out_image_gray, mask=maskImg)
            outBand = outDs.GetRasterBand(1)
            outBand.WriteArray(maskedPred, x_act, y_act)
            # outDs.GetRasterBand(1).FlushCache()
            # outDs.FlushCache()

            del image_R
            del image_G
            del image_B
            del input_image_gray
            del cur_out_image_gray
            # del output_image
            del label_pred
            del input_image
            gc.collect()

    print('Time used: {} sec'.format(time.time()-tt))




def pred_net():
    args = parse_args()

    # network
    if args.model == 'DeepLabV3plus_s16':
        network = DeepLabV3Plus('eval', args.num_classes, 16, args.freeze_bn)
    elif args.model == 'DeepLabV3plus_s8':
        network = DeepLabV3Plus('eval', args.num_classes, 8, args.freeze_bn)
    else:
        raise NotImplementedError('model [{:s}] not recognized'.format(args.model))

    eval_net = BuildEvalNetwork(network)
    eval_net.set_train(False)
    # load model
    param_dict = load_checkpoint(args.ckpt_path)
    load_param_into_net(eval_net, param_dict)
    input_img_path = args.input_img_path
    output_img_path = args.output_img_path

    #img_ = cv2.imread(input_img_path)
    #pred_result = eval_single_scales(args, eval_net, img_, scales=args.scales,
    #                  base_crop_size=args.crop_size)
    #pred_result[pred_result>0] = 255
    #cv2.imwrite(output_img_path, np.uint8(pred_result))
    readWriteAndProcessLargeBlockImageNOSPNOUncertaintySingleOut(input_img_path, output_img_path, eval_net, args, num_classes=args.num_classes)
    #simplePatchMerge(input_img_path, output_img_path, eval_net, args, num_classes=args.num_classes)


if __name__ == '__main__':
    pred_net()
