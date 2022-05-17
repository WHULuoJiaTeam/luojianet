import os
import luojianet_ms
from hrnetv2 import hrnetv2
from osgeo import gdal, gdalconst
import luojianet_ms.ops.operations as P
from luojianet_ms import dtype as mstype
import numpy as np
import luojianet_ms.dataset.vision.py_transforms as p_vision
from luojianet_ms.dataset.transforms import py_transforms



def run_spyd(ClassN, input_file, output_folder, checkpoint_path, x, y):
    model = hrnetv2(output_class=ClassN)
    luojianet_ms.load_param_into_net(model, luojianet_ms.load_checkpoint(checkpoint_path))
    #################   test model   ###################
    img_transform = py_transforms.Compose([
        p_vision.ToTensor(),
        p_vision.Normalize([0.3309, 0.3473, 0.3247], [0.2560, 0.2512, 0.2468])
    ])
    currentImgdata = gdal.Open(input_file, gdalconst.GA_ReadOnly)
    imgdata_band1 = currentImgdata.GetRasterBand(1)
    imgdata_band2 = currentImgdata.GetRasterBand(2)
    imgdata_band3 = currentImgdata.GetRasterBand(3)
    imagepatch_1 = imgdata_band1.ReadAsArray(x, y, 512, 512).astype('uint8')
    imagepatch_2 = imgdata_band2.ReadAsArray(x, y, 512, 512).astype('uint8')
    imagepatch_3 = imgdata_band3.ReadAsArray(x, y, 512, 512).astype('uint8')
    imagepatch_i = np.stack([imagepatch_1, imagepatch_2, imagepatch_3], axis=2)
    imagepatch_i = img_transform(imagepatch_i)
    imagepatch_i = luojianet_ms.Tensor(imagepatch_i)
    validate_spyd(imagepatch_i, model, os.path.join(output_folder, os.path.split(input_file)[-1]))



def validate_spyd(test_loader, model, output_folder):
    model.set_train(False)
    model.set_grad(False)

    input = test_loader
    input = luojianet_ms.ops.Cast()(input, mstype.float32)
    output = model(input)

    output_v = P.ArgMaxWithValue(axis=1, keep_dims=False)(output)[0].asnumpy()
    saveResult(output_v, output_folder)






def printIOU(dict_p):
    for key in dict_p:
        if key[:3] == 'IOU':
            print(key, '-->', dict_p[key])
def transform_result_dict_spyd(metrixlist, class_list=None):
    if class_list is None:
        class_list = ['_background', '_build', '_farmland', '_forest', '_meadow', '_water']
    dict_result = {}
    dict = [metric.validate() for metric in metrixlist]
    for ind, name in enumerate(class_list):
        for key in ["Precision", "Recall", "F1", "IOU"]:
            dict_result[key + name] = dict[ind][key]
            if key not in dict_result:
                dict_result[key] = dict[ind][key]
            else:
                dict_result[key] += dict[ind][key]
            if name != '_background':
                if key + '_target' not in dict_result:
                    dict_result[key + '_target'] = dict[ind][key]
                else:
                    dict_result[key + '_target'] += dict[ind][key]
    for key in ["Precision", "Recall", "F1", "IOU"]:
        dict_result[key] = dict_result[key] / len(class_list)
    for key in ["Precision_target", "Recall_target", "F1_target", "IOU_target"]:
        dict_result[key] = dict_result[key] / (len(class_list) - 1)
    return dict_result

def saveResult(data, output_folder):
    save_bandsize = 3
    data = data[0, :, :]

    if not os.path.exists(output_folder):
        gtif_driver = gdal.GetDriverByName("GTiff")
        out_ds = gtif_driver.Create(output_folder, 512, 512, save_bandsize, gdal.GDT_Byte)

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
        # print(im_data_i)
        write_ds.GetRasterBand(j + 1).WriteArray(im_data_i, 0, 0)



def saveBlockResult(data, save_info, train_name, Display=True):
    saveProot = os.path.join(r"D:\zz\SPYD_data\Predict", train_name)
    saveProot = saveProot if not Display else saveProot + '_RGB'
    if not os.path.exists(saveProot):
        os.makedirs(saveProot)
    save_bandsize = 3 if Display else 1
    B, _, _ = data.shape
    label_path = save_info['labelPath']
    x = save_info['x']
    y = save_info['y']
    block_x = save_info['block_x']
    block_y = save_info['block_y']
    for i in range(B):
        label_path_i = label_path[i]
        save_path_i = os.path.join(saveProot, os.path.split(label_path_i)[-1]).replace('.tif','_'+str(int(x[i]))+'_'+str(int(x[i]))+'.tif')
        block_x_i = block_x[i]
        block_y_i = block_y[i]
        data_i = data[i, :block_y_i, :block_x_i]

        if int(block_x_i) == 512 and int(block_y_i) == 512:
            gtif_driver = gdal.GetDriverByName("GTiff")
            out_ds = gtif_driver.Create(save_path_i, int(block_x_i), int(block_y_i), save_bandsize, gdal.GDT_Byte, options=["TILED=YES", "COMPRESS=PACKBITS"])
            out_ds.FlushCache()
            # del out_ds

            write_ds = gdal.Open(save_path_i, gdalconst.GA_Update)
            if Display:
                for j in range(save_bandsize):
                    im_data_i = data_i.copy()
                    if j == 0:
                        im_data_i[data_i == 0] = 0
                        im_data_i[data_i == 1] = 255
                        im_data_i[data_i == 2] = 0
                        im_data_i[data_i == 3] = 0
                        im_data_i[data_i == 4] = 255
                        im_data_i[data_i == 5] = 0
                    if j == 1:
                        im_data_i[data_i == 0] = 0
                        im_data_i[data_i == 1] = 0
                        im_data_i[data_i == 2] = 255
                        im_data_i[data_i == 3] = 255
                        im_data_i[data_i == 4] = 255
                        im_data_i[data_i == 5] = 0
                    if j == 2:
                        im_data_i[data_i == 0] = 0
                        im_data_i[data_i == 1] = 0
                        im_data_i[data_i == 2] = 0
                        im_data_i[data_i == 3] = 255
                        im_data_i[data_i == 4] = 0
                        im_data_i[data_i == 5] = 255
                    write_ds.GetRasterBand(j + 1).WriteArray(im_data_i)
            else:
                TowriteBand = write_ds.GetRasterBand(1)
                TowriteBand.WriteArray(data_i)
def saveBlockTarget(data, save_info, Display=True):
    saveProot = r"D:\zz\SPYD_data\Predict\Target"
    if not os.path.exists(saveProot):
        os.makedirs(saveProot)
    save_bandsize = 3 if Display else 1
    B, _, _ = data.shape
    label_path = save_info['labelPath']
    x = save_info['x']
    y = save_info['y']
    block_x = save_info['block_x']
    block_y = save_info['block_y']
    for i in range(B):
        label_path_i = label_path[i]
        save_path_i = os.path.join(saveProot, os.path.split(label_path_i)[-1]).replace('.tif','_'+str(int(x[i]))+'_'+str(int(x[i]))+'.tif')
        block_x_i = block_x[i]
        block_y_i = block_y[i]
        data_i = data[i, :block_y_i, :block_x_i]

        if int(block_x_i) == 512 and int(block_y_i)==512:
            gtif_driver = gdal.GetDriverByName("GTiff")
            out_ds = gtif_driver.Create(save_path_i, int(block_x_i), int(block_y_i), save_bandsize, gdal.GDT_Byte, options=["TILED=YES", "COMPRESS=PACKBITS"])
            out_ds.FlushCache()
            # del out_ds

            write_ds = gdal.Open(save_path_i, gdalconst.GA_Update)
            if Display:
                for j in range(save_bandsize):
                    im_data_i = data_i.copy()
                    if j == 0:
                        im_data_i[data_i == 0] = 255
                        im_data_i[data_i == 4] = 255
                        im_data_i[data_i == 1] = 50
                        im_data_i[data_i == 2] = 237
                        im_data_i[data_i == 3] = 140
                    if j == 1:
                        im_data_i[data_i == 0] = 255
                        im_data_i[data_i == 4] = 255
                        im_data_i[data_i == 1] = 67
                        im_data_i[data_i == 2] = 159
                        im_data_i[data_i == 3] = 124
                    if j == 2:
                        im_data_i[data_i == 0] = 255
                        im_data_i[data_i == 4] = 255
                        im_data_i[data_i == 1] = 138
                        im_data_i[data_i == 2] = 186
                        im_data_i[data_i == 3] = 65
                    write_ds.GetRasterBand(j + 1).WriteArray(im_data_i)
            else:
                TowriteBand = write_ds.GetRasterBand(1)
                TowriteBand.WriteArray(data_i)
def saveBlockImage(data, save_info, Display=True):
    saveProot = r"D:\zz\SPYD_data\Predict\Image"
    if not os.path.exists(saveProot):
        os.makedirs(saveProot)
    save_bandsize = 3
    B, C, _, _ = data.shape
    label_path = save_info['labelPath']
    x = save_info['x']
    y = save_info['y']
    block_x = save_info['block_x']
    block_y = save_info['block_y']
    for i in range(B):
        label_path_i = label_path[i]
        save_path_i = os.path.join(saveProot, os.path.split(label_path_i)[-1]).replace('.tif','_'+str(int(x[i]))+'_'+str(int(x[i]))+'.tif')
        block_x_i = block_x[i]
        block_y_i = block_y[i]
        data_i = data[i, :, :block_y_i, :block_x_i]

        if int(block_x_i) == 512 and int(block_y_i)==512:
            gtif_driver = gdal.GetDriverByName("GTiff")
            out_ds = gtif_driver.Create(save_path_i, int(block_x_i), int(block_y_i), save_bandsize, gdal.GDT_Byte, options=["TILED=YES", "COMPRESS=PACKBITS"])
            out_ds.FlushCache()
            # del out_ds
            write_ds = gdal.Open(save_path_i, gdalconst.GA_Update)
            for i in range(save_bandsize):
                TowriteBand = write_ds.GetRasterBand(i+1)
                TowriteBand.WriteArray(data_i[i, :, :])
if __name__ == '__main__':
    import argparse
    import luojianet_ms.context as context

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', default=r'/media/vhr/0D7A09740D7A0974/zz/GID/LCC5C/image_RGB_test/GF2_PMS2__L1A0001642620-MSS2.tif')
    parser.add_argument('--x', default='0')
    parser.add_argument('--y', default='0')
    parser.add_argument('--output_folder', default=r'/home/vhr/Zhangzhen/CPZL/GID_pre')
    parser.add_argument('--checkpoint_path', default=r'/media/vhr/0D7A09740D7A0974/zz/Luojianet_ckpt/hrnet/hrnet_best.ckpt')
    parser.add_argument('--device_target', default='GPU')
    args = parser.parse_args()

    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    run_spyd(6, args.input_file, args.output_folder, args.checkpoint_path, int(args.x), int(args.y))















