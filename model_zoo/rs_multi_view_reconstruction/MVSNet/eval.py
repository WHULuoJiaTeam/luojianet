
import argparse
import os
import numpy as np
import time
import luojianet_ms.dataset as ds
from src.dataset import MVSDatasetGenerator
from src.mvsnet import MVSNet
from luojianet_ms import Tensor
from luojianet_ms import Model
from luojianet_ms import load_checkpoint, load_param_into_net
import cv2


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser(description='A LuoJiaNET Implementation of MVSNet')
parser.add_argument('--dataset', default='whu', help='select dataset')

parser.add_argument('--data_root', default='/mnt/gj/stereo', help='train datapath')
parser.add_argument('--loadckpt', default='./checkpoint55/checkpoint_mvsnet_whu-30_3600.ckpt', help='load a specific checkpoint')
parser.add_argument('--normalize', type=str, default='mean', help='methods of center_image, mean[mean var] or standard[0-1].')

# input parameters
parser.add_argument('--view_num', type=int, default=3, help='Number of images (1 ref image and view_num - 1 view images).')
parser.add_argument('--ndepths', type=int, default=200, help='the number of depth values')
parser.add_argument('--max_w', type=int, default=768, help='Maximum image width')
parser.add_argument('--max_h', type=int, default=384, help='Maximum image height')
parser.add_argument('--resize_scale', type=float, default=1, help='output scale for depth and image (W and H)')
parser.add_argument('--sample_scale', type=float, default=0.25, help='Downsample scale for building cost volume (W and H)')
parser.add_argument('--interval_scale', type=float, default=1, help='the number of depth values')
parser.add_argument('--adaptive_scaling', type=bool, default=True, help='Let image size to fit the network, including scaling and cropping')

# parse arguments and check
args = parser.parse_args()


def create_dataset(mode, args):
    dataset_generator = MVSDatasetGenerator(args.data_root, mode, args.view_num, args.normalize, args)

    input_data = ds.GeneratorDataset(dataset_generator,
                                     column_names=["image", "camera", "target", "values", "mask"],
                                     shuffle=False)
    input_data = input_data.batch(1)

    return input_data


def Thres_metrics(depth_est, depth_gt, mask, thres):
    mask = mask > 0
    depth_est, depth_gt = depth_est[mask], depth_gt[mask]
    errors = np.abs(depth_est - depth_gt)
    err_mask = errors < thres
    return np.mean(err_mask.astype(np.float))


def AbsDepthError_metrics(depth_est, depth_gt, mask, depth_threshold):
    mask = mask > 0
    depth_est, depth_gt = depth_est[mask], depth_gt[mask]
    diff = np.fabs(depth_est - depth_gt)
    mask2 = (diff < depth_threshold)
    result = diff[mask2]

    return np.mean(result)


if __name__== "__main__":
    net = MVSNet(args.max_h, args.max_w, False)
    dataset_generator = MVSDatasetGenerator(args.data_root, "test", args.view_num, args.normalize, args)
    ds_eval = create_dataset("test", args)

    param_dict = load_checkpoint(args.loadckpt)
    load_param_into_net(net, param_dict)

    model = Model(net)

    mae = 0
    less_1 = 0
    less_3 = 0
    less_6 = 0

    i = 0
    for data in ds_eval.create_dict_iterator():
        start = time.time()

        img = Tensor(data['image'])
        camera = Tensor(data['camera'])
        values = Tensor(data['values'])

        output = model.predict(img, camera, values)
        end = time.time()

        output = output.squeeze(1)
        predict = output.asnumpy()[0]
        upsample_predict = cv2.resize(predict, None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
        label = data['target'].asnumpy()[0]
        mask = data['mask'].asnumpy()[0]

        sample_less_1 = Thres_metrics(upsample_predict, label, mask, 0.1)
        sample_less_3 = Thres_metrics(upsample_predict, label, mask, 0.3)
        sample_less_6 = Thres_metrics(upsample_predict, label, mask, 0.6)
        sample_mae = AbsDepthError_metrics(upsample_predict, label, mask, 0.1 * 100)

        print("[{}|{}] mae(m): {} less_0.1(%):{} less_0.3(%):{} less_0.6(%):{} costs(s):{}".format(
            i, ds_eval.get_dataset_size(), sample_mae, sample_less_1, sample_less_3, sample_less_6, end - start))

        mae += sample_mae
        less_1 += sample_less_1
        less_3 += sample_less_3
        less_6 += sample_less_6

        i += 1

    print("mae(m): {} less_0.1(%):{} less_0.3(%):{} less_0.6(%):{}".format(
        mae / ds_eval.get_dataset_size(), less_1 / ds_eval.get_dataset_size(),
        less_3 / ds_eval.get_dataset_size(), less_6 / ds_eval.get_dataset_size()))

