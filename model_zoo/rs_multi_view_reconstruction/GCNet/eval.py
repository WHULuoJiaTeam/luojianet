
from src.dataset import DatasetGenerator
import luojianet_ms.dataset as ds
from src.GCNet import GCNet
import os
import luojianet_ms.context as context
from luojianet_ms import load_checkpoint, load_param_into_net
from luojianet_ms import Tensor
import time
from luojianet_ms import Model
import numpy as np
import argparse


# gpu setting
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

# Set graph mode and target device
context.set_context(mode=context.GRAPH_MODE, device_target='GPU')

parser = argparse.ArgumentParser(description='LuoJiaNET GCNet Implement')
parser.add_argument("--eval_list", type=str, default="list/whu_validation.txt", help="the list for training")
parser.add_argument("--crop_h", type=int, default=384, help="crop height")
parser.add_argument("--crop_w", type=int, default=768, help="crop width")
parser.add_argument("--max_disp", type=int, default=160, help="max disparity")
parser.add_argument("--dataset_type", type=str, default="whu", help="dataset")
parser.add_argument('--model_path', type=str, default="checkpoint/checkpoint_gcnet_whu-20_8316.ckpt")
opt = parser.parse_args()


def create_dataset(list_file, batch_size, crop_w, crop_h, dataset):
    # define dataset
    ds.config.set_seed(1)
    dataset_generator = DatasetGenerator(list_file, crop_h, crop_w, dataset=dataset)
    input_data = ds.GeneratorDataset(dataset_generator, ["data", "label"], shuffle=False)
    input_data = input_data.batch(batch_size=batch_size)

    return input_data


def read_list(list_path):
    with open(list_path, "r") as f:
        data = f.read().splitlines()

    data = [d.split(",") for d in data]

    return data


def mean_absolute_error(gt, pred):
    difference = np.fabs(gt - pred)

    return np.sum(difference)


def less_than_n_pixel(gt, pred, n):
    mask = (gt > 0)
    difference = np.fabs(gt[mask] - pred[mask])
    valid_mask = (difference < n)

    return np.sum(valid_mask.astype(float)), np.sum(mask.astype(float))


if __name__ == "__main__":

    print("GCNet evaluating...")
    print(opt)

    # model
    net = GCNet(max_disp=opt.max_disp)
    param_dict = load_checkpoint(opt.model_path)
    load_param_into_net(net, param_dict)

    model = Model(net)

    dataset = create_dataset(opt.eval_list, 1, opt.crop_w, opt.crop_h, opt.dataset_type)

    # shuffle=False， please note！！
    data_list = read_list(opt.eval_list)
    idx = 0

    difference = 0
    valid_less_1 = 0
    valid_less_2 = 0
    valid_less_3 = 0
    all_pixel = 0
    for data in dataset.create_dict_iterator():
        start = time.time()

        input_data = Tensor(data['data'])
        output = net(input_data)

        end = time.time()

        disparity = output.asnumpy()[0]
        label = data['label'].asnumpy()[0]

        dir_name = data_list[idx][0].split("/")[-3]
        name = os.path.splitext(os.path.basename(data_list[idx][0]))[0]

        # mae
        mae = mean_absolute_error(label, disparity)
        difference += mae

        # less 1 pixel
        v1, a = less_than_n_pixel(label, disparity, 1)
        valid_less_1 += v1
        all_pixel += a

        # less 2 pixel
        v2, _ = less_than_n_pixel(label, disparity, 2)
        valid_less_2 += v2

        # less 3 pixel
        v3, _ = less_than_n_pixel(label, disparity, 3)
        valid_less_3 += v3

        print("Iteration[{}|{}] name: {}/{}, mae(pixel): {}, <1(%):{}, <2(%):{}, <3(%):{}".format(
            idx, len(data_list), dir_name, name, mae/a, v1/a, v2/a, v3/a))
        idx += 1

    print("Average: Mae(pixel): {}, <1(%):{}, <2(%):{}, <3(%):{}".format(
        difference/all_pixel, valid_less_1/ all_pixel, valid_less_2/all_pixel, valid_less_3 / all_pixel))
