
from src.dataset import DatasetGenerator
import luojianet_ms.dataset as ds
from src.GCNet import GCNet
import os
import luojianet_ms.context as context
from luojianet_ms import load_checkpoint, load_param_into_net
from luojianet_ms import Tensor
import time
from luojianet_ms import Model
from src.data_io import save_pfm
import matplotlib.pyplot as plt
import argparse


# gpu setting
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

# Set graph mode and target device
context.set_context(mode=context.GRAPH_MODE, device_target='GPU')

parser = argparse.ArgumentParser(description='LuoJiaNET GCNet Implement')
parser.add_argument("--predict_list", type=str, default="list/whu_validation.txt", help="the list for training")
parser.add_argument("--crop_h", type=int, default=384, help="crop height")
parser.add_argument("--crop_w", type=int, default=768, help="crop width")
parser.add_argument("--max_disp", type=int, default=160, help="max disparity")
parser.add_argument("--dataset_type", type=str, default="whu", help="dataset")
parser.add_argument('--model_path', type=str, default="checkpoint/checkpoint_gcnet_whu-20_8316.ckpt")
parser.add_argument('--save_path', type=str, default="/mnt/gj/stereo/WHU_epipolar/luojia_result")
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


if __name__ == "__main__":

    print("GCNet predicting...")
    print(opt)

    if not os.path.exists(opt.save_path):
        os.mkdir(opt.save_path)

    # model
    net = GCNet(max_disp=opt.max_disp)
    param_dict = load_checkpoint(opt.model_path)
    load_param_into_net(net, param_dict)

    model = Model(net)

    dataset = create_dataset(opt.predict_list, 1, opt.crop_w, opt.crop_h, opt.dataset_type)

    # shuffle=False， please note！！
    data_list = read_list(opt.predict_list)
    idx = 0
    for data in dataset.create_dict_iterator():
        start = time.time()

        input_data = Tensor(data['data'])
        output = net(input_data)

        end = time.time()

        disparity = output.asnumpy()[0]

        dir_name = data_list[idx][0].split("/")[-3]
        name = os.path.splitext(os.path.basename(data_list[idx][0]))[0]

        if not os.path.exists("{}/{}".format(opt.save_path, dir_name)):
            os.mkdir("{}/{}".format(opt.save_path, dir_name))

        save_pfm("{}/{}/{}.pfm".format(opt.save_path, dir_name, name), disparity)
        plt.imsave("{}/{}/{}.jpg".format(opt.save_path, dir_name, name), disparity)

        print("Iteration[{}|{}] name: {}/{}, cost time:{}s".format(idx, len(data_list), dir_name, name, end - start))
        idx += 1
