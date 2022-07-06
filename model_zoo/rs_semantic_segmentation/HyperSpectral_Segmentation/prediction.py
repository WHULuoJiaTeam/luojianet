
from dataset.dataset import WHU_Hi_test
from configs.FreeNet_HC_config import config as FreeNet_HC_config
from configs.FreeNet_HH_config import config as FreeNet_HH_config
from configs.FreeNet_LK_config import config as FreeNet_LK_config
from configs.S3ANet_HC_config import config as S3ANet_HC_config
from configs.S3ANet_HH_config import config as S3ANet_HH_config
from configs.S3ANet_LK_config import config as S3ANet_LK_config
import numpy as np
from luojianet_ms import ops, nn
from sklearn.metrics import accuracy_score, recall_score, cohen_kappa_score, confusion_matrix
from models.FreeNet import FreeNet
from models.S3ANet import S3ANET
import luojianet_ms as ms
from luojianet_ms import context
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from utils.label2rgb import label2rgb_hanchuan, label2rgb_honghu, label2rgb_longkou
import time
import os
import argparse
import importlib


class MyWithEvalCell(nn.Module):

        def __init__(self, network):
            super(MyWithEvalCell, self).__init__(auto_prefix=False)
            self.network = network

        def forward(self, data, label):
            outputs = self.network(data, label)
            return outputs, label


def prediction(config, ckpt_path='', IsFreeNet=True):
    x, y = WHU_Hi_test(config['test']['params']).return_data()
    y = y.astype(ms.dtype.int32) - 1

    if IsFreeNet:
        net = FreeNet(config['model']['params'])
    else:
        net = S3ANET(config['model']['params'], training=False)

    if ckpt_path == '':
        ms.load_param_into_net(net, ms.load_checkpoint(config['save_model_dir']))
    else:
        ms.load_param_into_net(net, ms.load_checkpoint(ckpt_path))

    eval_net = MyWithEvalCell(net)
    eval_net.set_train(False)
    start = time.time()
    y_hat, y = eval_net(x, y)
    end = time.time()
    print("time: " + str(end - start))

    pred_label = ms.ops.Argmax(axis=1)(y_hat)
    pred_label = pred_label.astype(np.float32)
    y_pred = pred_label
    reshape = ms.ops.Reshape()
    y = reshape(y, (-1,))
    y_pred = reshape(y_pred, (-1,))

    y_pred = y_pred.asnumpy() + 1
    y = y.asnumpy() + 1

    if 'HongHu' in config['dataset']['params']['train_gt_dir']:
        outrgb = label2rgb_honghu((pred_label.asnumpy()+1)[0,:,:])
    elif 'LongKou' in config['dataset']['params']['train_gt_dir']:
        outrgb = label2rgb_longkou((pred_label.asnumpy()+1)[0,:,:])
    elif 'HanChuan' in config['dataset']['params']['train_gt_dir']:
        outrgb = label2rgb_hanchuan((pred_label.asnumpy()+1)[0,:,:])

    plt.imshow(outrgb)
    plt.axis('off')
    plt.savefig(config['picture_save_dir'], bbox_inches='tight', dpi=300, )




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hyper-Image segmentation')
    parser = argparse.ArgumentParser(description='Image classification')

    parser.add_argument('-i','--input_file', type=str, default=None, help='Input file path')
    parser.add_argument('-o','--output_folder', type=str, default=None, help='Output file path')
    parser.add_argument('-c1','--checkpoint_path', type=str, default=None, help='Saved checkpoint file path')
    parser.add_argument('-c2','--classes_file', type=str, default=None, help='Classes saved txt path ')
    parser.add_argument('-t','--device_target', type=str, default="GPU", help='Device target')
    args_opt = parser.parse_args()
    context.set_context(mode=context.PYNATIVE_MODE, device_target=args_opt.device_target)

    input_file = args_opt.input_file
    checkpoint_path = args_opt.checkpoint_path
    output_pth = args_opt.output_folder

    config_file_name = os.path.basename(checkpoint_path).replace(".ckpt","_config")
    default_config = importlib.import_module("." + config_file_name, package='configs').config
    default_config['picture_save_dir'] = output_pth

    default_config['test']['params']['test_gt_dir'] = os.path.join(input_file, default_config['test']['params']['test_gt_dir'])
    default_config['test']['params']['test_data_dir'] = os.path.join(input_file, default_config['test']['params']['test_data_dir'])


    if 'FreeNet' in config_file_name:
        IsFreeNet = True
    else:
        IsFreeNet = False
    prediction(default_config, checkpoint_path, IsFreeNet)
