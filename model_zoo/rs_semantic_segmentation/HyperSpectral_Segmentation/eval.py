
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


def eval(config, ckpt_path='', IsFreeNet=True):
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

    index = np.where(y==0)
    y = np.delete(y, index)
    y_pred = np.delete(y_pred, index)
    print(y_pred.max(), y_pred.min())
    print(y.max(), y.min())


    OA = accuracy_score(y, y_pred)
    AA = recall_score(y, y_pred, average="macro")
    KA = cohen_kappa_score(y, y_pred)
    print("OA:{}".format(OA))
    print("AA:{}".format(AA))
    print("KA:{}".format(KA))
    matrix = confusion_matrix(y, y_pred)
    print(matrix.shape)
    a = np.zeros(config['num_class'])
    for i in range(config['num_class']):
        a[i] = matrix[i, i] / np.sum(matrix[i, :])


    #plot confusion matrix
    plt.matshow(matrix, cmap=plt.cm.Reds)

    for i in range(len(matrix)):
        for j in range(len(matrix)):
            plt.annotate(format(matrix[j, i]/sum(matrix[j, :]), '.2f'), xy=(i, j), horizontalalignment='center', verticalalignment='center', fontsize=5)

    plt.tick_params(labelsize=7)

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.ylabel('True label', fontdict={'family': 'Times New Roman', 'size': 10})
    plt.xlabel('Predicted label', fontdict={'family': 'Times New Roman', 'size':10})

    x_locator = MultipleLocator(1)
    y_locator = MultipleLocator(1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_locator)
    ax.yaxis.set_major_locator(y_locator)

    plt.show()
    plt.savefig("./confusion_matrix.png", dpi=300)


    return OA

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hyper-Image segmentation')
    parser.add_argument('-d','--dataset_path', type=str, default=None, help='Dataset path')
    parser.add_argument('-c','--checkpoint_path', type=str, default=None, help='Saved checkpoint file path')
    parser.add_argument('-t','--device_target', type=str, default="GPU", help='Device target')
    args_opt = parser.parse_args()
    context.set_context(mode=context.PYNATIVE_MODE, device_target=args_opt.device_target)

    dataset_path = args_opt.dataset_path
    checkpoint_path = args_opt.checkpoint_path

    config_file_name = os.path.basename(checkpoint_path).replace(".ckpt","_config")
    default_config = importlib.import_module("." + config_file_name, package='configs').config
    default_config['test']['params']['test_gt_dir'] = os.path.join(dataset_path, default_config['test']['params']['test_gt_dir'])
    default_config['test']['params']['test_data_dir'] = os.path.join(dataset_path, default_config['test']['params']['test_data_dir'])

    if 'FreeNet' in config_file_name:
        IsFreeNet = True
    else:
        IsFreeNet = False
    eval(default_config, checkpoint_path, IsFreeNet)
