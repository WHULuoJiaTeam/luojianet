# HRNet-3D网络测试

import os
import argparse
import numpy as np
import luojianet_ms.context as context
import luojianet_ms.dataset as ds
import luojianet_ms.nn as nn
import luojianet_ms.ops as ops
from luojianet_ms import load_checkpoint, load_param_into_net
from dataset import OHS_DatasetGenerator
from model import HigherHRNet_Binary
import time
from config import config

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP','.tif'
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

# 测试流程定义
class MyWithEvalCell(nn.Module):
    def __init__(self, network):
        super(MyWithEvalCell, self).__init__(auto_prefix=False)
        self.network = network
        self.softmax = ops.Softmax(axis=0)
        self.argmax = ops.Argmax(axis=0)

    def forward(self, data):
        output = self.network(data)
        output = output.squeeze()
        output = self.softmax(output)
        output = self.argmax(output)
        return output

# 精度评价指标定义（评价OA、Kappa、PA、UA、F1-score、IoU几个指标）
class MyMetric(nn.Metric):
    def __init__(self, classnum):
        super(MyMetric, self).__init__()
        self.classnum = classnum
        self.clear()

    def clear(self):
        self.confusion_matrix = np.zeros([self.classnum, self.classnum])

    def update(self, output, label):
        output = output.asnumpy().ravel()
        label = label.squeeze()
        label = label.asnumpy().ravel()

        mask = (label >= 0) & (label < self.classnum)

        label = label[mask]
        output = output[mask]

        label = self.classnum * label + output
        label = label.astype(np.uint16)
        count = np.bincount(label, minlength=self.classnum ** 2)
        confusionmat_cur = count.reshape(self.classnum, self.classnum)

        self.confusion_matrix = self.confusion_matrix + confusionmat_cur

    def eval(self):
        unique_index = np.where(np.sum(self.confusion_matrix, axis=1) != 0)[0]
        self.confusion_matrix = self.confusion_matrix[unique_index, :]
        self.confusion_matrix = self.confusion_matrix[:, unique_index]
        OA = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        PE = np.sum(self.confusion_matrix.sum(axis=0) * self.confusion_matrix.sum(axis=1)) / (np.sum(self.confusion_matrix.sum(axis=1)) ** 2)
        Kappa = (OA - PE) / (1 - PE)
        PA = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        UA = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=0)
        eps = 0.000001
        F1 = 2 * PA * UA / (PA + UA + eps)
        mF1 = np.nanmean(F1)
        intersection = np.diag(self.confusion_matrix)
        union = np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) - np.diag(
            self.confusion_matrix)
        IoU = intersection / union
        mIoU = np.nanmean(IoU)
        return dict(OA=OA, Kappa=Kappa, PA=PA, UA=UA, F1=F1, mF1=mF1, IoU=IoU, mIoU=mIoU)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_path', type=str, default=None, help='Dataset path')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Saved checkpoint file path')
    parser.add_argument('--device_target', type=str, default="GPU", help='Device target')

    args_opt = parser.parse_args()

    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target)

    # 读取测试数据并定义数据集
    print('Load data ...')

    data_path = args_opt.dataset_path
    data_path_test = os.path.join(data_path, 'test')

    test_image_list = []
    test_label_list = []

    for root, paths, fnames in sorted(os.walk(data_path_test)):
        for fname in fnames:
            if is_image_file(fname):
                label_path = os.path.join(data_path_test, fname)
                image_path = label_path.replace('test', 'image')
                assert os.path.exists(label_path)
                assert os.path.exists(image_path)
                test_image_list.append(image_path)
                test_label_list.append(label_path)

    assert len(test_image_list) == len(test_label_list)

    dataset_generator = OHS_DatasetGenerator(test_image_list, test_label_list, use_3D_input=True, normalize=config['normalize'])
    test_dataset = ds.GeneratorDataset(dataset_generator, column_names=['image', 'label'], shuffle=False)
    test_dataset = test_dataset.batch(1)

    # 加载网络模型
    print('Build model ...')
    net = HigherHRNet_Binary(num_classes=config['classnum'], hr_cfg='w18_3d2d_at')
    model_path = args_opt.checkpoint_path
    param_dict = load_checkpoint(model_path)
    load_param_into_net(net, param_dict)
    print('Loaded trained model.')

    net.set_train(False)
    net.set_grad(False)

    # 网络测试
    eval_net = MyWithEvalCell(net)
    eval_net.set_train(False)
    Metric = MyMetric(classnum=24)

    num = 0
    time_start = time.time()
    for image, label in test_dataset:
        num = num + 1

        output = eval_net(image)
        Metric.update(output, label)

    metrics = Metric.eval()

    print('OA:', metrics['OA'])
    print('Kapaa:', metrics['Kappa'])
    print('PA:', metrics['PA'])
    print('UA:', metrics['UA'])
    print('F1:', metrics['F1'])
    print('Mean F1:', metrics['mF1'])
    print('IoU', metrics['IoU'])
    print('mIoU', metrics['mIoU'])

    time_end = time.time()
    print('Time:', time_end - time_start)

if __name__ == '__main__':
    main()
