# HRNet-3D网络训练

import os
import numpy as np
import luojianet_ms
from luojianet_ms import Tensor
import luojianet_ms.context as context
import luojianet_ms.dataset as ds
import luojianet_ms.nn as nn
import luojianet_ms.ops as ops
from dataset import OHS_DatasetGenerator
from model import HigherHRNet_Binary
from osgeo import gdal
import time
from config import config

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

context.set_context(mode=context.GRAPH_MODE, device_target=config['device_target'])

# 加权交叉熵损失函数定义
class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, weight, ignore_index=None):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.weight = weight
        self.ignore_index = ignore_index

        self.logsoftmax = ops.LogSoftmax(axis=1)
        self.gather = ops.GatherD()
        self.nllloss = ops.NLLLoss(reduction='none')
        self.sum_all = ops.ReduceSum()

    def forward(self, output, label):
        eps = 1e-8

        output = output.transpose(0, 2, 3, 1)
        output = output.reshape(-1, output.shape[3])

        output_logsoftmax = self.logsoftmax(output)

        label = label.reshape(-1)
        label_tmp = label.copy()

        if self.ignore_index is not None:
            mask = (label == self.ignore_index)
            label[mask] = 0

        weight_label = self.gather(self.weight, 0, label.astype(luojianet_ms.int32))
        if self.ignore_index is not None:
            mask = (label_tmp == self.ignore_index)
            weight_label[mask] = 0

        weight_tmp = self.weight / (self.weight + eps)

        output_nllloss, _ = self.nllloss(output_logsoftmax, label.astype(luojianet_ms.int32), weight_tmp)
        output_nllloss_weighted = output_nllloss * weight_label
        loss_all = self.sum_all(output_nllloss_weighted) / self.sum_all(weight_label)

        return loss_all

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP','.tif'
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def main():
    batch_size = config['batch_size']
    num_epochs = config['num_epochs']

    # 读取训练数据并定义数据集
    print('Load data ...')

    data_path = config['dataset_path']
    data_path_train = os.path.join(data_path, 'train')

    train_image_list = []
    train_label_list = []

    for root, paths, fnames in sorted(os.walk(data_path_train)):
        for fname in fnames:
            if is_image_file(fname):
                label_path = os.path.join(data_path_train, fname)
                image_path = label_path.replace('train', 'image')
                assert os.path.exists(label_path)
                assert os.path.exists(image_path)
                train_image_list.append(image_path)
                train_label_list.append(label_path)

    assert len(train_image_list) == len(train_label_list)

    if (config['weight'] is not None):
        weight = np.array(config['weight'])
    else:
        weight = np.ones(config['classnum'])

    weight = Tensor(weight.astype(np.float32))

    print('Weights for each class:', weight)

    dataset_generator = OHS_DatasetGenerator(train_image_list, train_label_list, use_3D_input=True, normalize=config['normalize'])
    train_dataset = ds.GeneratorDataset(dataset_generator, column_names=['image', 'label'], shuffle=True)
    train_dataset = train_dataset.batch(batch_size, drop_remainder=True)

    step_size = train_dataset.get_dataset_size()

    net = HigherHRNet_Binary(num_classes=config['classnum'], hr_cfg='w18_3d2d_at')

    loss = WeightedCrossEntropyLoss(weight=weight, ignore_index=config['nodata_value']-1)

    # 定义学习率衰减
    initial_learning_rate = config['learning_rate']
    learning_rate = []
    for i in range(num_epochs):
        learning_rate_epoch = initial_learning_rate * ((1 - i / num_epochs) ** 0.9)
        for j in range(step_size):
            learning_rate.append(learning_rate_epoch)
    learning_rate = Tensor(np.array(learning_rate).astype(np.float32))

    optimizer = nn.Adam(net.trainable_params(), learning_rate=learning_rate, weight_decay=0.0001)

    model_path = config['save_model_path']
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    # 网络训练
    print('Training.')

    net_with_criterion = nn.WithLossCell(net, loss)
    train_net = nn.TrainOneStepCell(net_with_criterion, optimizer)

    for epoch in range(num_epochs):
        time_start = time.time()

        print('Epoch: {}'.format(epoch + 1))
        batch_idx = 0
        loss_sum = 0
        for image, label in train_dataset:
            train_net(image, label)
            loss_val = net_with_criterion(image, label)

            batch_idx = batch_idx + 1
            loss_sum = loss_sum + loss_val
            loss_avg = loss_sum / batch_idx

            if (batch_idx % 1 == 0):
                print('Epoch [{}/{}], Batch [{}/{}], Loss {}'.format(epoch + 1, num_epochs, batch_idx, step_size,
                                                                     loss_avg))

        luojianet_ms.save_checkpoint(net, model_path + '/HRNet3D_{}.ckpt'.format(epoch))

        time_end = time.time()
        print('Time:', time_end - time_start)
    
    # 训练模型保存
    luojianet_ms.save_checkpoint(net, model_path + '/HRNet3D_final.ckpt')

if __name__ == '__main__':
    main()


