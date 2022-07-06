from dataset.dataset import WHU_Hi_dataloader
from configs.FreeNet_HH_config import config as FreeNet_HH_config
from configs.FreeNet_LK_config import config as FreeNet_LK_config
from configs.FreeNet_HC_config import config as FreeNet_HC_config
import os
import luojianet_ms.dataset as ds
from luojianet_ms import ops, nn
from loss.CrossEntropy import SoftmaxCrossEntropyLoss
import luojianet_ms as ms
import eval
from luojianet_ms import context
from models.FreeNet import FreeNet
from models.S3ANet import S3ANET
from extra_modules import LossCell, TrainStep
import time
import argparse
import importlib


def get_argparse():
    parser = argparse.ArgumentParser(description='Hyper-Image training')
    parser.add_argument('-c', '--config', type=str, default='train_config', help='Configuration File')
    parser.add_argument('-t','--device_target', type=str, default="GPU", help='Device target')
    return parser.parse_args()


if __name__ == "__main__":
    config_name=get_argparse().config
    config = importlib.import_module("." + get_argparse().config, package='configs').config
    
    context.set_context(mode=context.PYNATIVE_MODE, device_target=get_argparse().device_target)

    if 'S3ANet' in get_argparse().config:
        net = S3ANET(config['model']['params'], training=True)
    else:
        net = FreeNet(config['model']['params'])

    dataset_path = r"/home/luojianet"
    config['dataset']['params']['train_gt_dir'] = os.path.join(dataset_path, config['dataset']['params']['train_gt_dir'])
    config['dataset']['params']['train_data_dir'] = os.path.join(dataset_path, config['dataset']['params']['train_data_dir'])
    config['test']['params']['test_gt_dir'] = os.path.join(dataset_path, config['test']['params']['test_gt_dir'])
    config['test']['params']['test_data_dir'] = os.path.join(dataset_path, config['test']['params']['test_data_dir'])
    
    # ms.load_param_into_net(net, ms.load_checkpoint(config['save_model_dir']))

    loss = SoftmaxCrossEntropyLoss(num_cls=config['model']['params']['num_classes'], ignore_label=-1)

    optimizer = nn.Adam(params=net.trainable_params(), learning_rate=0.001, )

    train_data = WHU_Hi_dataloader(config['dataset']['params'])
    train_data = ds.GeneratorDataset(train_data, ["data", "label"], num_parallel_workers=8, python_multiprocessing=False)

    net_with_criterion = LossCell(net, loss)
    train_net = TrainStep(net_with_criterion, optimizer)

    for epoch in range(300):
        train_loss = 0.0
        num = 0
        for data in train_data.create_dict_iterator():
            x = data['data']
            y = data['label']
            y = y.astype(ms.dtype.int32)
            start = time.time()
            train_net(x, y)
            end = time.time()
            loss = net_with_criterion(x, y)
            train_loss = train_loss + loss
            num = num + 1

        epoch = epoch + 1
        print('第%d次loss: %.4f' % (epoch, train_loss.asnumpy() / num))

    ms.save_checkpoint(net, config['save_model_dir'])