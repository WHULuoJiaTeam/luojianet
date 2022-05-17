import os
import numpy as np
import luojianet_ms as luojia
import luojianet_ms.nn as nn

from luojianet_ms import Model, ParameterTuple, ops, load_param_into_net
from tqdm import tqdm
from utils.config import hrnetw48_config
from utils.evaluator import Evaluator
from utils.saver import Saver
from dataloaders import make_retrain_data_loader
# from model.RetrainNet import RetrainNet
from model.RetrainNet1 import RetrainNet
from model.seg_hrnet import get_seg_model

class Trainer(object):
    def __init__(self, args):
        self.args = args

        # 定义保存
        self.saver = Saver(args)
        self.saver.save_experiment_config()

        # 定义dataloader
        kwargs = {'choice': 'train', 'run_distribute': False, 'raw': False}
        self.train_loader, self.image_size, self.num_classes = make_retrain_data_loader(args, args.batch_size, **kwargs)
        kwargs = {'choice': 'val', 'run_distribute': False, 'raw': False}
        self.val_loader, self.image_size, self.num_classes = make_retrain_data_loader(args, args.batch_size, **kwargs)

        self.trainloader = self.train_loader.create_dict_iterator()
        self.valloader = self.val_loader.create_dict_iterator()

        self.step_size = self.train_loader.get_dataset_size()
        self.val_step_size = self.val_loader.get_dataset_size()
        self.criterion = nn.SoftmaxCrossEntropyWithLogits(reduction='mean', sparse=True)

        if self.args.model_name == 'flexinet':
            layers = np.ones([14, 4])
            cell_arch = np.load(
                '/media/dell/DATA/wy/Seg_NAS/run/uadataset/search/experiment_0/cell_arch/2_cell_arch_epoch_nors24.npy')
            connections = np.load(
                '/media/dell/DATA/wy/Seg_NAS/run/uadataset/search/experiment_0/connections/2_connections_epoch37.npy')
            net = RetrainNet(layers, 4, connections, cell_arch, self.args.dataset, self.num_classes)
        elif self.args.model_name == 'hrnet':
            net = get_seg_model(hrnetw48_config, self.num_classes)

        self.net = net
        self.lr = nn.dynamic_lr.cosine_decay_lr(args.min_lr, args.lr, args.epochs * self.step_size,
                                           self.step_size, 2)
        # self.lr = 0.001

        self.evaluator = Evaluator(self.num_classes)

        self.optimizer = nn.SGD(self.net.trainable_params(), self.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        # 加载模型
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = luojia.load_checkpoint(args.resume)
            param_not_load = load_param_into_net(self.net, checkpoint)
            print(param_not_load)
            print("=> loaded checkpoint '{}'".format(args.resume))
        else:
            self.start_epoch = 0

        self.net_with_criterion = nn.WithLossCell(self.net, self.criterion)

        self.train_net = MyTrainStep(self.net_with_criterion, self.optimizer)

        self.val_net = MyWithEvalCell(self.net)

    def training(self, epoch):
        train_loss = 0.0
        tbar = tqdm(self.trainloader, ncols=80, total=self.step_size)
        self.net.set_train(True)
        for i, d in enumerate(tbar):
            self.train_net(d["image"], d["label"])
            loss = self.net_with_criterion(d["image"], d["label"])
            train_loss += float(loss.asnumpy())
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))

        if epoch > 10:
            self.validation(epoch)
        self.saver.save_checkpoint(self.net, False, 'current.ckpt')

    def validation(self, epoch):
        self.net.set_train(False)
        test_loss = 0.0
        tbar = tqdm(self.valloader, ncols=80, desc='Val', total=self.val_step_size)
        for i, d in enumerate(tbar):
            output, label = self.val_net(d["image"], d["label"])
            loss = self.net_with_criterion(d["image"], d["label"])
            output = ops.Transpose()(output, (0, 3, 1, 2))

            pred = output.asnumpy()
            target = label.asnumpy()

            pred = np.argmax(pred, axis=1)
            self.evaluator.add_batch(target, pred)

            test_loss += float(loss.asnumpy())
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))

        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU, IoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()

        print('Validation:')
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}, IoU:{}".format(Acc, Acc_class, mIoU, FWIoU, IoU))
        print('Loss: %.3f' % test_loss)
        new_pred = mIoU
        is_best = False

        if new_pred > self.best_pred:
            is_best = True
            self.saver.save_checkpoint(self.net, is_best, 'epoch{}_checkpoint.ckpt'.format(str(epoch)))

        self.saver.save_train_info(test_loss, epoch, Acc, mIoU, FWIoU, IoU, is_best)

    def test(self, epoch):
        self.net.set_train(False)
        test_loss = 0.0
        tbar = tqdm(self.valloader, ncols=80, desc='Val', total=self.val_step_size)
        for i, d in enumerate(tbar):
            output, label = self.val_net(d["image"], d["label"])
            loss = self.net_with_criterion(d["image"], d["label"])
            output = ops.Transpose()(output, (0, 3, 1, 2))

            pred = output.asnumpy()
            target = label.asnumpy()

            pred = np.argmax(pred, axis=1)
            self.evaluator.add_batch(target, pred)

            test_loss += float(loss.asnumpy())
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))

        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU, IoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()

        print('Test:')
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}, IoU:{}".format(Acc, Acc_class, mIoU, FWIoU, IoU))
        print('Loss: %.3f' % test_loss)
        new_pred = mIoU
        is_best = False

        if new_pred > self.best_pred:
            is_best = True
            self.saver.save_checkpoint(self.net, is_best, 'epoch{}_checkpoint.ckpt'.format(str(1)))

        self.saver.save_train_info(test_loss, epoch, Acc, mIoU, FWIoU, IoU, is_best)


class MyTrainStep(nn.TrainOneStepCell):
    """定义训练流程"""

    def __init__(self, network, optimizer):
        """参数初始化"""
        super(MyTrainStep, self).__init__(network, optimizer)
        self.grad = ops.GradOperation(get_by_list=True)

    def call(self, data, label):
        """构建训练过程"""
        weights = self.weights
        loss = self.network(data, label)
        grads = self.grad(self.network, weights)(data, label)
        return loss, self.optimizer(grads)

class MyWithEvalCell(nn.Module):
    """定义验证流程"""

    def __init__(self, network):
        super(MyWithEvalCell, self).__init__(auto_prefix=False)
        self.network = network

    def call(self, data, label):
        outputs = self.network(data)
        return outputs, label