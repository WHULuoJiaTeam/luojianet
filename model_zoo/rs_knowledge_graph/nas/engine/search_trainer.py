import os
import numpy as np
import luojianet_ms as luojia
import luojianet_ms.nn as nn

from luojianet_ms import Model, ParameterTuple, ops, load_param_into_net
from tqdm import tqdm
from utils.config import hrnetw48_config
from utils.evaluator import Evaluator
from utils.saver import Saver
from dataloaders import make_search_data_loader
from model.StageNet1 import SearchNet1
from model.seg_hrnet import get_seg_model
from model.StageNet2 import SearchNet2
from model.StageNet3 import SearchNet3
from model.cell import ReLUConvBN, MixedCell


class Trainer(object):
    def __init__(self, args):
        self.args = args

        # 定义保存
        self.saver = Saver(args)
        self.saver.save_experiment_config()

        # 定义dataloader
        kwargs = {'choice': 'train', 'run_distribute': False, 'raw': False}
        self.train_setA, self.train_setB, self.image_size, self.num_classes = make_search_data_loader(args, args.batch_size, **kwargs)

        # 定义dataloader
        kwargs = {'choice': 'val', 'run_distribute': False, 'raw': False}
        self.val_loader, _, _ = make_search_data_loader(args, args.batch_size, **kwargs)

        self.trainloader_A = self.train_setA.create_dict_iterator()

        self.valloader = self.val_loader.create_dict_iterator()

        self.step_size = self.train_setA.get_dataset_size()
        self.val_step_size = self.val_loader.get_dataset_size()
        self.criterion = nn.SoftmaxCrossEntropyWithLogits(reduction='mean', sparse=True)

        self.arch_para = None

        if self.args.search_stage == "first":
            layers = np.ones([14, 4])
            connections = np.load(self.args.model_encode_path)
            net = SearchNet1(layers, 4, connections, ReLUConvBN, self.args.dataset, self.num_classes)
            self.arch_para = 'betas'

        elif self.args.search_stage == "second":
            layers = np.ones([14, 4])
            connections = np.load(self.args.model_encode_path)
            core_path = np.load('/media/dell/DATA/wy/Seg_NAS/model/model_encode/core_path.npy').tolist()
            net = SearchNet2(layers, 4, connections, ReLUConvBN, self.args.dataset, self.num_classes,core_path=core_path)
            self.arch_para = 'betas'

        elif self.args.search_stage == "third":
            layers = np.ones([14, 4])
            connections = np.load(self.args.model_encode_path)
            # connections = 0
            net = SearchNet3(layers, 4, connections, MixedCell, self.args.dataset, self.num_classes)
            self.arch_para = 'alphas'

        else:
            model_config = hrnetw48_config
            net = get_seg_model(model_config, self.num_classes)
        self.net = net
        self.lr = nn.dynamic_lr.cosine_decay_lr(args.min_lr, args.lr, args.epochs * self.step_size,
                                           self.step_size, 2)
        # self.lr = 0.001

        self.evaluator = Evaluator(self.num_classes)

        self.optimizer = nn.SGD(self.net.weight_parameters(), self.lr, momentum=args.momentum, weight_decay=args.weight_decay)

        self.architect_optimizer = nn.Adam(self.net.arch_parameters(), args.arch_lr, weight_decay=args.arch_weight_decay)
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
        self.arch_with_criterion = nn.WithLossCell(self.net, self.criterion)

        self.train_net = MyTrainStep(self.net_with_criterion, self.optimizer)
        self.arch_net = ArchTrainStep(self.arch_with_criterion, self.architect_optimizer, self.arch_para)

        self.val_net = MyWithEvalCell(self.net)

    def training(self, epoch):
        print('Training Epoch: %d' % (epoch))
        train_loss = 0.0
        self.trainloader_B = self.train_setB.create_dict_iterator()
        tbar = tqdm(self.trainloader_A, ncols=80, total=self.step_size)

        self.net.set_train(True)
        for i, d in enumerate(tbar):
            self.train_net(d["image"], d["label"])
            loss = self.net_with_criterion(d["image"], d["label"])
            train_loss += float(loss.asnumpy())

            self.args.alpha_epochs = 0
            if epoch >= self.args.alpha_epochs:
                search = next(self.trainloader_B)
                self.arch_net(search["image"], search["label"])
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))

        if self.args.search_stage == "third":
            alphas = self.net.alphas.asnumpy()
            alphas_dir = '/media/dell/DATA/wy/Luojia-NAS/' + self.saver.experiment_dir + '/alphas'
            if not os.path.exists(alphas_dir):
                os.makedirs(alphas_dir)
            alphas_path = alphas_dir + '/alphas_{}.npy'.format(epoch)
            np.save(alphas_path, alphas, allow_pickle=True)
        else:
            betas = self.net.betas.asnumpy()
            betas_dir = '/media/dell/DATA/wy/Luojia-NAS/' + self.saver.experiment_dir + '/betas'
            if not os.path.exists(betas_dir):
                os.makedirs(betas_dir)
            betas_path = betas_dir + '/betas_{}.npy'.format(epoch)
            np.save(betas_path, betas, allow_pickle=True)

        if epoch > 55:
            self.validation(epoch)

    def validation(self, epoch):
        self.net.set_train(False)
        print('Val Epoch: %d' % (epoch))
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

class ArchTrainStep(nn.Module):
    def __init__(self, network, optimizer, arch_para):
        """参数初始化"""
        super(ArchTrainStep, self).__init__(auto_prefix=False)
        self.network = network
        # 使用tuple包装weight
        self.optimizer = optimizer
        self.weights = ParameterTuple(list(filter(lambda x: arch_para in x.name, self.get_parameters())))
        # print(self.weights)
        # 定义梯度函数
        self.grad = ops.GradOperation(get_by_list=True)

    def call(self, data, label):
        """构建训练过程"""
        weights = self.weights
        loss = self.network(data, label)
        grads = self.grad(self.network, weights)(data, label)
        # for grad in grads:
        #     print(grad)
        return loss, self.optimizer(grads)

class MyWithEvalCell(nn.Module):
    """定义验证流程"""

    def __init__(self, network):
        super(MyWithEvalCell, self).__init__(auto_prefix=False)
        self.network = network

    def call(self, data, label):
        outputs = self.network(data)
        return outputs, label