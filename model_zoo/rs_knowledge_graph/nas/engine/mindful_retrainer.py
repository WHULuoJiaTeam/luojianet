import os
import numpy as np
import luojianet_ms as luojia
import luojianet_ms.nn as nn
from luojianet_ms import Model, load_param_into_net, save_checkpoint
from luojianet_ms.train.loss_scale_manager import FixedLossScaleManager
from luojianet_ms.train.callback import ModelCheckpoint, CheckpointConfig
from utils.config import hrnetw48_config
from utils.callback import TimeLossMonitor, SegEvalCallback
from utils.loss import CrossEntropyWithLogits
from dataloaders import make_retrain_data_loader
from model.RetrainNet1 import RetrainNet
from model.seg_hrnet import get_seg_model
from easydict import EasyDict as ed
config = ed({
    "save_checkpoint_epochs": 5,
    "keep_checkpoint_max": 10,
    "total_epoch": 600,
    "loss_scale": 1024,
})
class Trainer(object):
    def __init__(self, args):
        self.args = args
        # 定义dataloader
        kwargs = {'choice': 'train', 'run_distribute': False, 'raw': False}
        self.train_loader, self.image_size, self.num_classes = make_retrain_data_loader(args, args.batch_size, **kwargs)
        kwargs = {'choice': 'val', 'run_distribute': False, 'raw': False}
        self.val_loader, self.image_size, self.num_classes = make_retrain_data_loader(args, args.batch_size, **kwargs)

        self.trainloader = self.train_loader
        self.valloader = self.val_loader

        self.step_size = self.train_loader.get_dataset_size()
        self.val_step_size = self.val_loader.get_dataset_size()
        self.criterion = nn.SoftmaxCrossEntropyWithLogits(reduction='mean', sparse=True)

        if self.args.model_name == 'flexinet':
            layers = np.ones([14, 4])
            cell_arch = np.load(
                './model/cell_arch.npy')
            connections = np.load(
                './model/connections.npy')
            net = RetrainNet(layers, 4, connections, cell_arch, self.args.dataset, self.num_classes)
        elif self.args.model_name == 'hrnet':
            net = get_seg_model(hrnetw48_config, self.num_classes)

        self.net = net
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = luojia.load_checkpoint(args.resume)
            param_not_load = load_param_into_net(self.net, checkpoint)
            print(param_not_load)
            print("=> loaded checkpoint '{}'".format(args.resume))

        self.loss = CrossEntropyWithLogits(num_classes=self.num_classes, ignore_label=255, image_size=self.image_size)
        loss_scale_manager = FixedLossScaleManager(config.loss_scale, False)

        self.lr = nn.dynamic_lr.cosine_decay_lr(args.min_lr, args.lr, args.epochs * self.step_size,
                                           self.step_size, 2)
        steps_per_epoch = self.trainloader.get_dataset_size()
        begin_step = 0
        self.lr = self.lr[begin_step:]

        self.optimizer = nn.SGD(self.net.trainable_params(), self.lr, momentum=args.momentum, weight_decay=args.weight_decay, loss_scale=config.loss_scale)
        self.model = Model(net, loss_fn=self.loss, optimizer=self.optimizer, loss_scale_manager=loss_scale_manager, amp_level='O0', keep_batchnorm_fp32=False)

        self.local_train_url = args.local_train_url

        # Callbacks
        time_loss_cb = TimeLossMonitor(lr_init=self.lr)
        self.cb = [time_loss_cb]
        # Save-checkpoint callback
        ckpt_config = CheckpointConfig(save_checkpoint_steps=steps_per_epoch * config.save_checkpoint_epochs,
                                       keep_checkpoint_max=config.keep_checkpoint_max)
        ckpt_cb = ModelCheckpoint(prefix="hrnetw48seg",
                                  directory=os.path.join(self.local_train_url, 'experiment_0'),
                                  config=ckpt_config)
        self.cb.append(ckpt_cb)
        # Self-defined callbacks

        eval_cb = SegEvalCallback(self.valloader, net, self.num_classes, start_epoch=0,
                                  save_path=self.local_train_url, interval=1)
        self.cb.append(eval_cb)

    def training(self, epochs):

        self.model.train(epochs, self.trainloader, callbacks=self.cb, dataset_sink_mode=True)
        last_checkpoint = os.path.join(self.local_train_url, "flexinetsegfinal.ckpt")
        save_checkpoint(self.net, last_checkpoint)




