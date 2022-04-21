from luojianet_ms import context
from luojianet_ms import Tensor
from luojianet_ms.nn import SGD, RMSProp
from luojianet_ms.context import ParallelMode
from luojianet_ms.train.model import Model
from luojianet_ms.train.callback import ModelCheckpoint, CheckpointConfig,TimeMonitor,LossMonitor
from luojianet_ms.train.serialization import load_checkpoint, load_param_into_net
from luojianet_ms.communication.management import init
from luojianet_ms.train.loss_scale_manager import FixedLossScaleManager
from luojianet_ms.common import dtype as mstype
from luojianet_ms.common import set_seed
import os

from utils import get_lr,create_dataset,CrossEntropySmooth
from config import config

from Resnet_se import *
set_seed(1)

if __name__ == '__main__':
    context.set_context(device_target=config.device_target)

    # define network
    net = se_resnet18(num_classes=config.class_num)
    
    # define loss
    if not config.use_label_smooth:
        config.label_smooth_factor = 0.0
    loss = CrossEntropySmooth(smooth_factor=config.label_smooth_factor, num_classes=config.class_num)

    # define dataset
    
    dataset = create_dataset(dataset_path=config.dataset_path,
                                do_train=True,
                                batch_size=config.batch_size)
    step_size = dataset.get_dataset_size()

    # resume
    if config.resume:
        ckpt = load_checkpoint(config.resume)
        load_param_into_net(net, ckpt)

    # get learning rate
    loss_scale = FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)
    lr = Tensor(get_lr(lr_init=config.lr_init,
                       lr_end=config.lr_end,
                       lr_max=config.lr_max,
                       warmup_epochs=config.warmup_epochs,
                       total_epochs=config.epoch_size,
                       steps_per_epoch=step_size,
                       lr_decay_mode=config.lr_decay_mode))

    # define optimization
    if config.opt == 'sgd':
        optimizer = SGD(net.trainable_params(), learning_rate=lr, momentum=config.momentum,
                        weight_decay=config.weight_decay, loss_scale=config.loss_scale)
    elif config.opt == 'rmsprop':
        optimizer = RMSProp(net.trainable_params(), learning_rate=lr, decay=0.9, weight_decay=config.weight_decay,
                            momentum=config.momentum, epsilon=config.opt_eps, loss_scale=config.loss_scale)

    # define model
    model = Model(net, loss_fn=loss, optimizer=optimizer, loss_scale_manager=loss_scale,
                  metrics={'acc'})

    time_cb = TimeMonitor(data_size=dataset.get_dataset_size())
    config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_epochs,
                                 keep_checkpoint_max=config.keep_checkpoint_max)
    save_ckpt_path = os.path.join(config.save_checkpoint_path, 'model' + '/')
    ckpoint_cb = ModelCheckpoint(prefix="net", directory=save_ckpt_path, config=config_ck)

    # begine train
    print("============== Starting Training ==============")
    model.train(config.epoch_size, dataset, callbacks=[time_cb,ckpoint_cb,LossMonitor()])