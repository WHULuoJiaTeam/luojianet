import os
import luojianet_ms.nn as nn
from luojianet_ms import context
from luojianet_ms.train import Model
from luojianet_ms.common import set_seed
from luojianet_ms.dataset import config
from luojianet_ms.train.callback import TimeMonitor, LossMonitor
from luojianet_ms import load_checkpoint, load_param_into_net
from luojianet_ms.train.callback import CheckpointConfig, ModelCheckpoint
from dataset import create_Dataset
from IFN import DSIFN
from config import config 
from loss import loss_fusion

import warnings
warnings.filterwarnings("ignore")
if __name__ == '__main__':
    '''trian'''
    context.set_context(mode=context.GRAPH_MODE,device_target=config.device_target,device_id=config.device_id)

    train_dataset, config.steps_per_epoch = create_Dataset(config.dataset_path, config.aug, config.batch_size, shuffle=True)
    net = DSIFN()
    
    if config.resume:
        ckpt = load_checkpoint('**.ckpt')
        load_param_into_net(net, ckpt)

    optimizer = nn.Adam(params=net.trainable_params(), learning_rate=config.max_lr,weight_decay=config.decay)

    time_cb = TimeMonitor(data_size=config.steps_per_epoch)
    loss_cb = LossMonitor(200)
    loss = loss_fusion()
    model = Model(net, loss_fn=loss, optimizer=optimizer, metrics={'acc'})

    config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_epochs,
                                 keep_checkpoint_max=config.keep_checkpoint_max)
    save_ckpt_path = os.path.join(config.save_checkpoint_path, 'model' + '/')
    ckpoint_cb = ModelCheckpoint(prefix="DSFIN", directory=save_ckpt_path, config=config_ck)
    callbacks = [time_cb, loss_cb, ckpoint_cb]
    
    print("============== Starting Training ==============")
    model.train(config.epoch_size, train_dataset, callbacks=callbacks)
    print("============== Training Finished ==============")