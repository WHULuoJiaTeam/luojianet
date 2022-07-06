import argparse
import importlib
import os
import shutil
from utils.deeplearning_dp import train_net
from utils.random_seed import setup_seed
from dataset.isprs_dataset import Isprs_Dataset
import dataset.isprs_transform as transform
from model import SegModel
from luojianet_ms import context
context.set_context(mode=context.PYNATIVE_MODE,device_id=0, device_target='GPU')

def get_argparse():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('-c', '--config', type=str, default='train_config', help='Configuration File')
    return parser.parse_args()


if __name__ == "__main__":
    # set random seed
    config_name=get_argparse().config
    param = importlib.import_module("." + get_argparse().config, package='config').param
    setup_seed(param['random_seed'])

    # data path
    data_dir = param['data_dir']#'./data'
    train_img_dir_path = os.path.join(data_dir, "img_dir/train")
    train_label_dir_path = os.path.join(data_dir, "ann_dir/train")
    val_img_dir_path = os.path.join(data_dir, "img_dir/val")
    val_label_dir_path = os.path.join(data_dir, "ann_dir/val")
    train_image_id_txt_path = param['train_image_id_txt_path']
    val_image_id_txt_path = param['val_image_id_txt_path']


    # dataset
    train_transform=getattr(transform,param['train_transform'])()
    val_transform = getattr(transform, param['val_transform'])()
    train_dataset = Isprs_Dataset(img_dir=train_img_dir_path, label_dir=train_label_dir_path,
                                      img_id_txt_path=train_image_id_txt_path, transform=train_transform)
    valid_dataset = Isprs_Dataset(img_dir=val_img_dir_path, label_dir=val_label_dir_path,img_id_txt_path=val_image_id_txt_path,
                                      transform=val_transform)

    # model
    model = SegModel(model_network=param['model_network'],
                     in_channels=param['in_channels'], n_class=param['n_class'])

    # model save path
    save_ckpt_dir = os.path.join('./checkpoint', param['save_dir'], 'ckpt')
    save_log_dir = os.path.join('./checkpoint', param['save_dir'])
    if not os.path.exists(save_ckpt_dir):
        os.makedirs(save_ckpt_dir)
    if not os.path.exists(save_log_dir):
        os.makedirs(save_log_dir)
    param['save_log_dir'] = save_log_dir
    old_config_name_path='./config'+'/'+config_name+'.py'
    new_config_name_path = param['save_log_dir'] + '/' + config_name + '.py'
    shutil.copyfile(src=old_config_name_path,dst=new_config_name_path)
    param['save_ckpt_dir'] = save_ckpt_dir

    # training
    train_net(param=param, model=model, train_dataset=train_dataset, valid_dataset=valid_dataset)
