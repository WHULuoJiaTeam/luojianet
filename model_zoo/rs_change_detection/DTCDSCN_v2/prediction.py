import luojianet_ms.ops as ops
from luojianet_ms import context
from luojianet_ms.common import set_seed
from luojianet_ms import load_checkpoint, load_param_into_net
import PIL
from PIL import Image
import numpy as np
from dataset import create_Dataset
from module import CDNet34
from config import config 
import argparse
import os
import tqdm
import luojianet_ms.dataset.vision.c_transforms as c_vision
import luojianet_ms.dataset.vision.py_transforms as py_vision

toTensor=py_vision.ToTensor()

def pred_dataset(model_path,data_path,result_path):
    '''pred_dataset'''
    model = CDNet34(3)
    model.set_train(False)
    print('load test weights from %s', str(model_path))
    load_param_into_net(model, load_checkpoint(model_path))
    print('loaded test weights from %s', str(model_path))
    img1_path=os.path.join(data_path,"A")
    img2_path=os.path.join(data_path,"B")
    left_imgs = os.listdir(img1_path) 
    right_imgs = os.listdir(img2_path)   
    test_dataset, _ = create_Dataset(data_path, aug = False, batch_size=1, shuffle=False)
    data_loader = test_dataset.create_dict_iterator()
    for i, data in tqdm.tqdm(enumerate(data_loader),ncols=100):
        left_imgarr=left_imgs[i].split('.')
        left_img=left_imgarr[0]
        right_imgarr=right_imgs[i].split('.')
        right_img=right_imgarr[0]
        output = model(data["image"])[2].asnumpy()
        output = np.array(output.squeeze())
        print(output.mean())
        output = ((output > 0.484)*255).astype('uint8')
        output = Image.fromarray(output)
        output.save(os.path.join(result_path, 'CD_'+left_img+'_'+right_img+'.tif'))
        print('saved'+' CD_'+left_img+'_'+right_img+'.tif')

def pred_single(model_path,left_path,right_path,result_path):
    '''pred_single'''
    leftarr=left_path.split('/')
    rightarr=right_path.split('/')
    img_left_arr=leftarr[len(leftarr)-1].split('.')
    img_right_arr=rightarr[len(rightarr)-1].split('.')
    img_left=img_left_arr[0]
    img_right=img_right_arr[1]
    model = CDNet34(3)
    model.set_train(False)
    print('load test weights from %s', str(model_path))
    load_param_into_net(model, load_checkpoint(model_path))
    print('loaded test weights from %s', str(model_path))
    image1=PIL.open(left_path)
    image2=PIL.open(right_path)       
    image1 = toTensor(image1)
    image2 = toTensor(image2)
    images = [image1,image2]  
    output = model(images)[2].asnumpy()
    output = np.array(output.squeeze())
    output = ((output > 0.4985)*255).astype('uint8')
    output = Image.fromarray(output)
    output.save(os.path.join(result_path, 'CD_'+img_left+'_'+img_right+'.tif'))
    print('saved'+' CD_'+img_left+'_'+img_right+'.tif')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Change Detection')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Saved checkpoint file path')
    parser.add_argument('--dataset_path', type=str, default=None, help='Predict dataset path')
    parser.add_argument('--left_input_file', type=str, default=None, help='Pre-period image')
    parser.add_argument('--right_input_file', type=str, default=None, help='Post-period image')
    parser.add_argument('--output_folder', type=str, default="./result", help='Results path')
    parser.add_argument('--device_target', type=str, default=config.device_target, help='Device target')
    parser.add_argument('--device_id', type=int, default=config.device_id, help='Device id')

    args_opt = parser.parse_args()
    set_seed(1)
    context.set_context(device_target=config.device_target,device_id=args_opt.device_id)
    if(args_opt.left_input_file):
        pred_single(args_opt.checkpoint_path,args_opt.left_input_file,args_opt.right_input_file,args_opt.output_folder)
    elif(args_opt.dataset_path):
        pred_dataset(args_opt.checkpoint_path,args_opt.dataset_path,args_opt.output_folder)
    else:
        print("Error:There are no images to predict")

