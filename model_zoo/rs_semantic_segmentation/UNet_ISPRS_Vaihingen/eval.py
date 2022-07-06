import albumentations as A
def val_transform():
    train_transform = A.Compose([
        A.ToFloat(max_value=1.0),
        A.Normalize(mean=(0.491, 0.482, 0.447),
                    std=(0.247, 0.243, 0.262), max_pixel_value=255.0),
    ])
    return train_transform
from skimage import io
from model import SegModel
import glob
import os
from luojianet_ms import load_param_into_net,load_checkpoint
from luojianet_ms import nn,Tensor
import argparse
from luojianet_ms import context, nn
from utils.metric_tool import SegEvaluator
class Eval_Net(nn.Module):
    def __init__(self, network):
        super(Eval_Net, self).__init__()
        self.network = network
    def forward(self, input_data):
        output = self.network(input_data)
        return output
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image segmentation')
    parser.add_argument('-d','--dataset_path', type=str, default=None, help='Dataset path')
    parser.add_argument('-c','--checkpoint_path', type=str, default=None, help='Saved checkpoint file path')
    parser.add_argument('-t','--device_target', type=str, default="CPU", help='Device target')
    args_opt = parser.parse_args()
    context.set_context(device_target=args_opt.device_target)
    param={}
    param['in_channels'] = 3
    param['model_network'] = "UNet"
    param['n_class'] = 6
    load_ckpt_dir = args_opt.checkpoint_path
    model = SegModel(model_network=param['model_network'],
                     in_channels=param['in_channels'], n_class=param['n_class'])
    param_dict = load_checkpoint(load_ckpt_dir)
    load_param_into_net(net=model, parameter_dict=param_dict)
    eval_net_step = Eval_Net(model)
    eval_net_step.set_train(False)
    label_image_dir = args_opt.dataset_path
    label_image_dir_path =os.path.join(label_image_dir,'img_dir/val/')
    label_mask_dir_path = os.path.join(label_image_dir, 'ann_dir/val/')
    label_image_filelist = glob.glob(os.path.join(label_image_dir_path, '*.png'))
    evaluator = SegEvaluator(class_num=param['n_class'])
    infer_transforms = val_transform()
    for image_path in label_image_filelist:
        data=infer_transforms(image=io.imread(image_path))['image'].transpose((2, 0, 1))[None,:]
        pred = eval_net_step(Tensor(data))
        pred=pred.argmax(1).asnumpy()[0].astype('uint8')
        file_name = os.path.split(image_path)[-1]
        label=io.imread(os.path.join(label_mask_dir_path,file_name))
        evaluator.add_batch(gt_image=label,pre_image=pred)
    #precision,recall,F1,IoU,mIoU,Kappa
    class_num=param['n_class']
    for class_index in range(class_num):
        p,r,f1=evaluator.one_class_classreport(class_index)
        class_iou=evaluator.class_iou(class_index)
        print('class_index:{},precision:{:.4f},recall:{:.4f},F1:{:.4f},IoU:{:.4f}'.format(class_index,p,r,f1,class_iou))
    MIoU = evaluator.mean_iou()
    Kappa = evaluator.kappa()
    print('MIoU:{:.4f},Kappa:{:.4f}'.format(MIoU,Kappa))
