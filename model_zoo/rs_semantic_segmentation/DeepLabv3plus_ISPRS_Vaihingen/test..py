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
import os
from luojianet_ms import load_param_into_net,load_checkpoint
from luojianet_ms import nn,Tensor
import argparse
from luojianet_ms import context, nn
class Eval_Net(nn.Module):
    def __init__(self, network):
        super(Eval_Net, self).__init__()
        self.network = network
    def call(self, input_data):
        output = self.network(input_data)
        return output
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image classification')

    parser.add_argument('-i','--input_file', type=str, default=None, help='Input file path')
    parser.add_argument('-o','--output_folder', type=str, default=None, help='Output file path')
    parser.add_argument('-c1','--checkpoint_path', type=str, default=None, help='Saved checkpoint file path')
    parser.add_argument('-c2','--classes_file', type=str, default=None, help='Classes saved txt path ')
    parser.add_argument('-t','--device_target', type=str, default="GPU", help='Device target')
    args = parser.parse_args()
    infer_dir=args.output_folder
    context.set_context(device_target=args.device_target)
    if not os.path.exists(infer_dir):
        os.makedirs(infer_dir)
    param={}
    param['in_channels'] = 3
    param['model_network'] = "DeepLabV3Plus"
    param['n_class'] = 6
    load_ckpt_dir = args.checkpoint_path
    model = SegModel(model_network=param['model_network'],
                     in_channels=param['in_channels'], n_class=param['n_class'])
    param_dict = load_checkpoint(load_ckpt_dir)
    load_param_into_net(net=model, parameter_dict=param_dict)
    eval_net_step = Eval_Net(model)
    eval_net_step.set_train(False)
    filepath = args.input_file
    infer_transforms = val_transform()
    data = infer_transforms(image=io.imread(filepath))['image'].transpose((2, 0, 1))[None, :]
    pred = eval_net_step(Tensor(data))
    pred = pred.argmax(1).asnumpy()[0].astype('uint8')
    file_name = os.path.split(filepath)[-1]
    file_name_save= file_name.replace('.png','.tif')
    output_path = os.path.join(infer_dir, file_name_save)
    io.imsave(output_path, pred, check_contrast=False)

