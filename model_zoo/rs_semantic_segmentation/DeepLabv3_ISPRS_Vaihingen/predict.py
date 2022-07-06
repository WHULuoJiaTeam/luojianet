import albumentations as A
import math
import numpy as np
from skimage import io
from model import SegModel
import os
from luojianet_ms import load_param_into_net,load_checkpoint
from luojianet_ms import nn,Tensor,ops
import argparse
from luojianet_ms import context, nn
class Eval_Net(nn.Module):
    def __init__(self, network):
        super(Eval_Net, self).__init__()
        self.network = network
    def forward(self, input_data):
        output = self.network(input_data)
        return output
def val_transform():
    train_transform = A.Compose([
        A.ToFloat(max_value=1.0),
        A.Normalize(mean=(0.491, 0.482, 0.447),
                    std=(0.247, 0.243, 0.262), max_pixel_value=255.0),
    ])
    return train_transform
def overlap_predict_image(model, image, grid, stride):
    overlap = grid - stride
    invalid_num = int(overlap / 2)

    n, b, r, c = image.shape
    rows = -((grid - r) // (stride + 1e-10)) * stride + grid
    cols = -((grid - c) // (stride + 1e-10)) * stride + grid
    rows = math.ceil(rows)
    cols = math.ceil(cols)
    pad_image = np.pad(image, ((0, 0), (0, 0), (0, rows - r), (0, cols - c)), 'symmetric')
    output = np.zeros((rows, cols), dtype=np.uint8)
    for i in range(0, rows, stride):
        print('Current row:', i)
        for j in range(0, cols, stride):
            patch = pad_image[0:, 0:, i:i + grid, j:j + grid]
            patch = Tensor(patch)
            pred = model(patch)
            pred=pred.argmax(1).asnumpy()[0].astype('uint8')

            output[i + invalid_num:i + grid - invalid_num, j + invalid_num:j + grid - invalid_num] = \
                pred[invalid_num:grid - invalid_num, invalid_num:grid - invalid_num]

    output = output[0:r, 0:c]

    return output
def label2rgb(label):
    """"""
    label2color_dict = {
        0: [255, 255, 255],
        1: [255, 0, 0],
        2: [255, 255, 0],
        3: [0, 255, 0],
        4: [0, 255, 255],
        5: [0, 0, 255],
    }
    visual_anno = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    for class_index in label2color_dict:
        visual_anno[label == class_index, :]=label2color_dict[class_index]
    return visual_anno
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
    param['model_network'] = "DeepLabV3"
    param['n_class'] = 6
    grid=512
    stride=512
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
    pred=overlap_predict_image(eval_net_step, data, grid, stride)
    file_name = os.path.split(filepath)[-1]
    file_name_save= file_name.replace('.png','.tif')
    file_name_save_rgb = 'rgb_'+file_name_save
    output_path = os.path.join(infer_dir, file_name_save)
    output_path_rgb = os.path.join(infer_dir, file_name_save_rgb)
    io.imsave(output_path, pred, check_contrast=False)
    io.imsave(output_path_rgb, label2rgb(pred), check_contrast=False)

