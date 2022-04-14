import luojianet_ms.ops as ops
from luojianet_ms import context
from luojianet_ms.common import set_seed
from luojianet_ms import load_checkpoint, load_param_into_net

from dataset import create_Dataset
from IFN import DSIFN
from config import config 
import argparse


def dice_coef(output, target):
    smooth = 1e-5
    intersection = (output * target).sum()
    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)

class AverageMeter():
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def test(model_path):
    '''test'''
    model = DSIFN()
    model.set_train(False)
    print('load test weights from %s', str(model_path))
    load_param_into_net(model, load_checkpoint(model_path))
    print('loaded test weights from %s', str(model_path))

    val_dataset, _ = create_Dataset(config.val_data_path, aug = False, batch_size=config.batch_size, shuffle=False)
    data_loader = val_dataset.create_dict_iterator()
    dices = AverageMeter()
    for _, data in enumerate(data_loader):
        output = model(data["image"]).asnumpy()
        dice = dice_coef(output, data["mask"].asnumpy())
        dices.update(dice, config.batch_size)
    print("Final dices: %s", str(dices.avg))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Change Detection')
    parser.add_argument('-m','--model_path', type=str, default=None, help='Saved checkpoint file path')

    args_opt = parser.parse_args()
    set_seed(1)
    context.set_context(device_target=config.device_target)
    test(args_opt.model_path)
