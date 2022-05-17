import os
import shutil
import luojianet_ms as luojia
from collections import OrderedDict
import glob


class Saver(object):

    def __init__(self, args):
        self.args = args
        self.directory = os.path.join('run', args.dataset, args.checkname)
        self.runs = sorted(glob.glob(os.path.join(self.directory, 'experiment_*')))
        run_id = max([int(x.split('_')[-1]) for x in self.runs]) + 1 if self.runs else 0

        self.experiment_dir = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)))
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        """Saves checkpoint to disk"""
        filename = os.path.join(self.experiment_dir, filename)
        luojia.save_checkpoint(state, filename)

    def save_experiment_config(self):
        logfile = os.path.join(self.experiment_dir, 'parameters.txt')
        log_file = open(logfile, 'w')
        p = OrderedDict()
        p['datset'] = self.args.dataset
        p['batchsize'] = self.args.batch_size
        p['lr'] = self.args.lr
        p['lr_scheduler'] = self.args.lr_scheduler
        p['epoch'] = self.args.epochs
        p['resize'] = self.args.resize
        p['crop_size'] = self.args.crop_size

        for key, val in p.items():
            log_file.write(key + ':' + str(val) + '\n')
        log_file.close()

    def save_train_info(self, loss, epoch, acc, miou, fwiou, iou, is_best):
        train_info = 'loss:{}, epoch:{}, acc:{}, miou:{}, fwiou:{}, iou:{}'.format(str(loss), str(epoch+1), str(acc), str(miou), str(fwiou), str(iou))
        if is_best:
            train_info = train_info + 'new best'

        info_file = os.path.join(self.experiment_dir, 'train_info.txt')
        file = open(info_file, 'a')

        file.write(train_info + '\n')
        file.close()

    def save_test_info(self, loss, epoch, acc, miou, fwiou, iou):
        train_info = 'loss: {}, epoch:{}, acc:{}, miou:{}, fwiou:{}, iou:{}'.format(str(loss), str(epoch + 1), str(acc),
                                                                                    str(miou), str(fwiou),
                                                                                    str(iou))
        info_file = os.path.join(self.experiment_dir, 'test_info.txt')
        file = open(info_file, 'a')
        file.write(train_info + '\n')
        file.close()
