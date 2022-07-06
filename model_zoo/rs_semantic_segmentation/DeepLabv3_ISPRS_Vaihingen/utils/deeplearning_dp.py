import os
import time
import luojianet_ms.dataset as ds
import luojianet_ms
from luojianet_ms import nn,ops
from utils.loss import SoftmaxCrossEntropyLoss
from utils.average_meter import AverageMeter
from utils.metric_tool import SegEvaluator
from utils.logger_tool import init_logger
from luojianet_ms import load_param_into_net,load_checkpoint
def create_dataset(dataset_class,batch_size,shuffle,num_workers):
    dataset = ds.GeneratorDataset(dataset_class, ["data", "label"], shuffle=shuffle,num_parallel_workers=num_workers,python_multiprocessing=False)
    dataset = dataset.batch(batch_size=batch_size,num_parallel_workers=num_workers)
    return dataset

class LossCell(nn.Module):
    def __init__(self, backbone, loss_fn):
        super(LossCell, self).__init__(auto_prefix=False)
        self.backbone = backbone
        self.loss_fn = loss_fn

    def forward(self, data, label):
        out = self.backbone(data)
        return self.loss_fn(out, label)

    def backbone_network(self):
        return self.backbone

class Eval_Net(nn.Module):
    def __init__(self, network):
        super(Eval_Net, self).__init__()
        self.network = network

    def forward(self, input_data):
        output = self.network(input_data)
        return output

class TrainStep(nn.TrainOneStepCell):
    """定义训练流程"""
    def __init__(self, network, optimizer):
        """参数初始化"""
        super(TrainStep, self).__init__(network, optimizer)
        self.grad = ops.GradOperation(get_by_list=True)

    def forward(self, data, label):
        """构建训练过程"""
        weights = self.weights
        loss = self.network(data, label)
        grads = self.grad(self.network, weights)(data, label)
        return loss, self.optimizer(grads)

def train_net(param, model, train_dataset, valid_dataset):
    """

    :param param:
    :param model:
    :param train_dataset:
    :param valid_dataset:
    :return:
    """

    # Initialization parameters
    epochs = param['epochs']
    batch_size = param['train_batch_size']
    test_batch_size = param['test_batch_size']
    lr = param['lr']
    weight_decay = param['weight_decay']

    save_inter = param['save_inter']
    min_inter = param['min_inter']
    iter_inter = param['iter_inter']

    save_log_dir = param['save_log_dir']
    save_ckpt_dir = param['save_ckpt_dir']
    load_ckpt_dir = param['load_ckpt_dir']
    num_workers = param['num_workers']

    train_data_size = train_dataset.__len__()
    valid_data_size = valid_dataset.__len__()

    train_loader = create_dataset(dataset_class=train_dataset, batch_size=batch_size, shuffle=True,num_workers=num_workers)
    valid_loader = create_dataset(dataset_class=valid_dataset, batch_size=test_batch_size, shuffle=False,num_workers=num_workers)
    train_loader_size = train_loader.get_dataset_size()
    # loss function
    criterion=SoftmaxCrossEntropyLoss(num_cls=param['n_class'], ignore_label=-100)
    # optimizer
    optimizer = nn.Adam(params=model.trainable_params(),learning_rate=lr,weight_decay=weight_decay)
    net_with_criterion = LossCell(model, criterion)
    train_net_step = TrainStep(net_with_criterion, optimizer)
    eval_net_step =Eval_Net(model)
    # logging
    logger = init_logger(
        os.path.join(save_log_dir, time.strftime("%m-%d-%H-%M-%S", time.localtime()) + '.log'))

    
    best_metric = 0.0
    epoch_start = 0

    # Subsequent training
    if load_ckpt_dir is not None:
        param_dict=load_checkpoint(load_ckpt_dir)
        load_param_into_net(net=model,parameter_dict=param_dict)
        load_param_into_net(net=optimizer, parameter_dict=param_dict)
        epoch_start = param['ckpt_epoch']
        best_metric = param['ckpt_best_metric']

    logger.info(
        'Total Epoch:{} Training num:{}  Validation num:{}'.format(epochs, train_data_size,valid_data_size))
    # main loop
    softmax_dim1=nn.Softmax(axis=1)
    for epoch in range(epoch_start, epochs):
        epoch_start_time = time.time()
        # training
        train_net_step.set_train(True)
        train_iter_loss = AverageMeter()
        batch_idx=0
        for batch_samples in train_loader.create_dict_iterator():
            data, label = batch_samples['data'], batch_samples['label']
            train_net_step(data, label)
            loss = net_with_criterion(data, label)
            image_loss = loss.asnumpy()
            train_iter_loss.update(image_loss)
            if batch_idx % iter_inter == 0:
                spend_time = time.time() - epoch_start_time
                logger.info('[train] epoch:{} iter:{}/{} {:.2f}% loss:{:.6f} ETA:{}min'.format(
                    epoch, batch_idx, train_loader_size, batch_idx / train_loader_size * 100,
                    train_iter_loss.avg, spend_time / (batch_idx + 1) * train_loader_size // 60 - spend_time // 60))
                train_iter_loss.reset()
            batch_idx=batch_idx+1
        # validation
        evaluator = SegEvaluator(param['n_class'])
        evaluator.reset()
        train_net_step.set_train(False)
        eval_net_step.set_train(False)
        for batch_samples in valid_loader.create_dict_iterator():
            data, label = batch_samples['data'], batch_samples['label']
            pred_label = softmax_dim1(eval_net_step(data)).argmax(1)
            evaluator.add_batch(gt_image=label.asnumpy().astype('int'), pre_image=pred_label.asnumpy().astype('int'))
        oa,kappa,miou = evaluator.pixel_oa(),evaluator.kappa(),evaluator.mean_iou()
        metric = kappa
        logger.info('[val] epoch:{},OA:{:.4f},kappa:{:.4f},mIoU:{:.4f}'.format(epoch, oa,kappa,miou))

        if epoch % save_inter == 0 and epoch > min_inter:
            filename = os.path.join(save_ckpt_dir, 'checkpoint-epoch{}.ckpt'.format(epoch))
            luojianet_ms.save_checkpoint(train_net_step,filename)
        # Save the best model
        if metric > best_metric:
            best_metric = metric
            filename = os.path.join(save_ckpt_dir, 'checkpoint-best.ckpt')
            luojianet_ms.save_checkpoint(train_net_step,filename)
            logger.info('[save] Best Model saved at epoch:{} ============================='.format(epoch))
