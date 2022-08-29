import os
from tqdm import tqdm
from dataloder_gid_aug import Dataset_RealUD
from hrnetv2 import hrnetv2
# from model1 import DeepLabV3Plus_RFB
import luojianet_ms
from luojianet_ms import dataset
import luojianet_ms.nn as ljnn
from losses import MCEloss, WeightedCrossEntropyLoss
import luojianet_ms.common.dtype as mstype
from metrics import SegmentationMetric
import luojianet_ms.ops.operations as P
from luojianet_ms.common import initializer

from geobject import get_objects
import os


def run_Gid(ClassN):
    ##################### define parameters #############################################################
    epochs = 120
    initial_learning_rate = 0.0001
    eta_min = 1e-5
    weight_decay = 1e-5
    batch_size_train = 2
    batch_size_test = 2
    mioubest = 55

    image_dir = '/media/xx/PortableSSD/GID5/image/'  # original image
    encode_label_dir = '/media/xx/PortableSSD/GID5/label/'  # encoded label
    color_label_dir = '/media/xx/PortableSSD/GID5/label_5classes/'  # color label

    def get_file_list(split='train'):

        id_list = os.path.join('datalist', split + '.txt')
        id_list = tuple(open(id_list, 'r'))

        image_files = [os.path.join(image_dir, id_.rstrip() + '.tif') for id_ in id_list]
        label_files = [os.path.join(encode_label_dir, id_.rstrip() + '_label.tif') for id_ in id_list]

        return image_files, label_files

    # Function get_objects: get quadtree search patch info in json file.
    # param[in] image_path, big_input image path.
    # param[in] label_path, big_input label path.
    # param[in] n_classes, number of dataset's land-cover categories.
    # param[in] ignore_label, ignore label value, default is 255.
    # param[in] seg_threshold, quadtree segmentation settings.
    # param[in] search_block_size, basic global processing unit for big_input data. This parameter is default, do not need to change.
    # param[in] max_searchsize, max output data size (max_searchsize*max_searchsize).
    # param[in] json_filename, output patch info. This parameter is default, do not need to change.
    # param[in] use_quadsearch, whether to use quadtree search.
    # return out, json file with patch info.
    train_image_path, train_label_path = get_file_list('train')
    get_objects(train_image_path, train_label_path, 6, 255, 150, 4096, 512, '/patch_info_train.json', 1)

    valid_image_path, valid_label_path = get_file_list('valid')
    get_objects(valid_image_path, valid_label_path, 6, 255, 150, 4096, 512, '/patch_info_valid.json', 0)

    # dataset_train = r'/media/xx/PortableSSD/QuadSearch/patch_info.json'
    # dataset_test = r'/media/xx/PortableSSD/QuadSearch/patch_info_test.json'
    ##################### load data ###################################################################
    train_dataset_generator = Dataset_RealUD(encode_label_dir=encode_label_dir, color_label_dir=color_label_dir, json_path='./patch_info_train.json', train=True)
    train_loader = dataset.GeneratorDataset(train_dataset_generator, ["data", "label"], shuffle=True)
    train_loader = train_loader.batch(batch_size_train)

    test_dataset_generator = Dataset_RealUD(encode_label_dir=encode_label_dir, color_label_dir=color_label_dir, json_path='./patch_info_valid.json')
    test_loader = dataset.GeneratorDataset(test_dataset_generator, ["data", "label"], shuffle=False)
    test_loader = test_loader.batch(batch_size_test)

    ###########################################   define model    #########################################
    model = hrnetv2(output_class=ClassN)
    # for _, cell in model.cells_and_names():
    #     if isinstance(cell, ljnn.Conv2d):
    #         cell.weight.set_data(initializer.initializer(initializer.XavierUniform(), cell.weight.shape, cell.weight.dtype))
    # luojianet_ms.load_param_into_net(model, luojianet_ms.load_checkpoint(
    #     '/media/vhr/0D7A09740D7A0974/zz/Luojianet_ckpt/hrnet/hrnet_newest.ckpt'))
    step_size = train_loader.get_dataset_size()
    criterion = WeightedCrossEntropyLoss(ignore_label=ClassN)
    scheduler = ljnn.cosine_decay_lr(min_lr=eta_min, max_lr=initial_learning_rate, total_step=epochs*step_size, step_per_epoch=step_size, decay_epoch=120)
    optimizer = ljnn.AdamWeightDecay(model.trainable_params(), learning_rate=scheduler, weight_decay=weight_decay)
    # optimizer = ljnn.AdamWeightDecay(model.trainable_params(), learning_rate=1e-5, weight_decay=weight_decay)

    net_with_criterion = luojianet_ms.nn.WithLossCell(model, criterion)
    train_net = luojianet_ms.nn.TrainOneStepCell(net_with_criterion, optimizer)

    metric = SegmentationMetric(ClassN)
    for epoch in range(epochs):
        model.set_train(True)
        model.set_grad(True)
        loss_val_sum = 0
        printcount = 0
        for train_x, train_y in tqdm(train_loader):
            printcount += 1
            train_x = train_x[:, 0, :, :, :]
            train_x = luojianet_ms.ops.Cast()(train_x, mstype.float32)
            train_y = luojianet_ms.ops.Cast()(train_y, mstype.int32)
            train_net(train_x, train_y)
            loss_val = net_with_criterion(train_x, train_y)
            loss_val_sum = loss_val_sum + loss_val
            if printcount % 100 == 0:
                print(loss_val_sum / printcount)

        # ///////////////

        metric.reset()
        model.set_train(False)
        model.set_grad(False)
        for input, target in tqdm(test_loader):
            input = input[:, 0, :, :, :]
            input = luojianet_ms.ops.Cast()(input, mstype.float32)
            target = luojianet_ms.ops.Cast()(target, mstype.int32)
            output = model(input)

            output_v = P.ArgMaxWithValue(axis=1, keep_dims=False)(output)[0].asnumpy()
            target = target.asnumpy()
            # target[target == 6] = 0
            metric.addBatch(output_v, target)

        validate_result = metric.IntersectionOverUnion() * 100
        print([round(i, 3) for i in validate_result])
        MIOU_target = sum(validate_result[1:])/(len(validate_result)-1)
        MIOU = sum(validate_result)/len(validate_result)
        print('Epoch:', epoch, 'MIOU-target:', MIOU_target, 'MIOU:', MIOU)

        luojianet_ms.save_checkpoint(model, '/media/xx/PortableSSD/QuadSearch/ckpt/hrnet_'+str(epoch)+'.ckpt')
        #if MIOU_target > mioubest:
        #     mioubest = MIOU_target
        #luojianet_ms.save_checkpoint(model, '/data02/zz/QuadSearch/ckpt/hrnet_best.ckpt')




if __name__ == '__main__':
    import luojianet_ms.context as context

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    run_Gid(6)

















