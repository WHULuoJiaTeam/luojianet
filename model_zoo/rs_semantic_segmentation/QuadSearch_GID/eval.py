import os
from tqdm import tqdm
from dataloder_gid_aug import Dataset_RealUD
from hrnetv2 import hrnetv2
import luojianet_ms
from luojianet_ms import dataset
import luojianet_ms.common.dtype as mstype
from metrics import SegmentationMetric
import luojianet_ms.ops.operations as P

def run_Gid(ClassN, checkpoint_path):
    ##################### define parameters #############################################################
    batch_size_test = 2
    encode_label_dir = '/media/xx/PortableSSD/GID5/label/'  # encoded label
    color_label_dir = '/media/xx/PortableSSD/GID5/label_5classes/'  # color label

    ##################### load data ##################################################################

    test_dataset_generator = Dataset_RealUD(encode_label_dir=encode_label_dir, color_label_dir=color_label_dir, json_path='./patch_info_valid.json')
    test_loader = dataset.GeneratorDataset(test_dataset_generator, ["data", "label"], shuffle=False)
    test_loader = test_loader.batch(batch_size_test)

    ###########################################   define model    #########################################
    model = hrnetv2(output_class=ClassN)
    luojianet_ms.load_param_into_net(model, luojianet_ms.load_checkpoint(checkpoint_path))
    metric = SegmentationMetric(ClassN)
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
    print('class_IOU:',[round(i, 3) for i in validate_result])
    # MIOU_target = sum(validate_result[1:])/(len(validate_result)-1)
    MIOU = sum(validate_result)/len(validate_result)
    print('MIOU:', MIOU)
    print('class_precision:', metric.classPixelAccuracy())
    print('class_recall:', metric.classPixelAccuracy2())
    print('OA:', metric.pixelAccuracy())
    print('KAPPA:', metric.kappa())





if __name__ == '__main__':
    import argparse
    import luojianet_ms.context as context

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', default='./ckpt/hrnet_21.ckpt')
    parser.add_argument('--device_target', default='GPU')
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    run_Gid(6, args.checkpoint_path)

















