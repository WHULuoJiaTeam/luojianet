import os
import numpy as np
import luojianet_ms.ops as ops
import luojianet_ms.nn as nn
from luojianet_ms import Model, Tensor, context, load_checkpoint, load_param_into_net
from densenet import densenet121
from utils import *
import argparse
import json

def txt2class(classes_path):
    # 将txt文件转换为class_name
    class_name = []
    with open(classes_path,'r') as f:
        for line in f.readlines():
            line = line.strip()
            class_name.append(line)
    f.close()
    return class_name

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image classification')

    parser.add_argument('-i','--input_file', type=str, default=None, help='Input file path')
    parser.add_argument('-o','--output_folder', type=str, default=None, help='Output file path')
    parser.add_argument('-c1','--checkpoint_path', type=str, default=None, help='Saved checkpoint file path')
    parser.add_argument('-c2','--classes_file', type=str, default=None, help='Classes saved txt path ')
    parser.add_argument('-t','--device_target', type=str, default="GPU", help='Device target')
    args = parser.parse_args()

    context.set_context(device_target=args.device_target)
    img_name = args.input_file.split('/')[-1].split('.')[0]
    out_dir = os.path.join(args.output_folder, img_name+'.json')
    class_name = txt2class(args.classes_file)
    img = Image.open(args.input_file).convert('RGB').resize((224,224))
    #更改网络模型
    net = densenet121(len(class_name)) # 网路模型
    
    param_dict = load_checkpoint(args.checkpoint_path)
    load_param_into_net(net,param_dict)
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True,reduction='mean')
    model = Model(net, loss, metrics={"Accuracy":nn.Accuracy()})
    img = np.array(img)/255.0
    img = img[np.newaxis,:,:,:].transpose((0,3,1,2))
    softmax = nn.Softmax()
    output = model.predict(Tensor(img,dtype=mstype.float32))
    output = softmax(output)
    topk = ops.TopK()
    value,indexs = topk(output, 5)
    indexs = indexs.squeeze(0).asnumpy()
    value = value.squeeze(0).asnumpy()
    with open(out_dir,'w') as f:
        for i in range(5):
            dicts = {"title": "Top-%d"%(i+1), "class_num": int(indexs[i]),"class_name": class_name[int(indexs[i])],"class_prob": float(value[i])}
            print(dicts)
            json.dump(dicts,f)
            f.write(',\n')
    f.close()
    print("Done!")