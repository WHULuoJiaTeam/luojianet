# Copyright 2021, 2022 LuoJiaNET Research and Development Group, Wuhan University
# Copyright 2021, 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""export checkpoint file into air, onnx, mindir models"""

import numpy as np
from src.luojia_detection.configuration.config import config
from src.luojia_detection.env.device_adapter import get_device_id
from src.luojia_detection.env.moxing_adapter import moxing_wrapper
from src.luojia_detection.detectors import Mask_Rcnn_Resnet, Faster_Rcnn_Resnet
from luojianet_ms import Tensor, context, load_checkpoint, load_param_into_net, export


if not config.enable_modelarts:
    config.ckpt_file = config.ckpt_file_local

context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
if config.device_target == "Ascend":
    context.set_context(device_id=get_device_id())

def modelarts_process():
    pass

@moxing_wrapper(pre_process=modelarts_process)
def export_maskrcnn():
    """ export_maskrcnn """
    config.test_batch_size = config.batch_size
    net = MaskRcnn(config=config)
    param_dict = load_checkpoint(config.ckpt_file)

    param_dict_new = {}
    for key, value in param_dict.items():
        param_dict_new["network." + key] = value

    load_param_into_net(net, param_dict_new)
    net.set_train(False)

    img = Tensor(np.zeros([config.batch_size, 3, config.img_height, config.img_width], np.float16))
    img_metas = Tensor(np.zeros([config.batch_size, 4], np.float16))

    input_data = [img, img_metas]
    export(net, *input_data, file_name=config.file_name, file_format=config.file_format)

@moxing_wrapper(pre_process=modelarts_pre_process)
def export_fasterrcnn():
    """ export_fasterrcnn """
    config.test_batch_size = config.batch_size
    net = FasterRcnn(config=config)
    param_dict = load_checkpoint(config.ckpt_file)

    param_dict_new = {}
    for key, value in param_dict.items():
        param_dict_new["network." + key] = value

    load_param_into_net(net, param_dict_new)

    device_type = "Ascend" if context.get_context("device_target") == "Ascend" else "Others"
    if device_type == "Ascend":
        net.to_float(mstype.float16)

    img = Tensor(np.zeros([config.test_batch_size, 3, config.img_height, config.img_width]), mstype.float32)
    img_metas = Tensor(np.random.uniform(0.0, 1.0, size=[config.test_batch_size, 4]), mstype.float32)

    export(net, img, img_metas, file_name=config.file_name, file_format=config.file_format)

if __name__ == '__main__':
    export_maskrcnn()
