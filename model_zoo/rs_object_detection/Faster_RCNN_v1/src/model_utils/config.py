# Copyright 2021 Huawei Technologies Co., Ltd
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

"""Parse arguments"""

import os
import ast
import argparse
from pprint import pprint, pformat
import yaml


class Config:
    """
    Configuration namespace. Convert dictionary to members.
    """
    def __init__(self, cfg_dict):
        for k, v in cfg_dict.items():
            if isinstance(v, (list, tuple)):
                setattr(self, k, [Config(x) if isinstance(x, dict) else x for x in v])
            else:
                setattr(self, k, Config(v) if isinstance(v, dict) else v)

    def __str__(self):
        return pformat(self.__dict__)

    def __repr__(self):
        return self.__str__()


def parse_cli_to_yaml(parser, cfg, helper=None, choices=None, cfg_path="default_config.yaml"):
    """
    Parse command line arguments to the configuration according to the default yaml.

    Args:
        parser: Parent parser.
        cfg: Base configuration.
        helper: Helper description.
        cfg_path: Path to the default yaml config.
    """
    parser = argparse.ArgumentParser(description="[REPLACE THIS at config.py]",
                                     parents=[parser])
    helper = {} if helper is None else helper
    choices = {} if choices is None else choices
    for item in cfg:
        if not isinstance(cfg[item], list) and not isinstance(cfg[item], dict):
            help_description = helper[item] if item in helper else "Please reference to {}".format(cfg_path)
            choice = choices[item] if item in choices else None
            if isinstance(cfg[item], bool):
                parser.add_argument("--" + item, type=ast.literal_eval, default=cfg[item], choices=choice,
                                    help=help_description)
            else:
                parser.add_argument("--" + item, type=type(cfg[item]), default=cfg[item], choices=choice,
                                    help=help_description)
    args = parser.parse_args()
    return args


def parse_yaml(yaml_path):
    """
    Parse the yaml config file.

    Args:
        yaml_path: Path to the yaml config.
    """
    with open(yaml_path, 'r') as fin:
        try:
            cfgs = yaml.load_all(fin.read(), Loader=yaml.FullLoader)
            cfgs = [x for x in cfgs]
            if len(cfgs) == 1:
                cfg_helper = {}
                cfg = cfgs[0]
                cfg_choices = {}
            elif len(cfgs) == 2:
                cfg, cfg_helper = cfgs
                cfg_choices = {}
            elif len(cfgs) == 3:
                cfg, cfg_helper, cfg_choices = cfgs
            else:
                raise ValueError("At most 3 docs (config, description for help, choices) are supported in config yaml")
            print(cfg_helper)
        except:
            raise ValueError("Failed to parse yaml")
    return cfg, cfg_helper, cfg_choices


def merge(args, cfg):
    """
    Merge the base config from yaml file and command line arguments.

    Args:
        args: Command line arguments.
        cfg: Base configuration.
    """
    args_var = vars(args)
    for item in args_var:
        cfg[item] = args_var[item]
    return cfg


def get_config():
    """
    Get Config according to the yaml file and cli arguments.
    """
    parser = argparse.ArgumentParser(description="default name", add_help=False)
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # train.py
    parser.add_argument("--config_path", type=str,
                        default=os.path.join(current_dir, "../../config/example_config.yaml"), help="Config file path.")
    parser.add_argument("--device", choices=['CPU', 'GPU', 'Ascend'], default='GPU',
                        help="Devices should be CPU/GPU/Ascend.")
    parser.add_argument("--GPU_id", type=str, default=None,
                        help="GPU ids.")
    # eval.py
    parser.add_argument("--eval_dataset", type=str, default=None, help="Dataset root path.")  # r"./mini_dataset"
    parser.add_argument("--annotation", type=str, default=r"./mini_dataset/train.json", help="Annotation file path.")
    parser.add_argument("--result_save_path", type=str, default=r"./eval_results/debug", help="Result save path.")
    parser.add_argument("--checkpoint_path", type=str,
                        default=r"/dat02/xwj/LuoJiaNET_applications/faster_rcnn/output_dir/resnet152_tianzhi_1024_all_retrain_pynative/ckpt_0/faster_rcnn-50_1126.ckpt",
                        help="Checkpoint file path.")
    # inference.py
    # eval.py
    parser.add_argument("--infer_img_dir", type=str, default=None, help="Inference images dir path.")  # r"/dat02/hhb/datasets/tianzhi/val/"
    parser.add_argument("--infer_save_dir", type=str, default=r"./inference_results/", help="Inference results save path.")
    parser.add_argument("--infer_checkpoint_path", type=str,
                        default=r"/dat02/xwj/LuoJiaNET_applications/faster_rcnn/output_dir/resnet152_tianzhi_1024_all_retrain_pynative/ckpt_0/faster_rcnn-50_1126.ckpt",
                        help="Inference checkpoint path.")


    path_args, _ = parser.parse_known_args()

    default, helper, choices = parse_yaml(path_args.config_path)
    args = parse_cli_to_yaml(parser=parser, cfg=default, helper=helper, choices=choices, cfg_path=path_args.config_path)
    default = Config(merge(args, default))
    default.feature_shapes = [
        [default.img_height // 4, default.img_width // 4],
        [default.img_height // 8, default.img_width // 8],
        [default.img_height // 16, default.img_width // 16],
        [default.img_height // 32, default.img_width // 32],
        [default.img_height // 64, default.img_width // 64],
    ]
    default.num_bboxes = default.num_anchors * sum([lst[0] * lst[1] for lst in default.feature_shapes])

    # rewrite config
    if args.device is not None:
        default.device_target = args.device
    if args.GPU_id is not None:
        default.device_id = args.GPU_id

    # eval mode
    if args.eval_dataset is not None:
        default.coco_root = args.eval_dataset
        default.anno_path = args.annotation
        default.eval_anno_path = args.annotation
        default.eval_save_dir = args.result_save_path
        default.eval_checkpoint_path = args.checkpoint_path
    # inference mode
    if args.infer_img_dir is not None:
        default.inference_save_dir = args.infer_save_dir
        default.inference_img_dir = args.infer_img_dir
        default.inference_checkpoint_path = args.infer_checkpoint_path

    pprint(default)
    print("Please check the above information for the configurations", flush=True)

    return default


config = get_config()

print(config)
