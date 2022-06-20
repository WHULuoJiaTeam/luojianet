# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
"""FasterRcnn Init."""

from .resnet import ResNetFea, ResidualBlockUsing
from .resnet50v1 import ResidualBlockUsing_V1
from .bbox_assign_sample import BboxAssignSample
from .bbox_assign_sample_stage2 import BboxAssignSampleForRcnn
from .fpn_neck import FeatPyramidNeck
from .proposal_generator import Proposal
from .rcnn import Rcnn
from .rpn import RPN
from .roi_align import SingleRoIExtractor
from .anchor_generator import AnchorGenerator
from .inceptionresnetv2 import InceptionResNetV2

__all__ = [
    "ResNetFea", "BboxAssignSample", "BboxAssignSampleForRcnn",
    "FeatPyramidNeck", "Proposal", "Rcnn",
    "RPN", "SingleRoIExtractor", "AnchorGenerator", "ResidualBlockUsing", "ResidualBlockUsing_V1",
    "InceptionResNetV2"
]
