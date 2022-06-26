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
"""
Boost provide auto accelerating for network, such as Less BN, Gradient Freeze, Gradient
accumulation and so on.

Note:
    This feature is a beta feature, and we are still improving its functionality.
"""
from .boost import AutoBoost
from .base import OptimizerProcess, ParameterProcess
from .boost_cell_wrapper import BoostTrainOneStepCell, BoostTrainOneStepWithLossScaleCell
from .less_batch_normalization import LessBN
from .grad_freeze import GradientFreeze, FreezeOpt, freeze_cell
from .grad_accumulation import GradientAccumulation
from .adasum import AdaSum
from .dim_reduce import DimReduce


__all__ = ['AutoBoost',
           'OptimizerProcess', 'ParameterProcess',
           'BoostTrainOneStepCell', 'BoostTrainOneStepWithLossScaleCell',
           'LessBN',
           'GradientFreeze', 'FreezeOpt', 'freeze_cell',
           'GradientAccumulation',
           'AdaSum', 'DimReduce']
