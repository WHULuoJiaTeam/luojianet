# Copyright 2020 Huawei Technologies Co., Ltd
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
Quantization module, including base class of the quantizer, the quantization aware training algorithm,
and quantization utils.

Note: This is an experimental interface that is subject to change and/or deletion.
"""

from .quantizer import OptimizeOption
from .qat import QuantizationAwareTraining, create_quant_config
from .quant_utils import load_nonquant_param_into_quant_net, query_quant_layers

__all__ = ["load_nonquant_param_into_quant_net", "query_quant_layers", "QuantizationAwareTraining",
           "create_quant_config", "OptimizeOption"]
