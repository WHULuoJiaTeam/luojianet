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
Parallel Config for the Parallel Training
This is an experimental interface that is subject to change and/or deletion.
"""
from mindspore.nn.transformer.op_parallel_config import OpParallelConfig, default_dpmp_config

__all__ = ["OpParallelConfig", "default_dpmp_config"]
