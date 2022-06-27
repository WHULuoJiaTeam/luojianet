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

"""Component that forward function that init params with random function and return gradients wrt params."""

from luojianet_ms.ops.composite import GradOperation
from ...components.icomponent import IBuilderComponent
from ...utils.block_util import run_block, gen_grad_net, create_funcs, get_uniform_with_shape


class RunBackwardBlockWrtParamsWithRandParamBC(IBuilderComponent):
    def __call__(self):
        grad_op = GradOperation(get_by_list=True, sens_param=True)
        return create_funcs(self.verification_set, gen_grad_net, run_block, grad_op, get_uniform_with_shape)
