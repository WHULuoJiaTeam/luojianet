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
# ===========================================================================
"""generate json desc for fused_adam_weight_decay"""
from ._utils import Expander, ExpanderInfoValidator as VLD


@VLD.check_all_formats_same
class FusedAdamWeightDecay(Expander):
    """FusedAdamWeightDecay expander"""

    def _expand(self, graph_builder):
        beta_1, one_sub_beta_1, beta_2, one_sub_beta_2, eps, lr, param, m, v, gradient, weight_decay = self.inputs

        # compute result
        beta_1_mul_m = graph_builder.emit('Mul', [beta_1, m])
        one_sub_beta_1_mul_grad = graph_builder.emit('Mul', [one_sub_beta_1, gradient])
        next_m = graph_builder.emit('Add', [beta_1_mul_m, one_sub_beta_1_mul_grad])
        beta_2_mul_v = graph_builder.emit('Mul', [beta_2, v])
        grad_square = graph_builder.emit('Mul', [gradient, gradient])
        one_sub_beta_2_mul_grad_square = graph_builder.emit('Mul', [one_sub_beta_2, grad_square])
        next_v = graph_builder.emit('Add', [beta_2_mul_v, one_sub_beta_2_mul_grad_square])
        sqrt_next_v = graph_builder.emit('Sqrt', [next_v])
        sqrt_next_v_add_eps = graph_builder.emit('Add', [sqrt_next_v, eps])
        update = graph_builder.emit('RealDiv', [next_m, sqrt_next_v_add_eps])
        param_with_weight_decay = graph_builder.emit('Mul', [weight_decay, param])
        update = graph_builder.emit('Add', [update, param_with_weight_decay])
        update_with_lr = graph_builder.emit('Mul', [lr, update])
        next_para = graph_builder.emit('Sub', [param, update_with_lr])

        para_result = graph_builder.emit(
            'InplaceAssign', [param, next_para, next_para], attrs={'fake_output': True})
        para_result = graph_builder.emit('InplaceAssign', [m, next_m, para_result], attrs={'fake_output': True})
        para_result = graph_builder.emit('InplaceAssign', [v, next_v, para_result], attrs={'fake_output': True})

        return para_result
