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
# ===========================================================================
"""generate json desc for LogSoftmax"""
from mindspore._extends.graph_kernel.model.model import DataFormat as DF
from ._utils import Expander, ExpanderInfoValidator as VLD


@VLD.add_format(DF.DEFAULT)
@VLD.check_attrs('axis')
class LogSoftmax(Expander):
    """LogSoftmax expander"""

    def _expand(self, graph_builder):
        input_x = self.inputs[0]
        axis = self.attrs['axis']
        processor = self.processor

        if isinstance(axis, int):
            axis = (axis,)

        ori_dtype = input_x.dtype
        if ori_dtype != "float16" and processor == "aicore":
            input_x_f16 = graph_builder.emit('Cast', [input_x], attrs={'dst_type': 'float16'})
            max_x_f16 = graph_builder.emit('ReduceMax', [input_x_f16], attrs={'reduce_axis': axis, 'keep_dims': True})
            max_x = graph_builder.emit('Cast', [max_x_f16], attrs={'dst_type': ori_dtype})
        else:
            max_x = graph_builder.emit('ReduceMax', [input_x], attrs={'reduce_axis': axis, 'keep_dims': True})
        data_sub = graph_builder.emit('Sub', [input_x, max_x])
        data_exp = graph_builder.emit('Exp', [data_sub])
        data_expsum = graph_builder.emit('ReduceSum', [data_exp], attrs={'reduce_axis': axis, 'keep_dims': True})
        log_expsum = graph_builder.emit('Log', [data_expsum])
        result = graph_builder.emit('Sub', [data_sub, log_expsum])

        return result
