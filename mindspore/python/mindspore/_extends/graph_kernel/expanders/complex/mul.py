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
# ===========================================================================
"""generate json desc for cmul"""
from mindspore._extends.graph_kernel.model.model import DataFormat as DF
from mindspore._extends.graph_kernel.expanders._utils import Expander, ExpanderInfoValidator as VLD


@VLD.add_format(DF.DEFAULT, DF.DEFAULT)
class CMul(Expander):
    """CMul expander"""

    def _expand(self, graph_builder):
        """CMul Implementation"""
        input_x, input_y = self.inputs
        x_real = graph_builder.emit('CReal', [input_x])
        y_real = graph_builder.emit('CReal', [input_y])
        x_imag = graph_builder.emit('CImag', [input_x])
        y_imag = graph_builder.emit('CImag', [input_y])
        x_real_mul_y_real = graph_builder.emit('Mul', [x_real, y_real])
        x_imag_mul_y_imag = graph_builder.emit('Mul', [x_imag, y_imag])
        x_real_mul_y_imag = graph_builder.emit('Mul', [x_real, y_imag])
        x_imag_mul_y_real = graph_builder.emit('Mul', [x_imag, y_real])
        result_real = graph_builder.emit('Sub', [x_real_mul_y_real, x_imag_mul_y_imag])
        result_imag = graph_builder.emit('Add', [x_real_mul_y_imag, x_imag_mul_y_real])
        result = graph_builder.emit('Complex', [result_real, result_imag])
        return result
