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
"""Precision."""
import sys

import numpy as np

from mindspore._checkparam import Validator as validator
from .metric import EvaluationBase, rearrange_inputs, _check_onehot_data


class Precision(EvaluationBase):
    r"""
    Calculates precision for classification and multilabel data.

    The precision function creates two local variables, :math:`\text{true_positive}` and
    :math:`\text{false_positive}`, that are used to compute the precision. The calculation formula is:

    .. math::
        \text{precision} = \frac{\text{true_positive}}{\text{true_positive} + \text{false_positive}}

    Note:
        In the multi-label cases, the elements of :math:`y` and :math:`y_{pred}` must be 0 or 1.

    Args:
        eval_type (str): 'classification' or 'multilabel' are supported. Default: 'classification'.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import nn, Tensor
        >>>
        >>> x = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]))
        >>> y = Tensor(np.array([1, 0, 1]))
        >>> metric = nn.Precision('classification')
        >>> metric.clear()
        >>> metric.update(x, y)
        >>> precision = metric.eval()
        >>> print(precision)
        [0.5 1. ]

    """
    def __init__(self, eval_type='classification'):
        super(Precision, self).__init__(eval_type)
        self.eps = sys.float_info.min
        self.clear()

    def clear(self):
        """Clears the internal evaluation result."""
        self._class_num = 0
        if self._type == "multilabel":
            self._true_positives = np.empty(0)
            self._positives = np.empty(0)
            self._true_positives_average = 0
            self._positives_average = 0
        else:
            self._true_positives = 0
            self._positives = 0

    @rearrange_inputs
    def update(self, *inputs):
        """
        Updates the internal evaluation result with `y_pred` and `y`.

        Args:
            inputs: Input `y_pred` and `y`. `y_pred` and `y` are Tensor, list or numpy.ndarray.
                For 'classification' evaluation type, `y_pred` is in most cases (not strictly) a list
                of floating numbers in range :math:`[0, 1]`
                and the shape is :math:`(N, C)`, where :math:`N` is the number of cases and :math:`C`
                is the number of categories. Shape of `y` can be :math:`(N, C)` with values 0 and 1 if one-hot
                encoding is used or the shape is :math:`(N,)` with integer values if index of category is used.
                For 'multilabel' evaluation type, `y_pred` and `y` can only be one-hot encoding with
                values 0 or 1. Indices with 1 indicate positive category. The shape of `y_pred` and `y`
                are both :math:`(N, C)`.

        Raises:
            ValueError: If the number of inputs is not 2.
        """
        if len(inputs) != 2:
            raise ValueError("For 'Precision.update', it needs 2 inputs (predicted value, true value), "
                             "but got {}.".format(len(inputs)))
        y_pred = self._convert_data(inputs[0])
        y = self._convert_data(inputs[1])
        if self._type == 'classification' and y_pred.ndim == y.ndim and _check_onehot_data(y):
            y = y.argmax(axis=1)
        self._check_shape(y_pred, y)
        self._check_value(y_pred, y)

        if self._class_num == 0:
            self._class_num = y_pred.shape[1]
        elif y_pred.shape[1] != self._class_num:
            raise ValueError("For 'Precision.update', class number not match, last input predicted data contain {} "
                             "classes, but current predicted data contain {} classes, please check your predicted "
                             "value(inputs[0])".format(self._class_num, y_pred.shape[1]))

        class_num = self._class_num
        if self._type == "classification":
            if y.max() + 1 > class_num:
                raise ValueError("For 'Precision.update', predicted value (input[0]) should have the same classes "
                                 "number as true value (input[1]), but got predicted value classes {}, true value "
                                 "classes {}.".format(class_num, y.max() + 1))
            y = np.eye(class_num)[y.reshape(-1)]
            indices = y_pred.argmax(axis=1).reshape(-1)
            y_pred = np.eye(class_num)[indices]
        elif self._type == "multilabel":
            y_pred = y_pred.swapaxes(1, 0).reshape(class_num, -1)
            y = y.swapaxes(1, 0).reshape(class_num, -1)

        positives = y_pred.sum(axis=0)
        true_positives = (y * y_pred).sum(axis=0)

        if self._type == "multilabel":
            self._true_positives_average += np.sum(true_positives / (positives + self.eps))
            self._positives_average += len(positives)
            self._true_positives = np.concatenate((self._true_positives, true_positives), axis=0)
            self._positives = np.concatenate((self._positives, positives), axis=0)

        else:
            self._true_positives += true_positives
            self._positives += positives

    def eval(self, average=False):
        """
        Computes the precision.

        Args:
            average (bool): Specify whether calculate the average precision. Default: False.

        Returns:
            numpy.float64, the computed result.
        """
        if self._class_num == 0:
            raise RuntimeError("The 'Precision' can not be calculated, because the number of samples is 0, "
                               "please check whether your inputs (predicted value, true value) are empty, or "
                               "has called update method before calling eval method.")

        validator.check_value_type("average", average, [bool], self.__class__.__name__)
        result = self._true_positives / (self._positives + self.eps)

        if average:
            if self._type == "multilabel":
                result = self._true_positives_average / (self._positives_average + self.eps)
            return result.mean()
        return result
