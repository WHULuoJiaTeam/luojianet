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
"""Invert Bijector"""
from luojianet_ms._checkparam import Validator as validator
from .bijector import Bijector


class Invert(Bijector):
    r"""
    Invert Bijector. Compute the inverse function of the input bijector.

    Args:
        bijector (Bijector): Base Bijector.
        name (str): The name of the Bijector. Default: "". When name is set to "", it is actually
            'Invert' + bijector.name.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> import luojianet_ms
        >>> import luojianet_ms.nn as nn
        >>> import luojianet_ms.nn.probability.bijector as msb
        >>> from luojianet_ms import Tensor
        >>> class Net(nn.Module):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self.origin = msb.ScalarAffine(scale=2.0, shift=1.0)
        ...         self.invert = msb.Invert(self.origin)
        ...
        ...     def forward(self, x_):
        ...         return self.invert.forward(x_)
        >>> forward = Net()
        >>> x = np.array([2.0, 3.0, 4.0, 5.0]).astype(np.float32)
        >>> ans = forward(Tensor(x, dtype=luojianet_ms.float32))
        >>> print(ans.shape)
        (4,)
    """

    def __init__(self,
                 bijector,
                 name=""):
        param = dict(locals())
        validator.check_value_type('bijector', bijector, [Bijector], "Invert")
        name = name or ('Invert' + bijector.name)
        param["name"] = name
        super(Invert, self).__init__(is_constant_jacobian=bijector.is_constant_jacobian,
                                     is_injective=bijector.is_injective,
                                     name=name,
                                     dtype=bijector.dtype,
                                     param=param)
        self._bijector = bijector
        self._batch_shape = self.bijector.batch_shape
        self._is_scalar_batch = self.bijector.is_scalar_batch

    @property
    def bijector(self):
        """Return base bijector."""
        return self._bijector

    def inverse(self, y):
        """
        Perform the inverse transformation of the inverse bijector,
        namely the forward transformation of the underlying bijector.

        Args:
            y (Tensor): the value of the transformed random variable.

        Output:
            Tensor, the value of the input random variable.
        """
        return self.bijector("forward", y)

    def forward(self, x):
        """
        Perform the forward transformation of the inverse bijector,
        namely the inverse transformation of the underlying bijector.

        Args:
            x (Tensor): the value of the input random variable.

        Output:
            Tensor, the value of the transformed random variable.
        """
        return self.bijector("inverse", x)

    def inverse_log_jacobian(self, y):
        """
        Logarithm of the derivative of the inverse transformation of the inverse bijector,
        namely logarithm of the derivative of the forward transformation of the underlying bijector.

        Args:
            y (Tensor): the value of the transformed random variable.

        Output:
            Tensor, logarithm of the derivative of the inverse transformation of the inverse bijector.
        """
        return self.bijector("forward_log_jacobian", y)

    def forward_log_jacobian(self, x):
        """
        Logarithm of the derivative of the forward transformation of the inverse bijector,
        namely logarithm of the derivative of the inverse transformation of the underlying bijector.

        Args:
            x (Tensor): the value of the input random variable.

        Output:
            Tensor, logarithm of the derivative of the forward transformation of the inverse bijector.
        """
        return self.bijector("inverse_log_jacobian", x)
