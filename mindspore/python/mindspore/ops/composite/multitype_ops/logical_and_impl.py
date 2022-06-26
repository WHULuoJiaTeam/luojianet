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

"""Implementation for internal polymorphism `logical and` operations."""
from mindspore.ops.composite import base
from mindspore.ops import functional as F

# logical_and is a metagraph object which will generate function according to input type
# using ".register" decorator
logical_and = base.MultitypeFuncGraph("logical_and", True)


@logical_and.register("Number", "Number")
def _logical_and_scala(x, y):
    """
    Return logical and operation result of x and y.

    Args:
       x(Number): Number.
       y(Number): Number.

    Returns:
       bool, Return logical and operation result of x and y.
   """
    return F.bool_and(x.__bool__(), y.__bool__())


@logical_and.register("Tensor", "Tensor")
def _logical_and_tensor(x, y):
    """
    Return logical and operation result of x and y.

    Args:
       x(Tensor): Tensor.
       y(Tensor): Tensor.

    Returns:
       Tensor, Return logical and operation result of x and y.
   """
    return F.logical_and(x, y)
