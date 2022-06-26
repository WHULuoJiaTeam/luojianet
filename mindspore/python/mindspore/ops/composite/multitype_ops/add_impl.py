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

"""Implementation for internal polymorphism `add` operations."""

from . import _compile_utils as utils
from ...composite import base
from ... import functional as F


add = base.MultitypeFuncGraph('add', True)
"""`add` is a metafuncgraph object which will add two objects according to input type using ".register" decorator."""


_add_backward = base.MultitypeFuncGraph('add_backward')
"""
`_add_backward` is an metafuncgraph object which will add_backward two objects according to input type
using ".register" decorator.
"""


class _TupleAdd(base.TupleAdd_):
    """
    Adding two tuples.

    Args:
        x (tuple): x
        y (tuple): y

    Returns:
        Tuple, consists of elements of x and elements of y.
    """

    def __init__(self, name):
        """Initialize _TupleAdd."""
        base.TupleAdd_.__init__(self, name)

    def __call__(self, *args):
        pass


_tuple_add = _TupleAdd('tuple_add')
"""`_tuple_add` is an metafuncgraph object which will concatenate two tuples to form a tuple."""


@add.register("Number", "Number")
@_add_backward.register("Number", "Number")
def _scalar_add_scalar(x, y):
    """
    Returns the sum of two numbers.

    Args:
        x (Number): x
        y (Number): y

    Returns:
        Number, equal to x + y, has the same type as x.
    """
    return F.scalar_add(x, y)


@add.register("String", "String")
def _string_concat_string(x, y):
    """
    Concatenate the string y to the string x.

    Args:
        x (str): The first input string.
        y (str): the second input string.

    Returns:
        str, concatenate the y to the x.
    """
    return F.string_concat(x, y)


@add.register("Number", "Tensor")
def _scalar_add_tensor(x, y):
    """
    Number is added to tensor.

    Args:
        x (Number): x
        y (Tensor): The dtype is same as x.

    Returns:
        Tensor, has the same dtype as x.
    """
    return F.add(x, y)


@add.register("Tensor", "Number")
def _tensor_add_scalar(x, y):
    """
    Tensor is added to number.

    Args:
        x (Tensor): x
        y (Number): The dtype is same as x.

    Returns:
        Tensor, has the same dtype as x.
    """
    return F.add(x, y)


@add.register("Tuple", "Tensor")
def _tuple_add_tensor(x, y):
    """
    Tuple is added to tensor.

    Args:
        x (Tuple): x
        y (Tensor): The dtype is same as x.

    Returns:
        Tensor, has the same dtype as x.
    """
    x = utils.sequence_to_tensor(x, y.dtype)
    return F.tensor_add(x, y)


@add.register("Tensor", "Tuple")
def _tensor_add_tuple(x, y):
    """
    Tensor is added to number.

    Args:
        x (Tensor): x
        y (Tuple): The dtype is same as x.

    Returns:
        Tensor, has the same dtype as x.
    """
    y = utils.sequence_to_tensor(y, x.dtype)
    return F.tensor_add(x, y)


@add.register("List", "Tensor")
def _list_add_tensor(x, y):
    """
    Tuple is added to tensor.

    Args:
        x (List): x
        y (Tensor): The dtype is same as x.

    Returns:
        Tensor, has the same dtype as x.
    """
    x = utils.sequence_to_tensor(x, y.dtype)
    return F.tensor_add(x, y)


@add.register("Tensor", "List")
def _tensor_add_list(x, y):
    """
    Tensor is added to number.

    Args:
        x (Tensor): x
        y (List): The dtype is same as x.

    Returns:
        Tensor, has the same dtype as x.
    """
    y = utils.sequence_to_tensor(y, x.dtype)
    return F.tensor_add(x, y)


@add.register("List", "List")
def _list_add_list(x, y):
    """
        list is added to list.

        Args:
            x (list): x
            y (list): y.

        Returns:
            list, has the same dtype as x.
    """
    for i in y:
        x.append(i)
    return x


@add.register("Tensor", "Tensor")
def _tensor_add_tensor(x, y):
    """
    Returns x + y element-wise.

    Args:
        x (Tensor): x
        y (Tensor): The dtype is same as x.

    Returns:
        Tensor, has the same dtype as x.
    """
    return F.add(x, y)


@add.register("RowTensor", "Tensor")
def add_rowtensor_tensor(x, y):
    """
   Adds RowTensor and Tensor.

   Args:
       x (RowTensor): x
       y (Tensor): y

   Returns:
       RowTensor, the dtype is same as x.
   """
    return F.row_tensor_add(x, y)


@add.register("None", "None")
def _none_add_none(x, y):
    """
   Adds None and None.

   Args:
       x (None): x
       y (None): y

   Returns:
       None.
   """
    return None


@_add_backward.register("EnvType", "EnvType")
def _add_env(x, y):
    """
    Adds two EnvType variables.

    Args:
        x (EnvType): x
        y (EnvType): y

    Returns:
        EnvType, equal to x + y.
    """
    return F.environ_add(x, y)


@add.register("Tuple", "Tuple")
def _add_tuple(x, y):
    """
    Adds two tuples.

    Args:
        x (tuple): x
        y (tuple): y

    Returns:
        Tuple, consists of elements of x and elements of y.
    """
    return _tuple_add(x, y)


@_add_backward.register("Tensor", "Tensor")
def _add_addn(x, y):
    """
   Adds two tensors by element.

   Args:
       x (Tensor): x
       y (Tensor): The dtype is same as x.

   Returns:
       Tensor, the dtype is same as x.
   """
    return F.addn((x, y))


@_add_backward.register("UMonad", "UMonad")
def _add_umonad_umonad(x, y):
    """
   Adds two monad.

   Args:
       x (UMonad): x
       y (UMonad): y

   Returns:
       Monad, the dtype is same as x.
   """
    return x


@_add_backward.register("IOMonad", "IOMonad")
def _add_iomonad_iomonad(x, y):
    """
   Adds two monad.

   Args:
       x (IOMonad): x
       y (IOMonad): y

   Returns:
       Monad, the dtype is same as x.
   """
    return x


@_add_backward.register("RowTensor", "Tensor")
def _add_rowtensor_tensor(x, y):
    """
   Adds RowTensor and Tensor.

   Args:
       x (RowTensor): x
       y (Tensor): y

   Returns:
       RowTensor, the dtype is same as x.
   """
    return x + y


@_add_backward.register("None", "None")
def _add_nonetensor_tensor(x, y):
    """
   Adds None and None.

   Args:
       x (None): x
       y (None): y

   Returns:
       None.
   """
    return x + y


@_add_backward.register("CSRTensor", "CSRTensor")
def _add_csrtensor_csrtensor(x, y):
    """
   Adds CSRTensor and CSRTensor.

   Args:
       x (CSRTensor): x
       y (CSRTensor): y

   Returns:
       CSRTensor.
   """
    return F.make_csr_tensor(x.indptr, x.indices, x.values + y.values, x.shape)


@_add_backward.register("COOTensor", "COOTensor")
def _add_cootensor_cootensor(x, y):
    """
   Adds COOTensor and COOTensor.

   Args:
       x (COOTensor): x
       y (COOTensor): y

   Returns:
       COOTensor.
   """
    return F.make_coo_tensor(x.indices, x.values + y.values, x.shape)

hyper_add = base.HyperMap(_add_backward)
