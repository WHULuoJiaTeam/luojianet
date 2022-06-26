# Copyright 2020-2022 Huawei Technologies Co., Ltd
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

"""Other operators."""
import functools
from mindspore import log as logger
from .. import signature as sig
from ..._checkparam import Validator as validator, Rel
from ...common import dtype as mstype
from ..primitive import Primitive, PrimitiveWithCheck, PrimitiveWithInfer, prim_attr_register
from ._pyfunc_registry import add_pyfunc


class Assign(Primitive):
    """
    Assigns `Parameter` with a value.

    Inputs of `variable` and `value` comply with the implicit type conversion rules to make the data types consistent.
    If they have different data types, the lower priority data type will be converted to
    the relatively highest priority data type.

    Inputs:
        - **variable** (Parameter) - The `Parameter`. :math:`(N,*)` where :math:`*` means,
          any number of additional dimensions, its rank should be less than 8.
        - **value** (Tensor) - The value to be assigned, has the same shape with `variable`.

    Outputs:
        Tensor, has the same data type and shape as original `variable`.

    Raises:
        TypeError: If `variable` is not a Parameter.
        TypeError: If `value` is not a Tensor.
        RuntimeError: If the data type of `variable` and `value` conversion of Parameter
                      is required when data type conversion of Parameter is not supported.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> value = Tensor([2.0], mindspore.float32)
        >>> variable = mindspore.Parameter(Tensor([1.0], mindspore.float32), name="variable")
        >>> assign = ops.Assign()
        >>> output = assign(variable, value)
        >>> print(output)
        [2.]
    """
    __mindspore_signature__ = (
        sig.make_sig('variable', sig.sig_rw.RW_WRITE, dtype=sig.sig_dtype.T),
        sig.make_sig('value', dtype=sig.sig_dtype.T)
    )

    @prim_attr_register
    def __init__(self):
        """Initialize Assign."""
        self.init_prim_io_names(inputs=['ref', 'value'], outputs=['output'])
        self.add_prim_attr('side_effect_mem', True)


class Load(PrimitiveWithCheck):
    """
    Load `Parameter` to a value.

    Inputs:
        - **variable** (Parameter) - The `Parameter`.

    Outputs:
        Tensor - The loaded parameter tensor value.
    """
    __mindspore_signature__ = (
        sig.make_sig('variable', sig.sig_rw.RW_READ, dtype=sig.sig_dtype.T),
        sig.make_sig('u', dtype=sig.sig_dtype.T1)
    )

    @prim_attr_register
    def __init__(self):
        """Initialize Load."""
        self.init_prim_io_names(inputs=['ref', 'u'], outputs=['output'])

    def check_dtype(self, variable):
        if variable != mstype.type_refkey:
            validator.check_tensors_dtypes_same_and_valid({"variable": variable}, mstype.number_type, self.name)


class BoundingBoxEncode(PrimitiveWithInfer):
    """
    Encodes bounding boxes locations.

    This operator will calculate the offset between the predicted bounding boxes and the real bounding boxes,
    and this offset will be used as a variable for the loss.

    Args:
        means (tuple): Means for encoding bounding boxes calculation. Default: (0.0, 0.0, 0.0, 0.0).
        stds (tuple): The standard deviations of deltas calculation. Default: (1.0, 1.0, 1.0, 1.0).

    Inputs:
        - **anchor_box** (Tensor) - Anchor boxes. The shape of anchor_box must be (n, 4).
        - **groundtruth_box** (Tensor) - Ground truth boxes. Which has the same shape with anchor_box.

    Outputs:
        Tensor, encoded bounding boxes. It has the same data type and shape as input `anchor_box`.

    Raises:
        TypeError: If `means` or `stds` is not a tuple.
        TypeError: If `anchor_box` or `groundtruth_box` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> anchor_box = Tensor([[2, 2, 2, 3], [2, 2, 2, 3]], mindspore.float32)
        >>> groundtruth_box = Tensor([[1, 2, 1, 4], [1, 2, 1, 4]], mindspore.float32)
        >>> boundingbox_encode = ops.BoundingBoxEncode(means=(0.0, 0.0, 0.0, 0.0), stds=(1.0, 1.0, 1.0, 1.0))
        >>> output = boundingbox_encode(anchor_box, groundtruth_box)
        >>> print(output)
        [[ -1.  0.25  0.  0.40551758]
         [ -1.  0.25  0.  0.40551758]]
    """

    @prim_attr_register
    def __init__(self, means=(0.0, 0.0, 0.0, 0.0), stds=(1.0, 1.0, 1.0, 1.0)):
        """Initialize BoundingBoxEncode."""
        validator.check_value_type('means', means, tuple, self.name)
        validator.check_value_type('stds', stds, tuple, self.name)
        for i, value in enumerate(means):
            validator.check_value_type("means[%d]" % i, value, [float], self.name)
        for i, value in enumerate(stds):
            validator.check_value_type("stds[%d]" % i, value, [float], self.name)
        validator.check_equal_int(len(means), 4, "means len", self.name)
        validator.check_equal_int(len(stds), 4, "stds len", self.name)

    def infer_shape(self, anchor_box, groundtruth_box):
        validator.check('anchor_box shape[0]', anchor_box[0], 'groundtruth_box shape[0]', groundtruth_box[0], Rel.EQ,
                        self.name)
        validator.check("anchor_box rank", len(anchor_box), "", 2, Rel.EQ, self.name)
        validator.check("groundtruth_box rank", len(groundtruth_box), "", 2, Rel.EQ, self.name)
        validator.check_equal_int(anchor_box[1], 4, 'anchor_box shape[1]', self.name)
        validator.check_equal_int(groundtruth_box[1], 4, 'groundtruth_box shape[1]', self.name)
        return anchor_box

    def infer_dtype(self, anchor_box, groundtruth_box):
        args = {"anchor_box": anchor_box, "groundtruth_box": groundtruth_box}
        validator.check_tensors_dtypes_same_and_valid(args, mstype.number_type, self.name)
        return anchor_box


class BoundingBoxDecode(Primitive):
    """
    Decodes bounding boxes locations.

    The function of the operator is to calculate the offset, and this operator converts the offset into a Bbox,
    which is used to mark the target in the subsequent images, etc.

    Args:
        means (tuple): The means of deltas calculation. Default: (0.0, 0.0, 0.0, 0.0).
        stds (tuple): The standard deviations of deltas calculation. Default: (1.0, 1.0, 1.0, 1.0).
        max_shape (tuple): The max size limit for decoding box calculation.
        wh_ratio_clip (float): The limit of width and height ratio for decoding box calculation. Default: 0.016.

    Inputs:
        - **anchor_box** (Tensor) - Anchor boxes. The shape of `anchor_box` must be (n, 4).
        - **deltas** (Tensor) - Delta of boxes. Which has the same shape with `anchor_box`.

    Outputs:
        Tensor, decoded boxes. It has the same data type and shape as `anchor_box`.

    Raises:
        TypeError: If `means`, `stds` or `max_shape` is not a tuple.
        TypeError: If `wh_ratio_clip` is not a float.
        TypeError: If `anchor_box` or `deltas` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> anchor_box = Tensor([[4, 1, 2, 1], [2, 2, 2, 3]], mindspore.float32)
        >>> deltas = Tensor([[3, 1, 2, 2], [1, 2, 1, 4]], mindspore.float32)
        >>> boundingbox_decode = ops.BoundingBoxDecode(means=(0.0, 0.0, 0.0, 0.0), stds=(1.0, 1.0, 1.0, 1.0),
        ...                                          max_shape=(768, 1280), wh_ratio_clip=0.016)
        >>> output = boundingbox_decode(anchor_box, deltas)
        >>> print(output)
        [[ 4.1953125  0.         0.         5.1953125]
         [ 2.140625   0.         3.859375  60.59375  ]]

    """

    @prim_attr_register
    def __init__(self, max_shape, means=(0.0, 0.0, 0.0, 0.0), stds=(1.0, 1.0, 1.0, 1.0), wh_ratio_clip=0.016):
        """Initialize BoundingBoxDecode."""
        validator.check_value_type('means', means, tuple, self.name)
        validator.check_value_type('stds', stds, tuple, self.name)
        for i, value in enumerate(means):
            validator.check_value_type("means[%d]" % i, value, [float], self.name)
        for i, value in enumerate(stds):
            validator.check_value_type("stds[%d]" % i, value, [float], self.name)
        validator.check_value_type('wh_ratio_clip', wh_ratio_clip, [float], self.name)
        validator.check_equal_int(len(means), 4, "means len", self.name)
        validator.check_equal_int(len(stds), 4, "stds len", self.name)
        if max_shape is not None:
            validator.check_value_type('max_shape', max_shape, [tuple], self.name)
            validator.check_equal_int(len(max_shape), 2, "max_shape len", self.name)


class CheckValid(PrimitiveWithInfer):
    """
    Checks bounding box.

    Checks whether the bounding box cross data and data border are valid.

    .. warning::
        specifying the valid boundary (heights x ratio, weights x ratio).

    Inputs:
        - **bboxes** (Tensor) - Bounding boxes tensor with shape (N, 4). "N" indicates the number of
          bounding boxes, the value "4" indicates "x0", "x1", "y0", and "y1". Data type must be float16 or float32.
        - **img_metas** (Tensor) - Raw image size information with the format of (height, width, ratio), specifying
          the valid boundary(height * ratio, width * ratio). Data type must be float16 or float32.

    Outputs:
        Tensor, with shape of (N,) and dtype of bool, specifying whether the bounding boxes is in the image.
        "True" indicates valid, while "False" indicates invalid.

    Raises:
        TypeError: If `bboxes` or `img_metas` is not a Tensor.
        TypeError: If dtype of `bboxes` or `img_metas` is neither float16 nor float32.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import mindspore.nn as nn
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self.check_valid = ops.CheckValid()
        ...     def construct(self, x, y):
        ...         valid_result = self.check_valid(x, y)
        ...         return valid_result
        ...
        >>> bboxes = Tensor(np.linspace(0, 6, 12).reshape(3, 4), mindspore.float32)
        >>> img_metas = Tensor(np.array([2, 1, 3]), mindspore.float32)
        >>> net = Net()
        >>> output = net(bboxes, img_metas)
        >>> print(output)
        [ True False False]
    """

    @prim_attr_register
    def __init__(self):
        """Initialize CheckValid."""
        self.init_prim_io_names(inputs=['bboxes', 'img_metas'], outputs=['output'])

    def infer_shape(self, bboxes_shape, metas_shape):
        validator.check("bboxes rank", len(bboxes_shape), "", 2, Rel.EQ, self.name)
        validator.check("bboxes_shape[-1]", bboxes_shape[-1], "", 4, Rel.EQ, self.name)
        validator.check("img_metas rank", len(metas_shape), "", 1, Rel.EQ, self.name)
        validator.check("img_metas shape[0]", metas_shape[0], "", 3, Rel.EQ, self.name)
        return bboxes_shape[:-1]

    def infer_dtype(self, bboxes_type, metas_type):
        valid_type = [mstype.float32, mstype.float16, mstype.int16, mstype.uint8]
        validator.check_tensor_dtype_valid("bboxes_type", bboxes_type, valid_type, self.name)
        validator.check_tensor_dtype_valid("metas_type", metas_type, valid_type, self.name)
        return mstype.bool_


class IOU(Primitive):
    r"""
    Calculates intersection over union for boxes.

    Computes the intersection over union (IOU) or the intersection over foreground (IOF) based on the ground-truth and
    predicted regions.

    .. math::
        \text{IOU} = \frac{\text{Area of Overlap}}{\text{Area of Union}}

        \text{IOF} = \frac{\text{Area of Overlap}}{\text{Area of Ground Truth}}

    .. warning::
        In Ascend, only computation of float16 data is supported. To avoid overflow, the input length
        and width are scaled by 0.2 internally.

    Args:
        mode (string): The mode is used to specify the calculation method,
                       now supporting 'iou' (intersection over union) or 'iof'
                       (intersection over foreground) mode. Default: 'iou'.

    Inputs:
        - **anchor_boxes** (Tensor) - Anchor boxes, tensor of shape (N, 4). "N" indicates the number of anchor boxes,
          and the value "4" refers to "x0", "y0", "x1", and "y1". Data type must be float16 or float32.
        - **gt_boxes** (Tensor) - Ground truth boxes, tensor of shape (M, 4). "M" indicates the number of ground
          truth boxes, and the value "4" refers to "x0", "y0", "x1", and "y1". Data type must be float16 or float32.

    Outputs:
        Tensor, the 'iou' values, tensor of shape (M, N), with the same data type as `anchor_boxes`.

    Raises:
        KeyError: When `mode` is not 'iou' or 'iof'.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> iou = ops.IOU()
        >>> anchor_boxes = Tensor(np.random.randint(1.0, 5.0, [3, 4]), mindspore.float16)
        >>> gt_boxes = Tensor(np.random.randint(1.0, 5.0, [3, 4]), mindspore.float16)
        >>> output = iou(anchor_boxes, gt_boxes)
        >>> print(output.shape)
        (3, 3)
    """

    @prim_attr_register
    def __init__(self, mode='iou'):
        """Initialize IOU."""
        if mode not in {'iou', 'iof'}:
            raise KeyError(f"For '{self.name}', only 'iou' or 'iof' are supported, but got 'mode': {mode}.")
        self.init_prim_io_names(inputs=['anchor_boxes', 'gt_boxes'], outputs=['overlap'])


class Partial(Primitive):
    """
    Makes a partial function instance. Partial function can be used to derived specialized
    functions from general functions by fixing the value of certain number of arguments.

    Inputs:
        - **args** (Union[FunctionType, Tensor]) - The function and bind arguments.

    Outputs:
        FunctionType, partial function bound with arguments.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import Tensor
        >>> from mindspore.ops import operations as P
        >>> def show_input(x, y, z):
        ...     return x, y, z
        >>> partial = P.Partial()
        >>> partial_show_input = partial(show_input, Tensor(1))
        >>> output1 = partial_show_input(Tensor(2), Tensor(3))
        >>> print(output1)
        (Tensor(shape=[], dtype=Int64, value= 1), Tensor(shape=[], dtype=Int64, value= 2), Tensor(shape=[], dtype=Int64,
         value= 3))
        >>> output2 = partial_show_input(Tensor(3), Tensor(4))
        >>> print(output2)
        (Tensor(shape=[], dtype=Int64, value= 1), Tensor(shape=[], dtype=Int64, value= 3), Tensor(shape=[], dtype=Int64,
         value= 4))
    """

    # Side effect will propagated from the first argument to return value.
    side_effect_propagate = 1

    @prim_attr_register
    def __init__(self):
        """Initialize Partial."""
        self.add_prim_attr('side_effect_propagate', 1)

    def __call__(self, *args):
        func = args[0].__call__
        partial_func = functools.partial(func, *args[1:])
        return partial_func


class Depend(Primitive):
    """
    Depend is used for processing dependency operations.

    In most scenarios, if operators have IO side effects or memory side effects,
    they will be executed according to the user's semantics. In some scenarios,
    if the two operators A and B have no order dependency, and A must be executed
    before B, we recommend using Depend to specify their execution order. The
    usage method is as follows::

        a = A(x)                --->        a = A(x)
        b = B(y)                --->        y = Depend(y, a)
                                --->        b = B(y)

    Inputs:
        - **value** (Tensor) - the real value to return for depend operator.
        - **expr** (Expression) - the expression to execute with no outputs.

    Outputs:
        Tensor, the value passed by last operator.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> import mindspore.nn as nn
        >>> import mindspore.ops as ops
        >>> from mindspore import Tensor
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self.softmax = ops.Softmax()
        ...         self.depend = ops.Depend()
        ...
        ...     def construct(self, x, y):
        ...         mul = x * y
        ...         y = self.depend(y, mul)
        ...         ret = self.softmax(y)
        ...         return ret
        ...
        >>> x = Tensor(np.ones([4, 5]), dtype=mindspore.float32)
        >>> y = Tensor(np.ones([4, 5]), dtype=mindspore.float32)
        >>> net = Net()
        >>> output = net(x, y)
        >>> print(output)
        [[0.2 0.2 0.2 0.2 0.2]
         [0.2 0.2 0.2 0.2 0.2]
         [0.2 0.2 0.2 0.2 0.2]
         [0.2 0.2 0.2 0.2 0.2]]
    """

    # Side effect will propagated from the first argument to return value.
    side_effect_propagate = 1

    @prim_attr_register
    def __init__(self):
        """Initialize Depend."""
        self.add_prim_attr('side_effect_propagate', 1)

    def __call__(self, value, expr):
        return value


class UpdateState(Primitive):
    """
    UpdateState is used for update side-effect state.

    Inputs:
        - **value** (State) - the state value to be updated.
        - **expr** (Expression) - the expression to evaluate before state changes.

    Outputs:
        State, the updated state value.
    """

    @prim_attr_register
    def __init__(self):
        pass

    def __call__(self, state, expr):
        return state


class CheckBprop(PrimitiveWithInfer):
    """
    Checks whether the data type and the shape of corresponding elements from tuples x and y are the same.

    Args:
        prim_to_check (str): The name of the primitive being checked. Default: ''.

    Inputs:
        - **input_x** (tuple[Tensor]) - The `input_x` contains the outputs of bprop to be checked.
        - **input_y** (tuple[Tensor]) - The `input_y` contains the inputs of bprop to check against.

    Outputs:
        Tuple[Tensor], the `input_x`,
        if data type and shape of corresponding elements from `input_x` and `input_y` are the same.

    Raises:
        TypeError: If `input_x` or `input_y` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self.op = ops.CheckBprop()
        ...     def construct(self, x, y):
        ...         return self.op(x, y)
        ...
        >>> net = Net()
        >>> input_x = (Tensor(np.array([[2, 2], [2, 2]]), mindspore.float32),)
        >>> input_y = (Tensor(np.array([[2, 2], [2, 2]]), mindspore.float32),)
        >>> output = net(input_x, input_y)
        >>> print(output)
        (Tensor(shape=[2, 2], dtype=Float32, value=
        [[ 2.00000000e+00,  2.00000000e+00],
         [ 2.00000000e+00,  2.00000000e+00]]),)
    """

    @prim_attr_register
    def __init__(self, prim_to_check=""):
        """Initialize CheckBprop"""
        self.prim_to_check = prim_to_check

    def infer_shape(self, xshapes, yshapes):
        tips = f"user defined method 'bprop'"
        validator.check_value_type('grads', xshapes, (tuple,), tips)
        validator.check_value_type('params', yshapes, (tuple,), tips)
        if not len(xshapes) == len(yshapes):
            raise ValueError(f"For {tips} the number of return values(gradients) should be equal to "
                             f"the number of input arguments except 'out' and 'dout', "
                             f"which is:{len(yshapes)} but got {len(xshapes)}.")
        checking_range = len(yshapes)
        for i in range(checking_range):
            xshape = xshapes[i]
            yshape = yshapes[i]
            if not xshape or not yshape:
                continue
            if xshape != yshape:
                raise ValueError(f"For {tips}, the {i}th return value(gradient of the {i}th argument) "
                                 f"should have the same shape as the {i}th argument, "
                                 f"which is:{yshape}, but got: {xshape}.")
        return xshapes

    def infer_dtype(self, xdtypes, ydtypes):
        tips = f"user defined method 'bprop'"
        validator.check_value_type('grads', xdtypes, (tuple,), tips)
        validator.check_value_type('params', ydtypes, (tuple,), tips)
        if not len(xdtypes) == len(ydtypes):
            raise ValueError(f"For {tips}, the number of return values(gradients) should be equal to "
                             f"the number of input arguments except 'out' and 'dout', "
                             f"which is:{len(ydtypes)} but got {len(xdtypes)}.")
        checking_range = len(ydtypes)
        for i in range(checking_range):
            xdtype = xdtypes[i]
            ydtype = ydtypes[i]
            if isinstance(xdtype, mstype.anything_type) or isinstance(ydtype, mstype.anything_type):
                continue
            if isinstance(ydtype, mstype.function_type):
                if not isinstance(xdtype, mstype.env_type_type):
                    raise TypeError(f"For {tips}, the {i}th return value(gradient of the {i}th argument) type "
                                    f"should be {mstype.env_type_type}, but got {xdtype}.")
                continue
            if xdtype != ydtype:
                raise TypeError(f"For {tips}, the {i}th return value(gradient of the {i}th argument) "
                                f"should have the same dtype as the {i}th argument, "
                                f"which is:{ydtype}, but got: {xdtype}.")
        return xdtypes


class ConfusionMatrix(PrimitiveWithInfer):
    r"""
    Calculates the confusion matrix from labels and predictions.

    Args:
        num_classes (int): The num of classes.
        dtype (str): Data type of confusion matrix. Default: 'int32'.

    Inputs:
        - **labels** (Tensor) - real labels, tensor of 1-D. the dtype must be non-negative Integer.
        - **predictions** (Tensor) - the labels from prediction, tensor of 1-D.
          the shape same as `labels` and the dtype must be non-negative Integer.
        - **weights** (Tensor) - tensor of 1-D. the shape same as `predictions`.

    Outputs:
        Tensor, the confusion matrix, with shape (`num_classes`, `num_classes`).

    Raises:
        TypeError: If `num_classes` is not an int.
        TypeError: If `dtype` is not a str.
        TypeError: If `labels`, `predictions` or weight` is not a Tensor.

    Examples:
        >>> confusion_matrix = ops.ConfusionMatrix(4)
        >>> labels = Tensor([0, 1, 1, 3], mindspore.int32)
        >>> predictions = Tensor([1, 2, 1, 3], mindspore.int32)
        >>> output = confusion_matrix(labels, predictions)
        >>> print(output)
        [[0 1 0 0]
         [0 1 1 0]
         [0 0 0 0]
         [0 0 0 1]]
    """

    @prim_attr_register
    def __init__(self, num_classes, dtype="int32"):
        """Initialize ConfusionMatrix."""
        validator.check_value_type("num_classes", num_classes, [int], self.name)
        validator.check_value_type("dtype", dtype, [str], self.name)

    def infer_shape(self, labels, predictions, weights=None):
        validator.check('labels dimension', len(labels), '', 1, Rel.EQ, self.name)
        validator.check('labels shape', labels, 'predictions shape', predictions, Rel.EQ, self.name)
        if weights is not None:
            validator.check('labels shape', labels, 'weights shape', weights, Rel.EQ, self.name)
        ret = (self.num_classes, self.num_classes)
        return ret

    def infer_dtype(self, labels, predictions, weights=None):
        validator.check_subclass('labels', labels, mstype.tensor, self.name)
        validator.check_subclass('predictions', predictions, mstype.tensor, self.name)
        if weights is not None:
            validator.check_subclass('weights', weights, mstype.tensor, self.name)
        args = {"labels": labels, "predictions": predictions}
        validator.check_tensors_dtypes_same_and_valid(args, (mstype.number_type), self.name)
        return labels


class PopulationCount(PrimitiveWithInfer):
    r"""
    Computes element-wise population count(a.k.a bitsum, bitcount).
    For each entry in `input` , calculates the number of 1 bits in the binary representation of that entry.

    Inputs:
        - **input** (Tensor) -  The data type must be int16 or uint16.

    Outputs:
        Tensor, with the same shape as the input.

    Raises:
        TypeError: If `input` is not a Tensor.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> population_count = ops.PopulationCount()
        >>> x_input = Tensor([0, 1, 3], mindspore.int16)
        >>> output = population_count(x_input)
        >>> print(output)
        [0 1 2]
    """

    @prim_attr_register
    def __init__(self):
        pass

    def infer_shape(self, x_shape):
        return x_shape

    def infer_dtype(self, x_dtype):
        validator.check_tensor_dtype_valid("x", x_dtype, (mstype.int16, mstype.uint16,), self.name)
        return mstype.tensor_type(mstype.uint8)


class Push(PrimitiveWithInfer):
    """
    Pushes the inputs of the corresponding optimizer to parameter server.

    Args:
        optim_type (string): The optimizer type. Default: 'ApplyMomentum'.
        only_shape_indices (list): The indices of input of which only shape
                                   will be pushed to parameter server. Default: None.

    Inputs:
        - **optim_inputs** (tuple) - The inputs for this kind of optimizer.
        - **optim_input_shapes** (tuple) - The shapes of the inputs.

    Outputs:
        Tensor, the key of the weight which needs to be updated.
    """

    @prim_attr_register
    def __init__(self, optim_type='ApplyMomentum', only_shape_indices=None):
        """Initialize Push"""
        self.add_prim_attr("primitive_target", "CPU")
        self.init_prim_io_names(inputs=['optim_inputs', 'optim_input_shapes'], outputs=['key'])
        self.add_prim_attr("side_effect_hidden", True)

    def infer_shape(self, inputs, shapes):
        return [1]

    def infer_dtype(self, inputs, shapes):
        return mstype.uint64


class Pull(PrimitiveWithInfer):
    """
    Pulls weight from parameter server.

    Inputs:
        - **key** (Tensor) - The key of the weight.
        - **weight** (Tensor) - The weight to be updated.

    Outputs:
        None.
    """

    @prim_attr_register
    def __init__(self):
        """Initialize Pull"""
        self.add_prim_attr("primitive_target", "CPU")
        self.init_prim_io_names(inputs=['key', 'weight'], outputs=['output'])

    def infer_shape(self, key_shape, weight_shape):
        return [1]

    def infer_dtype(self, key_dtype, weight_dtype):
        return mstype.float32


class PullWeight(PrimitiveWithInfer):
    """
    Pull weight by its names from server.

    Inputs:
        - **weight** (Tensor) - The weight to be pulled.
        - **name** (String) - The full name of the weight.
        - **index** (Int) - The index of the weight.

    Outputs:
        None.
    """

    @prim_attr_register
    def __init__(self):
        """Initialize PullWeight"""
        self.add_prim_attr("primitive_target", "CPU")
        self.init_prim_io_names(inputs=['weight', "name", "index"], outputs=['output'])

    def infer_shape(self, weight, name, index):
        return [1]

    def infer_dtype(self, weight, name, index):
        return mstype.float32


class PushWeight(PrimitiveWithInfer):
    """
    Upload weight by its names to server.

    Inputs:
        - **weight** (Tensor) - The weight to be uploaded.
        - **name** (String) - The full name of the weight.
        - **index** (Int) - The index of the weight.

    Outputs:
        None.
    """

    @prim_attr_register
    def __init__(self):
        """Initialize PushWeight"""
        self.add_prim_attr("primitive_target", "CPU")
        self.init_prim_io_names(inputs=["weight", "name", "index"], outputs=["output"])

    def infer_shape(self, weight, name, index):
        return [1]

    def infer_dtype(self, weight, ps_key, index):
        return mstype.float32


class PushMetrics(PrimitiveWithInfer):
    """
    Push metrics like loss and accuracy for federated learning worker.

    Inputs:
        - **loss** (Tensor) - The loss.
        - **accuracy** (Tensor) - The accuracy.

    Outputs:
        None.
    """

    @prim_attr_register
    def __init__(self):
        """Initialize PushMetrics"""
        self.add_prim_attr("primitive_target", "CPU")
        self.add_prim_attr("side_effect_mem", True)
        self.init_prim_io_names(inputs=["loss", "accuracy"], outputs=["result"])

    def infer_shape(self, loss, accuracy):
        return [1]

    def infer_dtype(self, loss, accuracy):
        return mstype.float32


class StartFLJob(PrimitiveWithInfer):
    """
    StartFLJob for federated learning worker.
    """
    @prim_attr_register
    def __init__(self, data_size):
        self.add_prim_attr("primitive_target", "CPU")
        self.add_prim_attr("data_size", data_size)
        self.init_prim_io_names(inputs=[], outputs=["result"])

    def infer_shape(self):
        return [1]

    def infer_dtype(self):
        return mstype.float32


class UpdateModel(PrimitiveWithInfer):
    """
    UpdateModel for federated learning worker.
    """
    @prim_attr_register
    def __init__(self, encrypt_mode=""):
        self.add_prim_attr("primitive_target", "CPU")
        self.add_prim_attr('side_effect_mem', True)
        self.add_prim_attr('encrypt_mode', encrypt_mode)
        self.init_prim_io_names(inputs=["weights"], outputs=["result"])

    def infer_shape(self, weights):
        return [1]

    def infer_dtype(self, weights):
        return mstype.float32


class GetModel(PrimitiveWithInfer):
    """
    GetModel for federated learning worker.
    """
    @prim_attr_register
    def __init__(self):
        self.add_prim_attr("primitive_target", "CPU")
        self.add_prim_attr('side_effect_mem', True)
        self.init_prim_io_names(inputs=["weights"], outputs=["result"])

    def infer_shape(self, weights):
        return [1]

    def infer_dtype(self, weights):
        return mstype.float32


class ExchangeKeys(PrimitiveWithInfer):
    """
    Exchange pairwise public keys for federated learning worker.
    """
    @prim_attr_register
    def __init__(self):
        self.add_prim_attr("primitive_target", "CPU")
        self.add_prim_attr('side_effect_mem', True)
        self.init_prim_io_names(inputs=[], outputs=["result"])

    def infer_shape(self):
        return [1]

    def infer_dtype(self):
        return mstype.float32


class GetKeys(PrimitiveWithInfer):
    """
    Get pairwise public keys for federated learning worker.
    """
    @prim_attr_register
    def __init__(self):
        self.add_prim_attr("primitive_target", "CPU")
        self.add_prim_attr('side_effect_mem', True)
        self.init_prim_io_names(inputs=[], outputs=["result"])

    def infer_shape(self):
        return [1]

    def infer_dtype(self):
        return mstype.float32


class identity(Primitive):
    """
    Makes a identify primitive, used for pynative mode.

    Inputs:
        - **x** (Any) - identity input value.

    Outputs:
        The same as input.
    """

    # Side effect will propagated from the first argument to return value.
    side_effect_propagate = 1

    @prim_attr_register
    def __init__(self):
        """Initialize identity."""
        self.add_prim_attr('side_effect_propagate', 1)

    def __call__(self, x):
        return x


class PyFunc(PrimitiveWithInfer):
    r"""
    Execute Python function.

    `PyFunc` encapsulates Python functions as an operator which could be compiled into computation graph.
    Unlike normal operators, it cannot be exported to MindIR as it is executed in current Python context.
    As only the weights of the network is stored in the checkpoint, network include `PyFunc` could save
    checkpoint and load to the network again, but will lose any Python function state.

    .. warning::
        This is an experimental prototype that is subject to change and/or deletion.

    Args:
        fn (function): Python function which inputs and outputs should be Python built-in scalar or numpy ndarray.
        in_types (list[:class:`mindspore.dtype`]): The type of the inputs.
        in_shapes (list[tuple[int]]): The dimensionality of the inputs. An empty list represents a scalar, otherwise it
                                      represent a numpy array.
        out_types (list[:class:`mindspore.dtype`]): The type of the outputs.
        out_shapes (list[tuple[int]]): The dimensionality of the outputs. An empty list represents a scalar, otherwise
                                       it represent a numpy array.
        stateful (bool): Whether the function is stateful or not.
                         If True, the execution order is same with model definition.

    Inputs:
        - **input_x** (Union(tuple[Tensor], list[Tensor])) - The input tuple or list
          is made up of multiple tensors.

    Outputs:
        tuple[Tensor], execution results Python functions.

    Raises:
        TypeError: The Python function execution failed.
        TypeError: The attributes(in_types/in_shapes/out_types/out_shapes) are inconsistent with Python function
                   specifications.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> def func(x1, x2):
        >>>     return x1 + x2
        >>> x1 = Tensor(np.array([1, 2, 3]).astype(np.float32))
        >>> x2 = Tensor(np.array([1, 2, 3]).astype(np.float32))
        >>> op = P.PyFunc(func, [x1.dtype, x2.dtype], [x1.shape, x2.shape], [x1.dtype], [x1.dtype])
        >>> output = op((x1, x2))
        >>> print(output[0].asnumpy())
        [2. 4. 6.]
    """

    def __init__(self, fn, in_types, in_shapes, out_types, out_shapes, stateful=True):
        super(PyFunc, self).__init__(self.__class__.__name__)
        add_pyfunc(id(fn), fn)
        self.add_prim_attr('fn_id', id(fn))
        self.add_prim_attr('in_types', in_types)
        self.add_prim_attr('in_shapes', in_shapes)
        self.add_prim_attr('out_types', out_types)
        self.add_prim_attr('out_shapes', out_shapes)
        validator.check_value_type("in_types", in_types, [list, tuple], self.name)
        validator.check_value_type("in_shapes", in_shapes, [list, tuple], self.name)
        validator.check("in_types length", len(in_types), "in_shapes length", len(in_shapes), Rel.EQ, self.name)
        validator.check_value_type("out_types", out_types, [list, tuple], self.name)
        validator.check_value_type("out_shapes", out_shapes, [list, tuple], self.name)
        validator.check("out_types length", len(out_types), "out_shapes length", len(out_shapes), Rel.EQ, self.name)
        self.add_prim_attr("side_effect_io", stateful)
        self.add_prim_attr("primitive_target", "CPU")
        fake_output = False
        single_scalar_output = False
        if not out_types:
            fake_output = True
        elif not out_shapes:
            single_scalar_output = True
        self.add_prim_attr("fake_output", fake_output)
        self.add_prim_attr("single_scalar_output", single_scalar_output)

    def infer_shape(self, *args):
        if self.out_shapes:
            return tuple(self.out_shapes)

        logger.warning("The function output are empty tuple. Add a placeholder instead. "
                       "Do not use it as it could be any uninitialized data.")
        return ((1,),)

    def infer_dtype(self, *args):
        if self.out_shapes:
            return tuple(self.out_types)

        logger.warning("The function output are empty tuple. Add a placeholder instead. "
                       "Do not use it as it could be any uninitialized data.")
        return (mstype.int32,)
