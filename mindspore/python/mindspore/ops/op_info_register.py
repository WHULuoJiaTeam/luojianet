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

"""Operators info register."""

import inspect
import json
import os
import functools

from mindspore._c_expression import Oplib

from mindspore._checkparam import Validator as validator

# path of built-in op info register.
BUILT_IN_OPS_REGISTER_PATH = "mindspore/ops/_op_impl"
BUILT_IN_CUSTOM_OPS_REGISTER_PATH = "mindspore/ops/_op_impl/_custom_op"


def op_info_register(op_info):
    r"""
    A decorator which is used to register an operator.

    Note:
        'op_info' should represent the operator information by string with json format.
        The 'op_info' will be added into oplib.

    Args:
        op_info (Union[str, dict]): operator information in json format.

    Examples:
        >>> from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType
        >>> abs_op_info = TBERegOp("Abs") \
        ...    .fusion_type("ELEMWISE") \
        ...    .async_flag(False) \
        ...    .binfile_name("abs.so") \
        ...    .compute_cost(10) \
        ...    .kernel_name("abs") \
        ...    .partial_flag(True) \
        ...    .op_pattern("formatAgnostic") \
        ...    .input(0, "x", None, "required", None) \
        ...    .output(0, "y", True, "required", "all") \
        ...    .dtype_format(DataType.F16_None, DataType.F16_None) \
        ...    .dtype_format(DataType.F32_None, DataType.F32_None) \
        ...    .dtype_format(DataType.I32_None, DataType.I32_None) \
        ...    .get_op_info()
        >>>
        >>> @op_info_register(abs_op_info)
        ... def _abs_tbe():
        ...    return
        ...

    Returns:
        Function, returns a decorator for op info register.
    """

    def register_decorator(func):
        if isinstance(op_info, dict):
            op_info_real = json.dumps(op_info)
        else:
            op_info_real = op_info
        validator.check_value_type("op_info", op_info_real, [str])
        op_lib = Oplib()
        file_path = os.path.realpath(inspect.getfile(func))
        # keep the path custom ops implementation.
        if BUILT_IN_CUSTOM_OPS_REGISTER_PATH in file_path:
            imply_path = file_path
        else:
            imply_path = "" if BUILT_IN_OPS_REGISTER_PATH in file_path else file_path
        if not op_lib.reg_op(op_info_real, imply_path):
            raise ValueError('Invalid op info {}:\n{}\n'.format(file_path, op_info_real))

        def wrapped_function(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapped_function

    return register_decorator


def custom_info_register(*reg_info):
    r"""
    A decorator which is used to bind the registration information to the `func` parameter of
    :class:`mindspore.ops.Custom`.

    Note:
        The 'reg_info' will be added into oplib.

    Args:
        reg_info (tuple[str, dict]): Each item represents registration information in json format.

    Returns:
        Function, returns a decorator for op info register.

    Examples:
        >>> from mindspore.ops import custom_info_register, CustomRegOp, DataType
        >>> custom_func_ascend_info = CustomRegOp() \
        ...     .input(0, "x", "dynamic") \
        ...     .output(0, "y") \
        ...     .dtype_format(DataType.F16_Default, DataType.F16_Default) \
        ...     .dtype_format(DataType.F32_Default, DataType.F32_Default) \
        ...     .target("Ascend") \
        ...     .get_op_info()
        >>>
        >>> @custom_info_register(custom_func_ascend_info)
        ... def custom_func(x):
        ...     pass
    """

    def decorator(func):
        setattr(func, "reg_info", reg_info)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorator


class RegOp:
    """
    Base class for op info register.

    Args:
        op_name (str): Name of op.
    """

    def __init__(self, op_name=""):
        if not isinstance(op_name, str):
            raise ValueError("op name value must be string")
        if not op_name.strip():
            raise ValueError("op name is empty")
        self.op_name = op_name
        self.inputs = []
        self.outputs = []
        self.attr_ = []
        self.fusion_type_ = ''
        self.dtype_format_ = []

    def _is_string(self, value):
        """
        Check if the value is a str type.

        Args:
            value: Parameter to be checked.

        Raises:
            TypeError: If the type of value is not a str.
        """
        if not isinstance(value, str):
            raise TypeError("%s value must be str" % str(value))
        return True

    def _is_int(self, value):
        """
        Check if the value is an int.

        Args:
            value: Parameter to be checked.

        Raises:
            TypeError: If the type of value is not an int.
        """
        if not isinstance(value, int):
            raise TypeError("%s value must be int" % str(value))
        return True

    def _is_bool(self, value):
        """
        Check if the value is a bool.

        Args:
            value: Parameter to be checked.

        Raises:
            TypeError: If the type of value is not a bool.
        """
        if not isinstance(value, bool):
            raise TypeError("%s value must be bool" % str(value))
        return True

    def _check_param(self, param_list, key_list, fn_list, kwargs):
        """
        Check if the parameter type is correct.

        Args:
            param_list (list): Parameter list to be checked.
            key_list (list): The keys of output dict.
            fn_list (list): Function used for parameter checking. If the function list has only one element,
                            all parameters will use the same function.
            kwargs (dict): Other parameter information.

        Raises:
            TypeError: If the type of value is not list.
            ValueError: If the size of param list is not equal to the size of key list, or
                        the size of param list is not equal to the size of function list.
        """
        for i in [param_list, key_list, fn_list]:
            if not isinstance(i, list):
                raise TypeError("%s value must be list type" % str(i))
        if len(param_list) != len(key_list) or (len(fn_list) != 1 and len(param_list) != len(fn_list)):
            raise ValueError("param_list size {}, key_list size {}, must be equal.And fn_list size {}.".
                             format(len(param_list), len(key_list), len(fn_list)))
        out_dict = {}
        for idx, element in enumerate(param_list):
            if element is not None:
                if len(fn_list) == 1:
                    fn_list[0](element)
                else:
                    fn_list[idx](element)
                out_dict[key_list[idx]] = element
        if kwargs:
            out_dict = dict(out_dict, **kwargs)
        return out_dict

    def fusion_type(self, fusion_type):
        """
        Fusion type of the operator.

        Args:
            fusion_type (str): Value of fusion type.
        """
        self._is_string(fusion_type)
        self.fusion_type_ = fusion_type
        return self

    def dtype_format(self, *args):
        """
        A dtype and format supported by the operator.

        Args:
            args (tuple): Value of dtype and format.

        Raises:
            ValueError: If the size of args not equal to input size add output size.
            TypeError: If the type of args is not tuple.
        """
        if len(self.inputs) + len(self.outputs) != len(args):
            raise ValueError("input size add output size must be equal to dtype format size")
        dtype_format = []
        for arg in args:
            if not isinstance(arg, tuple) or len(arg) != 2:
                raise ValueError("dtype and format value must be tuple of two elements")
            self._is_string(arg[0])
            self._is_string(arg[1])
            dtype_format.append(arg)
        self.dtype_format_.append(tuple(dtype_format))
        return self

    def get_op_info(self):
        """
        Return all registration information for this instance.

        The '_' character ending the key is removed here for compatibility with previous version.

        Key will be unified into an underlined form later.
        """
        op_info = {}
        for key, value in self.__dict__.items():
            if isinstance(key, str) and key.endswith('_'):
                op_info[key.rstrip('_')] = value
            else:
                op_info[key] = value
        return op_info


class CpuRegOp(RegOp):
    """Class for Cpu op info register"""

    def __init__(self, op_name):
        super(CpuRegOp, self).__init__(op_name)
        self.imply_type = "CPU"

    def input(self, index=None, name=None, param_type=None, **kwargs):
        """
        Register Cpu op input information.

        Args:
            index (int): Order of the input. Default: None.
            name (str): Name of the input. Default: None.
            param_type (str): Param type of the input. Default: None.
            kwargs (dict): Other information of the input.
        """
        param_list = [index, name, param_type]
        key_list = ["index", "name", "param_type"]
        fn_list = [self._is_int, self._is_string, self._is_string]
        input_dict = self._check_param(param_list, key_list, fn_list, kwargs)
        self.inputs.append(input_dict)
        return self

    def output(self, index=None, name=None, param_type=None, **kwargs):
        """
        Register AiCPU op output information.

        Args:
            index (int): Order of the output. Default: None.
            name (str): Name of the output. Default: None.
            param_type (str): Param type of the output. Default: None.
            kwargs (dict): Other information of the output.
        """
        param_list = [index, name, param_type]
        key_list = ["index", "name", "param_type"]
        fn_list = [self._is_int, self._is_string, self._is_string]
        output_dict = self._check_param(param_list, key_list, fn_list, kwargs)
        self.outputs.append(output_dict)
        return self

    def attr(self, name=None, value_type=None, value=None, **kwargs):
        """
        Register AiCPU op attribute information.

        Args:
            name (str): Name of the attribute. Default: None.
            value_type (str): Value type of the attribute. Default: None.
            value (str): Value of the attribute. Default: None.
            kwargs (dict): Other information of the attribute.
        """
        param_list = [name, value_type, value]
        key_list = ["name", "type", "value"]
        fn_list = [self._is_string]
        attr_dict = self._check_param(param_list, key_list, fn_list, kwargs)
        self.attr_.append(attr_dict)
        return self


class AkgRegOp(RegOp):
    """Class for Akg op info register."""

    def __init__(self, op_name, processor):
        super(AkgRegOp, self).__init__(op_name)
        self.imply_type = "AKG"
        self.processor = processor

    def input(self, index=None, name=None, param_type=None, **kwargs):
        """
        Register Akg op input information.

        Args:
            index (int): Order of the input. Default: None.
            name (str): Name of the input. Default: None.
            param_type (str): Param type of the input. Default: None.
            kwargs (dict): Other information of the input.
        """
        param_list = [index, name, param_type]
        key_list = ["index", "name", "param_type"]
        fn_list = [self._is_int, self._is_string, self._is_string]
        input_dict = self._check_param(param_list, key_list, fn_list, kwargs)
        self.inputs.append(input_dict)
        return self

    def output(self, index=None, name=None, **kwargs):
        """
        Register Akg op output information.

        Args:
            index (int): Order of the output. Default: None.
            name (str): Name of the output. Default: None.
            kwargs (dict): Other information of the output.
        """
        param_list = [index, name]
        key_list = ["index", "name"]
        fn_list = [self._is_int, self._is_string]
        output_dict = self._check_param(param_list, key_list, fn_list, kwargs)
        self.outputs.append(output_dict)
        return self

    def attr(self, name=None, param_type=None, value_type=None, **kwargs):
        """
        Register Akg op attribute information.

        Args:
            name (str): Name of the attribute. Default: None.
            param_type (str): Param type of the attribute. Default: None.
            value_type (str): Value type of the attribute. Default: None.
            kwargs (dict): Other information of the attribute.
        """
        param_list = [name, param_type, value_type]
        key_list = ["name", "param_type", "type"]
        fn_list = [self._is_string]
        attr_dict = self._check_param(param_list, key_list, fn_list, kwargs)
        self.attr_.append(attr_dict)
        return self


class AkgGpuRegOp(AkgRegOp):
    """Class for AkgGpu op info register"""

    def __init__(self, op_name):
        super(AkgGpuRegOp, self).__init__(op_name, "CUDA")


class AkgAscendRegOp(AkgRegOp):
    """Class for AkgAscend op info register"""

    def __init__(self, op_name):
        super(AkgAscendRegOp, self).__init__(op_name, "AiCore")


class AkgCpuRegOp(AkgRegOp):
    """Class for AkgCpu op info register"""

    def __init__(self, op_name):
        super(AkgCpuRegOp, self).__init__(op_name, "LLVM")


class AiCPURegOp(CpuRegOp):
    r"""
    Class for AiCPU operator information register.

    Args:
        op_name (str):kernel name.

    Examples:
        >>> from mindspore.ops.op_info_register import op_info_register, AiCPURegOp, DataType
        >>> stack_op_info = AiCPURegOp("Stack") \
        ...    .fusion_type("OPAQUE") \
        ...    .attr("axis", "int") \
        ...    .input(0, "x", "dynamic") \
        ...    .output(0, "y", "required") \
        ...    .dtype_format(DataType.I8_Default, DataType.I8_Default) \
        ...    .dtype_format(DataType.I16_Default, DataType.I16_Default) \
        ...    .dtype_format(DataType.I32_Default, DataType.I32_Default) \
        ...    .dtype_format(DataType.I64_Default, DataType.I64_Default) \
        ...    .dtype_format(DataType.U8_Default, DataType.U8_Default) \
        ...    .dtype_format(DataType.U16_Default, DataType.U16_Default) \
        ...    .dtype_format(DataType.U32_Default, DataType.U32_Default) \
        ...    .dtype_format(DataType.U64_Default, DataType.U64_Default) \
        ...    .dtype_format(DataType.F16_Default, DataType.F16_Default) \
        ...    .dtype_format(DataType.F32_Default, DataType.F32_Default) \
        ...    .dtype_format(DataType.F64_Default, DataType.F64_Default) \
        ...    .dtype_format(DataType.BOOL_Default, DataType.BOOL_Default) \
        ...    .get_op_info()
        >>>
    """

    def __init__(self, op_name):
        super(AiCPURegOp, self).__init__(op_name)
        self.imply_type = "AiCPU"


class TBERegOp(RegOp):
    r"""
    Class for TBE operator information register.

    Args:
        op_name (str):kernel name.

    Examples:
        >>> from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType
        >>> abs_op_info = TBERegOp("Abs") \
        ...    .fusion_type("ELEMWISE") \
        ...    .async_flag(False) \
        ...    .binfile_name("abs.so") \
        ...    .compute_cost(10) \
        ...    .kernel_name("abs") \
        ...    .partial_flag(True) \
        ...    .op_pattern("formatAgnostic") \
        ...    .input(0, "x", None, "required", None) \
        ...    .output(0, "y", True, "required", "all") \
        ...    .dtype_format(DataType.F16_None, DataType.F16_None) \
        ...    .dtype_format(DataType.F32_None, DataType.F32_None) \
        ...    .dtype_format(DataType.I32_None, DataType.I32_None) \
        ...    .get_op_info()
        >>>
    """

    def __init__(self, op_name):
        super(TBERegOp, self).__init__(op_name)
        self.imply_type = "TBE"
        self.async_flag_ = False
        self.binfile_name_ = ''
        self.compute_cost_ = 10
        self.kernel_name_ = ''
        self.partial_flag_ = False
        self.reshape_type_ = ''
        self.dynamic_shape_ = False
        self.dynamic_compile_static_ = False
        self.need_check_supported_ = False
        self.is_dynamic_format_ = False
        self.op_pattern_ = ""

    def async_flag(self, async_flag=False):
        """
        Define the calculation efficiency of the operator, whether the asynchronous calculation is supported.

        Args:
            async_flag (bool): Value of async flag. Default: false.
        """
        self._is_bool(async_flag)
        self.async_flag_ = async_flag
        return self

    def binfile_name(self, binfile_name):
        """
        Set the binary file name of the operator, it is optional.

        Args:
            binfile_name (str): The binary file name of the operator.
        """
        self._is_string(binfile_name)
        self.binfile_name_ = binfile_name
        return self

    def compute_cost(self, compute_cost=10):
        """
        Define the calculation efficiency of operator, which refers to the value of the cost model
        in the tiling module.

        Args:
            compute_cost (int): Value of compute cost. Default: 10.
        """
        self._is_int(compute_cost)
        self.compute_cost_ = compute_cost
        return self

    def kernel_name(self, kernel_name):
        """
        The name of operator kernel.

        Args:
            kernel_name (str): Name of operator kernel.
        """
        self._is_string(kernel_name)
        self.kernel_name_ = kernel_name
        return self

    def partial_flag(self, partial_flag=True):
        """
        Define the calculation efficiency of operator, whether the partial calculation is supported.

        Args:
            partial_flag (bool): Value of partial flag. Default: true.
        """
        self._is_bool(partial_flag)
        self.partial_flag_ = partial_flag
        return self

    def reshape_type(self, reshape_type):
        """
        Reshape type of operator.

        Args:
            reshape_type (str): Value of reshape type.
        """
        self._is_string(reshape_type)
        self.reshape_type_ = reshape_type
        return self

    def dynamic_shape(self, dynamic_shape=False):
        """
        Whether the operator supports dynamic shape.

        Args:
            dynamic_shape (bool): Value of dynamic shape. Default: false.
        """
        self._is_bool(dynamic_shape)
        self.dynamic_shape_ = dynamic_shape
        return self

    def dynamic_compile_static(self, dynamic_compile_static=False):
        """
        Whether the operator supports dynamic compile static.

        Args:
            dynamic_compile_static (bool): Value of dynamic compile static. Default: false.
        """
        self._is_bool(dynamic_compile_static)
        self.dynamic_compile_static_ = dynamic_compile_static
        return self

    def need_check_supported(self, need_check_supported=False):
        """
        Whether the operator needs check supports.

        Args:
            need_check_supported (bool): Value of need_check_supported. Default: false.
        """
        self._is_bool(need_check_supported)
        self.need_check_supported_ = need_check_supported
        return self

    def is_dynamic_format(self, is_dynamic_format=False):
        """
        Whether the operator needs calop_select_format api.

        Args:
            is_dynamic_format (bool): Value of is_dynamic_format. Default: false.
        """
        self._is_bool(is_dynamic_format)
        self.is_dynamic_format_ = is_dynamic_format
        return self

    def op_pattern(self, pattern=None):
        """
        The behavior type of operator, such as broadcast, reduce and so on.

        Args:
            pattern (str): Value of op pattern. Default: None.
        """
        if pattern is not None and self._is_string(pattern):
            self.op_pattern_ = pattern
        return self

    def attr(self, name=None, param_type=None, value_type=None, value=None, default_value=None, **kwargs):
        """
        Register TBE op attribute information.

        Args:
            name (str): Name of the attribute. Default: None.
            param_type (str): Param type of the attribute. Default: None.
            value_type (str): Type of the attribute. Default: None.
            value (str): Value of the attribute. Default: None.
            default_value (str): Default value of attribute. Default: None.
            kwargs (dict): Other information of the attribute.
        """
        param_list = [name, param_type, value_type, value, default_value]
        key_list = ["name", "param_type", "type", "value", "default_value"]
        fn_list = [self._is_string]
        attr_dict = self._check_param(param_list, key_list, fn_list, kwargs)
        self.attr_.append(attr_dict)
        return self

    def input(self, index=None, name=None, need_compile=None, param_type=None, shape=None, value_depend=None, **kwargs):
        """
        Register TBE op input information.

        Args:
            index (int): Order of the input. Default: None.
            name (str): Name of the input. Default: None.
            need_compile (bool): Whether the input needs to be compiled or not. Default: None.
            param_type (str): Type of the input. Default: None.
            shape (str): Shape of the input. Default: None.
            value_depend (str): Whether the input is constant value depend. Default: None.
            kwargs (dict): Other information of the input.
        """
        param_list = [index, name, need_compile, param_type, shape, value_depend]
        key_list = ["index", "name", "need_compile", "param_type", "shape", "value_depend"]
        fn_list = [self._is_int, self._is_string, self._is_bool, self._is_string, self._is_string, self._is_string]
        input_dict = self._check_param(param_list, key_list, fn_list, kwargs)
        value_depend_values = ("ignored", "optional", "required")
        if value_depend and value_depend.lower() not in value_depend_values:
            raise ValueError("Operator {} input{}'s value_depend's value ({}) is not in {}.".
                             format(self.op_name, index, value_depend, value_depend_values))
        self.inputs.append(input_dict)
        return self

    def output(self, index=None, name=None, need_compile=None, param_type=None, shape=None, **kwargs):
        """
        Register TBE op output information.

        Args:
            index (int): Order of the output. Default: None.
            name (str): Name of the output. Default: None.
            need_compile (bool): Whether the output needs to be compiled or not. Default: None.
            param_type (str): Type of the output. Default: None.
            shape (str): Shape of the output. Default: None.
            kwargs (dict): Other information of the output.
        """
        param_list = [index, name, need_compile, param_type, shape]
        key_list = ["index", "name", "need_compile", "param_type", "shape"]
        fn_list = [self._is_int, self._is_string, self._is_bool, self._is_string, self._is_string]
        output_dict = self._check_param(param_list, key_list, fn_list, kwargs)
        self.outputs.append(output_dict)
        return self


class CustomRegOp(RegOp):
    r"""
    Class used for generating the registration information for the `func` parameter of :class:`mindspore.ops.Custom`.

    Args:
        op_name (str): kernel name. No need to set this value as `Custom`, operator will generate a unique name
            automatically. Default: "Custom".

    Examples:
        >>> from mindspore.ops import CustomRegOp, DataType
        >>> custom_op_ascend_info = CustomRegOp() \
        ...     .input(0, "x", "dynamic") \
        ...     .output(0, "y") \
        ...     .dtype_format(DataType.F16_Default, DataType.F16_Default) \
        ...     .dtype_format(DataType.F32_Default, DataType.F32_Default) \
        ...     .target("Ascend") \
        ...     .get_op_info()
    """

    def __init__(self, op_name="Custom"):
        super(CustomRegOp, self).__init__(op_name)
        self.target_ = "UnKnown"

    def input(self, index=None, name=None, param_type="required", **kwargs):
        """
        Register Custom op input information.

        Args:
            index (int): Order of the input. Default: None.
            name (str): Name of the input. Default: None.
            param_type (str): Param type of the input. Default: "required".
            kwargs (dict): Other information of the input.
        """
        param_list = [index, name, param_type]
        key_list = ["index", "name", "param_type"]
        fn_list = [self._is_int, self._is_string, self._is_string]
        input_dict = self._check_param(param_list, key_list, fn_list, kwargs)
        self.inputs.append(input_dict)
        return self

    def output(self, index=None, name=None, param_type="required", **kwargs):
        """
        Register Custom op output information.

        Args:
            index (int): Order of the output. Default: None.
            name (str): Name of the output. Default: None.
            param_type (str): Param type of the output. Default: "required".
            kwargs (dict): Other information of the output.
        """
        param_list = [index, name, param_type]
        key_list = ["index", "name", "param_type"]
        fn_list = [self._is_int, self._is_string, self._is_string]
        output_dict = self._check_param(param_list, key_list, fn_list, kwargs)
        self.outputs.append(output_dict)
        return self

    def attr(self, name=None, param_type=None, value_type=None, default_value=None, **kwargs):
        """
        Register Custom op attribute information.

        Args:
            name (str): Name of the attribute. Default: None.
            param_type (str): Param type of the attribute. Default: None.
            value_type (str): Value type of the attribute. Default: None.
            default_value (str): Default value of attribute. If value is a list, each item should split by ','.
                For example, if `value_type` is "listInt", then `default_value` can be "1,2,3". Default: None.
            kwargs (dict): Other information of the attribute.
        """
        param_list = [name, param_type, value_type, default_value]
        key_list = ["name", "param_type", "type", "default_value"]
        fn_list = [self._is_string]
        attr_dict = self._check_param(param_list, key_list, fn_list, kwargs)
        self.attr_.append(attr_dict)
        return self

    def target(self, target=None):
        """
        Register Custom op target information.

        Args:
            target (str): Device target for current operator information, should be one of ["Ascend", "GPU", "CPU"].
                Default: None.
        """
        self._is_string(target)
        self.target_ = target
        return self


class DataType:
    r"""
    Various combinations of dtype and format of Ascend ops.

    current support:

    .. code-block::

        None_None = ("", "")
        None_Default = ("", "DefaultFormat")
        BOOL_None = ("bool", "")
        BOOL_Default = ("bool", "DefaultFormat")
        BOOL_5HD = ("bool", "NC1HWC0")
        BOOL_FracZ = ("bool", "FracZ")
        BOOL_FracNZ = ("bool", "FRACTAL_NZ")
        BOOL_C1HWNCoC0 = ("bool", "C1HWNCoC0")
        BOOL_NCHW = ("bool", "NCHW")
        BOOL_NHWC = ("bool", "NHWC")
        BOOL_HWCN = ("bool", "HWCN")
        BOOL_NDHWC = ("bool", "NDHWC")
        BOOL_ChannelLast = ("bool", "ChannelLast")

        I8_None = ("int8", "")
        I8_Default = ("int8", "DefaultFormat")
        I8_5HD = ("int8", "NC1HWC0")
        I8_FracZ = ("int8", "FracZ")
        I8_FracNZ = ("int8", "FRACTAL_NZ")
        I8_C1HWNCoC0 = ("int8", "C1HWNCoC0")
        I8_NCHW = ("int8", "NCHW")
        I8_NHWC = ("int8", "NHWC")
        I8_HWCN = ("int8", "HWCN")
        I8_NDHWC = ("int8", "NDHWC")
        I8_ChannelLast = ("int8", "ChannelLast")
        I8_NDC1HWC0 = ("int8", "NDC1HWC0")

        U8_None = ("uint8", "")
        U8_Default = ("uint8", "DefaultFormat")
        U8_5HD = ("uint8", "NC1HWC0")
        U8_FracZ = ("uint8", "FracZ")
        U8_FracNZ = ("uint8", "FRACTAL_NZ")
        U8_C1HWNCoC0 = ("uint8", "C1HWNCoC0")
        U8_NCHW = ("uint8", "NCHW")
        U8_NHWC = ("uint8", "NHWC")
        U8_HWCN = ("uint8", "HWCN")
        U8_NDHWC = ("uint8", "NDHWC")
        U8_ChannelLast = ("uint8", "ChannelLast")
        U8_NDC1HWC0 = ("uint8", "NDC1HWC0")

        I16_None = ("int16", "")
        I16_Default = ("int16", "DefaultFormat")
        I16_5HD = ("int16", "NC1HWC0")
        I16_FracZ = ("int16", "FracZ")
        I16_FracNZ = ("int16", "FRACTAL_NZ")
        I16_C1HWNCoC0 = ("int16", "C1HWNCoC0")
        I16_NCHW = ("int16", "NCHW")
        I16_NHWC = ("int16", "NHWC")
        I16_HWCN = ("int16", "HWCN")
        I16_NDHWC = ("int16", "NDHWC")
        I16_ChannelLast = ("int16", "ChannelLast")

        U16_None = ("uint16", "")
        U16_Default = ("uint16", "DefaultFormat")
        U16_5HD = ("uint16", "NC1HWC0")
        U16_FracZ = ("uint16", "FracZ")
        U16_FracNZ = ("uint16", "FRACTAL_NZ")
        U16_C1HWNCoC0 = ("uint16", "C1HWNCoC0")
        U16_NCHW = ("uint16", "NCHW")
        U16_NHWC = ("uint16", "NHWC")
        U16_HWCN = ("uint16", "HWCN")
        U16_NDHWC = ("uint16", "NDHWC")
        U16_ChannelLast = ("uint16", "ChannelLast")

        I32_None = ("int32", "")
        I32_Default = ("int32", "DefaultFormat")
        I32_5HD = ("int32", "NC1HWC0")
        I32_FracZ = ("int32", "FracZ")
        I32_FracNZ = ("int32", "FRACTAL_NZ")
        I32_C1HWNCoC0 = ("int32", "C1HWNCoC0")
        I32_NCHW = ("int32", "NCHW")
        I32_NHWC = ("int32", "NHWC")
        I32_HWCN = ("int32", "HWCN")
        I32_NDHWC = ("int32", "NDHWC")
        I32_ChannelLast = ("int32", "ChannelLast")

        U32_None = ("uint32", "")
        U32_Default = ("uint32", "DefaultFormat")
        U32_5HD = ("uint32", "NC1HWC0")
        U32_FracZ = ("uint32", "FracZ")
        U32_FracNZ = ("uint32", "FRACTAL_NZ")
        U32_C1HWNCoC0 = ("uint32", "C1HWNCoC0")
        U32_NCHW = ("uint32", "NCHW")
        U32_NHWC = ("uint32", "NHWC")
        U32_HWCN = ("uint32", "HWCN")
        U32_NDHWC = ("uint32", "NDHWC")
        U32_ChannelLast = ("uint32", "ChannelLast")

        I64_None = ("int64", "")
        I64_Default = ("int64", "DefaultFormat")
        I64_5HD = ("int64", "NC1HWC0")
        I64_FracZ = ("int64", "FracZ")
        I64_FracNZ = ("int64", "FRACTAL_NZ")
        I64_C1HWNCoC0 = ("int64", "C1HWNCoC0")
        I64_NCHW = ("int64", "NCHW")
        I64_NHWC = ("int64", "NHWC")
        I64_HWCN = ("int64", "HWCN")
        I64_NDHWC = ("int64", "NDHWC")
        I64_ChannelLast = ("int64", "ChannelLast")

        U64_None = ("uint64", "")
        U64_Default = ("uint64", "DefaultFormat")
        U64_5HD = ("uint64", "NC1HWC0")
        U64_FracZ = ("uint64", "FracZ")
        U64_FracNZ = ("uint64", "FRACTAL_NZ")
        U64_C1HWNCoC0 = ("uint64", "C1HWNCoC0")
        U64_NCHW = ("uint64", "NCHW")
        U64_NHWC = ("uint64", "NHWC")
        U64_HWCN = ("uint64", "HWCN")
        U64_NDHWC = ("uint64", "NDHWC")
        U64_ChannelLast = ("uint64", "ChannelLast")

        F16_None = ("float16", "")
        F16_Default = ("float16", "DefaultFormat")
        F16_5HD = ("float16", "NC1HWC0")
        F16_FracZ = ("float16", "FracZ")
        F16_FracNZ = ("float16", "FRACTAL_NZ")
        F16_C1HWNCoC0 = ("float16", "C1HWNCoC0")
        F16_NCHW = ("float16", "NCHW")
        F16_NHWC = ("float16", "NHWC")
        F16_HWCN = ("float16", "HWCN")
        F16_NDHWC = ("float16", "NDHWC")
        F16_NCDHW = ("float16", "NCDHW")
        F16_DHWCN = ("float16", "DHWCN")
        F16_NDC1HWC0 = ("float16", "NDC1HWC0")
        F16_FRACTAL_Z_3D = ("float16", "FRACTAL_Z_3D")
        F16_FracZNLSTM = ("float16", "FRACTAL_ZN_LSTM")
        F16_FracZNRNN = ("float16", "FRACTAL_ZN_RNN")
        F16_ND_RNNBIAS = ("float16", "ND_RNN_BIAS")
        F16_ChannelLast = ("float16", "ChannelLast")

        F32_None = ("float32", "")
        F32_Default = ("float32", "DefaultFormat")
        F32_5HD = ("float32", "NC1HWC0")
        F32_FracZ = ("float32", "FracZ")
        F32_FracNZ = ("float32", "FRACTAL_NZ")
        F32_C1HWNCoC0 = ("float32", "C1HWNCoC0")
        F32_NCHW = ("float32", "NCHW")
        F32_NHWC = ("float32", "NHWC")
        F32_HWCN = ("float32", "HWCN")
        F32_NDHWC = ("float32", "NDHWC")
        F32_NCDHW = ("float32", "NCDHW")
        F32_DHWCN = ("float32", "DHWCN")
        F32_NDC1HWC0 = ("float32", "NDC1HWC0")
        F32_FRACTAL_Z_3D = ("float32", "FRACTAL_Z_3D")
        F32_FracZNLSTM = ("float32", "FRACTAL_ZN_LSTM")
        F32_FracZNRNN = ("float32", "FRACTAL_ZN_RNN")
        F32_ND_RNNBIAS = ("float32", "ND_RNN_BIAS")
        F32_ChannelLast = ("float32", "ChannelLast")

        F64_None = ("float64", "")
        F64_Default = ("float64", "DefaultFormat")
        F64_5HD = ("float64", "NC1HWC0")
        F64_FracZ = ("float64", "FracZ")
        F64_FracNZ = ("float64", "FRACTAL_NZ")
        F64_C1HWNCoC0 = ("float64", "C1HWNCoC0")
        F64_NCHW = ("float64", "NCHW")
        F64_NHWC = ("float64", "NHWC")
        F64_HWCN = ("float64", "HWCN")
        F64_NDHWC = ("float64", "NDHWC")
        F64_ChannelLast = ("float64", "ChannelLast")

        C64_Default = ("complex64", "DefaultFormat")
        C128_Default = ("complex128", "DefaultFormat")
    """

    None_None = ("", "")
    None_Default = ("", "DefaultFormat")
    BOOL_None = ("bool", "")
    BOOL_Default = ("bool", "DefaultFormat")
    BOOL_5HD = ("bool", "NC1HWC0")
    BOOL_FracZ = ("bool", "FracZ")
    BOOL_FracNZ = ("bool", "FRACTAL_NZ")
    BOOL_C1HWNCoC0 = ("bool", "C1HWNCoC0")
    BOOL_NCHW = ("bool", "NCHW")
    BOOL_NHWC = ("bool", "NHWC")
    BOOL_HWCN = ("bool", "HWCN")
    BOOL_NDHWC = ("bool", "NDHWC")
    BOOL_ChannelLast = ("bool", "ChannelLast")

    I8_None = ("int8", "")
    I8_Default = ("int8", "DefaultFormat")
    I8_5HD = ("int8", "NC1HWC0")
    I8_FracZ = ("int8", "FracZ")
    I8_FracNZ = ("int8", "FRACTAL_NZ")
    I8_C1HWNCoC0 = ("int8", "C1HWNCoC0")
    I8_NCHW = ("int8", "NCHW")
    I8_NHWC = ("int8", "NHWC")
    I8_HWCN = ("int8", "HWCN")
    I8_NDHWC = ("int8", "NDHWC")
    I8_ChannelLast = ("int8", "ChannelLast")
    I8_NDC1HWC0 = ("int8", "NDC1HWC0")

    U8_None = ("uint8", "")
    U8_Default = ("uint8", "DefaultFormat")
    U8_5HD = ("uint8", "NC1HWC0")
    U8_FracZ = ("uint8", "FracZ")
    U8_FracNZ = ("uint8", "FRACTAL_NZ")
    U8_C1HWNCoC0 = ("uint8", "C1HWNCoC0")
    U8_NCHW = ("uint8", "NCHW")
    U8_NHWC = ("uint8", "NHWC")
    U8_HWCN = ("uint8", "HWCN")
    U8_NDHWC = ("uint8", "NDHWC")
    U8_ChannelLast = ("uint8", "ChannelLast")
    U8_NDC1HWC0 = ("uint8", "NDC1HWC0")

    I16_None = ("int16", "")
    I16_Default = ("int16", "DefaultFormat")
    I16_5HD = ("int16", "NC1HWC0")
    I16_FracZ = ("int16", "FracZ")
    I16_FracNZ = ("int16", "FRACTAL_NZ")
    I16_C1HWNCoC0 = ("int16", "C1HWNCoC0")
    I16_NCHW = ("int16", "NCHW")
    I16_NHWC = ("int16", "NHWC")
    I16_HWCN = ("int16", "HWCN")
    I16_NDHWC = ("int16", "NDHWC")
    I16_ChannelLast = ("int16", "ChannelLast")

    U16_None = ("uint16", "")
    U16_Default = ("uint16", "DefaultFormat")
    U16_5HD = ("uint16", "NC1HWC0")
    U16_FracZ = ("uint16", "FracZ")
    U16_FracNZ = ("uint16", "FRACTAL_NZ")
    U16_C1HWNCoC0 = ("uint16", "C1HWNCoC0")
    U16_NCHW = ("uint16", "NCHW")
    U16_NHWC = ("uint16", "NHWC")
    U16_HWCN = ("uint16", "HWCN")
    U16_NDHWC = ("uint16", "NDHWC")
    U16_ChannelLast = ("uint16", "ChannelLast")

    I32_None = ("int32", "")
    I32_Default = ("int32", "DefaultFormat")
    I32_5HD = ("int32", "NC1HWC0")
    I32_FracZ = ("int32", "FracZ")
    I32_FracNZ = ("int32", "FRACTAL_NZ")
    I32_C1HWNCoC0 = ("int32", "C1HWNCoC0")
    I32_NCHW = ("int32", "NCHW")
    I32_NHWC = ("int32", "NHWC")
    I32_HWCN = ("int32", "HWCN")
    I32_NDHWC = ("int32", "NDHWC")
    I32_ChannelLast = ("int32", "ChannelLast")

    U32_None = ("uint32", "")
    U32_Default = ("uint32", "DefaultFormat")
    U32_5HD = ("uint32", "NC1HWC0")
    U32_FracZ = ("uint32", "FracZ")
    U32_FracNZ = ("uint32", "FRACTAL_NZ")
    U32_C1HWNCoC0 = ("uint32", "C1HWNCoC0")
    U32_NCHW = ("uint32", "NCHW")
    U32_NHWC = ("uint32", "NHWC")
    U32_HWCN = ("uint32", "HWCN")
    U32_NDHWC = ("uint32", "NDHWC")
    U32_ChannelLast = ("uint32", "ChannelLast")

    I64_None = ("int64", "")
    I64_Default = ("int64", "DefaultFormat")
    I64_5HD = ("int64", "NC1HWC0")
    I64_FracZ = ("int64", "FracZ")
    I64_FracNZ = ("int64", "FRACTAL_NZ")
    I64_C1HWNCoC0 = ("int64", "C1HWNCoC0")
    I64_NCHW = ("int64", "NCHW")
    I64_NHWC = ("int64", "NHWC")
    I64_HWCN = ("int64", "HWCN")
    I64_NDHWC = ("int64", "NDHWC")
    I64_ChannelLast = ("int64", "ChannelLast")

    U64_None = ("uint64", "")
    U64_Default = ("uint64", "DefaultFormat")
    U64_5HD = ("uint64", "NC1HWC0")
    U64_FracZ = ("uint64", "FracZ")
    U64_FracNZ = ("uint64", "FRACTAL_NZ")
    U64_C1HWNCoC0 = ("uint64", "C1HWNCoC0")
    U64_NCHW = ("uint64", "NCHW")
    U64_NHWC = ("uint64", "NHWC")
    U64_HWCN = ("uint64", "HWCN")
    U64_NDHWC = ("uint64", "NDHWC")
    U64_ChannelLast = ("uint64", "ChannelLast")

    F16_None = ("float16", "")
    F16_Default = ("float16", "DefaultFormat")
    F16_5HD = ("float16", "NC1HWC0")
    F16_FracZ = ("float16", "FracZ")
    F16_FracNZ = ("float16", "FRACTAL_NZ")
    F16_C1HWNCoC0 = ("float16", "C1HWNCoC0")
    F16_NCHW = ("float16", "NCHW")
    F16_NHWC = ("float16", "NHWC")
    F16_HWCN = ("float16", "HWCN")
    F16_NDHWC = ("float16", "NDHWC")
    F16_NCDHW = ("float16", "NCDHW")
    F16_DHWCN = ("float16", "DHWCN")
    F16_NDC1HWC0 = ("float16", "NDC1HWC0")
    F16_FRACTAL_Z_3D = ("float16", "FRACTAL_Z_3D")
    F16_FracZNLSTM = ("float16", "FRACTAL_ZN_LSTM")
    F16_FracZNRNN = ("float16", "FRACTAL_ZN_RNN")
    F16_ND_RNNBIAS = ("float16", "ND_RNN_BIAS")
    F16_ChannelLast = ("float16", "ChannelLast")

    F32_None = ("float32", "")
    F32_Default = ("float32", "DefaultFormat")
    F32_5HD = ("float32", "NC1HWC0")
    F32_FracZ = ("float32", "FracZ")
    F32_FracNZ = ("float32", "FRACTAL_NZ")
    F32_C1HWNCoC0 = ("float32", "C1HWNCoC0")
    F32_NCHW = ("float32", "NCHW")
    F32_NHWC = ("float32", "NHWC")
    F32_HWCN = ("float32", "HWCN")
    F32_NDHWC = ("float32", "NDHWC")
    F32_NCDHW = ("float32", "NCDHW")
    F32_DHWCN = ("float32", "DHWCN")
    F32_NDC1HWC0 = ("float32", "NDC1HWC0")
    F32_FRACTAL_Z_3D = ("float32", "FRACTAL_Z_3D")
    F32_FracZNLSTM = ("float32", "FRACTAL_ZN_LSTM")
    F32_FracZNRNN = ("float32", "FRACTAL_ZN_RNN")
    F32_ND_RNNBIAS = ("float32", "ND_RNN_BIAS")
    F32_ChannelLast = ("float32", "ChannelLast")

    F64_None = ("float64", "")
    F64_Default = ("float64", "DefaultFormat")
    F64_5HD = ("float64", "NC1HWC0")
    F64_FracZ = ("float64", "FracZ")
    F64_FracNZ = ("float64", "FRACTAL_NZ")
    F64_C1HWNCoC0 = ("float64", "C1HWNCoC0")
    F64_NCHW = ("float64", "NCHW")
    F64_NHWC = ("float64", "NHWC")
    F64_HWCN = ("float64", "HWCN")
    F64_NDHWC = ("float64", "NDHWC")
    F64_ChannelLast = ("float64", "ChannelLast")

    C64_Default = ("complex64", "DefaultFormat")
    C128_Default = ("complex128", "DefaultFormat")
