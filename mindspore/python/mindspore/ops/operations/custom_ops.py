# Copyright 2021-2022 Huawei Technologies Co., Ltd
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

"""Custom operator"""
import json
import os
import re
import ast
import hashlib
import inspect
import numpy as np
from mindspore._c_expression import Oplib, typing
from mindspore import context
from mindspore.common import Tensor
from mindspore.common import dtype as mstype
from mindspore.ops import DataType
from mindspore import log as logger
from mindspore import ops
from ._ms_hybrid import determine_variable_usage
from ._custom_grad import autodiff_bprop
from ._pyfunc_registry import add_pyfunc


class Custom(ops.PrimitiveWithInfer):
    r"""
    `Custom` primitive is used for user defined operators and is to enhance the expressive ability of built-in
    primitives. You can construct a `Custom` object with a predefined function, which describes the computation
    logic of a user defined operator. You can also construct another `Custom` object with another predefined
    function if needed. Then these `Custom` objects can be directly used in neural networks.

    .. warning::
        This is an experimental prototype that is subject to change.

    Args:
        func (Union[function, str]):

            - function: If func is of function type, then func should be a Python function which describes the
              computation logic of a user defined operator. The function can be one of the following:

              1. A AKG operator implementation function, which can use ir builder/tvm compute/hybrid grammar.
              2. A TBE operator implementation function.
              3. A pure python function
              4. An ms_hybrid decorated function written by the Hybrid DSL.

            - str: If func is of str type, then str should be a path of file along with a function name.
              This could be used when func_type is "aot" or "julia".

              1. for "aot":

                 Currently "aot" supports GPU/CPU(linux only) platform.
                 "aot" means ahead of time, in which case Custom directly launches user defined "xxx.so" file as an
                 operator. Users need to compile a handwriting "xxx.cu"/"xxx.cc" file into "xxx.so" ahead of time,
                 and offer the path of the file along with a function name.

                 - "xxx.so" file generation:

                   1) GPU Platform: Given user defined "xxx.cu" file (ex. "{path}/add.cu"), use nvcc command to compile
                   it.(ex. "nvcc --shared -Xcompiler -fPIC -o add.so add.cu")

                   2) CPU Platform: Given user defined "xxx.cc" file (ex. "{path}/add.cc"), use g++/gcc command to
                   compile it.(ex. "g++ --shared -fPIC  -o add.so add.cc")

                 - Define a "xxx.cc"/"xxx.cu" file:

                   "aot" is a cross-platform identity. The functions defined in "xxx.cc" or "xxx.cu" share
                   the same args. Typically, the function should be as:

                   .. code-block::

                       int func(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes,
                               void *stream, void *extra)

                   Parameters:

                   - nparam(int): total number of inputs plus outputs; suppose the operator has 2 inputs and 3 outputs,
                     then nparam=5
                   - params(void \*\*): a pointer to the array of inputs and outputs' pointer; the pointer type of
                     inputs and outputs is void \* ; suppose the operator has 2 inputs and 3 outputs, then the first
                     input's pointer is params[0] and the second output's pointer is params[3]
                   - ndims(int \*): a pointer to the array of inputs and outputs' dimension num; suppose params[i] is a
                     1024x1024 tensor and params[j] is a 77x83x4 tensor, then ndims[i]=2, ndims[j]=3.
                   - shapes(int64_t \*\*): a pointer to the array of inputs and outputs' shapes(int64_t \*); the ith
                     input's jth dimension's size is shapes[i][j](0<=j<ndims[i]); suppose params[i] is a 2x3 tensor and
                     params[j] is a 3x3x4 tensor, then shapes[i][0]=2, shapes[j][2]=4.
                   - dtypes(const char \*\*): a pointer to the array of inputs and outputs' types(const char \*);
                     (ex. "float32", "float16", "float", "float64", "int", "int8", "int16", "int32", "int64", "uint",
                     "uint8", "uint16", "uint32", "uint64", "bool")
                   - stream(void \*): stream pointer, only used in cuda file
                   - extra(void \*): used for further extension

                   Return Value(int):

                   - 0: MindSpore will continue to run if this aot kernel is successfully executed
                   - others: MindSpore will raise exception and exit

                   Examples: see details in tests/st/ops/graph_kernel/custom/aot_test_files/

                 - Use it in Custom:

                   .. code-block::

                       Custom(func="{dir_path}/{file_name}:{func_name}",...)
                       (ex. Custom(func="./reorganize.so:CustomReorganize", out_shape=[1], out_dtype=mstype.float32,
                       "aot"))

              2. for "julia":

                 Currently "julia" supports CPU(linux only) platform.
                 For julia use JIT compiler, and julia support c api to call julia code.
                 The Custom can directly launches user defined "xxx.jl" file as an operator.
                 Users need to write a "xxx.jl" file which include modules and functions,
                 and offer the path of the file along with a module name and function name.

                 Examples: see details in tests/st/ops/graph_kernel/custom/julia_test_files/

                 - Use it in Custom:

                   .. code-block::

                       Custom(func="{dir_path}/{file_name}:{module_name}:{func_name}",...)
                       (ex. Custom(func="./add.jl:Add:add", out_shape=[1], out_dtype=mstype.float32, "julia"))

        out_shape (Union[function, list, tuple]): The output shape infer function or the value of output shape of
            `func`. Default: None.

            If func has single output, then the value of output shape is a list or tuple of int.

            If func has multiple outputs, then the value of output shape is a tuple, each item represents the shape
            of each output.

            The input can be None only when the func_type input is "hybrid". In this case, the automatic infer
            shape mechanic will be enabled.

        out_dtype (Union[function, :class:`mindspore.dtype`, tuple[:class:`mindspore.dtype`]]): The output data type
            infer function or the value of output data type of `func`. Default: None.

            If func has single output, then the value of output shape is a `mindspore.dtype`.

            If func has multiple outputs, then the value of output shape is a tuple of `mindspore.dtype`, each item
            represents the data type of each output.

            The input can be None only when the func_type input is "hybrid". In this case, the automatic infer
            value mechanic will be enabled.

        func_type (str): The implementation type of `func`, should be one of

            ["hybrid", "akg", "tbe", "aot", "pyfunc", "julia", "aicpu"].

            Each `func_type` only supports specific platforms(targets). Default: "hybrid".
            The supported platforms of `func_type`:

            - "hybrid": supports ["Ascend", "GPU"].
            - "akg": supports ["Ascend", "GPU"].
            - "tbe": supports ["Ascend"].
            - "aot": supports ["GPU", "CPU"].
            - "pyfunc": supports ["CPU"].
            - "julia": supports ["CPU"].
            - "aicpu": supports ["Ascend"].

        bprop (function): The back propagation function of `func`. Default: None.
        reg_info (Union[str, dict, list, tuple]): Represents the registration information(reg info) of `func` with
            json format of type str or dict. The reg info specifies supported data types and formats of inputs and
            outputs, attributes and target of `func`. Default: None.

            If reg info is a list or tuple, then each item should be with json format of type str or dict, which
            represents the registration information of `func` in a specific target. You need to invoke `CustomRegOp`
            or the subclass of `RegOp` to generate the reg info for `func`. Then you can invoke
            `custom_info_register` to bind the reg info to `func` or just pass the reg info to `reg_info` parameter.
            The `reg_info` parameter takes higher priority than `custom_info_register` and the reg info in a
            specific target will be registered only once.

            If reg info is not set, then we will infer the data types and formats from the inputs of `Custom` operator.

            Please note that, if `func_type` is "tbe" or the `func` only supports some specified data types and formats,
            or it has attribute inputs, then you should set the reg info for `func`.

    Inputs:
        - **input** (Union(tuple, list)) - The input tuple or list is made up of multiple tensors, and attributes
          value(optional).

    Outputs:
        Tensor or tuple[Tensor], execution results.

    Raises:
        TypeError: If the type of `func` is invalid or the type of register information for `func` is invalid.
        ValueError: If `func_type` is invalid.
        ValueError: If the register information is invalid, including the target is not supported, the input numbers
            or the attributes of `func` differs in different targets.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.ops as ops
        >>> import numpy as np
        >>> from mindspore.ops import CustomRegOp, custom_info_register, DataType, ms_hybrid
        >>> from mindspore.common import dtype as mstype
        >>> from mindspore.nn import Cell
        >>> input_x = Tensor(np.ones([16, 16]).astype(np.float32))
        >>> input_y = Tensor(np.ones([16, 16]).astype(np.float32))
        >>>
        >>> # Example, func_type = "hybrid"
        >>> # This is the default func_type in Custom,
        >>> # and both out_shape and out_dtype can be None(default value).
        >>> # In this case, the input func must be a function written in the Hybrid DSL
        >>> # and decorated by @ms_hybrid.
        >>> @ms_hybrid
        ... def outer_product_script(a, b):
        ...     c = output_tensor(a.shape, a.dtype)
        ...     for i0 in range(a.shape[0]):
        ...         for i1 in range(b.shape[1]):
        ...             c[i0, i1] = 0.0
        ...             for i2 in range(a.shape[1]):
        ...                 c[i0, i1] = c[i0, i1] + (a[i0, i2] * b[i2, i1])
        ...     return c
        >>>
        >>> test_op_hybrid = ops.Custom(outer_product_script)
        >>> output = test_op_hybrid(input_x, input_y)
        >>> # the result will be a 16 * 16 tensor with all elements 16
        >>> print(output.shape)
        (16, 16)
        >>> # Example, func_type = "akg"
        >>> def outer_product(a_1, b_1):
        ...     d = output_tensor(a_1.shape, a_1.dtype)
        ...     for i0 in range(a_1.shape[0]):
        ...         for i1 in range(b_1.shape[1]):
        ...             d[i0, i1] = 0.0
        ...             for i2 in range(a.shape[1]):
        ...                 d[i0, i1] = d[i0, i1] + (a_1[i0, i2] * b_1[i2, i1])
        ...     return d
        >>>
        >>> # Example, func_type = "tbe"
        >>> square_with_bias_op_info = CustomRegOp() \
        ...     .fusion_type("OPAQUE") \
        ...     .attr("bias", "required", "float") \
        ...     .input(0, "x") \
        ...     .output(0, "y") \
        ...     .dtype_format(DataType.F32_Default, DataType.F32_Default) \
        ...     .dtype_format(DataType.F16_Default, DataType.F16_Default) \
        ...     .target("Ascend") \
        ...     .get_op_info()
        >>>
        >>> @custom_info_register(square_with_bias_op_info)
        ... def square_with_bias(input_x, output_y, bias=0.0, kernel_name="square_with_bias"):
        ...     import te.lang.cce
        ...     from te import tvm
        ...     from topi.cce import util
        ...
        ...     shape = input_x.get("shape")
        ...     dtype = input_x.get("dtype").lower()
        ...
        ...     shape = util.shape_refine(shape)
        ...     data = tvm.placeholder(shape, name="data", dtype=dtype)
        ...
        ...     with tvm.target.cce():
        ...         res0 = te.lang.cce.vmul(data, data)
        ...         res = te.lang.cce.vadds(res0, bias)
        ...         sch = te.lang.cce.auto_schedule(res)
        ...
        ...     config = {"print_ir": False,
        ...               "name": kernel_name,
        ...               "tensor_list": [data, res]}
        ...
        ...     te.lang.cce.cce_build_code(sch, config)
        >>>
        >>> def test_tbe():
        ...     square_with_bias = ops.Custom(square_with_bias, out_shape=lambda x, _: x, \
        ...                                   out_dtype=lambda x, _: x, func_type="tbe")
        ...     res = self.square_with_bias(input_x, 1.0)
        ...     return res
        >>>
        >>> # Example, func_type = "aicpu"
        >>> resize_bilinear_op_info = CustomRegOp("ResizeBilinear") \
        ...     .fusion_type("OPAQUE") \
        ...     .input(0, "input", "required") \
        ...     .output(1, "output", "required") \
        ...     .attr("align_corners", "required", "bool") \
        ...     .attr("cust_aicpu", "optional", "str", "aicpu_kernels") \
        ...     .dtype_format(DataType.F32_Default, DataType.F32_Default) \
        ...     .dtype_format(DataType.F16_Default, DataType.F32_Default) \
        ...     .target("Ascend") \
        ...     .get_op_info()
        >>>
        >>> @custom_info_register(resize_bilinear_op_info)
        ... def resize_bilinear_aicpu():
        ...     return
        >>>
        >>> def test_aicpu(x):
        ...     resize_bilinear_op = ops.Custom(resize_bilinear_aicpu, out_shape=[1, 1, 9, 9], \
        ...                                     out_dtype=mstype.float32, func_type="aicpu")
        ...     res = resize_bilinear_op(x, True, "aicpu_kernels")
        ...     return res
        >>>
        >>> # Example, func_type = "aot"
        >>> def test_aot(x, y, out_shapes, out_types):
        ...     program = ops.Custom("./reorganize.so:CustomReorganize", out_shapes, out_types, "aot")
        ...     out = program(x, y)
        ...     return out
        >>>
        >>> # Example, func_type = "pyfunc"
        >>> def func_multi_output(x1, x2):
        ...     return (x1 + x2), (x1 - x2)
        >>>
        >>> test_pyfunc = ops.Custom(func_multi_output, lambda x, _: (x, x), lambda x, _: (x, x), "pyfunc")
        >>> output = test_pyfunc(input_x, input_y)
        >>>
        >>> # Example, func_type = "julia"
        >>> # julia code:
        >>> # add.jl
        >>> # module Add
        >>> # function add(x, y, z)
        >>> #   z .= x + y
        >>> #   return z
        >>> # end
        >>> # end
        >>> def test_julia(x, y, out_shapes, out_types):
        ...     program = ops.Custom("./add.jl:Add:add", out_shapes, out_types, "julia")
        ...     out = program(x, y)
        ...     return out
    """

    registered_func = {}
    attr_dict = {}  # Save input_names and attr_names for func.

    def __init__(self, func, out_shape=None, out_dtype=None, func_type="hybrid", bprop=None, reg_info=None):
        ops.PrimitiveWithInfer.__init__(self, "Custom")

        self.supported_targets = ["Ascend", "GPU", "CPU"]
        self.supported_func_type = ["hybrid", "akg", "tbe", "aicpu", "aot", "pyfunc", "julia"]
        self.func = func
        self.func_type = func_type
        self.func_name = ""
        self.uniq_name = ""
        self.imply_path = ""
        self.func_source_str = ""
        self._func_compile_attrs = {}
        self._is_ms_hybrid = False

        self._check_func()
        self._update_func_info()
        self.add_prim_attr("func_name", self.func_name)
        self.add_prim_attr("uniq_name", self.uniq_name)
        self.add_prim_attr("func_compile_attrs", self._func_compile_attrs)

        self.add_prim_attr("imply_path", self.imply_path)
        if self.func_type == "pyfunc":
            func_id = id(self.func)
            add_pyfunc(func_id, self.func)
            self.add_prim_attr("fn_id", func_id)

        self.out_shape = out_shape
        self.out_dtype = out_dtype
        self.bprop = bprop
        self.fake_output = False
        self.single_scalar_output = False
        if not self.out_dtype:
            self.fake_output = True
        elif not self.out_shape:
            self.single_scalar_output = True
        self.add_prim_attr("fake_output", self.fake_output)
        self.add_prim_attr("single_scalar_output", self.single_scalar_output)

        # Register info
        self._register_info(reg_info)

        if func_type == "akg":
            self.add_prim_attr('func_source_str', self.func_source_str)
            if "ir_builder" in self.func_source_str:
                self.func_type = "ir_builder"
            elif "compute" in self.func_source_str:
                self.func_type = "tvm_compute"
            else:
                self.func_type = "hybrid"
                self._hybrid_func_analyser()

        if not self.bprop and self.func_type == "hybrid":
            self._hybrid_autodiff(func_type)

        self.add_prim_attr("func_type", self.func_type)
        self._update_attr()

    def get_bprop(self):
        return self.bprop

    def _check_julia_func(self):
        """Check the validity of julia func"""
        if not isinstance(self.func, str):
            raise TypeError("{} func should be of type str, but got {}".format(
                self.func_type, type(self.func)))
        if self.func.count(':') != 2:
            raise Exception("func format in julia custom op should be file:module:func.")
        file, module, func = self.func.split(':')
        with open(file, 'r') as f:
            jl = f.read()
            if 'module ' + module not in jl:
                raise Exception("module: " + module + " not found!!!")
            if 'function ' + func not in jl:
                raise Exception("function: " + func + " not found!!!")

    def _check_func(self):
        """Check the validity of func_type and type of func"""
        if self.func_type not in self.supported_func_type:
            raise ValueError("func_type should be one of {}, but got {}"
                             .format(self.supported_func_type, self.func_type))
        if self.func_type == "aot":
            if not isinstance(self.func, str):
                raise TypeError("{} func should be of type str, but got {}".format(
                    self.func_type, type(self.func)))
        elif self.func_type == "julia":
            self._check_julia_func()
        elif self.func_type == "hybrid":
            if not hasattr(self.func, "ms_hybrid_flag"):
                raise TypeError(
                    "To use the mode hybrid, the input func should a function decorated by ms_hybrid")
            self._is_ms_hybrid = True
            self._func_compile_attrs = getattr(self.func, "compile_attrs", {})
        elif self.func_type == "akg":
            if hasattr(self.func, "ms_hybrid_flag"):
                logger.warning("To have a better user experience, the mode hybrid is suggested "
                               "for the input function with decorator @ms_hybrid"
                               "To enable this mode, set the func_type to be \"hybrid\"")
        elif self.func_type == "pyfunc":
            if hasattr(self.func, "ms_hybrid_flag"):
                logger.warning("Now you are using the function with decorator @ms_hybrid in the mode pyfunc"
                               "The kernel will be executed as a native python function, which might lead to "
                               "low efficiency. To accelerate the kernel, set the func_type to be \"ms_hybrid\""
                               )
        else:
            if not callable(self.func):
                raise TypeError("{} func should be of type function, but got {}"
                                .format(self.func_type, type(self.func)))

    def _update_func_info(self):
        """Update information of func"""
        if callable(self.func):
            # For the func_type other then hybrid, get the original function if func is decorated
            if "__wrapped__" in self.func.__dict__ and not self.func_type in ["hybrid", "pyfunc"]:
                self.func = self.func.__dict__["__wrapped__"]
            # func name
            self.func_name = self.func.__name__
            # path of func
            self.imply_path = os.path.realpath(inspect.getfile(self.func))
            # source code of func, not include the decorator before def
            self.func_source_str = inspect.getsource(self.func)
            index = self.func_source_str.find("def ")
            if index != -1:
                self.func_source_str = self.func_source_str[index:]

            if self._is_ms_hybrid:
                # static check for the Hybrid DSL in hybrid
                root = ast.parse(self.func_source_str)
                inplace_assign_output = determine_variable_usage(root, self.func_name)
                if inplace_assign_output:
                    self.add_prim_attr("inplace_assign_output",
                                       " ".join((str(j) for i in inplace_assign_output for j in i)))
                self.add_prim_attr('func_source_str', self.func_source_str)

            # unique func name
            sha256 = hashlib.sha256()
            sha256.update(self.imply_path.encode("utf-8"))
            sha256.update(self.func_source_str.encode("utf-8"))
            hash_str = sha256.hexdigest()
            self.uniq_name = self.name + "_" + self.func_name + "_" + hash_str
        elif isinstance(self.func, str):
            # func name
            self.func_name = self.func
            # uniq func name
            self.uniq_name = self.name + "_" + self.func_name
        else:
            raise TypeError("func should be of type function or str, but got {}".format(type(self.func)))

    def _register_info(self, info):
        """Register reg_info."""
        reg_info = info
        if reg_info is None and hasattr(self.func, "reg_info"):
            reg_info = getattr(self.func, "reg_info")
        if self.func_type == "aicpu" and reg_info is None:
            raise ValueError("custom aicpu ops must set reg_info, but current reg_info is None.")
        reg_info_list = self._get_expanded_list(reg_info)
        for reg_info in reg_info_list:
            if not isinstance(reg_info, (str, dict)):
                continue
            if isinstance(reg_info, str):
                reg_info = json.loads(reg_info)
            if self.fake_output:
                reg_info["outputs"].append(dict({"index": 0, "name": "y", "param_type": "required"}))
                new_dtype_format = []
                for i in reg_info["dtype_format"]:
                    new_dtype_format.append(i + (DataType.I32_Default,))
                reg_info["dtype_format"] = new_dtype_format
            target = self._get_target(reg_info)
            # Reg info for func is only registered once for a certain target
            if self._has_registered(target):
                continue
            # Register
            reg_info = self._reformat_reg_info(reg_info, target)
            reg_info_str = json.dumps(reg_info)
            op_lib = Oplib()
            if not op_lib.reg_op(reg_info_str, self.imply_path):
                raise ValueError('Invalid reg info {}: {}\n'.format(self.imply_path, reg_info_str))
            self._save_attr(reg_info)
            self._save_register_status(target)

    def _get_expanded_list(self, data):
        """Recursive function to parse elements in list or tuple."""
        data_list = []
        if isinstance(data, (list, tuple)):
            for i in data:
                tmp_list = self._get_expanded_list(i)
                for ii in tmp_list:
                    data_list.append(ii)
        else:
            data_list.append(data)
        return data_list

    def _get_registered_targets(self):
        """Get the registered targets of func."""
        targets = []
        if callable(self.func):
            targets = getattr(self.func, "registered_targets", [])
        elif isinstance(self.func, str):
            targets = Custom.registered_func.get(self.func, [])
        if not isinstance(targets, list):
            targets = [targets]
        return targets

    def _has_registered(self, target):
        """Check if registration information is registered in target."""
        registered_targets = self._get_registered_targets()
        return target in registered_targets

    def _save_register_status(self, target):
        """Save registration status for target."""
        if callable(self.func):
            registered_targets = getattr(self.func, "registered_targets", [])
            registered_targets.append(target)
            setattr(self.func, "registered_targets", registered_targets)
        elif isinstance(self.func, str):
            if isinstance(Custom.registered_func.get(self.func), list):
                Custom.registered_func[self.func].append(target)
            else:
                Custom.registered_func[self.func] = [target]

    def _get_op_name(self, reg_info):
        if self.func_type == "aicpu":
            self.uniq_name = reg_info["op_name"]
            self.add_prim_attr("uniq_name", self.uniq_name)
        return self.uniq_name

    def _reformat_reg_info(self, reg_info, target):
        """Reformat registration information."""
        if not isinstance(reg_info, dict):
            raise TypeError("reg_info should be of type dict, but got {}".format(type(reg_info)))
        reg_info["op_name"] = self._get_op_name(reg_info)
        reg_info["imply_type"] = self._get_imply_type(reg_info, target)
        if not isinstance(reg_info.get("fusion_type"), str) or not reg_info["fusion_type"].strip():
            reg_info["fusion_type"] = "OPAQUE"
        # Supplement necessary info for TBE if these information is missing in reg_info
        if reg_info["imply_type"] == "TBE":
            if reg_info.get("attr") is not None and isinstance(reg_info["attr"], list):
                for i, item in enumerate(reg_info["attr"]):
                    if isinstance(item, dict) and item.get("value") is None:
                        reg_info["attr"][i]["value"] = "all"
            reg_info["async_flag"] = reg_info.get("async_flag", False)
            reg_info["binfile_name"] = "%s.so" % self.func_name
            reg_info["compute_cost"] = reg_info.get("compute_cost", 10)
            reg_info["kernel_name"] = self.func_name
            reg_info["partial_flag"] = reg_info.get("partial_flag", True)
            reg_info["need_check_supported"] = reg_info.get("need_check_supported", False)
        # Supplement necessary info for AKG if these information is missing in reg_info
        if reg_info["imply_type"] == "AKG":
            target_to_processor = {"Ascend": "AiCore", "GPU": "CUDA", "CPU": "CPU"}
            reg_info["processor"] = reg_info.get("processor", target_to_processor.get(target))
        return reg_info

    def _get_target(self, reg_info):
        """Get target information."""
        target = None
        if isinstance(reg_info, dict):
            # Get target from reg_info["target"]
            target = reg_info.get("target")
            # Infer target from reg_info["processor"], reg_info generated from AkgGpuRegOp or AkgAscendRegOp
            #   will have the processor information.
            if target not in self.supported_targets:
                processor_to_target = {"AiCore": "Ascend", "CUDA": "GPU", "CPU": "CPU"}
                target = processor_to_target.get(reg_info.get("processor"))
            # Infer target from reg_info["imply_type"]
            if target not in self.supported_targets:
                imply_type_to_target = {"TBE": "Ascend", "GPU": "GPU", "CPU": "CPU"}
                target = imply_type_to_target.get(reg_info.get("imply_type"))
        # Infer target from func_type
        if target not in self.supported_targets:
            func_type_to_target = {"tbe": "Ascend", "pyfunc": "CPU"}
            target = func_type_to_target.get(self.func_type)
        if target not in self.supported_targets:
            raise ValueError("target should be one of {}, but got {}".format(self.supported_targets, target))
        return target

    def _get_imply_type(self, reg_info, target):
        """Get imply_typ information."""
        # Get imply_type from reg_info["imply_type"]
        if isinstance(reg_info, dict) and isinstance(reg_info.get("imply_type"), str) and \
                reg_info["imply_type"].strip():
            return reg_info["imply_type"]
        # Infer imply_type from func_type
        func_type_to_imply_type = {"hybrid": "AKG", "akg": "AKG", "tbe": "TBE", "aicpu": "AiCPU", "aot": target,
                                   "pyfunc": target, "julia": target}
        return func_type_to_imply_type.get(self.func_type, "AKG")

    def _save_attr(self, reg_info):
        """Save input_names and attr_names of current func."""
        if not isinstance(reg_info, dict):
            return
        tensor_inputs = reg_info.get("inputs", [])
        attr = reg_info.get("attr", [])
        if not isinstance(tensor_inputs, (list, tuple)):
            tensor_inputs = [tensor_inputs]
        if not isinstance(attr, (list, tuple)):
            attr = [attr]
        # input_names include tensor input names and attr input names
        input_names = []
        # attr_names only includes attr input names
        attr_names = []
        for item in tensor_inputs:
            if isinstance(item, dict) and item.get("name") is not None:
                input_names.append(item["name"])
        for item in attr:
            if isinstance(item, dict) and item.get("name") is not None:
                input_names.append(item["name"])
                attr_names.append(item["name"])
        cur_attr = {"input_names": input_names, "attr_names": attr_names}
        # If func does not have attr, save current attr.
        # Else, check if current attr is same as previous saved one.
        prev_input_names = input_names
        prev_attr_names = attr_names
        if callable(self.func):
            func_attr = getattr(self.func, "func_attr", None)
            if not isinstance(func_attr, dict):
                setattr(self.func, "func_attr", cur_attr)
            else:
                prev_input_names = func_attr.get("input_names")
                prev_attr_names = func_attr.get("attr_names")
        elif isinstance(self.func, str):
            func_attr = Custom.attr_dict.get(self.func)
            if not isinstance(func_attr, dict):
                Custom.attr_dict[self.func] = cur_attr
            else:
                prev_input_names = func_attr.get("input_names")
                prev_attr_names = func_attr.get("attr_names")
        if not isinstance(prev_input_names, list):
            raise TypeError("func {}: previous saved input_names should be a list, but got {}"
                            .format(self.func, type(prev_input_names)))
        if len(input_names) != len(prev_input_names):
            raise ValueError("func {}: input_names's length {} is different from previous saved one: {}"
                             .format(self.func, len(input_names), len(prev_input_names)))
        if attr_names != prev_attr_names:
            raise ValueError("func {}: attr_names {} is different from previous saved one: {}"
                             .format(self.func, attr_names, prev_attr_names))

    def _add_prim_target(self):
        """Add primitive_target to primitive's attr."""
        registered_targets = self._get_registered_targets()
        if self.func_type == "pyfunc":
            self.add_prim_attr("primitive_target", "CPU")
            if registered_targets and registered_targets != ["CPU"]:
                logger.warning("CustomPyfunc only supports CPU platform, but gets registered target as {}."
                               "We will run CustomPyfunc on CPU".format(registered_targets))
        elif self.func_type == "aot":
            if len(registered_targets) != 1:
                logger.info("Target of CustomAOT will be set according to context.")
            elif registered_targets == ["GPU"]:
                self.add_prim_attr("primitive_target", "GPU")
            elif registered_targets == ["CPU"]:
                self.add_prim_attr("primitive_target", "CPU")
        elif self.func_type == "julia":
            self.add_prim_attr("primitive_target", "CPU")
            device_target = context.get_context('device_target')
            if device_target == "CPU":
                pass
            elif device_target == "GPU" and registered_targets and registered_targets == ["CPU"]:
                logger.warning("CustomJulia only supports CPU platform, but gets registered target as {}."
                               "We will run CustomJulia on CPU".format(registered_targets))
            else:
                raise ValueError("CustomJulia only supports CPU platform, but gets target as {}.".format(device_target))

    def _update_attr(self):
        """Add input_names, attr_names, primitive_target to primitive's attr."""
        # add input_names, attr_names
        func_attr = {}
        if callable(self.func):
            inputs_num = len(inspect.signature(self.func).parameters)
            self.add_prim_attr("inputs_num", inputs_num)
            func_attr = getattr(self.func, "func_attr", None)
        elif isinstance(self.func, str):
            func_attr = Custom.attr_dict.get(self.func)
        if isinstance(func_attr, dict):
            input_names = func_attr.get("input_names")
            attr_names = func_attr.get("attr_names")
            if input_names:
                self.add_prim_attr("input_names", input_names)
            if attr_names:
                self.add_prim_attr("attr_names", attr_names)
        self._add_prim_target()
        if callable(self.func) and callable(self.out_shape):
            if hasattr(self.out_shape, "type") and getattr(self.out_shape, "type") == "autodiff":
                self.add_prim_attr("autodiff", True)
        else:
            self.add_prim_attr("autodiff", False)

    def _hybrid_autodiff(self, input_func_type):
        """generate backward op for a custom hybrid op"""
        inputs_num = len(inspect.signature(self.func).parameters)
        if inputs_num == 0:
            logger.warning("Function with no input has no backward op.")
        elif inputs_num > 10:
            logger.warning("Currently autodiff for function with more than 10 inputs is not supported.")
        else:
            grad_func = autodiff_bprop(inputs_num)

            def infer_func(*args):
                return args[:inputs_num]

            setattr(infer_func, "type", "autodiff")
            op = Custom(func=self.func, out_shape=infer_func, out_dtype=infer_func,
                        func_type=input_func_type, bprop=True)
            self.bprop = grad_func(op)

    def _hybrid_func_analyser(self):
        """analyze hybrid source string and add corresponding attrs."""
        args = {val: idx for idx, val in enumerate(list(inspect.signature(self.func).parameters))}
        if self.func_source_str.count('return') != 1:
            logger.warning("Hybrid function code should have only one 'return' syntax.")
        else:
            sentences = [s for s in self.func_source_str.split('\n') if s.count("return") == 1]
            symbols = re.sub(r"return|\s|\[|\]|\(|\)", "", sentences[-1]).split(',')
            inplace_assign_output = [[idx, args[val]] if val in args else [idx, -1]
                                     for idx, val in enumerate(symbols)]

            if any(i[1] != -1 for i in inplace_assign_output):
                self.add_prim_attr("inplace_assign_output", " ".join(
                    (str(j) for i in inplace_assign_output for j in i)))

    def _auto_infer(self, *args):
        """
        the automatic infer function for functions with @ms_hybrid decorator
        """
        fake_input = []
        enable_infer_value = True
        for arg in args:
            if arg["value"] is not None:
                fake_input.append(arg["value"].asnumpy())
            else:
                arg_dtype = arg["dtype"]
                # if any value is missing from input, disable infer value
                enable_infer_value = False
                if isinstance(arg_dtype, mstype.tensor_type):
                    arg_dtype = arg_dtype.element_type()
                fake_arg = np.zeros(arg["shape"]).astype(
                    mstype.dtype_to_nptype(arg_dtype))
                fake_input.append(fake_arg)

        fake_output = self.func(*fake_input)

        if hasattr(fake_output, 'shape'):
            infer_shape = fake_output.shape
            infer_dtype = mstype.tensor_type(mstype.pytype_to_dtype(fake_output.dtype))
        else:
            infer_shape = (1,)
            infer_dtype = mstype.pytype_to_dtype(fake_output.dtype)

        infer_value = Tensor(fake_output) if enable_infer_value else None

        return infer_shape, infer_dtype, infer_value

    def __infer__(self, *args):
        if callable(self.out_shape):
            infer_shape = self.out_shape(*(x["shape"] for x in args))
        else:
            infer_shape = self.out_shape

        if callable(self.out_shape):
            infer_dtype = self.out_dtype(*(x["dtype"] for x in args))
        else:
            infer_dtype = self.out_dtype

        infer_value = None

        # deal with the case of ms script
        # enable auto infer function if any infer information is missing
        if self._is_ms_hybrid and (infer_dtype is None or infer_shape is None):
            logger.warning("Now Custom Op is inferring the shape and dtype automatically. "
                           "There might be some Python RuntimeWarning but it wouldn't influence the result.")

            auto_infer_result = self._auto_infer(*args)

            # use automatically inferred shape/dtype if the input infer values are null
            infer_shape = auto_infer_result[0] if infer_shape is None else infer_shape
            infer_dtype = auto_infer_result[1] if infer_dtype is None else infer_dtype
            infer_value = auto_infer_result[2]

        # deal with case that the custom op is of type pyfunc with empty output
        if self.func_type == "pyfunc":
            if infer_shape == ():
                logger.warning("The function output are empty tuple. Add a placeholder instead. "
                               "Do not use it as it could be any uninitialized data.")
                infer_shape = (1,)
            if infer_dtype == ():

                logger.warning("The function output are empty tuple. Add a placeholder instead. "
                               "Do not use it as it could be any uninitialized data.")
                infer_dtype = mstype.int32

        # after all automatic infer information fulfillment, throw error if infer_shape/infer_dtype is still None
        if not isinstance(infer_shape, (tuple, list)):
            raise TypeError(
                "The input 'out_shape' should be one of a tuple, list, and function, "
                "but get a {} for the Custom Op {}".format(type(infer_shape), self.func_name))

        if not isinstance(infer_dtype, (typing.Type, tuple, list)):
            raise TypeError(
                "The input 'out_dtype' should be one of a mindspore.dtype, tuple, list, and function, "
                "but get a {} for the Custom Op {}".format(type(infer_dtype), self.func_name))

        out = {
            "shape": infer_shape,
            "dtype": infer_dtype,
            "value": infer_value,
        }
        return out
