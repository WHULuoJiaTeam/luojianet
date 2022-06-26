# Copyright 2022 Huawei Technologies Co., Ltd
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
"""ms_hybrid decorator and related util functions"""

import ast
import json
from functools import wraps
from itertools import product
import numpy
from mindspore import context


def _allocate(shape, dtype='float32', scope='global'):
    """Allocate a buffer with given shape

    Parameters
    ----------
    shape: Tuple
        The shape of the tensor to be allocated
    dtype: string
        The data type of the tensor
    scope: string
        The storage scope of the tensor

    Returns
    -------
    tensor: numpy.array
        The tensor allocated
    """
    del scope
    return numpy.zeros(shape).astype(dtype)


def _rsqrt(x):
    """
    Computes reciprocal of square root of x element-wise

    Parameters
    ----------
    x: Tensor

    Returns
    -------
    res: Tensor
        The result of reciprocal of square root of x
    """
    return numpy.ones_like(x) / numpy.sqrt(x)


def _erf(x):
    """
    Erf function of x, aka erf(x) = 2 / sqrt(pi) * integral(exp(-t*t), t = 0..x).
    The algorithm comes from Handbook of Mathematical Functions, formula 7.1.26.

    Parameters
    ----------
    x: a real number

    Returns
    -------
    res: a real number
        The result of erf function
    """
    # save the sign of x
    sign = 1 if x >= 0 else -1
    x = numpy.abs(x)

    # constants
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    # A&S formula 7.1.26
    t = 1.0/(1.0 + p*x)
    y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*numpy.exp(-x*x)
    return sign*y  # erf(-x) = -erf(x)


def _grid(extents):
    extents_list = []
    for ext in extents:
        extents_list.append(range(ext))
    return product(*extents_list)


class WithStub:
    """
    Runtime support for with scrop intrin in Hybrid DSL
    """

    def __init__(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        del exc_type, exc_value, exc_traceback
        return self

    def __del__(self):
        return self

    def __call__(self, *arg, **kwargs):
        return self


INTRIN_BUFFER = {
    'allocate': _allocate,
    'output_tensor': _allocate
}

INTRIN_LOOP = {
    'range': range,
    'grid': _grid,
}

INTRIN_WITH_SCOPE = {
    'attr': WithStub(),
    'block_realize': WithStub(),
}

INTRIN_UNARY_OP = {
    'sqrt': numpy.sqrt,
    'sign': numpy.sign,
    'log': numpy.log,
    'tanh': numpy.tanh,
    'exp': numpy.exp,
    'abs': numpy.abs,
    'int32': numpy.int32,
    'float16': numpy.float16,
    'float32': numpy.float32,
}

INTRIN_BINARY_OP = {
    'power': numpy.power,
}

INTRIN_GLOBALS = {
    **INTRIN_BUFFER,
    **INTRIN_LOOP,
    **INTRIN_WITH_SCOPE,
    **INTRIN_UNARY_OP,
    **INTRIN_BINARY_OP,
}

INTRIN_GPU_UNARY_OP = {
    'rsqrt': _rsqrt,
    'erf': _erf,
    'isnan': numpy.isnan,
    'int8': numpy.int8,
    'int16': numpy.int16,
    'int64': numpy.int64,
    'float64': numpy.float64,
    'sin': numpy.sin,
    'cos': numpy.cos,
    'isinf': numpy.isinf,
    'isfinite': numpy.isfinite,
    'atan': numpy.arctan,
    'atan2': numpy.arctan2,
    'expm1': numpy.expm1,
    'floor': numpy.floor,
    'ceil': numpy.ceil,
    'trunc': numpy.trunc,
    'round': numpy.round,
}

INTRIN_GPU_BINARY_OP = {
    'ceil_div': lambda a, b: (a + b - 1) // b,
}

INTRIN_GPU = {
    **INTRIN_GPU_UNARY_OP,
    **INTRIN_GPU_BINARY_OP
}

INTRIN_RUNTIME = {
    **INTRIN_GLOBALS,
    **INTRIN_GPU
}


class VariableUsage(ast.NodeVisitor):
    """
    The ast visitor to perform static check for the source code,
    and determine the index of inplace assign outputs
    """

    def __init__(self, func_name):
        self.func_name = func_name
        self.scope_level = []
        self.inplace_assign_output = []
        self.args_index = {}
        self.status = {}
        self.output_tensor = []
        self.temp_tensor = []
        self.device = context.get_context('device_target')

    def visit_FunctionDef(self, node):
        """
        Ast visitor for FunctionDef

        collect all input tensors
        """
        self.scope_level.append(node)
        for idx, arg in enumerate(node.args.args):
            self.args_index[arg.arg] = idx
        for elem in node.body:
            self.visit(elem)

    def visit_For(self, node):
        """
        Ast visitor for For loop

        append and pop Ast.For node as scope
        """
        self.visit(node.iter)
        self.scope_level.append(node)
        for i in node.body:
            self.visit(i)
        self.scope_level.pop()

    def visit_Name(self, node):
        """
        Ast visitor for Name

        Check the use of variables, including
        - whether it is defined
        - whether it is used inside its scope
        """
        # If it is from the argument list or loop variable, we do not worry about it!
        if node.id in self.args_index.keys():
            return
        fors = list(loop.target.id for loop in self.scope_level if isinstance(loop, ast.For))
        if node.id in fors:
            # The loop variable cannot be overwritten when iteration
            if isinstance(node.ctx, ast.Store):
                raise ValueError(
                    "In the function {} written in the Hybrid DSL, "
                    "iter var cannot be overwritten: {}".format(self.func_name, node.id))
            return

        if node.id not in self.status.keys():
            if not isinstance(node.ctx, ast.Store):
                raise ValueError(
                    "In the function {} written in the Hybrid DSL, there is "
                    "a undeclared variable: {}".format(self.func_name, node.id))
            self.status[node.id] = (node, self.scope_level[-1], set())
        else:
            decl, loop, usage = self.status.get(node.id, (None, None, None))
            usage.add(type(node.ctx))
            if not loop in self.scope_level:
                raise ValueError(
                    "In the function {} written in the Hybrid DSL, there is "
                    "a variable used out of the scope it is defined: {}".format(self.func_name, node.id))
            self.status[node.id] = (decl, loop, usage)

    def visit_Call(self, node):
        """
        Ast visitor for Call

        Check the func call used in the DSL. Only those in INTRIN_RUNTIME are supported for now.
        """

        func_id = node.func.id
        if not (func_id in list(INTRIN_RUNTIME.keys()) +
                ['max', 'min', 'len', 'ms_hybrid']):
            raise ValueError(
                "In the function {} written in the Hybrid DSL, function call id {} "
                "not in intrinsics' list".format(self.func_name, func_id))
        if self.device != "GPU" and func_id in list(INTRIN_GPU.keys()):
            raise ValueError(
                "In the function {} written in the Hybrid DSL, function {} is not available on the "
                "device {}".format(self.func_name, func_id, self.device))
        if func_id in list(INTRIN_UNARY_OP.keys()) + list(INTRIN_GPU_UNARY_OP.keys()) + list(INTRIN_LOOP.keys()) \
                and len(node.args) != 1:
            raise TypeError(
                "In the function {} written in the Hybrid DSL, function {} "
                "expects one input, but get {}".format(self.func_name, func_id, len(node.args)))
        if func_id in list(INTRIN_BINARY_OP.keys()) + list(INTRIN_GPU_BINARY_OP.keys()) + \
                list(INTRIN_BUFFER.keys()) and len(node.args) != 2:
            raise TypeError(
                "In the function {} written in the Hybrid DSL, function {} "
                "expects two inputs, but get {}".format(self.func_name, func_id, len(node.args)))
        for elem in node.args:
            self.visit(elem)

    def visit_With(self, node):
        """
        Ast visitor for With

        Check the func used in the with scope. Only attr and block_realize are supported for now.
        """
        context_expr = node.items[0].context_expr
        if context_expr.func.id == "attr":
            if len(context_expr.args) != 2:
                raise ValueError(
                    "In the function {} written in the Hybrid DSL, two inputs are expected by 'attr', "
                    "but get {}".format(self.func_name, len(context_expr.args)))
            if not isinstance(context_expr.args[0], ast.Str):
                raise ValueError(
                    "In the function {} written in the Hybrid DSL, the first input of 'attr' should be a string, "
                    "but get {}".format(self.func_name, type(context_expr.args[0])))
            if not (isinstance(context_expr.args[1], (ast.Str, ast.Num, ast.NameConstant)) and
                    context_expr.args[1].value is not None):
                raise ValueError(
                    "In the function {} written in the Hybrid DSL, the second input of 'attr' should be a string, "
                    "number or bool value, but get {}".format(self.func_name, type(context_expr.args[1])))
        elif context_expr.func.id == "block_realize":
            if len(context_expr.args) != 1:
                raise ValueError(
                    "In the function {} written in the Hybrid DSL, only one input is accepted by 'block_realize', "
                    "but get {}".format(self.func_name, len(context_expr.args)))
            if not isinstance(context_expr.args[0], ast.Name):
                raise ValueError(
                    "In the function {} written in the Hybrid DSL, the input of 'block_realize' should be "
                    "a tensor name, but get {}".format(self.func_name, type(context_expr.args[0])))

        else:
            raise ValueError(
                "Unsupported function in With scope in the function {} written in the Hybrid DSL: "
                "{} ".format(self.func_name, context_expr.func.id))

        for stmt in node.body:
            if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Str)):
                self.visit(stmt)

    def visit_Assign(self, node):
        """
        Ast visitor for Assign

        Collect all tensor declared by allocate and output_tensor
        """
        if len(node.targets) > 1:
            raise ValueError(
                "One statement with multiple assignments is not allowed in the function {} "
                "written in the Hybrid DSL.".format(self.func_name))
        if isinstance(node.targets[0], ast.Name) and isinstance(node.value, ast.Call) and \
                isinstance(node.value.func, ast.Name):
            assign_id = node.targets[0].id
            func_name = node.value.func.id
            if assign_id in self.output_tensor + self.temp_tensor:
                raise ValueError(
                    "In the function {} written in the Hybrid DSL, the tensor is "
                    "redefined: {}".format(self.func_name, assign_id))
            if func_name == "allocate":
                self.temp_tensor.append(assign_id)
            if func_name == "output_tensor":
                self.output_tensor.append(assign_id)

        return self.generic_visit(node)

    def visit_Break(self, node):
        """
        Ast visitor for Break

        Throw an error if the key word break in the DSL
        """
        del node
        raise TypeError(
            "Keyword 'break' not accepted in the function {} written in the Hybrid DSL!".format(self.func_name))

    def visit_Continue(self, node):
        """
        Ast visitor for Continue

        Throw an error if the key word continue in the DSL
        """
        del node
        raise TypeError(
            "Keyword 'continue' not accepted in the function {} written in the Hybrid DSL!".format(self.func_name))

    def visit_While(self, node):
        """
        Ast visitor for While

        Throw an error if the key word while in the DSL
        """
        del node
        raise TypeError(
            "Keyword 'while' not accepted in the function {} written in the Hybrid DSL!".format(self.func_name))

    def visit_Attribute(self, node):
        """
        Ast visitor for Attribute

        Throw an error if the attribute is neither shape nor dtype.
        """
        if not isinstance(node.value, ast.Name):
            raise ValueError(
                "In the function {} written in the Hybrid DSL, getattr is only supported for a tensor object, "
                "not for the object with type: {}".format(self.func_name, type(node.value)))

        if not node.value.id in self.output_tensor + self.temp_tensor + list(self.args_index.keys()):
            raise ValueError(
                "In the function {} written in the Hybrid DSL, getattr is only supported for a tensor variable "
                "after its declaration, not for: {}".format(self.func_name, node.value.id))

        if not (node.attr in ['shape', 'dtype']):
            raise ValueError(
                "In the function {} written in the Hybrid DSL, a tensor object "
                "has no attribute called {}".format(self.func_name, node.attr))

    def visit_Return(self, node):
        """
        Ast visitor for Return

        Calculate all inplace_assign index, namely which output is in fact an input
        """
        symbols = []
        if isinstance(node.value, ast.Name):
            symbols = [node.value.id]
        else:
            if not isinstance(node.value, ast.Tuple):
                raise TypeError(
                    "In the function {} written in the Hybrid DSL, the return value should be "
                    "either a single tensor or a tuple, but get a {}.".format(self.func_name, type(node.value)))
            for i in node.value.elts:
                if not isinstance(i, ast.Name):
                    raise TypeError("In the function {} written in the Hybrid DSL, the element in the return value "
                                    "should be the name of a tensor, but get a {}.".format(self.func_name, type(i)))
            symbols = list(i.id for i in node.value.elts)
        for sy in symbols:
            if not sy in list(self.args_index.keys()) + self.output_tensor:
                raise TypeError("In the function {} written in the Hybrid DSL, the element in the return value "
                                "should be either an input tensor or a tensor allocated by output_tensor, "
                                "but get name: {}".format(self.func_name, sy))
        for sy in self.output_tensor:
            if not sy in symbols:
                raise TypeError("In the function {} written in the Hybrid DSL, the tensor is allocated as an output "
                                "tensor but not in the return value: {}".format(self.func_name, sy))
        self.inplace_assign_output = list([idx, self.args_index.get(val, -1)]
                                          for idx, val in enumerate(symbols) if val in self.args_index)


def determine_variable_usage(root, func_name):
    """
    The function to perform static check for the source code,
    and determine the index of inplace assign outputs

    Parameters
    ----------
    root: an ast tree root

    Returns
    -------
    inplace_assign_output: a list
        The list of index about inplace assign outputs
    """
    visitor = VariableUsage(func_name)
    visitor.visit(root)
    return visitor.inplace_assign_output


def ms_hybrid(fn=None, reg_info=None, compile_attrs=None):
    """
    The decorator of the Hybrid DSL function for the Custom Op.
    When a function written by the Hybrid DSL is decorated by ms_hybrid,
    it can be run as a usual Python function.
    Also, this function can be used in the api Custom and to create a Custom op, with func_type
    "hybrid" or "pyfunc". Creating a custom op with mode "hybrid" by the Hybrid DSL function
    will enjoy the automatic dtype/shape infer for free.

    Args:
        fn (Function): The Python function that will be run as a custom operator. Default: None.
        reg_info (tuple[str, dict]): Each item represents registration information in json format. Default: None.
        compile_attrs (Dict): The Python object is used to distinguish the compiled function. Default: None.

    Returns:
        Function, if `fn` is not None, returns a callable function that will execute the Hybrid DSL function;
        If `fn` is None, returns a decorator and when this decorator invokes with a single `fn` argument, the
        callable function is equal to the case when `fn` is not None.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import ops, Tensor
        >>> from mindspore.ops import ms_hybrid, DataType, CustomRegOp
        ...
        >>> # Create a dict for the compile flags.
        >>> attrs = {
        ...     "test1": True,
        ...     "test2": "good",
        ...     "test3": 12,
        ... }
        >>> # Create the reg info json string.
        >>> op_gpu_info = CustomRegOp() \\
        ...     .input(0, "a") \\
        ...     .input(0, "b") \\
        ...     .output(0, "y") \\
        ...     .dtype_format(DataType.F32_None, DataType.F32_None, DataType.F32_None) \\
        ...     .target("GPU") \\
        ...     .get_op_info()
        >>>
        >>> # Create inputs for the custom op.
        >>> input_x = np.ones([4, 4]).astype(np.float32)
        >>> input_y = np.ones([4, 4]).astype(np.float32)
        ...
        >>> # Write a Hybrid DSL function through the decorator @ms_hybrid.
        >>> # We can also pass the compile attrs and the reg info through the decorator.
        >>> @ms_hybrid(reg_info=op_gpu_info, compile_attrs=attrs)
        ... def outer_product(a, b):
        ...     c = output_tensor(a.shape, a.dtype)
        ...
        ...     with block_realize(c):
        ...         for i0 in range(a.shape[0]):
        ...             for i1 in range(b.shape[1]):
        ...                 c[i0, i1] = 0.0
        ...                 for i2 in range(a.shape[1]):
        ...                     c[i0, i1] = c[i0, i1] + (a[i0, i2] * b[i2, i1])
        ...     return c
        ...
        >>> # We can use the function directly as a python function.
        >>> # In this case, the inputs should be numpy arrays.
        >>> result = outer_product(input_x, input_y)
        ...
        >>> # Create a custom op with mode "hybrid" (default value) by the Hybrid DSL function.
        >>> # In this case, we will enjoy the automatic dtype/shape infer for free.
        >>> # The inputs should be mindspore tensors.
        >>> test_op_hybrid = ops.Custom(outer_product)
        >>> output = test_op_hybrid(Tensor(input_x), Tensor(input_y))
    """
    if compile_attrs is None:
        compile_attrs = {}

    if not isinstance(compile_attrs, dict):
        raise TypeError("The input 'compile_attrs' of @ms_hybrid should be a dict, "
                        "but get a {}".format(type(compile_attrs)))

    for key in compile_attrs.keys():
        if not isinstance(key, str):
            raise TypeError("The key of 'compile_attrs' of @ms_hybrid should be a str, "
                            "but get a {}".format(type(key)))

    if reg_info is not None and not isinstance(reg_info, (str, dict, tuple)):
        raise TypeError(
            "The input 'reg_info' of @ms_hybrid should be one of "
            "str, dict and tuple, but get a {}".format(type(reg_info)))

    def wrap_ms_hybrid(func):
        setattr(func, "ms_hybrid_flag", True)
        setattr(func, "compile_attrs", json.dumps(compile_attrs))
        if reg_info is not None:
            setattr(func, "reg_info", reg_info)

        @wraps(func)
        def _patch_intrins_to_runtime(*args):
            _globals = func.__globals__
            for elem in list(INTRIN_RUNTIME.keys()):
                _globals[elem] = INTRIN_RUNTIME[elem]
            return func(*args)

        return _patch_intrins_to_runtime

    if fn is not None:
        return wrap_ms_hybrid(fn)
    return wrap_ms_hybrid
