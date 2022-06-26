# This is the Python adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
#
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
"""The module of parser python object, called by c++."""

import os
import sys
import ast
import hashlib
import inspect
import types
import importlib
from dataclasses import is_dataclass
from textwrap import dedent

import asttokens

from mindspore import Tensor
from mindspore import log as logger
from mindspore import nn
from mindspore import ops
from mindspore.common.api import _MindsporeFunctionExecutor, _convert_data
from mindspore.common.dtype import pytype_to_dtype
from .namespace import CellNamespace, ClosureNamespace, ClassMemberNamespace, ClassAttrNamespace
from .resources import parse_object_map, ops_symbol_map, convert_object_map, trope_ns, SYMBOL_UNDEFINE, NO_IMPLEMENT
from .jit_fallback_modules import jit_fallback_third_party_modules_whitelist

# define return value
RET_SUCCESS = 0
RET_FAILURE = 0xFF

# define resolve type
RESOLVE_TYPE_NONE = 0                   # resolve None
RESOLVE_TYPE_FUNCTION = 1               # resolve function
RESOLVE_TYPE_METHOD = 2                 # resolve class method
RESOLVE_TYPE_CLASS_TYPE = 3             # resolve class type
RESOLVE_TYPE_CLASS_INSTANCE = 4         # resolve the class instance of common class
RESOLVE_TYPE_INVALID = 0xFF

# define the class instance detail type
# When the type is RESOLVE_TYPE_CLASS_INSTANCE
CLASS_INSTANCE_TYPE_CELL = 0            # class instance type is Cell
CLASS_INSTANCE_TYPE_PRIMITIVE = 1       # class instance type is Primitive
CLASS_INSTANCE_TYPE_INVALID = 0xFF

# Ast main type
AST_MAIN_TYPE_STMT = 0                  # ast.Stmt
AST_MAIN_TYPE_EXPR = 1                  # ast.Expr
AST_MAIN_TYPE_SLICE = 2                 # ast.Slice
AST_MAIN_TYPE_UNKNOWN = 0xFF            # unknown

# Ast sub type
AST_SUB_TYPE_AND = 3                   # ast.And
AST_SUB_TYPE_OR = 4                    # ast.Or
AST_SUB_TYPE_NAME = 5                  # ast.Name
AST_SUB_TYPE_TUPLE = 6                 # ast.Tuple
AST_SUB_TYPE_SUBSCRIPT = 7             # ast.Subscript
AST_SUB_TYPE_STARRED = 8               # ast.Starred
AST_SUB_TYPE_ATTRIBUTE = 9             # ast.Attribute
AST_SUB_TYPE_UNKNOWN = 0xFF            # unknown

# Syntax support
SYNTAX_SUPPORTED = 0                   # supported syntax
SYNTAX_UNSUPPORTED_INTERNAL_TYPE = 1   # unsupported internal type
SYNTAX_UNSUPPORTED_EXTERNAL_TYPE = 2   # unsupported external type
SYNTAX_UNSUPPORTED_SPECIAL_TYPE = 3    # unsupported special type
SYNTAX_UNSUPPORTED_NAMESPACE = 4       # unsupported namespace

# Process expr statement white list
# add as needed, eg: "clear", "extend", "insert", "remove", "reverse"
parse_expr_statement_white_list = (
    "append", "insert",
)

_builtin_function_or_method_type = type(abs)

_unsupported_python_builtin_type = (
    list, tuple, set, dict, slice, bool, int, float, str, complex, reversed, type,
)

_unsupported_internal_type = (
    Tensor,
)

_unsupported_special_type = (
    print,
)


def create_slice_obj(start, end, step):
    """Create slice object"""
    return slice(start, end, step)


def parse_cb(func, parse_method=None):
    """Implements the function of parse."""
    return Parser(func, parse_method)


def get_parse_method_of_class(obj, parse_method=None):
    """
    Het parse method of class.

    Args:
        obj(Object): Instance of class.
        parse_method(str): Save the method name. Cell object has default method named 'construct'.

    Returns:
        Function, obj's method.
    """
    method = None
    method_name = None
    if parse_method is not None:
        method_name = parse_method
    elif isinstance(obj, nn.Cell):
        if obj._enable_backward_hook:
            method_name = "_backward_hook_construct"
        else:
            method_name = "construct"
    if method_name is not None:
        if hasattr(obj, method_name):
            method = getattr(obj, method_name)
    return method


def get_bprop_method_of_class(obj, parse_method=None):
    """
    Get bprop method of class.

    Args:
        obj (Object): Instance of class.
        parse_method(str): Save the method name. Cell object has default method named 'bprop'.

    Returns:
        Function, obj's method.
    """
    method = None
    if isinstance(obj, nn.Cell):
        method_name = "bprop"
        if hasattr(obj, method_name):
            method = getattr(obj, method_name)
    return method


def get_env_support_modules():
    """Get support modules from environment variable."""
    support_modules = os.getenv('MS_DEV_SUPPORT_MODULES')
    if support_modules is None:
        return []
    env_support_modules = []
    modules = support_modules.split(',')
    for module in modules:
        try:
            module_spec = importlib.util.find_spec(module)
        except (ModuleNotFoundError, ValueError):
            module = module[0:module.rfind('.')]
            module_spec = importlib.util.find_spec(module)
        finally:
            pass
        if module_spec is None:
            raise ModuleNotFoundError(f"Cannot find module: {module}. " \
                f"Please check if {module} is installed, or if MS_DEV_SUPPORT_MODULES is set correctly.")
        # Add the outermost module.
        env_support_modules.append(module.split('.')[0])
    logger.debug(f"Get support modules from env: {env_support_modules}")
    return env_support_modules


# The fallback feature is enabled in default.
# Not support change the flag during the process is alive.
support_fallback_ = os.getenv('MS_DEV_ENABLE_FALLBACK')
support_modules_ = get_env_support_modules()


def resolve_symbol(namespace, symbol):
    """
    Resolve a symbol.

    Note:
        Can't get function when use closure function. So save the fn on namespace.

    Args:
        namespace (Object): Symbol's namespace.
        symbol (str): Need resolve symbol.

    Returns:
        Object, resolve result of symbol.
    """
    # All exceptions need to be caught in this function
    try:
        resolve_ = namespace[symbol]

        # list and dict is not hashable ,it can not be key for the map, just return the result
        if isinstance(resolve_, (tuple, list, dict)):
            return resolve_

        # dataclass may not be hashable
        if getattr(resolve_, "__hash__") is None:
            return resolve_

        # Raise a proper error if not using Fallback feature.
        if support_fallback_ == '0':
            # Raise NotImplementedError when parsing the numpy methods, but not the numpy constant.
            if namespace.name == "numpy" and \
                isinstance(resolve_, (types.FunctionType, types.MethodType, types.ModuleType)):
                raise NotImplementedError("Mindspore does not support to use the numpy methods " \
                                          "within the construct() or @ms_function decorated function in graph mode.")

        # If need trope the obj
        if resolve_ in convert_object_map:
            resolve_ = convert_object_map.get(resolve_)
            logger.debug("Convert resolve: %r", resolve_)
            if resolve_ == NO_IMPLEMENT:
                raise NotImplementedError(f"Not support for '{symbol}'.")
    except Exception as e:
        if isinstance(e, NotImplementedError):
            raise e
        resolve_ = None
        logger.debug("Resolve exception occurred, value: %r", e)
        logger.debug("Resolve type is invalid, namespace: %s, symbol: %s",
                     namespace.__str__(), symbol)

    if isinstance(resolve_, _MindsporeFunctionExecutor):
        logger.debug("Resolve class _MindsporeFunctionExecutor, resolve fn instead.")
        resolve_ = resolve_.fn
    logger.debug(f"Found '{symbol}' in {namespace.__str__()}, resolved: {resolve_} / {type(resolve_)}")
    return resolve_


def generate_scope(obj):
    """Generate the scope for every cell object in the network."""
    if isinstance(obj, nn.Cell):
        obj.generate_scope()


def get_scope_name(obj):
    """Returns the scope of a cell object in one network."""
    if isinstance(obj, nn.Cell):
        return obj.get_scope()
    return None


def get_object_key(obj):
    """Return the function key: module + name."""
    obj_key = ""
    if hasattr(obj, "__name__"):
        if hasattr(obj, "cell_init_args"):
            obj_key = "%s_ID" % (str(obj.__class__.__name__) + str(obj.__name__) + obj.cell_init_args)
        obj_id = "%s_ID%d" % (str(obj.__class__.__name__) + str(obj.__name__), id(obj))
    else:
        # `<class 'xxxxxxx'>`
        # -> `xxxxxxx`
        tag = str(obj.__class__)[8:-2]
        if hasattr(obj, "cell_init_args"):
            obj_key = "%s_ID" % (tag + obj.cell_init_args)
        obj_id = "%s_ID%d" % (tag, id(obj))
    logger.debug("obj_key: %s, obj_id: %s", obj_key, obj_id)

    # method has same id of different instance
    if isinstance(obj, types.MethodType):
        method_instance = obj.__self__
        instance_id = "%s_ID%d" % (str(method_instance.__class__.__name__), id(method_instance))
        obj_id = instance_id + obj_id + str(obj.__hash__())
    return obj_id, obj_key


def is_class_member(node):
    """Check the attr is class member variable."""
    type_ = node.__class__.__name__
    if type_ == "Attribute":
        if not hasattr(node.value, "id"):
            return False
        id_ = node.value.id
        if id_ == "self":
            return True
    return False


def get_obj_id(obj):
    """Get the obj id."""
    return str(id(obj))


def get_obj_type(obj):
    """Get the obj type."""
    logger.debug("Get object type: %r", obj)
    obj_type = RESOLVE_TYPE_INVALID
    if obj is None:
        obj_type = RESOLVE_TYPE_NONE
    elif isinstance(obj, types.FunctionType):
        obj_type = RESOLVE_TYPE_FUNCTION
    elif isinstance(obj, types.MethodType):
        obj_type = RESOLVE_TYPE_METHOD
    elif isinstance(obj, type):
        obj_type = RESOLVE_TYPE_CLASS_TYPE
    elif _is_class_instance(obj):
        obj_type = RESOLVE_TYPE_CLASS_INSTANCE
    else:
        # Raise a proper error if not using Fallback feature.
        if support_fallback_ != '0':
            obj_type = RESOLVE_TYPE_INVALID
        else:
            # here for ndarray, just print its shape (in case of the array to large and print many data in screen)
            is_ndarray = type(obj).__name__ == 'ndarray' and hasattr(obj, 'shape')
            raise TypeError(f"Not support for this object with type '{type(obj)}' and "
                            f"{'shape' if is_ndarray else 'value'} '{obj.shape if is_ndarray else obj}'.")
    return obj_type


def get_class_instance_type(obj):
    """Get the class instance detail type."""
    # check the obj type
    logger.debug("Get the class type(%r)", obj)
    class_type = CLASS_INSTANCE_TYPE_INVALID
    if _is_class_instance(obj):
        if isinstance(obj, nn.Cell):
            class_type = CLASS_INSTANCE_TYPE_CELL
        elif isinstance(obj, ops.Primitive):
            class_type = CLASS_INSTANCE_TYPE_PRIMITIVE
        # Add the other type base requirement
    return class_type


def _is_class_instance(obj):
    """Confirm the obj is class instance."""
    return isinstance(obj, (nn.Cell, ops.Primitive)) or _is_dataclass_instance(obj)


def _is_dataclass_instance(obj):
    """Check whether a class is an instance of a dataclass (and not a dataclass itself)"""
    return is_dataclass(obj) and not isinstance(obj, type)


def _convert_tuple_to_args_kwargs(params):
    """Convert tuple to args and kwargs."""
    args = tuple()
    kwargs = dict()
    for param in params:
        if isinstance(param, dict):
            kwargs.update(param)
        else:
            args += (param,)
    return (args, kwargs)


def is_supported_create_instance_type(cls_type):
    """Check if cls_type is a supported instance type."""
    return issubclass(cls_type, (nn.Cell, ops.Primitive))


def create_instance(cls_type, params=None):
    """Create python instance."""
    if not isinstance(cls_type, type):
        logger.warning(f"create_instance(), cls_type is not a type, cls_type: {cls_type}")
        return None

    # Check the type, now only support nn.Cell and Primitive.
    obj = None
    if is_supported_create_instance_type(cls_type):
        # Check arguments, only support *args or **kwargs.
        if params is None:
            obj = cls_type()
        elif isinstance(params, tuple):
            args, kwargs = _convert_tuple_to_args_kwargs(params)
            logger.debug(f"create_instance(), args: {args}, kwargs: {kwargs}")
            if args and kwargs:
                obj = cls_type(*args, **kwargs)
            elif args:
                obj = cls_type(*args)
            elif kwargs:
                obj = cls_type(**kwargs)
        # If invalid parameters.
        if obj is None:
            raise ValueError(f"When call 'create_instance', the parameter should be *args or **kwargs, "
                             f"but got {params.__class__.__name__}, params: {params}")
    return obj


def get_obj_from_sequence(obj, index):
    """Implement `tuple_getitem`."""
    if not isinstance(obj, (tuple, list)):
        raise TypeError(f"Should not get item from a object that not sequence type, obj: {obj}")
    # Not check index out of range by self.
    return obj[index]


def get_module_namespace(obj):
    """Get the module's namespace."""
    logger.debug("get module namespace, module: %r", obj)
    mod_namespace = None
    if isinstance(obj, types.ModuleType):
        mod_namespace = CellNamespace(obj.__name__)
    else:
        logger.warning("Module(%r) is invalid, get namespace failure!", obj)
    return mod_namespace


def get_class_attr_namespace_symbol(obj):
    """Get class namespace."""
    logger.debug("get class namespace, object: %r", obj)
    class_namespace = ClassAttrNamespace(obj)
    logger.debug("class namespace: %r", class_namespace)
    return class_namespace


def get_class_member_namespace_symbol(obj):
    """Get obj class member type."""
    logger.debug("get class instance namespace, object: %r", obj)
    class_namespace = ClassMemberNamespace(obj)
    logger.debug("class namespace: %r", class_namespace)
    return class_namespace


def get_dataclass_attributes(cls):
    """Get attributes of dataclass."""
    fields = cls.__dataclass_fields__
    attributes = {name: pytype_to_dtype(field.type)
                  for name, field in fields.items()}
    return attributes


def get_dataclass_methods(cls):
    """Get functions of dataclass."""
    methods = {name: getattr(cls, name)
               for name in dir(cls)
               if isinstance(getattr(cls, name), (types.FunctionType,))}
    return methods


def get_ms_class_name(cls):
    """Get the name of the class instance decorated by ms_class."""
    # Check if cls is nn.Cell.
    if isinstance(cls, nn.Cell):
        raise TypeError(f"ms_class is used for user-defined classes and cannot be used for nn.Cell: {cls}.")
    if isinstance(cls, type):
        name = cls.__name__
    else:
        name = cls.__class__.__name__
    # Get the name of cls.
    cls_name = cls.__module__ + '.' + name
    return cls_name


def get_ms_class_attr(cls, name: str):
    """Get attribute or method of ms_class obj."""
    # Don't take into account python magic methods and private variables.
    if name.startswith('_'):
        raise AttributeError(f"{name} is a private variable or magic method, which is not supported.")
    if not hasattr(cls, name):
        raise AttributeError(f"{cls} has no attribute: {name}.")
    return getattr(cls, name)


def convert_to_ms_tensor(data):
    """Convert C++ tensor to mindspore tensor."""
    return Tensor(data)


def get_object_description(obj, fname, fline):
    """return method or funcition description for error report, include location, class name, etc."""
    if isinstance(obj, types.MethodType):
        obj_cls = obj.__self__.__class__
        class_name = f"{obj_cls.__module__}.{obj_cls.__qualname__}"
        cls_fname = inspect.getfile(obj_cls)
        _, cls_fline = inspect.getsourcelines(obj_cls)
        class_loc = f"{cls_fname}:{cls_fline}"
        return f"bound method '{obj.__name__}' at {fname}:{fline} of <{class_name} at {class_loc} object>"
    if isinstance(obj, types.FunctionType):
        return f"function '{obj.__name__}' at {fname}:{fline}"
    if isinstance(obj, ast.FunctionDef):
        return f"function '{obj.name}' at {fname}:{fline}"
    if isinstance(obj, ast.Attribute):
        return f"attribute "
    return str(obj)


def expand_expr_statement(node):
    """
    Process the expr statement and expand it.

    Returns:
        tuple, (True, expr.value, x)/(False, None, None).
    """
    if isinstance(node, ast.Expr):
        expr_value = node.value
        if isinstance(expr_value, ast.Call):
            func = expr_value.func
            if isinstance(func, ast.Attribute) and \
                    hasattr(func, "attr") and \
                    hasattr(func, "value"):
                method = func.attr
                target = func.value
                if method in parse_expr_statement_white_list:
                    logger.debug("Expand expr, target:%s, method:%s", target, method)
                    return True, expr_value, target
        if not isinstance(expr_value, ast.Str):
            return True, expr_value
    return (False,)


def get_ast_namespace_symbol(obj):
    """Get obj type and namespace and symbol."""
    # step 1:get symbol from object map
    ops_info = parse_object_map.get(type(obj), SYMBOL_UNDEFINE)
    logger.debug("ops info: %r", ops_info)
    return ops_info


def get_operation_symbol(obj):
    """Get obj operation symbol."""
    ops_symbol = ops_symbol_map.get(type(obj), SYMBOL_UNDEFINE)
    logger.debug("ops symbol: %s", ops_symbol)
    return ops_symbol


def get_operation_namespace_symbol(var: str):
    """Get operation namespace and symbol."""
    ops_info = (trope_ns, var)
    logger.debug("get operation ops info: %r", ops_info)
    return ops_info


def get_ast_type(node):
    """Get the ast type."""
    ast_type = AST_SUB_TYPE_UNKNOWN
    if isinstance(node, ast.And):
        ast_type = AST_SUB_TYPE_AND
    elif isinstance(node, ast.Or):
        ast_type = AST_SUB_TYPE_OR
    elif isinstance(node, ast.Name):
        ast_type = AST_SUB_TYPE_NAME
    elif isinstance(node, ast.Tuple):
        ast_type = AST_SUB_TYPE_TUPLE
    elif isinstance(node, ast.Subscript):
        ast_type = AST_SUB_TYPE_SUBSCRIPT
    elif isinstance(node, ast.Starred):
        ast_type = AST_SUB_TYPE_STARRED
    elif isinstance(node, ast.Attribute):
        ast_type = AST_SUB_TYPE_ATTRIBUTE
    else:
        ast_type = AST_SUB_TYPE_UNKNOWN
    return ast_type


def get_node_type(node):
    """Process an ast node."""
    method_name = f"{node.__class__.__name__}"
    node_type = [method_name]
    # judge the ast main type
    if isinstance(node, ast.stmt):
        node_type.append(AST_MAIN_TYPE_STMT)
    elif isinstance(node, (ast.expr, ast.slice)) or node is None:
        # ast.slice and ast.expr should be expr
        node_type.append(AST_MAIN_TYPE_EXPR)
    else:
        node_type.append(AST_MAIN_TYPE_UNKNOWN)
    return node_type


def get_args_default_values(node):
    """get the args'default values of parse object."""
    nondefaults = [None] * (len(node.args.args) - len(node.args.defaults))
    defaults = nondefaults + node.args.defaults + node.args.kw_defaults
    if node.args.vararg:
        defaults.append(None)
    if node.args.kwarg:
        defaults.append(None)
    return defaults


def get_args(node):
    """Get the arg of parse object."""
    args = []
    # process position args
    for arg in node.args.args:
        args.append(arg)

    # process kwonlyargs: kwonlyargs is append after position args
    if node.args.kwonlyargs:
        for kwarg in node.args.kwonlyargs:
            args.append(kwarg)
    # process vararg: vararg is append after kwonlyargs
    if node.args.vararg:
        args.append(node.args.vararg)
    # process kwarg: kwarg is append after vararg
    if node.args.kwarg:
        args.append(node.args.kwarg)
    return args


def _in_sys_path(file_path):
    """To check if file_path is under system path."""
    for path in list(sys.path):
        if file_path.startswith(path):
            return True
    return False


def is_third_party_module(value):
    """To check if value is a third-party module."""
    # Check if value is a module or package, check if module file is under the sys path.
    if not inspect.ismodule(value) or not hasattr(value, '__file__') or not _in_sys_path(value.__file__):
        return False

    # Get module leftmost name.
    if not hasattr(value, '__name__'):
        return False
    module_name = value.__name__
    module_leftmost_name = module_name.split('.')[0]
    # Ignore mindspore package.
    if module_leftmost_name == "mindspore":
        return False
    # Check if module is in whitelist.
    if module_leftmost_name in support_modules_:
        logger.debug(f"Found support modules from env: {module_name}")
        return True
    if module_leftmost_name in jit_fallback_third_party_modules_whitelist:
        logger.debug(f"Found third-party module: {module_name}")
        return True
    return False


def eval_script(exp_str, params):
    """Evaluate a python expression."""
    if not isinstance(params, tuple):
        raise ValueError(f"eval_script(), params is not a tuple, params: {params}")
    if len(params) != 2:
        raise ValueError(f"eval_script(), params tuple length is wrong, params: {params}")

    # Eval function parses the expression argument and evaluates it as a python expression.
    logger.debug(f"exp_str: '{exp_str}', params: '{params}'")
    global_params = params[0]
    local_params = params[1]
    try:
        local_params = _convert_data(local_params)
        obj = eval(exp_str, global_params, local_params)
    except Exception as e:
        error_info = f"When eval '{exp_str}' by using JIT Fallback feature, an error occurred: " + str(e) + \
            ". You can try to turn off JIT Fallback feature by 'export MS_DEV_ENABLE_FALLBACK=0'."
        logger.error(error_info)
        raise e

    # Convert set to tuple.
    if isinstance(obj, set):
        return tuple(obj)
    return obj


class Parser:
    """
    Parser python code to ast tree.

    Args:
        fn(FunctionType/MethodType): Need parse object instance.
        parse_method(ExtendInfoOfParseObj): Extend information for parse the function.
        ast_cache: Dictionary for caching ast tree.
    """
    ast_cache = {}

    def __init__(self, fn: (types.FunctionType, types.MethodType), parse_method=None) -> None:
        self.fn = fn
        self.parse_method = parse_method
        self.line_offset = 0
        self.filename: str = inspect.getfile(inspect.unwrap(self.fn))

        # Used to resolve the function's globals namespace.
        self.global_namespace = CellNamespace(fn.__module__)
        self.function_module = fn.__module__
        # Used to resolve the function's nonlocals.
        self.closure_namespace = ClosureNamespace(inspect.unwrap(self.fn))
        self.function_name = fn.__name__
        self.col_offset = 0

    @staticmethod
    def is_unsupported_special_type(value):
        """To check if not supported special type, such as print"""
        if value in _unsupported_special_type:
            logger.debug(f"Found unsupported special type: '{value}'.")
            return True
        return False

    @staticmethod
    def is_unsupported_namespace(value):
        """To check if not supported for namespace"""
        unsupported = isinstance(value, _builtin_function_or_method_type) and value not in convert_object_map
        logger.debug(f"'{value}' unsupported: {unsupported}.")
        return unsupported

    @staticmethod
    def is_unsupported_python_builtin_type(value):
        """To check if not supported for builtin type"""
        unsupported = value in _unsupported_python_builtin_type
        logger.debug(f"value: '{value}', unsupported builtin type: {unsupported}.")
        return unsupported

    @staticmethod
    def is_unsupported_internal_type(value):
        """To check if not supported internal type, such as Tensor"""
        for item in _unsupported_internal_type:
            if value == item:
                logger.debug(f"Found unsupported internal type: '{value}'.")
                return True
        return False

    def parse(self):
        """Parse the function or method."""
        logger.debug("fn: %r", self.fn)
        if isinstance(self.fn, (types.FunctionType, types.MethodType)):
            try:
                lines, self.line_offset = inspect.getsourcelines(self.fn)
            except OSError as e:
                if e.__str__() == "could not get source code":
                    raise OSError(f"Mindspore can not compile temporary source code in terminal. "
                                  f"Please write source code to a python file and run the file.")
                raise e
            original_src = ''.join(lines)
            hexstr = hashlib.sha256(original_src.encode()).hexdigest()
            ast_tokens_cache = Parser.ast_cache.get(hexstr)
            if not ast_tokens_cache:
                src = dedent(original_src)
                self.col_offset = \
                    len(original_src.split('\n')[0]) - len(src.split('\n')[0])
                logger.debug("Get source: %s", src)
                try:
                    ast_tokens = asttokens.ASTTokens(src, parse=True)
                except IndentationError as idt_err:
                    idt_err.filename = self.filename
                    idt_err.lineno = self.line_offset
                    idt_err.msg = f"There are incorrect indentations in definition or comment of function: " \
                                  f"'{self.fn.__qualname__}'."
                    raise idt_err
                ast_tokens_cache = (ast_tokens, self.col_offset)
                Parser.ast_cache[hexstr] = ast_tokens_cache
            else:
                self.col_offset = ast_tokens_cache[1]
            return ast_tokens_cache[0], ast_tokens_cache[0].tree

        logger.error("Fn type is invalid")
        return None, None

    def is_constant_value(self, var, attr):
        """Check whether the value is a constant."""
        if var in self.global_namespace:
            module = self.global_namespace[var]
            if hasattr(module, attr):
                value = getattr(module, attr)
                # Check if value is constant.
                return isinstance(value, (int, float, bool))
        return False

    def get_namespace_symbol(self, var: str):
        """Get symbol type and namespace and symbol."""
        if var in self.closure_namespace:
            logger.debug(f"Found '{var}' in closure_namespace {self.closure_namespace.__str__()}")
            return self.closure_namespace, var
        if var in self.global_namespace:
            logger.debug(f"Found '{var}' in global_namespace {self.global_namespace.__str__()}")
            value = self.global_namespace[var]
            if self.is_unsupported_namespace(value):
                error_info = f"The builtin function '{var}' of python is not supported in graph mode."
                return None, error_info
            return self.global_namespace, var

        error_info = f"The name '{var}' is not defined in function '{self.function_name}'."
        return None, error_info

    def get_builtin_namespace_symbol(self, var: str):
        """Get mindspore builtin namespace and symbol."""
        if var in self.closure_namespace:
            logger.debug(f"Found '{var}' in closure_namespace {self.closure_namespace.__str__()}.")
            return self.closure_namespace, var
        if var in self.global_namespace:
            logger.debug(f"Found '{var}' in global_namespace {self.global_namespace.__str__()}.")
            value = self.global_namespace[var]
            value_str = value.__name__ if hasattr(value, '__name__') else str(value)
            logger.debug(f"value: {type(value)}, '{value_str}', hasattr(__name__): {hasattr(value, '__name__')}.")
            # To check if allowed to support.
            if self.is_unsupported_internal_type(value):
                support_info = self.global_namespace, var, value, SYNTAX_UNSUPPORTED_INTERNAL_TYPE
            elif self.is_unsupported_python_builtin_type(value):
                support_info = self.global_namespace, var, value, SYNTAX_UNSUPPORTED_EXTERNAL_TYPE
            elif self.is_unsupported_special_type(value):
                support_info = self.global_namespace, var, value, SYNTAX_UNSUPPORTED_SPECIAL_TYPE
            elif self.is_unsupported_namespace(value) or is_third_party_module(value):
                support_info = self.global_namespace, var, value, SYNTAX_UNSUPPORTED_NAMESPACE
            else:
                support_info = self.global_namespace, var, value, SYNTAX_SUPPORTED
            return support_info

        error_info = f"The name '{var}' is not defined, or not supported in graph mode."
        logger.debug(f"error_info: {error_info}")
        return None, error_info

    def analyze_super(self, class_type_node, subclass_instance):
        """Analyze super and return a class instance."""
        sub_class = type(subclass_instance)
        if class_type_node is None:
            return super(sub_class, subclass_instance)
        if isinstance(class_type_node, ast.Name):
            class_name = getattr(class_type_node, 'id')
        elif isinstance(class_type_node, ast.Attribute):
            class_name = getattr(class_type_node, 'attr')
        else:
            raise ValueError(f"The first argument of 'super()' must be a class type, "
                             f"but got {class_type_node.__class__.__name__}.")

        target_father_class = None
        for class_element in sub_class.mro():
            if class_element.__name__ == class_name:
                target_father_class = class_element
                break
        if target_father_class is None:
            raise ValueError(f"The second argument of 'super()' must be 'self', "
                             f"but got {subclass_instance}.")
        return super(target_father_class, subclass_instance)

    def get_location(self, node):
        """
        Get location of node start and end line no.

        Args:
            node: AST op node or tuple or List. This is a node in the ANF diagram,
                  here is the code location to get this node.

        Returns:
            List, [fileName, linestart, colstart, lineend, colend].
        """
        ret = [self.filename]
        err_exit = 0
        if isinstance(node, (list, tuple)):
            node_size = len(node)
            if node_size == 0:
                err_exit = 1
            else:
                start_node = node[0]
                end_node = node[-1]
        else:
            start_node = node
            end_node = node

        if err_exit == 0:
            if hasattr(start_node, "first_token") and \
                    hasattr(end_node, "last_token"):
                start_lineno, start_colno = start_node.first_token.start
                end_lineno, end_colno = end_node.last_token.end
                start_lineno += self.line_offset - 1
                start_colno += self.col_offset
                end_lineno += self.line_offset - 1
                end_colno += self.col_offset
                ret = ret + [start_lineno, start_colno, end_lineno, end_colno]
            else:
                ret = ret + [0, 0, 0, 0]
        return ret
