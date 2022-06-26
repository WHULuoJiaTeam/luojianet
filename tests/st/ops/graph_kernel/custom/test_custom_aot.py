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
# ============================================================================

import os
import platform
import numpy as np
import pytest
from mindspore import context, Tensor
from mindspore.common import dtype as mstype
from mindspore.nn import Cell
import mindspore.ops as ops
from mindspore.ops import DataType, CustomRegOp


class AOTSingleOutputNet(Cell):
    def __init__(self, func, out_shapes, out_types, reg=None):
        super(AOTSingleOutputNet, self).__init__()

        self.program = ops.Custom(func, out_shapes, out_types, "aot", reg_info=reg)

    def construct(self, x, y):
        return self.program(x, y)


def get_file_path_gpu(cuda, so):
    dir_path = os.path.dirname(os.path.abspath(__file__))
    cmd = "nvcc --shared -Xcompiler -fPIC  -o " + dir_path + "/aot_test_files/" + so + " " + dir_path + \
          "/aot_test_files/" + cuda
    func_path = dir_path + "/aot_test_files/" + so
    return cmd, func_path


def get_file_path_cpu(cc, so):
    dir_path = os.path.dirname(os.path.abspath(__file__))
    cmd = "g++ -std=c++17 --shared -fPIC -o " + dir_path + "/aot_test_files/" + so + " " + dir_path + \
          "/aot_test_files/" + cc
    func_path = dir_path + "/aot_test_files/" + so
    return cmd, func_path


def check_exec_file(cmd, func_path, source, execf):
    with os.popen(cmd) as f:
        r = f.read()
    if os.path.exists(func_path) and not r:
        pass
    else:
        if os.path.exists(func_path):
            os.remove(func_path)
        assert False, "Failed to compile " + source + " to " + execf


def aot_single_output(get_file_path, source, execf, reg):
    shape = (4, 5)
    input_x = np.random.normal(0, 1, shape).astype(np.float32)
    input_y = np.random.normal(0, 1, shape).astype(np.float32)
    cmd, func_path = get_file_path(source, execf)
    check_exec_file(cmd, func_path, source, execf)
    try:
        test = AOTSingleOutputNet(func_path + ":CustomAdd", (shape,), (mstype.float32,), reg)
        output = test(Tensor(input_x), Tensor(input_y))[0]
    except Exception as e:
        if os.path.exists(func_path):
            os.remove(func_path)
        raise e
    os.remove(func_path)
    assert np.allclose(input_x + input_y, output.asnumpy(), 0.001, 0.001)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_aot_single_output_gpu():
    """
    Feature: custom aot operator, multiple inputs, single output, GPU
    Description: pre-compile xxx.cu to xxx.so, custom operator launches xxx.so
    Expectation: nn result matches numpy result
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    aot_single_output(get_file_path_gpu, "add.cu", "add.so", None)


add_cpu_info = CustomRegOp() \
    .input(0, "x1") \
    .input(1, "x2") \
    .output(0, "y") \
    .dtype_format(DataType.None_None, DataType.None_None, DataType.None_None) \
    .target("CPU") \
    .get_op_info()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_aot_single_output_cpu():
    """
    Feature: custom aot operator, multiple inputs, single output, CPU, GRAPH_MODE
    Description: pre-compile xxx.cc to xxx.so, custom operator launches xxx.so
    Expectation: nn result matches numpy result
    """
    sys = platform.system()
    if sys == 'Windows':
        pass
    else:
        context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
        aot_single_output(get_file_path_cpu, "add.cc", "add.so", add_cpu_info)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_reorganize():
    """
    Feature: custom aot operator, multiple inputs(dtypes:float32,int64_t), single output, GPU
    Description: pre-compile xxx.cu to xxx.so, custom operator launches xxx.so
    Expectation: nn result matches numpy result
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    shape = [5]
    input_x = np.array([1.0, 4.0, 9.0, 16.0, 25.0]).astype(np.float32)
    input_y = np.array([3, 2, 0, 1, 4]).astype(np.int64)
    expect = np.array([16.0, 9.0, 1.0, 4.0, 25.0]).astype(np.float32)

    cmd, func_path = get_file_path_gpu("reorganize.cu", "reorganize.so")
    check_exec_file(cmd, func_path, "reorganize.cu", "reorganize.so")
    try:
        test = AOTSingleOutputNet(func_path + ":CustomReorganize", (shape,), (mstype.float32,), None)
        output = test(Tensor(input_x), Tensor(input_y))[0]
    except Exception as e:
        if os.path.exists(func_path):
            os.remove(func_path)
        raise e
    os.remove(func_path)
    assert np.allclose(expect, output.asnumpy(), 0.001, 0.001)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_hetero_square_mul():
    """
    Feature: custom aot operator, multiple inputs(dtypes:float32,float16), single output(dtype:float16), GPU
    Description: pre-compile xxx.cu to xxx.so, custom operator launches xxx.so
    Expectation: nn result matches numpy result
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    shape = [5]
    input_x = np.random.normal(0, 1, shape).astype(np.float32)
    input_y = np.random.normal(0, 1, shape).astype(np.float16)
    expect = (input_x * input_x * input_y.astype(np.float32)).astype(np.float16)
    cmd, func_path = get_file_path_gpu("hetero_square_mul.cu", "hetero_square_mul.so")
    check_exec_file(cmd, func_path, "hetero_square_mul.cu", "hetero_square_mul.so")
    try:
        test = AOTSingleOutputNet(func_path + ":CustomHSquareMul", (shape,), (mstype.float16,), None)
        output = test(Tensor(input_x), Tensor(input_y))[0]
    except Exception as e:
        if os.path.exists(func_path):
            os.remove(func_path)
        raise e
    os.remove(func_path)
    assert np.allclose(expect, output.asnumpy(), 0.001, 0.001)


class SquareGradNet(Cell):
    def __init__(self, func, out_shapes, out_types, bprop, reg):
        super(SquareGradNet, self).__init__()
        self.square = ops.Custom(func, out_shapes, out_types, "aot", bprop, reg)

    def construct(self, x):
        res = self.square(x)
        res2 = self.square(res)
        return res2


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_square_py_bprop():
    """
    Feature: custom aot operator, bprop(pyfunc), GPU
    Description: pre-compile xxx.cu to xxx.so, custom operator launches xxx.so
    Expectation: nn result matches numpy result
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    x = np.array([1.0, 4.0, 9.0]).astype(np.float32)
    sens = np.array([1.0, 1.0, 1.0]).astype(np.float32)
    expect = np.array([4.0, 256.0, 2916.0]).astype(np.float32)
    cmd, func_path = get_file_path_gpu("square.cu", "square_py.so")
    check_exec_file(cmd, func_path, "square.cu", "square_py.so")

    def bprop(x, out, dout):
        gradient = x * 2
        dx = gradient * dout
        return (dx,)

    try:
        net = SquareGradNet(func_path + ":CustomSquare", (3,), mstype.float32, bprop=bprop, reg=None)
        dx = ops.GradOperation(sens_param=True)(net)(Tensor(x), Tensor(sens))
    except Exception as e:
        if os.path.exists(func_path):
            os.remove(func_path)
        raise e
    os.remove(func_path)
    dx_np = dx.asnumpy()
    assert np.allclose(expect, dx_np, 0.0001, 0.0001)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_square_aot_bprop():
    """
    Feature: custom aot operator, bprop(Cell), GPU
    Description: pre-compile xxx.cu to xxx.so, custom operator launches xxx.so
    Expectation: nn result matches numpy result
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    x = np.array([1.0, 4.0, 9.0]).astype(np.float32)
    sens = np.array([1.0, 1.0, 1.0]).astype(np.float32)
    expect = np.array([4.0, 256.0, 2916.0]).astype(np.float32)
    cmd_bprop, func_path_bprop = get_file_path_gpu("square_bprop.cu", "square_bprop.so")
    check_exec_file(cmd_bprop, func_path_bprop, "square_bprop.cu", "square_bprop.so")
    try:
        aot_bprop = ops.Custom(func_path_bprop + ":CustomSquareBprop",
                               (3,), mstype.float32, "aot", reg_info=None)
    except Exception as e:
        if os.path.exists(func_path_bprop):
            os.remove(func_path_bprop)
        raise e

    def bprop(x, out, dout):
        res = aot_bprop(x, out, dout)
        return (res,)

    cmd, func_path = get_file_path_gpu("square.cu", "square.so")
    check_exec_file(cmd, func_path, "square_bprop.cu", "square_bprop.so")
    try:
        net = SquareGradNet(func_path + ":CustomSquare", (3,), mstype.float32, bprop=bprop, reg=None)
        dx = ops.GradOperation(sens_param=True)(net)(Tensor(x), Tensor(sens))
    except Exception as e:
        if os.path.exists(func_path):
            os.remove(func_path)
        if os.path.exists(func_path_bprop):
            os.remove(func_path_bprop)
        raise e
    os.remove(func_path)
    os.remove(func_path_bprop)
    dx_np = dx.asnumpy()
    assert np.allclose(expect, dx_np, 0.0001, 0.0001)


class AOTMultiOutputNet(Cell):
    def __init__(self, func, out_shapes, out_types, bprop=None, reg=None):
        super(AOTMultiOutputNet, self).__init__()

        self.program = ops.Custom(func, out_shapes, out_types, "aot", bprop, reg)
        self.add = ops.Add()
        self.mul = ops.Mul()

    def construct(self, x, y):
        aot = self.program(x, y)
        add_res = self.add(aot[0], aot[1])
        mul_res = self.mul(add_res, aot[2])
        return mul_res


multioutput_gpu_info = CustomRegOp() \
    .input(0, "x1") \
    .input(1, "x2") \
    .output(0, "y1") \
    .output(1, "y2") \
    .output(2, "y3") \
    .dtype_format(DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default, DataType.F32_Default, DataType.F32_Default) \
    .target("GPU") \
    .get_op_info()

multioutput_bprop_gpu_info = CustomRegOp() \
    .input(0, "x1") \
    .input(1, "x2") \
    .input(2, "x3") \
    .input(3, "x4") \
    .input(4, "x5") \
    .output(0, "y1") \
    .output(1, "y2") \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default, DataType.F32_Default,
                  DataType.F32_Default, DataType.F32_Default, DataType.F32_Default) \
    .target("GPU") \
    .get_op_info()


def add_mul_div_bprop(source, execf, source_prop, execf_prop):
    x = np.array([1.0, 4.0, 9.0]).astype(np.float32)
    y = np.array([1.0, 1.0, 1.0]).astype(np.float32)
    sens = np.array([1.0, 1.0, 1.0]).astype(np.float32)
    expect_dx = np.array([5.0, 17.0, 37.0]).astype(np.float32)
    expect_dy = np.array([-1.0, -16.0, -81.0]).astype(np.float32)

    cmd_bprop, func_path_bprop = get_file_path_gpu(source_prop, execf_prop)
    check_exec_file(cmd_bprop, func_path_bprop, source_prop, execf_prop)
    try:
        aot_bprop = ops.Custom(func_path_bprop + ":CustomAddMulDivBprop",
                               ([3], [3]), (mstype.float32, mstype.float32), "aot", reg_info=multioutput_bprop_gpu_info)
    except Exception as e:
        if os.path.exists(func_path_bprop):
            os.remove(func_path_bprop)
        raise e

    def bprop(x, y, out, dout):
        res = aot_bprop(x, y, dout[0], dout[1], dout[2])
        return res

    cmd, func_path = get_file_path_gpu(source, execf)
    check_exec_file(cmd, func_path, source, execf)
    try:
        net = AOTMultiOutputNet(func_path + ":CustomAddMulDiv", ([3], [3], [3]),
                                (mstype.float32, mstype.float32, mstype.float32), bprop=bprop, reg=multioutput_gpu_info)

        dx, dy = ops.GradOperation(sens_param=True, get_all=True)(net)(Tensor(x), Tensor(y), Tensor(sens))
    except Exception as e:
        if os.path.exists(func_path):
            os.remove(func_path)
        if os.path.exists(func_path_bprop):
            os.remove(func_path_bprop)
        raise e
    os.remove(func_path)
    os.remove(func_path_bprop)
    dx_np = dx.asnumpy()
    dy_np = dy.asnumpy()
    assert np.allclose(expect_dx, dx_np, 0.0001, 0.0001)
    assert np.allclose(expect_dy, dy_np, 0.0001, 0.0001)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
def test_add_mul_div_bprop_graph():
    """
    Feature: custom aot operator, bprop(Cell), multiple outputs, GPU, GRAPH_MODE
    Description: pre-compile xxx.cu to xxx.so, custom operator launches xxx.so
    Expectation: nn result matches numpy result
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    add_mul_div_bprop("add_mul_div.cu", "add_mul_div.so", "add_mul_div_bprop.cu", "add_mul_div_bprop.so")


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
def test_add_mul_div_bprop_pynative():
    """
    Feature: custom aot operator, bprop(Cell), multiple outputs, GPU, PYNATIVE_MODE
    Description: pre-compile xxx.cu to xxx.so, custom operator launches xxx.so
    Expectation: nn result matches numpy result
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    add_mul_div_bprop("add_mul_div.cu", "add_mul_div_pynative.so",
                      "add_mul_div_bprop.cu", "add_mul_div_bprop_pynative.so")
