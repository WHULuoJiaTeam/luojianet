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

import numpy as np
import pytest

from mindspore import ops, nn, ParameterTuple, context, set_seed
from mindspore.train import DatasetHelper, connect_network_with_dataset
import mindspore.dataset as ds

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
set_seed(2)


def _exec_preprocess(network, is_train, dataset, dataset_sink_mode, epoch_num, sink_size):
    if dataset_sink_mode and not is_train:
        dataset.__loop_size__ = 1

    dataset_helper = DatasetHelper(dataset, dataset_sink_mode, sink_size, epoch_num)

    if dataset_sink_mode:
        network = connect_network_with_dataset(network, dataset_helper)

    return dataset_helper, network


def dynamic_shape_sink_process(network, dataset, is_train=True):
    # epoch_num=1 sink_size=1: exec one step
    dataset_sink_mode = True
    sink_size = 1
    epoch_num = 1
    dataset_helper, network = _exec_preprocess(network, is_train, dataset, dataset_sink_mode, epoch_num, sink_size)
    network.set_train(is_train)
    for inputs in dataset_helper:
        outputs = network(*inputs)
        return outputs


def fixed_shape_process(network, dataset, is_train=True):
    network.set_train(is_train)
    for inputs in dataset.create_tuple_iterator():
        outputs = network(*inputs)
        return outputs


def dataset_generator(data_list):
    for data in data_list:
        yield data


def compare(output, expect):
    if isinstance(output, (tuple, list)):
        assert isinstance(expect, (tuple, list))
        for output_, expect_ in zip(output, expect):
            if not compare(output_, expect_):
                return False
    else:
        if not np.allclose(output.asnumpy(), expect.asnumpy(), rtol=1.0e-4, atol=1.0e-4):
            return False
    return True


class GradNetWrtX(nn.Cell):
    def __init__(self, net):
        super(GradNetWrtX, self).__init__()
        self.net = net
        self.grad_op = ops.GradOperation(get_all=True, get_by_list=True, sens_param=True)
        self.params = ParameterTuple(net.trainable_params())

    def construct(self, *inputs):
        gradient_function = self.grad_op(self.net, self.params)
        return gradient_function(*inputs)


def common_func(dynamic_range, input_shape, data_type, op_net):
    data_list = []
    for i in dynamic_range:
        cur_data = []
        for data_shape in input_shape:
            cur_shape = [dim if dim is not None else i for dim in data_shape]
            cur_data.append(np.random.random(cur_shape).astype(data_type))
        data_list.append(tuple(cur_data))

    dynamic_data_map = {}
    for i, val in enumerate(input_shape):
        dynamic_data_map["data" + str(i + 1)] = val

    dataset = ds.GeneratorDataset(data_list, list(dynamic_data_map.keys()))
    dataset.set_dynamic_columns(columns=dynamic_data_map)
    net = GradNetWrtX(op_net)

    gradients = dynamic_shape_sink_process(net, dataset)
    gradients_cmp = fixed_shape_process(net, dataset)
    assert compare(gradients, gradients_cmp)


class LayerNormNet(nn.Cell):
    def __init__(self, last_dim):
        super(LayerNormNet, self).__init__()
        self.layernorm = nn.LayerNorm([last_dim])

    def construct(self, x):
        return self.layernorm(x)


class Conv2dNet(nn.Cell):
    def __init__(self):
        super(Conv2dNet, self).__init__()
        self.conv = nn.Conv2d(3, 10, 4, pad_mode="valid", has_bias=False, weight_init='normal')

    def construct(self, x):
        return self.conv(x)


class DropoutNet(nn.Cell):
    def __init__(self):
        super(DropoutNet, self).__init__()
        self.drop = nn.Dropout(0.5)
        self.relu = ops.ReLU()

    def construct(self, x):
        x = self.relu(x)
        return self.relu(self.drop(x))


class ReduceSumNet(nn.Cell):
    def __init__(self, axis=()):
        super(ReduceSumNet, self).__init__()
        self.reduce = ops.ReduceSum()
        self.axis = axis

    def construct(self, x):
        return self.reduce(x, self.axis)


class AddNet(nn.Cell):
    def construct(self, x, y):
        return ops.add(x, y)


class SoftmaxNet(nn.Cell):
    def construct(self, x):
        return ops.Softmax(axis=-1)(x)


class BatchNormNet(nn.Cell):
    def __init__(self, channels):
        super(BatchNormNet, self).__init__()
        self.bn = nn.BatchNorm2d(channels)

    def construct(self, x):
        return self.bn(x)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dynamic_layernorm():
    """
    Feature: Test LayerNorm and its backward. The input shape is dynamic.
    Description: The second dim of input is unknown.
    Expectation: Assert that results are consistent with fixed shape.
    """
    last_dim = 32
    batch_size = 16
    dynamic_range = range(20, 23)
    data_type = np.float32
    input_shape = [(batch_size, None, last_dim), (batch_size, None, last_dim)]
    net = LayerNormNet(last_dim)
    common_func(dynamic_range, input_shape, data_type, net)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dynamic_conv2d():
    """
    Feature: Test Conv2d and its backward. The input shape is dynamic.
    Description: Input dim of `H `or `W` is unknown. Conv2d's attr[pad] set to "valid".
    Expectation: Assert that results are consistent with fixed shape.
    """
    batch_size = 16
    dynamic_range = range(220, 224)
    data_type = np.float32
    input_shape = [(batch_size, 3, None, 112), (batch_size, 10, 219, 109)]
    net = Conv2dNet()
    common_func(dynamic_range, input_shape, data_type, net)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dynamic_dropout():
    """
    Feature: Test Dropout and its backward.
    Description: The input shape is dynamic.
    Expectation: Dropout result is random, assert gradient shape.
    """
    batch_size = 16
    data_list = []
    for i in range(48, 50):
        data_list.append((np.random.rand(batch_size, i, 256).astype(np.float32),
                          np.random.rand(batch_size, i, 256).astype(np.float32)))

    dataset = ds.GeneratorDataset(data_list, ["data1", "data2"])
    dataset.set_dynamic_columns(columns={"data1": [batch_size, None, 256], "data2": [batch_size, None, 256]})
    net = GradNetWrtX(DropoutNet())
    net.set_train()
    gradients = dynamic_shape_sink_process(net, dataset)
    assert gradients[0][0].shape == (batch_size, 49, 256)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dynamic_reducesum1():
    """
    Feature: Test ReduceSum and its backward. The input shape is dynamic.
    Description: axis=(), result of reduce sum is a scalar, gradient shape is the same as input, value is all one.
    Expectation: Assert gradient shape.
    """
    batch_size = 16
    data_list = []
    for i in range(48, 50):
        data_list.append((np.random.rand(batch_size, i, i + 2).astype(np.float32),
                          np.array(1).astype(np.float32)))

    dataset = ds.GeneratorDataset(data_list, ["data1", "data2"])
    dataset.set_dynamic_columns(columns={"data1": [batch_size, None, None], "data2": []})
    net = GradNetWrtX(ReduceSumNet())

    gradients = dynamic_shape_sink_process(net, dataset)
    assert gradients[0][0].shape == (batch_size, 49, 51)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dynamic_reducesum2():
    """
    Feature: Test ReduceSum and its backward. The input shape is dynamic.
    Description: axis is a scalar, not tuple.
    Expectation: Assert that results are consistent with fixed shape.
    """
    batch_size = 16
    data_list = []
    for i in range(48, 50):
        data_list.append((np.random.rand(batch_size, i, i + 2).astype(np.float32),
                          np.random.rand(batch_size, i + 2).astype(np.float32)))

    dataset = ds.GeneratorDataset(data_list, ["data1", "data2"])
    dataset.set_dynamic_columns(columns={"data1": [batch_size, None, None], "data2": [batch_size, None]})
    net = GradNetWrtX(ReduceSumNet(1))

    gradients = dynamic_shape_sink_process(net, dataset)
    gradients_cmp = fixed_shape_process(net, dataset)
    assert compare(gradients, gradients_cmp)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dynamic_add1():
    """
    Feature: Test Add and its backward. The input shape is dynamic.
    Description: Second input is a scalar. Shape of forward result is the same as first input.
    Expectation: Assert that results are consistent with fixed shape.
    """
    batch_size = 16
    data_list = []
    for i in range(48, 50):
        data_list.append((np.random.rand(batch_size, i).astype(np.float32),
                          np.array(1).astype(np.float32),
                          np.random.rand(batch_size, i).astype(np.float32)))

    dataset = ds.GeneratorDataset(data_list, ["data1", "data2", "data3"])
    dataset.set_dynamic_columns(columns={"data1": [batch_size, None], "data2": [], "data3": [batch_size, None]})
    net = GradNetWrtX(AddNet())

    gradients = dynamic_shape_sink_process(net, dataset)
    gradients_cmp = fixed_shape_process(net, dataset)
    assert compare(gradients, gradients_cmp)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dynamic_add2():
    """
    Feature: Test Add and its backward. The input shape is dynamic.
    Description: Shape of forward result is the same as first input. The axis of reduce_sum in add's bprop will be a
                 empty Tensor.
    Expectation: Assert that results are consistent with fixed shape.
    """
    batch_size = 16
    data_list = []
    for i in range(48, 50):
        data_list.append((np.random.rand(batch_size, 2, i).astype(np.float32),
                          np.random.rand(2, i).astype(np.float32),
                          np.random.rand(batch_size, 2, i).astype(np.float32)))

    dataset = ds.GeneratorDataset(data_list, ["data1", "data2", "data3"])
    dataset.set_dynamic_columns(columns=
                                {"data1": [batch_size, 2, None], "data2": [2, None], "data3": [batch_size, 2, None]})
    net = GradNetWrtX(AddNet())

    gradients = dynamic_shape_sink_process(net, dataset)
    gradients_cmp = fixed_shape_process(net, dataset)
    assert compare(gradients, gradients_cmp)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dynamic_softmax():
    """
    Feature: Test Softmax and its backward. The input shape is dynamic.
    Description: The input shape is dynamic.
    Expectation: Assert that results are consistent with fixed shape.
    """
    batch_size = 16
    dynamic_range = range(48, 50)
    data_type = np.float32
    input_shape = [(batch_size, 2, None), (batch_size, 2, None)]
    net = SoftmaxNet()
    common_func(dynamic_range, input_shape, data_type, net)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_dynamic_batchnorm():
    """
    Feature: Test Batchnorm2D and its backward. The input shape is dynamic.
    Description: The input shape is dynamic.
    Expectation: Assert that results are consistent with fixed shape.
    """
    batch_size = 1
    dynamic_range = range(2, 64)
    data_type = np.float32
    input_shape = [(batch_size, 256, None, 12), (batch_size, 256, None, 12)]
    net = BatchNormNet(256)
    common_func(dynamic_range, input_shape, data_type, net)
