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
# ==============================================================================

import math
import pytest
import numpy as np
from mindspore import context
from mindspore import nn
from mindspore import Tensor
from mindspore.common.parameter import ParameterTuple
from mindspore.common.parameter import Parameter
from mindspore.ops import composite as c


class GradOfAllInputsAndParams(nn.Cell):
    def __init__(self, network, sens_param):
        super().__init__()
        self.grad = c.GradOperation(get_all=True, get_by_list=True, sens_param=sens_param)
        self.network = network
        self.params = ParameterTuple(self.network.trainable_params())

    def construct(self, *inputs):
        gout = self.grad(self.network, self.params)(*inputs)
        return gout


class LSTM(nn.Cell):
    def __init__(self, input_s, hidden_s, num_layers, has_bias, batch_first, bidirectional, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_s, hidden_size=hidden_s, num_layers=num_layers, has_bias=has_bias,
                            batch_first=batch_first, bidirectional=bidirectional, dropout=dropout)

    def construct(self, inp, h0, c0):
        return self.lstm(inp, (h0, c0))


class LSTMWeightBias():
    def __init__(self, num_layers, has_bias, input_size, num_directions, hidden_size, bidirectional):
        self.num_layers = num_layers
        self.has_bias = has_bias
        self.input_size = input_size
        self.num_directions = num_directions
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

    def get_weight_bias(self):
        gate_size = 4 * self.hidden_size

        w_ih_list = []
        w_hh_list = []
        b_ih_list = []
        b_hh_list = []
        stdv = 1 / math.sqrt(self.hidden_size)
        for layer in range(self.num_layers):
            for direction in range(self.num_directions):
                layer_input_size = self.input_size if layer == 0 else self.hidden_size * self.num_directions
                suffix = '_reverse' if direction == 1 else ''

                w_ih_list.append(Parameter(
                    Tensor(np.random.uniform(-stdv, stdv, (gate_size, layer_input_size)).astype(np.float32)),
                    name='weight_ih_l{}{}'.format(layer, suffix)))
                w_hh_list.append(Parameter(
                    Tensor(np.random.uniform(-stdv, stdv, (gate_size, self.hidden_size)).astype(np.float32)),
                    name='weight_hh_l{}{}'.format(layer, suffix)))
                if self.has_bias:
                    b_ih_list.append(Parameter(
                        Tensor(np.random.uniform(-stdv, stdv, (gate_size)).astype(np.float32)),
                        name='bias_ih_l{}{}'.format(layer, suffix)))
                    b_hh_list.append(Parameter(
                        Tensor(np.random.uniform(-stdv, stdv, (gate_size)).astype(np.float32)),
                        name='bias_hh_l{}{}'.format(layer, suffix)))
        w_ih_list = ParameterTuple(w_ih_list)
        w_hh_list = ParameterTuple(w_hh_list)
        b_ih_list = ParameterTuple(b_ih_list)
        b_hh_list = ParameterTuple(b_hh_list)
        return w_ih_list, w_hh_list, b_ih_list, b_hh_list

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_sit_lstm_forward_input_3_32_32_is_32_hs_16():
    """
    Feature: LSTM forward
    Description: LSTM with input (3, 32, 32)
    Expectation: Graph mode equal to pynative mode
    """
    input_s = 32
    hidden_s = 16
    has_bias = True
    bidirectional = False
    num_layers = 1
    num_directions = 1

    fact = LSTMWeightBias(num_layers, has_bias, input_s, num_directions, hidden_s, bidirectional)
    w_ih_list, w_hh_list, b_ih_list, b_hh_list = fact.get_weight_bias()

    h0 = Tensor(np.random.randn(num_layers * 1, 32, 16).astype(np.float32))
    c0 = Tensor(np.random.randn(num_layers * 1, 32, 16).astype(np.float32))
    input_ms = Tensor(np.random.randn(3, 32, 32).astype(np.float32))

    # graph mode
    context.set_context(mode=context.GRAPH_MODE)
    net = LSTM(input_s=input_s, hidden_s=16, num_layers=num_layers, has_bias=has_bias, batch_first=False,
               bidirectional=bidirectional, dropout=0.0)
    net.lstm.w_ih_list = w_ih_list
    net.lstm.w_hh_list = w_hh_list
    net.lstm.b_ih_list = b_ih_list
    net.lstm.b_hh_list = b_hh_list
    out, (hy, cy) = net(input_ms, h0, c0)

    # pynative mode
    context.set_context(mode=context.PYNATIVE_MODE)
    net_pynative = LSTM(input_s=input_s, hidden_s=16, num_layers=num_layers, has_bias=has_bias, batch_first=False,
                        bidirectional=bidirectional, dropout=0.0)
    net_pynative.lstm.w_ih_list = w_ih_list
    net_pynative.lstm.w_hh_list = w_hh_list
    net_pynative.lstm.b_ih_list = b_ih_list
    net_pynative.lstm.b_hh_list = b_hh_list
    out_pynative, (hy_pynative, cy_pynative) = net_pynative(input_ms, h0, c0)
    context.set_context(mode=context.GRAPH_MODE)

    assert np.allclose(out.asnumpy(), out_pynative.asnumpy(), 0.0001, 0.0001)
    assert np.allclose(hy.asnumpy(), hy_pynative.asnumpy(), 0.0001, 0.0001)
    assert np.allclose(cy.asnumpy(), cy_pynative.asnumpy(), 0.0001, 0.0001)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_sit_lstm_grad_input_3_32_32_is_32_hs_16():
    """
    Feature: LSTM backward
    Description: LSTM with input (3, 32, 32)
    Expectation: Graph mode equal to pynative mode
    """
    input_s = 32
    hidden_s = 16
    has_bias = True
    bidirectional = False
    num_layers = 1
    num_directions = 1

    fact = LSTMWeightBias(num_layers, has_bias, input_s, num_directions, hidden_s, bidirectional)
    w_ih_list, w_hh_list, b_ih_list, b_hh_list = fact.get_weight_bias()

    h0 = Tensor(np.random.randn(num_layers * 1, 32, 16).astype(np.float32))
    c0 = Tensor(np.random.randn(num_layers * 1, 32, 16).astype(np.float32))
    input_ms = Tensor(np.random.randn(3, 32, 32).astype(np.float32))

    # graph mode
    context.set_context(mode=context.GRAPH_MODE)
    net = LSTM(input_s=input_s, hidden_s=16, num_layers=num_layers, has_bias=has_bias, batch_first=False,
               bidirectional=bidirectional, dropout=0.0)
    net.lstm.w_ih_list = w_ih_list
    net.lstm.w_hh_list = w_hh_list
    net.lstm.b_ih_list = b_ih_list
    net.lstm.b_hh_list = b_hh_list

    grad_net_inp = GradOfAllInputsAndParams(net, sens_param=False)
    grad_net_inp.set_train()
    out_grad, _ = grad_net_inp(input_ms, h0, c0)
    x_grad = out_grad[0].asnumpy()
    h_grad = out_grad[1].asnumpy()
    c_grad = out_grad[2].asnumpy()

    # pynative mode
    context.set_context(mode=context.PYNATIVE_MODE)
    net_pynative = LSTM(input_s=input_s, hidden_s=16, num_layers=num_layers, has_bias=has_bias, batch_first=False,
                        bidirectional=bidirectional, dropout=0.0)
    net_pynative.lstm.w_ih_list = w_ih_list
    net_pynative.lstm.w_hh_list = w_hh_list
    net_pynative.lstm.b_ih_list = b_ih_list
    net_pynative.lstm.b_hh_list = b_hh_list

    grad_net_inp_pynative = GradOfAllInputsAndParams(net_pynative, sens_param=False)
    grad_net_inp_pynative.set_train()
    out_grad_pynative, _ = grad_net_inp_pynative(input_ms, h0, c0)
    x_grad_pynative = out_grad_pynative[0].asnumpy()
    h_grad_pynative = out_grad_pynative[1].asnumpy()
    c_grad_pynative = out_grad_pynative[2].asnumpy()
    context.set_context(mode=context.GRAPH_MODE)

    assert np.allclose(x_grad, x_grad_pynative, 0.001, 0.001)
    assert np.allclose(h_grad, h_grad_pynative, 0.001, 0.001)
    assert np.allclose(c_grad, c_grad_pynative, 0.001, 0.001)
