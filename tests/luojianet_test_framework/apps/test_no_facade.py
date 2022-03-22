# Copyright 2021, 2022 LuoJiaNET Research and Development Group, Wuhan University
# Copyright 2021, 2022 Huawei Technologies Co., Ltd
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

"""Core verification config for luojianet_ms_test_framework"""

# pylint: disable=invalid-name

import numpy as np

from ..luojianet_ms_test import luojianet_ms_test
from ..pipeline.forward.verify_shapetype import pipeline_for_verify_shapetype_for_group_by_group_config


# from ...vm_impl import *

# functions could be operations or NN cell
def Add(x, y):
    return x + y


def Sub(x, y):
    return x - y


def Mul(x, y):
    return x * y


def Div(x, y):
    return x / y


# given Add/Sub/Mul/Div operations, verify that the result's shape and type is correct
verification_set = {
    'function': [
        {
            'id': 'Add',
            'group': 'op-test',
            'block': Add
        },
        {
            'id': 'Sub',
            'group': 'op-test',
            'block': Sub
        },
        {
            'id': 'Mul',
            'group': 'op-test',
            'block': Mul
        },
        {
            'id': 'Div',
            'group': 'op-test',
            'block': Div
        }
    ],
    'inputs': [
        {
            'id': '1',
            'group': 'op-test',
            'desc_inputs': [
                np.array([[1, 1], [1, 1]]).astype(np.float32),
                np.array([[2, 2], [2, 2]]).astype(np.float32)
            ]
        },
        {
            'id': '2',
            'group': 'op-test',
            'desc_inputs': [
                np.array([[3, 3], [3, 3]]).astype(np.float32),
                np.array([[4, 4], [4, 4]]).astype(np.float32)
            ]
        },
        {
            'id': '3',
            'group': 'op-test',
            'desc_inputs': [
                np.array([[5, 5], [5, 5]]).astype(np.float32),
                np.array([[6, 6], [6, 6]]).astype(np.float32)
            ]
        }
    ],
    'expect': [
        {
            'id': '1',
            'group': 'op-test-op-test',
            'desc_expect': {
                'shape_type': [
                    {
                        'type': np.float32,
                        'shape': (2, 2)
                    }
                ]
            }
        }
    ],
    'ext': {}
}


@luojianet_ms_test(pipeline_for_verify_shapetype_for_group_by_group_config)
def test_no_facade():
    return verification_set
