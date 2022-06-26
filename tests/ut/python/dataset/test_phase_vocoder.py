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
# ==============================================================================

import numpy as np
import pytest

import mindspore.dataset as ds
import mindspore.dataset.audio.transforms as audio
from mindspore import log as logger


def gen(shape):
    np.random.seed(0)
    data = np.random.random(shape)
    yield (np.array(data, dtype=np.float32),)


def count_unequal_element(data_expected, data_me, rtol, atol):
    assert data_expected.shape == data_me.shape
    total_count = len(data_expected.flatten())
    error = np.abs(data_expected - data_me)
    greater = np.greater(error, atol + np.abs(data_expected) * rtol)
    loss_count = np.count_nonzero(greater)
    assert (loss_count / total_count) < rtol, \
        "\ndata_expected_std:{0}\ndata_me_error:{1}\nloss:{2}". \
        format(data_expected[greater], data_me[greater], error[greater])


def allclose_nparray(data_expected, data_me, rtol, atol, equal_nan=True):
    if np.any(np.isnan(data_expected)):
        assert np.allclose(data_me, data_expected, rtol, atol, equal_nan=equal_nan)
    elif not np.allclose(data_me, data_expected, rtol, atol, equal_nan=equal_nan):
        count_unequal_element(data_expected, data_me, rtol, atol)


def test_phase_vocoder_compare():
    """
    Feature: PhaseVocoder
    Description: mindspore eager mode checking precision
    Expectation: the returned result is as expected
    """
    indata_0 = np.array([[[[0.43189, 2.3049924],
                           [-0.01202229, 0.9176453],
                           [-0.6258611, 0.66475236],
                           [0.13541847, 1.2829605],
                           [0.9725325, 1.1669061]],
                          [[-0.35001752, -1.0989336],
                           [-1.4930767, 0.86829656],
                           [0.3355314, -0.41216415],
                           [-1.1828239, 1.0075365],
                           [-0.19343425, 0.38364533]]]]).astype('float32')
    indata_1 = np.array([[[[0.43189, 2.3049924],
                           [-0.01202229, 0.9176453],
                           [-0.6258611, 0.66475236],
                           [0.13541847, 1.2829605],
                           [0.9725325, 1.1669061]],
                          [[-0.35001752, -1.0989336],
                           [-1.4930767, 0.86829656],
                           [0.3355314, -0.41216415],
                           [-1.1828239, 1.0075365],
                           [-0.19343425, 0.38364533]]]]).astype('float64')
    rate = 2.
    phase_advance_0 = np.array([[0.0000], [3.9270]]).astype('float32')
    op_0 = audio.PhaseVocoder(rate, phase_advance_0)
    phase_advance_1 = np.array([[0.0000], [3.9270]]).astype('float64')
    op_1 = audio.PhaseVocoder(rate, phase_advance_1)
    outdata_0 = op_0(indata_0)
    outdata_1 = op_1(indata_1)
    stand_outdata = np.array([[[[0.43189007, 2.3049924],
                                [-0.01196056, 0.9129374],
                                [1.1385509, 1.00558]],
                               [[-0.35001755, -1.0989336],
                                [-0.4594292, 0.26718047],
                                [0.404371, -0.14520557]]]]).astype('float32')
    allclose_nparray(outdata_0, stand_outdata, 0.0001, 0.0001)
    allclose_nparray(outdata_1, stand_outdata, 0.0001, 0.0001)


def test_phase_vocoder_eager():
    """
    Feature: PhaseVocoder
    Description: mindspore eager mode with normal testcase
    Expectation: the returned result is as expected
    """
    logger.info("test PhaseVocoder op in eager mode")
    stft = next(gen([10, 10, 10, 2]))[0]
    out_put = audio.PhaseVocoder(1.3, np.random.randn(10, 1).astype('float32'))(stft)
    assert out_put.shape == (10, 10, 8, 2)


def test_phase_vocoder_pipeline():
    """
    Feature: PhaseVocoder
    Description: mindspore pipeline mode with normal testcase
    Expectation: the returned result is as expected
    """
    logger.info("test PhaseVocoder op in pipeline mode")

    generator = gen([32, 33, 333, 2])
    data1 = ds.GeneratorDataset(source=generator, column_names=["input"])

    transforms = [audio.PhaseVocoder(0.8, np.random.randn(33, 1).astype('float32'))]
    data1 = data1.map(operations=transforms, input_columns=["input"])

    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        out_put = item["input"]
    assert out_put.shape == (32, 33, 417, 2)


def test_phase_vocoder_invalid_input():
    """
    Feature: PhaseVocoder
    Description: mindspore eager mode with invalid input
    Expectation: the returned result is as expected
    """
    def test_invalid_param(test_name, rate, phase_advance, error, error_msg):
        logger.info("Test PhaseVocoder with wrong params: {0}".format(test_name))
        with pytest.raises(error) as error_info:
            _ = audio.PhaseVocoder(rate, phase_advance)
        assert error_msg in str(error_info.value)

    def test_invalid_input(test_name, spec, rate, phase_advance, error, error_msg):
        logger.info("Test PhaseVocoder with wrong params: {0}".format(test_name))
        with pytest.raises(error) as error_info:
            _ = audio.PhaseVocoder(rate, phase_advance)(spec)
        assert error_msg in str(error_info.value)

    test_invalid_param("invalid phase_advance", 2, None, TypeError,
                       "Argument phase_advance with value None is not of type")
    test_invalid_param("invalid phase_advance", 0, np.random.randn(4, 1), ValueError,
                       "Input rate is not within the required interval of (0, 16777216].")
    spec = next(gen([1, 2, 2]))[0]
    test_invalid_input("invalid phase_advance", spec, 1.23, np.random.randn(4), RuntimeError,
                       "PhaseVocoder: invalid parameter, 'phase_advance' should be in shape of <freq, 1>.")
    test_invalid_input("invalid phase_advance", spec, 1.1, np.random.randn(4, 4, 1), RuntimeError,
                       "PhaseVocoder: invalid parameter, 'phase_advance' should be in shape of <freq, 1>.")
    test_invalid_input("invalid input tensor", spec, 2, np.random.randn(3, 1), RuntimeError,
                       "PhaseVocoder: invalid parameter, 'first dimension of 'phase_advance'' should be equal")
    input_tensor = np.random.randn(4, 4, 2).astype('float32')
    input_phase_advance = np.random.randn(4, 1).astype('float64')
    test_invalid_input("invalid input tensor", input_tensor, 2, input_phase_advance, RuntimeError,
                       "PhaseVocoder: invalid parameter, data type of phase_advance should be equal to data")


if __name__ == "__main__":
    test_phase_vocoder_compare()
    test_phase_vocoder_eager()
    test_phase_vocoder_pipeline()
    test_phase_vocoder_invalid_input()
