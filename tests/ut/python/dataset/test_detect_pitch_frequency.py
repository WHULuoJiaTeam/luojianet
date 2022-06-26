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
import numpy as np
import pytest

import mindspore.dataset as ds
import mindspore.dataset.audio.transforms as audio
from mindspore import log as logger


def count_unequal_element(data_expected, data_me, rtol, atol):
    assert data_expected.shape == data_me.shape
    total_count = len(data_expected.flatten())
    error = np.abs(data_expected - data_me)
    greater = np.greater(error, atol + np.abs(data_expected) * rtol)
    loss_count = np.count_nonzero(greater)
    assert (loss_count / total_count) < rtol, \
        "\ndata_expected_std:{0}\ndata_me_error:{1}\nloss:{2}". \
        format(data_expected[greater], data_me[greater], error[greater])


def test_detect_pitch_frequency_eager():
    """ mindspore eager mode normal testcase:detect_pitch_frequency op"""
    # Original waveform
    waveform = np.array([[2.716064453125e-03, 6.34765625e-03, 9.246826171875e-03, 1.0894775390625e-02,
                          1.1383056640625e-02, 1.1566162109375e-02, 1.3946533203125e-02, 1.55029296875e-02,
                          1.6143798828125e-02, 1.8402099609375e-02],
                         [1.7181396484375e-02, 1.59912109375e-02, 1.64794921875e-02, 1.5106201171875e-02,
                          1.385498046875e-02, 1.3458251953125e-02, 1.4190673828125e-02, 1.2847900390625e-02,
                          1.0528564453125e-02, 9.368896484375e-03]], dtype=np.float64)
    # Expect waveform
    expect_waveform = np.array(
        [[10., 10., 10.], [5., 5., 10.]], dtype=np.float64)
    detect_pitch_frequency_op = audio.DetectPitchFrequency(30, 0.1, 3, 5, 25)
    # Detect pitch frequence
    output = detect_pitch_frequency_op(waveform)
    count_unequal_element(expect_waveform, output, 0.0001, 0.0001)


def test_detect_pitch_frequency_pipeline():
    """ mindspore pipeline mode normal testcase:detect_pitch_frequency op"""
    # Original waveform
    waveform = np.array([[0.716064453125e-03, 5.34765625e-03, 6.246826171875e-03, 2.0894775390625e-02,
                          7.1383056640625e-02], [4.1566162109375e-02, 1.3946533203125e-02, 3.55029296875e-02,
                                                 0.6143798828125e-02, 3.8402099609375e-02]], dtype=np.float64)
    # Expect waveform
    expect_waveform = np.array([[10.0000], [7.5000]], dtype=np.float64)
    dataset = ds.NumpySlicesDataset(waveform, ["audio"], shuffle=False)
    detect_pitch_frequency_op = audio.DetectPitchFrequency(30, 0.1, 3, 5, 25)
    # Detect pitch frequence
    dataset = dataset.map(input_columns=["audio"],
                          operations=detect_pitch_frequency_op, num_parallel_workers=8)
    i = 0
    for item in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        count_unequal_element(expect_waveform[i, :],
                              item['audio'], 0.0001, 0.0001)
        i += 1


def test_detect_pitch_frequency_invalid_input():
    def test_invalid_input(test_name, sample_rate, frame_time, win_length, freq_low, freq_high, error, error_msg):
        logger.info(
            "Test DetectPitchFrequency with bad input: {0}".format(test_name))
        with pytest.raises(error) as error_info:
            audio.DetectPitchFrequency(
                sample_rate, frame_time, win_length, freq_low, freq_high)
        assert error_msg in str(error_info.value)

    test_invalid_input("invalid sample_rate parameter type as a float", 44100.5, 0.01, 30, 85, 3400, TypeError,
                       "Argument sample_rate with value 44100.5 is not of type [<class 'int'>],"
                       " but got <class 'float'>.")
    test_invalid_input("invalid sample_rate parameter type as a String", "44100", 0.01, 30, 85, 3400, TypeError,
                       "Argument sample_rate with value 44100 is not of type [<class 'int'>], but got <class 'str'>.")
    test_invalid_input("invalid frame_time parameter type as a String", 44100, "0.01", 30, 85, 3400, TypeError,
                       "Argument frame_time with value 0.01 is not of type [<class 'float'>, <class 'int'>],"
                       " but got <class 'str'>.")
    test_invalid_input("invalid win_length parameter type as a float", 44100, 0.01, 30.1, 85, 3400, TypeError,
                       "Argument win_length with value 30.1 is not of type [<class 'int'>], but got <class 'float'>.")
    test_invalid_input("invalid win_length parameter type as a String", 44100, 0.01, "30", 85, 3400, TypeError,
                       "Argument win_length with value 30 is not of type [<class 'int'>], but got <class 'str'>.")
    test_invalid_input("invalid freq_low parameter type as a String", 44100, 0.01, 30, "85", 3400, TypeError,
                       "Argument freq_low with value 85 is not of type [<class 'int'>, <class 'float'>],"
                       " but got <class 'str'>.")
    test_invalid_input("invalid freq_high parameter type as a String", 44100, 0.01, 30, 85, "3400", TypeError,
                       "Argument freq_high with value 3400 is not of type [<class 'int'>, <class 'float'>],"
                       " but got <class 'str'>.")
    test_invalid_input("invalid sample_rate parameter value", 0, 0.01, 30, 85, 3400, ValueError,
                       "Input sample_rate is not within the required interval of [-2147483648, 0) and (0, 2147483647].")
    test_invalid_input("invalid frame_time parameter value", 44100, 0, 30, 85, 3400, ValueError,
                       "Input frame_time is not within the required interval of (0, 16777216].")
    test_invalid_input("invalid win_length parameter value", 44100, 0.01, 0, 85, 3400, ValueError,
                       "Input win_length is not within the required interval of [1, 2147483647].")
    test_invalid_input("invalid freq_low parameter value", 44100, 0.01, 30, 0, 3400, ValueError,
                       "Input freq_low is not within the required interval of (0, 16777216].")
    test_invalid_input("invalid freq_high parameter value", 44100, 0.01, 30, 85, 0, ValueError,
                       "Input freq_high is not within the required interval of (0, 16777216].")


if __name__ == "__main__":
    test_detect_pitch_frequency_eager()
    test_detect_pitch_frequency_pipeline()
    test_detect_pitch_frequency_invalid_input()
