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
"""
Testing LowpassBiquad op in DE
"""

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
    assert (loss_count / total_count) < rtol, "\ndata_expected_std:{0}\ndata_me_error:{1}\nloss:{2}".format(
        data_expected[greater], data_me[greater], error[greater])


def test_lowpass_biquad_eager():
    """ mindspore eager mode normal testcase:lowpass_biquad op"""
    # Original waveform
    waveform = np.array([[0.8236, 0.2049, 0.3335], [0.5933, 0.9911, 0.2482],
                         [0.3007, 0.9054, 0.7598], [0.5394, 0.2842, 0.5634], [0.6363, 0.2226, 0.2288]])
    # Expect waveform
    expect_waveform = np.array([[0.2745, 0.6174, 0.4308], [0.1978, 0.7259, 0.8753],
                                [0.1002, 0.5023, 0.9237], [0.1798, 0.4543, 0.4971], [0.2121, 0.4984, 0.3661]])
    lowpass_biquad_op = audio.LowpassBiquad(4000, 1000.0, 1)
    # Filtered waveform by lowpass_biquad
    output = lowpass_biquad_op(waveform)
    count_unequal_element(expect_waveform, output, 0.0001, 0.0001)


def test_lowpass_biquad_pipeline():
    """ mindspore pipeline mode normal testcase:lowpass_biquad op"""
    # Original waveform
    waveform = np.array([[3.5, 3.2, 2.5, 7.1], [5.5, 0.3, 4.9, 5.0],
                         [1.3, 7.4, 7.1, 3.8], [3.4, 3.3, 3.7, 1.1]])
    # Expect waveform
    expect_waveform = np.array([[0.0481, 0.2029, 0.4180, 0.6830], [0.0755, 0.2538, 0.4555, 0.7107],
                                [0.0178, 0.1606, 0.5220, 0.9729], [0.0467, 0.1997, 0.4322, 0.6546]])
    dataset = ds.NumpySlicesDataset(waveform, ["col1"], shuffle=False)
    lowpass_biquad_op = audio.LowpassBiquad(44100, 2000.0, 0.3)
    # Filtered waveform by lowpass_biquad
    dataset = dataset.map(
        input_columns=["col1"], operations=lowpass_biquad_op, num_parallel_workers=4)
    i = 0
    for _ in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        count_unequal_element(expect_waveform[i, :],
                              _["col1"], 0.0001, 0.0001)
        i += 1


def test_lowpass_biquad_invalid_input():
    """
    Test invalid input of LowpassBiquad
    """
    def test_invalid_input(test_name, sample_rate, cutoff_freq, Q, error, error_msg):
        logger.info("Test LowpassBiquad with bad input: {0}".format(test_name))
        with pytest.raises(error) as error_info:
            audio.LowpassBiquad(sample_rate, cutoff_freq, Q)
        assert error_msg in str(error_info.value)
    test_invalid_input("invalid sample_rate parameter type as a float", 44100.5, 1000, 0.707, TypeError,
                       "Argument sample_rate with value 44100.5 is not of type [<class 'int'>],"
                       " but got <class 'float'>.")
    test_invalid_input("invalid sample_rate parameter type as a String", "44100", 1000, 0.707, TypeError,
                       "Argument sample_rate with value 44100 is not of type [<class 'int'>],"
                       " but got <class 'str'>.")
    test_invalid_input("invalid cutoff_freq parameter type as a String", 44100, "1000", 0.707, TypeError,
                       "Argument cutoff_freq with value 1000 is not of type [<class 'float'>, <class 'int'>],"
                       " but got <class 'str'>.")
    test_invalid_input("invalid Q parameter type as a String", 44100, 1000, "0.707", TypeError,
                       "Argument Q with value 0.707 is not of type [<class 'float'>, <class 'int'>],"
                       " but got <class 'str'>.")

    test_invalid_input("invalid sample_rate parameter value", 441324343243242342345300, 1000, 0.707, ValueError,
                       "Input sample_rate is not within the required interval of [-2147483648, 0) and (0, 2147483647].")
    test_invalid_input("invalid cutoff_freq parameter value", 44100, 32434324324234321, 0.707, ValueError,
                       "Input cutoff_freq is not within the required interval of [-16777216, 16777216].")

    test_invalid_input("invalid sample_rate parameter value", None, 1000, 0.707, TypeError,
                       "Argument sample_rate with value None is not of type [<class 'int'>], "
                       "but got <class 'NoneType'>.")
    test_invalid_input("invalid cutoff_rate parameter value", 44100, None, 0.707, TypeError,
                       "Argument cutoff_freq with value None is not of type [<class 'float'>, <class 'int'>],"
                       " but got <class 'NoneType'>.")

    test_invalid_input("invalid Q parameter value", 44100, 1000, 0, ValueError,
                       "Input Q is not within the required interval of (0, 1].")


if __name__ == "__main__":
    test_lowpass_biquad_eager()
    test_lowpass_biquad_pipeline()
    test_lowpass_biquad_invalid_input()
