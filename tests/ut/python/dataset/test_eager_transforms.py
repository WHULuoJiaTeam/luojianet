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
Eager Tests for Transform Tensor ops
"""

import numpy as np
import mindspore.common.dtype as mstype
import mindspore.dataset.transforms.c_transforms as data_trans


def test_eager_concatenate():
    """
    Test Concatenate op is callable
    """
    prepend_tensor = np.array([1.4, 2., 3., 4., 4.5], dtype=np.float)
    append_tensor = np.array([9., 10.3, 11., 12.], dtype=np.float)
    concatenate_op = data_trans.Concatenate(0, prepend_tensor, append_tensor)
    expected = np.array([1.4, 2., 3., 4., 4.5, 5., 6., 7., 8., 9., 10.3,
                         11., 12.])
    assert np.array_equal(concatenate_op([5., 6., 7., 8.]), expected)


def test_eager_fill():
    """
    Test Fill op is callable
    """
    fill_op = data_trans.Fill(3)
    expected = np.array([3, 3, 3, 3])
    assert np.array_equal(fill_op([4, 5, 6, 7]), expected)


def test_eager_mask():
    """
    Test Mask op is callable
    """
    mask_op = data_trans.Mask(data_trans.Relational.EQ, 3, mstype.bool_)
    expected = np.array([False, False, True, False, False])
    assert np.array_equal(mask_op([1, 2, 3, 4, 5]), expected)


def test_eager_pad_end():
    """
    Test PadEnd op is callable
    """
    pad_end_op = data_trans.PadEnd([3], -1)
    expected = np.array([1, 2, -1])
    assert np.array_equal(pad_end_op([1, 2]), expected)


def test_eager_slice():
    """
    Test Slice op is callable
    """
    indexing = [[0], [0, 3]]
    slice_op = data_trans.Slice(*indexing)
    expected = np.array([[1, 4]])
    assert np.array_equal(slice_op([[1, 2, 3, 4, 5]]), expected)


if __name__ == "__main__":
    test_eager_concatenate()
    test_eager_fill()
    test_eager_mask()
    test_eager_pad_end()
    test_eager_slice()
