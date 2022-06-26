# Copyright 2020 Huawei Technologies Co., Ltd
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
import mindspore.dataset.vision.c_transforms as vision


DATA_DIR_TF2 = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR_TF2 = "../data/dataset/test_tf_file_3_images/datasetSchema.json"


def test_tf_skip():
    """
    a simple skip operation.
    """
    data1 = ds.TFRecordDataset(DATA_DIR_TF2, SCHEMA_DIR_TF2, shuffle=False)

    resize_height, resize_width = 32, 32
    decode_op = vision.Decode()
    resize_op = vision.Resize((resize_height, resize_width), interpolation=ds.transforms.vision.Inter.LINEAR)
    data1 = data1.map(operations=decode_op, input_columns=["image"])
    data1 = data1.map(operations=resize_op, input_columns=["image"])
    data1 = data1.skip(2)

    num_iter = 0
    for _ in data1.create_dict_iterator(num_epochs=1):
        num_iter += 1
    assert num_iter == 1


def generator_md():
    """
    create a dataset with [0, 1, 2, 3, 4]
    """
    for i in range(5):
        yield (np.array([i]),)


def test_generator_skip():
    ds1 = ds.GeneratorDataset(generator_md, ["data"], num_parallel_workers=4)

    # Here ds1 should be [3, 4]
    ds1 = ds1.skip(3)

    buf = []
    for data in ds1.create_tuple_iterator(num_epochs=1, output_numpy=True):
        buf.append(data[0][0])
    assert len(buf) == 2
    assert buf == [3, 4]


def test_skip_1():
    ds1 = ds.GeneratorDataset(generator_md, ["data"])

    # Here ds1 should be []
    ds1 = ds1.skip(7)

    buf = []
    for data in ds1.create_tuple_iterator(num_epochs=1, output_numpy=True):
        buf.append(data[0][0])
    assert buf == []


def test_skip_2():
    ds1 = ds.GeneratorDataset(generator_md, ["data"])

    # Here ds1 should be [0, 1, 2, 3, 4]
    ds1 = ds1.skip(0)

    buf = []
    for data in ds1.create_tuple_iterator(num_epochs=1, output_numpy=True):
        buf.append(data[0][0])
    assert len(buf) == 5
    assert buf == [0, 1, 2, 3, 4]


def test_skip_repeat_1():
    ds1 = ds.GeneratorDataset(generator_md, ["data"])

    # Here ds1 should be [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
    ds1 = ds1.repeat(2)

    # Here ds1 should be [3, 4, 0, 1, 2, 3, 4]
    ds1 = ds1.skip(3)

    buf = []
    for data in ds1.create_tuple_iterator(num_epochs=1, output_numpy=True):
        buf.append(data[0][0])
    assert len(buf) == 7
    assert buf == [3, 4, 0, 1, 2, 3, 4]


def test_skip_repeat_2():
    ds1 = ds.GeneratorDataset(generator_md, ["data"])

    # Here ds1 should be [3, 4]
    ds1 = ds1.skip(3)

    # Here ds1 should be [3, 4, 3, 4]
    ds1 = ds1.repeat(2)

    buf = []
    for data in ds1.create_tuple_iterator(num_epochs=1, output_numpy=True):
        buf.append(data[0][0])
    assert len(buf) == 4
    assert buf == [3, 4, 3, 4]


def test_skip_repeat_3():
    ds1 = ds.GeneratorDataset(generator_md, ["data"])

    # Here ds1 should be [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
    ds1 = ds1.repeat(2)

    # Here ds1 should be [3, 4]
    ds1 = ds1.skip(8)

    # Here ds1 should be [3, 4, 3, 4, 3, 4]
    ds1 = ds1.repeat(3)

    buf = []
    for data in ds1.create_tuple_iterator(num_epochs=1, output_numpy=True):
        buf.append(data[0][0])
    assert len(buf) == 6
    assert buf == [3, 4, 3, 4, 3, 4]


def test_skip_take_1():
    ds1 = ds.GeneratorDataset(generator_md, ["data"])

    # Here ds1 should be [0, 1, 2, 3]
    ds1 = ds1.take(4)

    # Here ds1 should be [2, 3]
    ds1 = ds1.skip(2)

    buf = []
    for data in ds1.create_tuple_iterator(num_epochs=1, output_numpy=True):
        buf.append(data[0][0])
    assert len(buf) == 2
    assert buf == [2, 3]


def test_skip_take_2():
    ds1 = ds.GeneratorDataset(generator_md, ["data"])

    # Here ds1 should be [2, 3, 4]
    ds1 = ds1.skip(2)

    # Here ds1 should be [2, 3]
    ds1 = ds1.take(2)

    buf = []
    for data in ds1.create_tuple_iterator(num_epochs=1, output_numpy=True):
        buf.append(data[0][0])
    assert len(buf) == 2
    assert buf == [2, 3]


def generator_1d():
    for i in range(64):
        yield (np.array([i]),)


def test_skip_filter_1():
    dataset = ds.GeneratorDataset(generator_1d, ['data'])
    dataset = dataset.skip(5)
    dataset = dataset.filter(predicate=lambda data: data < 11, num_parallel_workers=4)

    buf = []
    for item in dataset.create_tuple_iterator(num_epochs=1, output_numpy=True):
        buf.append(item[0][0])
    assert buf == [5, 6, 7, 8, 9, 10]


def test_skip_filter_2():
    dataset = ds.GeneratorDataset(generator_1d, ['data'])
    dataset = dataset.filter(predicate=lambda data: data < 11, num_parallel_workers=4)
    dataset = dataset.skip(5)

    buf = []
    for item in dataset.create_tuple_iterator(num_epochs=1, output_numpy=True):
        buf.append(item[0][0])
    assert buf == [5, 6, 7, 8, 9, 10]


def test_skip_exception_1():
    data1 = ds.GeneratorDataset(generator_md, ["data"])

    try:
        data1 = data1.skip(count=-1)
        num_iter = 0
        for _ in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
            num_iter += 1

    except ValueError as e:
        assert "Input count is not within the required interval" in str(e)


def test_skip_exception_2():
    ds1 = ds.GeneratorDataset(generator_md, ["data"])

    with pytest.raises(ValueError) as e:
        ds1 = ds1.skip(-2)
    assert "Input count is not within the required interval" in str(e.value)



if __name__ == "__main__":
    test_tf_skip()
    test_generator_skip()
    test_skip_1()
    test_skip_2()
    test_skip_repeat_1()
    test_skip_repeat_2()
    test_skip_repeat_3()
    test_skip_take_1()
    test_skip_take_2()
    test_skip_filter_1()
    test_skip_filter_2()
    test_skip_exception_1()
    test_skip_exception_2()
