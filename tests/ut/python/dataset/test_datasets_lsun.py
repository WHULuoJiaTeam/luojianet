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
Test LSUN dataset operators
"""
import pytest

import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as vision
from mindspore import log as logger

DATA_DIR = "../data/dataset/testLSUN"


def test_lsun_basic():
    """
    Feature: LSUN
    Description: test basic usage of LSUN
    Expectation: the dataset is as expected
    """
    logger.info("Test Case basic")
    # define parameters
    repeat_count = 1

    # apply dataset operations
    data1 = ds.LSUNDataset(DATA_DIR)
    data1 = data1.repeat(repeat_count)

    num_iter = 0
    # each data is a dictionary
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        # in this example, each dictionary has keys "image" and "label"
        logger.info("image is {}".format(item["image"]))
        logger.info("label is {}".format(item["label"]))
        num_iter += 1

    logger.info("Number of data in data1: {}".format(num_iter))
    assert num_iter == 4


def test_lsun_num_samples():
    """
    Feature: LSUN
    Description: test basic usage of LSUN
    Expectation: the dataset is as expected
    """
    logger.info("Test Case num_samples")
    # define parameters
    repeat_count = 1

    # apply dataset operations
    data1 = ds.LSUNDataset(DATA_DIR, num_samples=10, num_parallel_workers=2)
    data1 = data1.repeat(repeat_count)

    num_iter = 0
    # each data is a dictionary
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        # in this example, each dictionary has keys "image" and "label"
        logger.info("image is {}".format(item["image"]))
        logger.info("label is {}".format(item["label"]))
        num_iter += 1

    logger.info("Number of data in data1: {}".format(num_iter))
    assert num_iter == 4

    random_sampler = ds.RandomSampler(num_samples=3, replacement=True)
    data1 = ds.LSUNDataset(DATA_DIR, num_parallel_workers=2, sampler=random_sampler)

    num_iter = 0
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        num_iter += 1

    assert num_iter == 3

    random_sampler = ds.RandomSampler(num_samples=3, replacement=False)
    data1 = ds.LSUNDataset(DATA_DIR, num_parallel_workers=2, sampler=random_sampler)

    num_iter = 0
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        num_iter += 1

    assert num_iter == 3


def test_lsun_num_shards():
    """
    Feature: LSUN
    Description: test basic usage of LSUN
    Expectation: the dataset is as expected
    """
    logger.info("Test Case numShards")
    # define parameters
    repeat_count = 1

    # apply dataset operations
    data1 = ds.LSUNDataset(DATA_DIR, num_shards=2, shard_id=1)
    data1 = data1.repeat(repeat_count)

    num_iter = 0
    # each data is a dictionary
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        # in this example, each dictionary has keys "image" and "label"
        logger.info("image is {}".format(item["image"]))
        logger.info("label is {}".format(item["label"]))
        num_iter += 1
    logger.info("Number of data in data1: {}".format(num_iter))
    assert num_iter == 2


def test_lsun_shard_id():
    """
    Feature: LSUN
    Description: test basic usage of LSUN
    Expectation: the dataset is as expected
    """
    logger.info("Test Case withShardID")
    # define parameters
    repeat_count = 1

    # apply dataset operations
    data1 = ds.LSUNDataset(DATA_DIR, num_shards=2, shard_id=0)
    data1 = data1.repeat(repeat_count)

    num_iter = 0
    # each data is a dictionary
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        # in this example, each dictionary has keys "image" and "label"
        logger.info("image is {}".format(item["image"]))
        logger.info("label is {}".format(item["label"]))
        num_iter += 1
    logger.info("Number of data in data1: {}".format(num_iter))
    assert num_iter == 2


def test_lsun_no_shuffle():
    """
    Feature: LSUN
    Description: test basic usage of LSUN
    Expectation: the dataset is as expected
    """
    logger.info("Test Case noShuffle")
    # define parameters
    repeat_count = 1

    # apply dataset operations
    data1 = ds.LSUNDataset(DATA_DIR, shuffle=False)
    data1 = data1.repeat(repeat_count)

    num_iter = 0
    # each data is a dictionary
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        # in this example, each dictionary has keys "image" and "label"
        logger.info("image is {}".format(item["image"]))
        logger.info("label is {}".format(item["label"]))
        num_iter += 1

    logger.info("Number of data in data1: {}".format(num_iter))
    assert num_iter == 4


def test_lsun_extra_shuffle():
    """
    Feature: LSUN
    Description: test basic usage of LSUN
    Expectation: the dataset is as expected
    """
    logger.info("Test Case extra_shuffle")
    # define parameters
    repeat_count = 2

    # apply dataset operations
    data1 = ds.LSUNDataset(DATA_DIR, shuffle=True)
    data1 = data1.shuffle(buffer_size=5)
    data1 = data1.repeat(repeat_count)

    num_iter = 0
    # each data is a dictionary
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        # in this example, each dictionary has keys "image" and "label"
        logger.info("image is {}".format(item["image"]))
        logger.info("label is {}".format(item["label"]))
        num_iter += 1
    logger.info("Number of data in data1: {}".format(num_iter))
    assert num_iter == 8


def test_lsun_decode():
    """
    Feature: LSUN
    Description: test basic usage of LSUN
    Expectation: the dataset is as expected
    """
    logger.info("Test Case decode")
    # define parameters
    repeat_count = 1

    # apply dataset operations
    data1 = ds.LSUNDataset(DATA_DIR, decode=True)
    data1 = data1.repeat(repeat_count)

    num_iter = 0
    # each data is a dictionary
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        # in this example, each dictionary has keys "image" and "label"
        logger.info("image is {}".format(item["image"]))
        logger.info("label is {}".format(item["label"]))
        num_iter += 1

    logger.info("Number of data in data1: {}".format(num_iter))
    assert num_iter == 4


def test_sequential_sampler():
    """
    Feature: LSUN
    Description: test basic usage of LSUN
    Expectation: the dataset is as expected
    """
    logger.info("Test Case SequentialSampler")
    # define parameters
    repeat_count = 1
    # apply dataset operations
    sampler = ds.SequentialSampler(num_samples=10)
    data1 = ds.LSUNDataset(DATA_DIR, usage="train", sampler=sampler)
    data1 = data1.repeat(repeat_count)
    result = []
    num_iter = 0
    # each data is a dictionary
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        # in this example, each dictionary has keys "image" and "label"
        result.append(item["label"])
        num_iter += 1

    assert num_iter == 2
    logger.info("Result: {}".format(result))


def test_random_sampler():
    """
    Feature: LSUN
    Description: test basic usage of LSUN
    Expectation: the dataset is as expected
    """
    logger.info("Test Case RandomSampler")
    # define parameters
    repeat_count = 1

    # apply dataset operations
    sampler = ds.RandomSampler()
    data1 = ds.LSUNDataset(DATA_DIR, usage="train", sampler=sampler)
    data1 = data1.repeat(repeat_count)

    num_iter = 0
    # each data is a dictionary
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        # in this example, each dictionary has keys "image" and "label"
        logger.info("image is {}".format(item["image"]))
        logger.info("label is {}".format(item["label"]))
        num_iter += 1
    logger.info("Number of data in data1: {}".format(num_iter))
    assert num_iter == 2


def test_distributed_sampler():
    """
    Feature: LSUN
    Description: test basic usage of LSUN
    Expectation: the dataset is as expected
    """
    logger.info("Test Case DistributedSampler")
    # define parameters
    repeat_count = 1

    # apply dataset operations
    sampler = ds.DistributedSampler(2, 1)
    data1 = ds.LSUNDataset(DATA_DIR, usage="train", sampler=sampler)
    data1 = data1.repeat(repeat_count)

    num_iter = 0
    # each data is a dictionary
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        # in this example, each dictionary has keys "image" and "label"
        logger.info("image is {}".format(item["image"]))
        logger.info("label is {}".format(item["label"]))
        num_iter += 1
    logger.info("Number of data in data1: {}".format(num_iter))
    assert num_iter == 1


def test_pk_sampler():
    """
    Feature: LSUN
    Description: test basic usage of LSUN
    Expectation: the dataset is as expected
    """
    logger.info("Test Case PKSampler")
    # define parameters
    repeat_count = 1

    # apply dataset operations
    sampler = ds.PKSampler(1)
    data1 = ds.LSUNDataset(DATA_DIR, usage="train", sampler=sampler)
    data1 = data1.repeat(repeat_count)

    num_iter = 0
    # each data is a dictionary
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        # in this example, each dictionary has keys "image" and "label"
        logger.info("image is {}".format(item["image"]))
        logger.info("label is {}".format(item["label"]))
        num_iter += 1
    logger.info("Number of data in data1: {}".format(num_iter))
    assert num_iter == 2


def test_chained_sampler():
    """
    Feature: LSUN
    Description: test basic usage of LSUN
    Expectation: the dataset is as expected
    """
    logger.info("Test Case Chained Sampler - Random and Sequential, with repeat")

    # Create chained sampler, random and sequential
    sampler = ds.RandomSampler()
    child_sampler = ds.SequentialSampler()
    sampler.add_child(child_sampler)
    # Create LSUNDataset with sampler
    data1 = ds.LSUNDataset(DATA_DIR, usage="train", sampler=sampler)

    data1 = data1.repeat(count=3)

    # Verify dataset size
    data1_size = data1.get_dataset_size()
    logger.info("dataset size is: {}".format(data1_size))
    assert data1_size == 6

    # Verify number of iterations
    num_iter = 0
    # each data is a dictionary
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        # in this example, each dictionary has keys "image" and "label"
        logger.info("image is {}".format(item["image"]))
        logger.info("label is {}".format(item["label"]))
        num_iter += 1
    logger.info("Number of data in data1: {}".format(num_iter))
    assert num_iter == 6


def test_lsun_test_dataset():
    """
    Feature: LSUN
    Description: test basic usage of LSUN
    Expectation: the dataset is as expected
    """
    logger.info("Test Case usage")
    # apply dataset operations
    data1 = ds.LSUNDataset(DATA_DIR, usage="test", num_samples=8)
    num_iter = 0
    # each data is a dictionary
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        # in this example, each dictionary has keys "image" and "label"
        logger.info("image is {}".format(item["image"]))
        logger.info("label is {}".format(item["label"]))
        num_iter += 1

    logger.info("Number of data in data1: {}".format(num_iter))
    assert num_iter == 1


def test_lsun_valid_dataset():
    """
    Feature: LSUN
    Description: test basic usage of LSUN
    Expectation: the dataset is as expected
    """
    logger.info("Test Case usage")
    # apply dataset operations
    data1 = ds.LSUNDataset(DATA_DIR, usage="valid", num_samples=8)
    num_iter = 0
    # each data is a dictionary
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        # in this example, each dictionary has keys "image" and "label"
        logger.info("image is {}".format(item["image"]))
        logger.info("label is {}".format(item["label"]))
        num_iter += 1

    logger.info("Number of data in data1: {}".format(num_iter))
    assert num_iter == 2


def test_lsun_train_dataset():
    """
    Feature: LSUN
    Description: test basic usage of LSUN
    Expectation: the dataset is as expected
    """
    logger.info("Test Case usage")
    # apply dataset operations
    data1 = ds.LSUNDataset(DATA_DIR, usage="train", num_samples=8)
    num_iter = 0
    # each data is a dictionary
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        # in this example, each dictionary has keys "image" and "label"
        logger.info("image is {}".format(item["image"]))
        logger.info("label is {}".format(item["label"]))
        num_iter += 1

    logger.info("Number of data in data1: {}".format(num_iter))
    assert num_iter == 2


def test_lsun_all_dataset():
    """
    Feature: LSUN
    Description: test basic usage of LSUN
    Expectation: the dataset is as expected
    """
    logger.info("Test Case usage")
    # apply dataset operations
    data1 = ds.LSUNDataset(DATA_DIR, usage="all", num_samples=8)
    num_iter = 0
    # each data is a dictionary
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        # in this example, each dictionary has keys "image" and "label"
        logger.info("image is {}".format(item["image"]))
        logger.info("label is {}".format(item["label"]))
        num_iter += 1

    logger.info("Number of data in data1: {}".format(num_iter))
    assert num_iter == 4


def test_lsun_classes():
    """
    Feature: LSUN
    Description: test classes of LSUN
    Expectation: the dataset is as expected
    """
    logger.info("Test Case usage")
    # apply dataset operations
    data1 = ds.LSUNDataset(DATA_DIR, usage="train", classes=["bedroom"], num_samples=8)
    num_iter = 0
    # each data is a dictionary
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        # in this example, each dictionary has keys "image" and "label"
        logger.info("image is {}".format(item["image"]))
        logger.info("label is {}".format(item["label"]))
        num_iter += 1

    logger.info("Number of data in data1: {}".format(num_iter))
    assert num_iter == 1


def test_lsun_zip():
    """
    Feature: LSUN
    Description: test basic usage of LSUN
    Expectation: the dataset is as expected
    """
    logger.info("Test Case zip")
    # define parameters
    repeat_count = 2

    # apply dataset operations
    data1 = ds.LSUNDataset(DATA_DIR, num_samples=10)
    data2 = ds.LSUNDataset(DATA_DIR, num_samples=10)

    data1 = data1.repeat(repeat_count)
    # rename dataset2 for no conflict
    data2 = data2.rename(input_columns=["image", "label"], output_columns=["image1", "label1"])
    data3 = ds.zip((data1, data2))

    num_iter = 0
    # each data is a dictionary
    for item in data3.create_dict_iterator(num_epochs=1, output_numpy=True):
        # in this example, each dictionary has keys "image" and "label"
        logger.info("image is {}".format(item["image"]))
        logger.info("label is {}".format(item["label"]))
        num_iter += 1

    logger.info("Number of data in data1: {}".format(num_iter))
    assert num_iter == 4


def test_lsun_exception():
    """
    Feature: LSUN
    Description:  test error cases for LSUN
    Expectation: throw exception correctly
    """
    logger.info("Test lsun exception")

    error_msg_1 = "sampler and shuffle cannot be specified at the same time"
    with pytest.raises(RuntimeError, match=error_msg_1):
        ds.LSUNDataset(DATA_DIR, shuffle=False, sampler=ds.PKSampler(3))

    error_msg_2 = "sampler and sharding cannot be specified at the same time"
    with pytest.raises(RuntimeError, match=error_msg_2):
        ds.LSUNDataset(DATA_DIR, sampler=ds.PKSampler(3), num_shards=2, shard_id=0)

    error_msg_3 = "num_shards is specified and currently requires shard_id as well"
    with pytest.raises(RuntimeError, match=error_msg_3):
        ds.LSUNDataset(DATA_DIR, num_shards=10)

    error_msg_4 = "shard_id is specified but num_shards is not"
    with pytest.raises(RuntimeError, match=error_msg_4):
        ds.LSUNDataset(DATA_DIR, shard_id=0)

    error_msg_5 = "Input shard_id is not within the required interval"
    with pytest.raises(ValueError, match=error_msg_5):
        ds.LSUNDataset(DATA_DIR, num_shards=5, shard_id=-1)
    with pytest.raises(ValueError, match=error_msg_5):
        ds.LSUNDataset(DATA_DIR, num_shards=5, shard_id=5)
    with pytest.raises(ValueError, match=error_msg_5):
        ds.LSUNDataset(DATA_DIR, num_shards=2, shard_id=5)

    error_msg_6 = "num_parallel_workers exceeds"
    with pytest.raises(ValueError, match=error_msg_6):
        ds.LSUNDataset(DATA_DIR, shuffle=False, num_parallel_workers=0)
    with pytest.raises(ValueError, match=error_msg_6):
        ds.LSUNDataset(DATA_DIR, shuffle=False, num_parallel_workers=256)
    with pytest.raises(ValueError, match=error_msg_6):
        ds.LSUNDataset(DATA_DIR, shuffle=False, num_parallel_workers=-2)

    error_msg_7 = "Argument shard_id"
    with pytest.raises(TypeError, match=error_msg_7):
        ds.LSUNDataset(DATA_DIR, num_shards=2, shard_id="0")



def test_lsun_exception_map():
    """
    Feature: LSUN
    Description:  test error cases for LSUN
    Expectation: throw exception correctly
    """
    logger.info("Test lsun exception map")
    def exception_func(item):
        raise Exception("Error occur!")

    def exception_func2(image, label):
        raise Exception("Error occur!")

    try:
        data = ds.LSUNDataset(DATA_DIR)
        data = data.map(operations=exception_func, input_columns=["image"], num_parallel_workers=1)
        for _ in data.__iter__():
            pass
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data files" in str(e)

    try:
        data = ds.LSUNDataset(DATA_DIR)
        data = data.map(operations=exception_func2,
                        input_columns=["image", "label"],
                        output_columns=["image", "label", "label1"],
                        column_order=["image", "label", "label1"],
                        num_parallel_workers=1)
        for _ in data.__iter__():
            pass
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data files" in str(e)

    try:
        data = ds.LSUNDataset(DATA_DIR)
        data = data.map(operations=vision.Decode(), input_columns=["image"], num_parallel_workers=1)
        data = data.map(operations=exception_func, input_columns=["image"], num_parallel_workers=1)
        for _ in data.__iter__():
            pass
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data files" in str(e)


if __name__ == '__main__':
    test_lsun_basic()
    test_lsun_num_samples()
    test_sequential_sampler()
    test_random_sampler()
    test_distributed_sampler()
    test_pk_sampler()
    test_lsun_num_shards()
    test_lsun_shard_id()
    test_lsun_no_shuffle()
    test_lsun_extra_shuffle()
    test_lsun_decode()
    test_lsun_test_dataset()
    test_lsun_valid_dataset()
    test_lsun_train_dataset()
    test_lsun_all_dataset()
    test_lsun_classes()
    test_lsun_zip()
    test_lsun_exception()
    test_lsun_exception_map()
