# Copyright 2019 Huawei Technologies Co., Ltd
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
Test Cifar10 and Cifar100 dataset operators
"""
import os
import pytest
import numpy as np
import matplotlib.pyplot as plt
import mindspore.dataset as ds
from mindspore import log as logger

DATA_DIR_10 = "../data/dataset/testCifar10Data"
DATA_DIR_100 = "../data/dataset/testCifar100Data"
NO_BIN_DIR = "../data/dataset/testMnistData"


def load_cifar(path, kind="cifar10"):
    """
    load Cifar10/100 data
    """
    raw = np.empty(0, dtype=np.uint8)
    for file_name in os.listdir(path):
        if file_name.endswith(".bin"):
            with open(os.path.join(path, file_name), mode='rb') as file:
                raw = np.append(raw, np.fromfile(file, dtype=np.uint8), axis=0)
    if kind == "cifar10":
        raw = raw.reshape(-1, 3073)
        labels = raw[:, 0]
        images = raw[:, 1:]
    elif kind == "cifar100":
        raw = raw.reshape(-1, 3074)
        labels = raw[:, :2]
        images = raw[:, 2:]
    else:
        raise ValueError("Invalid parameter value")
    images = images.reshape(-1, 3, 32, 32)
    images = images.transpose(0, 2, 3, 1)
    return images, labels


def visualize_dataset(images, labels):
    """
    Helper function to visualize the dataset samples
    """
    num_samples = len(images)
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(images[i])
        plt.title(labels[i])
    plt.show()


### Testcases for Cifar10Dataset Op ###


def test_cifar10_content_check():
    """
    Validate Cifar10Dataset image readings
    """
    logger.info("Test Cifar10Dataset Op with content check")
    data1 = ds.Cifar10Dataset(DATA_DIR_10, num_samples=100, shuffle=False)
    images, labels = load_cifar(DATA_DIR_10)
    num_iter = 0
    # in this example, each dictionary has keys "image" and "label"
    for i, d in enumerate(data1.create_dict_iterator(num_epochs=1, output_numpy=True)):
        np.testing.assert_array_equal(d["image"], images[i])
        np.testing.assert_array_equal(d["label"], labels[i])
        num_iter += 1
    assert num_iter == 100


def test_cifar10_basic():
    """
    Validate CIFAR10
    """
    logger.info("Test Cifar10Dataset Op")

    # case 0: test loading the whole dataset
    data0 = ds.Cifar10Dataset(DATA_DIR_10)
    num_iter0 = 0
    for _ in data0.create_dict_iterator(num_epochs=1):
        num_iter0 += 1
    assert num_iter0 == 10000

    # case 1: test num_samples
    data1 = ds.Cifar10Dataset(DATA_DIR_10, num_samples=100)
    num_iter1 = 0
    for _ in data1.create_dict_iterator(num_epochs=1):
        num_iter1 += 1
    assert num_iter1 == 100

    # case 2: test num_parallel_workers
    data2 = ds.Cifar10Dataset(DATA_DIR_10, num_samples=50, num_parallel_workers=1)
    num_iter2 = 0
    for _ in data2.create_dict_iterator(num_epochs=1):
        num_iter2 += 1
    assert num_iter2 == 50

    # case 3: test repeat
    data3 = ds.Cifar10Dataset(DATA_DIR_10, num_samples=100)
    data3 = data3.repeat(3)
    num_iter3 = 0
    for _ in data3.create_dict_iterator(num_epochs=1):
        num_iter3 += 1
    assert num_iter3 == 300

    # case 4: test batch with drop_remainder=False
    data4 = ds.Cifar10Dataset(DATA_DIR_10, num_samples=100)
    assert data4.get_dataset_size() == 100
    assert data4.get_batch_size() == 1
    data4 = data4.batch(batch_size=7)  # drop_remainder is default to be False
    assert data4.get_dataset_size() == 15
    assert data4.get_batch_size() == 7
    num_iter4 = 0
    for _ in data4.create_dict_iterator(num_epochs=1):
        num_iter4 += 1
    assert num_iter4 == 15

    # case 5: test batch with drop_remainder=True
    data5 = ds.Cifar10Dataset(DATA_DIR_10, num_samples=100)
    assert data5.get_dataset_size() == 100
    assert data5.get_batch_size() == 1
    data5 = data5.batch(batch_size=7, drop_remainder=True)  # the rest of incomplete batch will be dropped
    assert data5.get_dataset_size() == 14
    assert data5.get_batch_size() == 7
    num_iter5 = 0
    for _ in data5.create_dict_iterator(num_epochs=1):
        num_iter5 += 1
    assert num_iter5 == 14


def test_cifar10_pk_sampler():
    """
    Test Cifar10Dataset with PKSampler
    """
    logger.info("Test Cifar10Dataset Op with PKSampler")
    golden = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4,
              5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9]
    sampler = ds.PKSampler(3)
    data = ds.Cifar10Dataset(DATA_DIR_10, sampler=sampler)
    num_iter = 0
    label_list = []
    for item in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        label_list.append(item["label"])
        num_iter += 1
    np.testing.assert_array_equal(golden, label_list)
    assert num_iter == 30


def test_cifar10_sequential_sampler():
    """
    Test Cifar10Dataset with SequentialSampler
    """
    logger.info("Test Cifar10Dataset Op with SequentialSampler")
    num_samples = 30
    sampler = ds.SequentialSampler(num_samples=num_samples)
    data1 = ds.Cifar10Dataset(DATA_DIR_10, sampler=sampler)
    data2 = ds.Cifar10Dataset(DATA_DIR_10, shuffle=False, num_samples=num_samples)
    num_iter = 0
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        np.testing.assert_equal(item1["label"], item2["label"])
        num_iter += 1
    assert num_iter == num_samples


def test_cifar10_exception():
    """
    Test error cases for Cifar10Dataset
    """
    logger.info("Test error cases for Cifar10Dataset")
    error_msg_1 = "sampler and shuffle cannot be specified at the same time"
    with pytest.raises(RuntimeError, match=error_msg_1):
        ds.Cifar10Dataset(DATA_DIR_10, shuffle=False, sampler=ds.PKSampler(3))

    error_msg_2 = "sampler and sharding cannot be specified at the same time"
    with pytest.raises(RuntimeError, match=error_msg_2):
        ds.Cifar10Dataset(DATA_DIR_10, sampler=ds.PKSampler(3), num_shards=2, shard_id=0)

    error_msg_3 = "num_shards is specified and currently requires shard_id as well"
    with pytest.raises(RuntimeError, match=error_msg_3):
        ds.Cifar10Dataset(DATA_DIR_10, num_shards=10)

    error_msg_4 = "shard_id is specified but num_shards is not"
    with pytest.raises(RuntimeError, match=error_msg_4):
        ds.Cifar10Dataset(DATA_DIR_10, shard_id=0)

    error_msg_5 = "Input shard_id is not within the required interval"
    with pytest.raises(ValueError, match=error_msg_5):
        ds.Cifar10Dataset(DATA_DIR_10, num_shards=2, shard_id=-1)
    with pytest.raises(ValueError, match=error_msg_5):
        ds.Cifar10Dataset(DATA_DIR_10, num_shards=2, shard_id=5)

    error_msg_6 = "num_parallel_workers exceeds"
    with pytest.raises(ValueError, match=error_msg_6):
        ds.Cifar10Dataset(DATA_DIR_10, shuffle=False, num_parallel_workers=0)
    with pytest.raises(ValueError, match=error_msg_6):
        ds.Cifar10Dataset(DATA_DIR_10, shuffle=False, num_parallel_workers=256)

    error_msg_7 = r"cifar\(.bin\) files are missing"
    with pytest.raises(RuntimeError, match=error_msg_7):
        ds1 = ds.Cifar10Dataset(NO_BIN_DIR)
        for _ in ds1.__iter__():
            pass


def test_cifar10_visualize(plot=False):
    """
    Visualize Cifar10Dataset results
    """
    logger.info("Test Cifar10Dataset visualization")

    data1 = ds.Cifar10Dataset(DATA_DIR_10, num_samples=10, shuffle=False)
    num_iter = 0
    image_list, label_list = [], []
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        image = item["image"]
        label = item["label"]
        image_list.append(image)
        label_list.append("label {}".format(label))
        assert isinstance(image, np.ndarray)
        assert image.shape == (32, 32, 3)
        assert image.dtype == np.uint8
        assert label.dtype == np.uint32
        num_iter += 1
    assert num_iter == 10
    if plot:
        visualize_dataset(image_list, label_list)


### Testcases for Cifar100Dataset Op ###

def test_cifar100_content_check():
    """
    Validate Cifar100Dataset image readings
    """
    logger.info("Test Cifar100Dataset with content check")
    data1 = ds.Cifar100Dataset(DATA_DIR_100, num_samples=100, shuffle=False)
    images, labels = load_cifar(DATA_DIR_100, kind="cifar100")
    num_iter = 0
    # in this example, each dictionary has keys "image", "coarse_label" and "fine_image"
    for i, d in enumerate(data1.create_dict_iterator(num_epochs=1, output_numpy=True)):
        np.testing.assert_array_equal(d["image"], images[i])
        np.testing.assert_array_equal(d["coarse_label"], labels[i][0])
        np.testing.assert_array_equal(d["fine_label"], labels[i][1])
        num_iter += 1
    assert num_iter == 100


def test_cifar100_basic():
    """
    Test Cifar100Dataset
    """
    logger.info("Test Cifar100Dataset")

    # case 1: test num_samples
    data1 = ds.Cifar100Dataset(DATA_DIR_100, num_samples=100)
    num_iter1 = 0
    for _ in data1.create_dict_iterator(num_epochs=1):
        num_iter1 += 1
    assert num_iter1 == 100

    # case 2: test repeat
    data1 = data1.repeat(2)
    num_iter2 = 0
    for _ in data1.create_dict_iterator(num_epochs=1):
        num_iter2 += 1
    assert num_iter2 == 200

    # case 3: test num_parallel_workers
    data2 = ds.Cifar100Dataset(DATA_DIR_100, num_samples=100, num_parallel_workers=1)
    num_iter3 = 0
    for _ in data2.create_dict_iterator(num_epochs=1):
        num_iter3 += 1
    assert num_iter3 == 100

    # case 4: test batch with drop_remainder=False
    data3 = ds.Cifar100Dataset(DATA_DIR_100, num_samples=100)
    assert data3.get_dataset_size() == 100
    assert data3.get_batch_size() == 1
    data3 = data3.batch(batch_size=3)
    assert data3.get_dataset_size() == 34
    assert data3.get_batch_size() == 3
    num_iter4 = 0
    for _ in data3.create_dict_iterator(num_epochs=1):
        num_iter4 += 1
    assert num_iter4 == 34

    # case 4: test batch with drop_remainder=True
    data4 = ds.Cifar100Dataset(DATA_DIR_100, num_samples=100)
    data4 = data4.batch(batch_size=3, drop_remainder=True)
    assert data4.get_dataset_size() == 33
    assert data4.get_batch_size() == 3
    num_iter5 = 0
    for _ in data4.create_dict_iterator(num_epochs=1):
        num_iter5 += 1
    assert num_iter5 == 33


def test_cifar100_pk_sampler():
    """
    Test Cifar100Dataset with PKSampler
    """
    logger.info("Test Cifar100Dataset with PKSampler")
    golden = [i for i in range(20)]
    sampler = ds.PKSampler(1)
    data = ds.Cifar100Dataset(DATA_DIR_100, sampler=sampler)
    num_iter = 0
    label_list = []
    for item in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        label_list.append(item["coarse_label"])
        num_iter += 1
    np.testing.assert_array_equal(golden, label_list)
    assert num_iter == 20


def test_cifar100_exception():
    """
    Test error cases for Cifar100Dataset
    """
    logger.info("Test error cases for Cifar100Dataset")
    error_msg_1 = "sampler and shuffle cannot be specified at the same time"
    with pytest.raises(RuntimeError, match=error_msg_1):
        ds.Cifar100Dataset(DATA_DIR_100, shuffle=False, sampler=ds.PKSampler(3))

    error_msg_2 = "sampler and sharding cannot be specified at the same time"
    with pytest.raises(RuntimeError, match=error_msg_2):
        ds.Cifar100Dataset(DATA_DIR_100, sampler=ds.PKSampler(3), num_shards=2, shard_id=0)

    error_msg_3 = "num_shards is specified and currently requires shard_id as well"
    with pytest.raises(RuntimeError, match=error_msg_3):
        ds.Cifar100Dataset(DATA_DIR_100, num_shards=10)

    error_msg_4 = "shard_id is specified but num_shards is not"
    with pytest.raises(RuntimeError, match=error_msg_4):
        ds.Cifar100Dataset(DATA_DIR_100, shard_id=0)

    error_msg_5 = "Input shard_id is not within the required interval"
    with pytest.raises(ValueError, match=error_msg_5):
        ds.Cifar100Dataset(DATA_DIR_100, num_shards=2, shard_id=-1)
    with pytest.raises(ValueError, match=error_msg_5):
        ds.Cifar10Dataset(DATA_DIR_100, num_shards=2, shard_id=5)

    error_msg_6 = "num_parallel_workers exceeds"
    with pytest.raises(ValueError, match=error_msg_6):
        ds.Cifar100Dataset(DATA_DIR_100, shuffle=False, num_parallel_workers=0)
    with pytest.raises(ValueError, match=error_msg_6):
        ds.Cifar100Dataset(DATA_DIR_100, shuffle=False, num_parallel_workers=256)

    error_msg_7 = r"cifar\(.bin\) files are missing"
    with pytest.raises(RuntimeError, match=error_msg_7):
        ds1 = ds.Cifar100Dataset(NO_BIN_DIR)
        for _ in ds1.__iter__():
            pass


def test_cifar100_visualize(plot=False):
    """
    Visualize Cifar100Dataset results
    """
    logger.info("Test Cifar100Dataset visualization")

    data1 = ds.Cifar100Dataset(DATA_DIR_100, num_samples=10, shuffle=False)
    num_iter = 0
    image_list, label_list = [], []
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        image = item["image"]
        coarse_label = item["coarse_label"]
        fine_label = item["fine_label"]
        image_list.append(image)
        label_list.append("coarse_label {}\nfine_label {}".format(coarse_label, fine_label))
        assert isinstance(image, np.ndarray)
        assert image.shape == (32, 32, 3)
        assert image.dtype == np.uint8
        assert coarse_label.dtype == np.uint32
        assert fine_label.dtype == np.uint32
        num_iter += 1
    assert num_iter == 10
    if plot:
        visualize_dataset(image_list, label_list)


def test_cifar_usage():
    """
    test usage of cifar
    """
    logger.info("Test Cifar100Dataset usage flag")

    # flag, if True, test cifar10 else test cifar100
    def test_config(usage, flag=True, cifar_path=None):
        if cifar_path is None:
            cifar_path = DATA_DIR_10 if flag else DATA_DIR_100
        try:
            data = ds.Cifar10Dataset(cifar_path, usage=usage) if flag else ds.Cifar100Dataset(cifar_path, usage=usage)
            num_rows = 0
            for _ in data.create_dict_iterator(num_epochs=1, output_numpy=True):
                num_rows += 1
        except (ValueError, TypeError, RuntimeError) as e:
            return str(e)
        return num_rows

    # test the usage of CIFAR100
    assert test_config("train") == 10000
    assert test_config("all") == 10000
    assert "usage is not within the valid set of ['train', 'test', 'all']" in test_config("invalid")
    assert "Argument usage with value ['list'] is not of type [<class 'str'>]" in test_config(["list"])
    assert "Cifar10Dataset API can't read the data file (interface mismatch or no data found)" in test_config("test")

    # test the usage of CIFAR10
    assert test_config("test", False) == 10000
    assert test_config("all", False) == 10000
    assert "Cifar100Dataset API can't read the data file" in test_config("train", False)
    assert "usage is not within the valid set of ['train', 'test', 'all']" in test_config("invalid", False)

    # change this directory to the folder that contains all cifar10 files
    all_cifar10 = None
    if all_cifar10 is not None:
        assert test_config("train", True, all_cifar10) == 50000
        assert test_config("test", True, all_cifar10) == 10000
        assert test_config("all", True, all_cifar10) == 60000
        assert ds.Cifar10Dataset(all_cifar10, usage="train").get_dataset_size() == 50000
        assert ds.Cifar10Dataset(all_cifar10, usage="test").get_dataset_size() == 10000
        assert ds.Cifar10Dataset(all_cifar10, usage="all").get_dataset_size() == 60000

    # change this directory to the folder that contains all cifar100 files
    all_cifar100 = None
    if all_cifar100 is not None:
        assert test_config("train", False, all_cifar100) == 50000
        assert test_config("test", False, all_cifar100) == 10000
        assert test_config("all", False, all_cifar100) == 60000
        assert ds.Cifar100Dataset(all_cifar100, usage="train").get_dataset_size() == 50000
        assert ds.Cifar100Dataset(all_cifar100, usage="test").get_dataset_size() == 10000
        assert ds.Cifar100Dataset(all_cifar100, usage="all").get_dataset_size() == 60000


def test_cifar_exception_file_path():
    def exception_func(item):
        raise Exception("Error occur!")

    try:
        data = ds.Cifar10Dataset(DATA_DIR_10)
        data = data.map(operations=exception_func, input_columns=["image"], num_parallel_workers=1)
        num_rows = 0
        for _ in data.create_dict_iterator(num_epochs=1):
            num_rows += 1
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data files" in str(e)

    try:
        data = ds.Cifar10Dataset(DATA_DIR_10)
        data = data.map(operations=exception_func, input_columns=["label"], num_parallel_workers=1)
        num_rows = 0
        for _ in data.create_dict_iterator(num_epochs=1):
            num_rows += 1
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data files" in str(e)

    try:
        data = ds.Cifar100Dataset(DATA_DIR_100)
        data = data.map(operations=exception_func, input_columns=["image"], num_parallel_workers=1)
        num_rows = 0
        for _ in data.create_dict_iterator(num_epochs=1):
            num_rows += 1
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data files" in str(e)

    try:
        data = ds.Cifar100Dataset(DATA_DIR_100)
        data = data.map(operations=exception_func, input_columns=["coarse_label"], num_parallel_workers=1)
        num_rows = 0
        for _ in data.create_dict_iterator(num_epochs=1):
            num_rows += 1
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data files" in str(e)

    try:
        data = ds.Cifar100Dataset(DATA_DIR_100)
        data = data.map(operations=exception_func, input_columns=["fine_label"], num_parallel_workers=1)
        num_rows = 0
        for _ in data.create_dict_iterator(num_epochs=1):
            num_rows += 1
        assert False
    except RuntimeError as e:
        assert "map operation: [PyFunc] failed. The corresponding data files" in str(e)


def test_cifar10_pk_sampler_get_dataset_size():
    """
    Test Cifar10Dataset with PKSampler and get_dataset_size
    """
    sampler = ds.PKSampler(3)
    data = ds.Cifar10Dataset(DATA_DIR_10, sampler=sampler)
    num_iter = 0
    ds_sz = data.get_dataset_size()
    for _ in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        num_iter += 1

    assert ds_sz == num_iter == 30


def test_cifar10_with_chained_sampler_get_dataset_size():
    """
    Test Cifar10Dataset with PKSampler chained with a SequentialSampler and get_dataset_size
    """
    sampler = ds.SequentialSampler(start_index=0, num_samples=5)
    child_sampler = ds.PKSampler(4)
    sampler.add_child(child_sampler)
    data = ds.Cifar10Dataset(DATA_DIR_10, sampler=sampler)
    num_iter = 0
    ds_sz = data.get_dataset_size()
    for _ in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        num_iter += 1
    assert ds_sz == num_iter == 5


if __name__ == '__main__':
    test_cifar10_content_check()
    test_cifar10_basic()
    test_cifar10_pk_sampler()
    test_cifar10_sequential_sampler()
    test_cifar10_exception()
    test_cifar10_visualize(plot=False)

    test_cifar100_content_check()
    test_cifar100_basic()
    test_cifar100_pk_sampler()
    test_cifar100_exception()
    test_cifar100_visualize(plot=False)

    test_cifar_usage()
    test_cifar_exception_file_path()

    test_cifar10_with_chained_sampler_get_dataset_size()
    test_cifar10_pk_sampler_get_dataset_size()
