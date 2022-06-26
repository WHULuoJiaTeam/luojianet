# Copyright 2019 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""test mnist to mindrecord tool"""
import gzip
import os
import pytest
import numpy as np
import cv2

from mindspore import log as logger
from mindspore.mindrecord import FileReader
from mindspore.mindrecord import MnistToMR

MNIST_DIR = "../data/mindrecord/testMnistData"
PARTITION_NUM = 4
IMAGE_SIZE = 28
NUM_CHANNELS = 1

@pytest.fixture
def fixture_file():
    """add/remove file"""
    def remove_one_file(x):
        if os.path.exists(x):
            os.remove(x)
    def remove_file(file_name):
        remove_one_file(file_name + '_train.mindrecord')
        remove_one_file(file_name + '_train.mindrecord.db')
        remove_one_file(file_name + '_test.mindrecord')
        remove_one_file(file_name + '_test.mindrecord.db')
        for i in range(PARTITION_NUM):
            x = file_name + "_train.mindrecord" + str(i)
            remove_one_file(x)
            x = file_name + "_train.mindrecord" + str(i) + ".db"
            remove_one_file(x)
            x = file_name + "_test.mindrecord" + str(i)
            remove_one_file(x)
            x = file_name + "_test.mindrecord" + str(i) + ".db"
            remove_one_file(x)

    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    remove_file(file_name)
    yield "yield_fixture_data"
    remove_file(file_name)

def read(file_name, partition=False):
    """test file reader"""
    count = 0
    if partition:
        train_name = file_name + "_train.mindrecord0"
        test_name = file_name + "_test.mindrecord0"
    else:
        train_name = file_name + "_train.mindrecord"
        test_name = file_name + "_test.mindrecord"
    reader = FileReader(train_name)
    for _, x in enumerate(reader.get_next()):
        assert len(x) == 2
        count = count + 1
        if count == 1:
            logger.info("data: {}".format(x))
    assert count == 20
    reader.close()

    count = 0
    reader = FileReader(test_name)
    for _, x in enumerate(reader.get_next()):
        assert len(x) == 2
        count = count + 1
        if count == 1:
            logger.info("data: {}".format(x))
    assert count == 10
    reader.close()


def test_mnist_to_mindrecord(fixture_file):
    """test transform mnist dataset to mindrecord."""
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    mnist_transformer = MnistToMR(MNIST_DIR, file_name)
    mnist_transformer.transform()
    assert os.path.exists(file_name + "_train.mindrecord")
    assert os.path.exists(file_name + "_test.mindrecord")

    read(file_name)

def test_mnist_to_mindrecord_compare_data(fixture_file):
    """test transform mnist dataset to mindrecord and compare data."""
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    mnist_transformer = MnistToMR(MNIST_DIR, file_name)
    mnist_transformer.transform()
    assert os.path.exists(file_name + "_train.mindrecord")
    assert os.path.exists(file_name + "_test.mindrecord")

    train_name = file_name + "_train.mindrecord"
    test_name = file_name + "_test.mindrecord"

    def _extract_images(filename, num_images):
        """Extract the images into a 4D tensor [image index, y, x, channels]."""
        with gzip.open(filename) as bytestream:
            bytestream.read(16)
            buf = bytestream.read(
                IMAGE_SIZE * IMAGE_SIZE * num_images * NUM_CHANNELS)
            data = np.frombuffer(buf, dtype=np.uint8)
            data = data.reshape(
                num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
            return data

    def _extract_labels(filename, num_images):
        """Extract the labels into a vector of int64 label IDs."""
        with gzip.open(filename) as bytestream:
            bytestream.read(8)
            buf = bytestream.read(1 * num_images)
            labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
            return labels

    train_data_filename_ = os.path.join(MNIST_DIR,
                                        'train-images-idx3-ubyte.gz')
    train_labels_filename_ = os.path.join(MNIST_DIR,
                                          'train-labels-idx1-ubyte.gz')
    test_data_filename_ = os.path.join(MNIST_DIR,
                                       't10k-images-idx3-ubyte.gz')
    test_labels_filename_ = os.path.join(MNIST_DIR,
                                         't10k-labels-idx1-ubyte.gz')
    train_data = _extract_images(train_data_filename_, 20)
    train_labels = _extract_labels(train_labels_filename_, 20)
    test_data = _extract_images(test_data_filename_, 10)
    test_labels = _extract_labels(test_labels_filename_, 10)

    reader = FileReader(train_name)
    for x, data, label in zip(reader.get_next(), train_data, train_labels):
        _, img = cv2.imencode(".jpeg", data)
        assert np.array(x['data']) == img.tobytes()
        assert np.array(x['label']) == label
    reader.close()

    reader = FileReader(test_name)
    for x, data, label in zip(reader.get_next(), test_data, test_labels):
        _, img = cv2.imencode(".jpeg", data)
        assert np.array(x['data']) == img.tobytes()
        assert np.array(x['label']) == label
    reader.close()


def test_mnist_to_mindrecord_multi_partition(fixture_file):
    """test transform mnist dataset to multiple mindrecord files."""
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    mnist_transformer = MnistToMR(MNIST_DIR, file_name, PARTITION_NUM)
    mnist_transformer.transform()

    read(file_name, partition=True)
