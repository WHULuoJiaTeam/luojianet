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
"""
Test Python Multiprocessing with Python functions/ops
"""
import numpy as np
import pytest
import mindspore.dataset as ds
import mindspore.dataset.transforms.py_transforms as py_transforms
import mindspore.dataset.vision.py_transforms as py_vision
from util import visualize_list

MNIST_DATA_DIR = "../data/dataset/testMnistData"
TF_DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
TF_SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"
PYFUNCMAP_DATA_DIR = ["../data/dataset/testPyfuncMap/data.data"]
PYFUNCMAP_SCHEMA_DIR = "../data/dataset/testPyfuncMap/schema.json"


def skip_test_pyfunc_multiproc_shrmem():
    """
    Feature: PyFunc in Map op
    Description: Test python_multiprocessing=True with shared memory enabled
    Expectation: Data results are correct
    """

    def pyfunc(x):
        return x

    # Confirm shared memory optimization is enabled by default
    mem_original = ds.config.get_enable_shared_mem()
    assert mem_original

    # Reduce memory needed by reducing queue size
    prefetch_original = ds.config.get_prefetch_size()
    ds.config.set_prefetch_size(1)

    max_elements = 2000
    np_data = list(range(0, max_elements))

    data1 = ds.NumpySlicesDataset(np_data, shuffle=False)

    data1 = data1.map(pyfunc, num_parallel_workers=8, python_multiprocessing=True, max_rowsize=1)

    for i, data in enumerate(data1):
        np.testing.assert_equal(data[0].asnumpy(), np_data[i])

    assert data1.get_dataset_size() == max_elements

    ds.config.set_prefetch_size(prefetch_original)


def create_dataset_pyop_multiproc(num_parallel_workers=None, max_rowsize=16, batch_size=32, repeat_size=1,
                                  num_samples=None):
    """
    Create dataset with Python ops list and python_multiprocessing=True for Map op
    """

    # Define dataset
    data1 = ds.MnistDataset(MNIST_DATA_DIR, num_samples=num_samples)

    data1 = data1.map(operations=[py_vision.ToType(np.int32)], input_columns="label",
                      num_parallel_workers=num_parallel_workers,
                      python_multiprocessing=True, max_rowsize=max_rowsize)

    # Setup transforms list which include Python ops
    transforms_list = [
        py_vision.ToTensor(),
        lambda x: x,
        py_vision.HWC2CHW(),
        py_vision.RandomErasing(0.9, value='random'),
        py_vision.Cutout(4, 2),
        lambda y: y
    ]
    compose_op = py_transforms.Compose(transforms_list)
    data1 = data1.map(operations=compose_op, input_columns="image", num_parallel_workers=num_parallel_workers,
                      python_multiprocessing=True, max_rowsize=max_rowsize)

    # Apply Dataset Ops
    buffer_size = 10000
    data1 = data1.shuffle(buffer_size=buffer_size)
    data1 = data1.batch(batch_size, drop_remainder=True)
    data1 = data1.repeat(repeat_size)

    return data1


def test_pyfunc_multiproc_noshrmem():
    """
    Feature: Python Multiprocessing
    Description: Test Map op with python_multiprocessing=True
    Expectation: Number of return data rows is correct
    """
    # Reduce memory required by disabling the shared memory optimization
    mem_original = ds.config.get_enable_shared_mem()
    ds.config.set_enable_shared_mem(False)

    mydata1 = create_dataset_pyop_multiproc(num_parallel_workers=12, repeat_size=2)
    mycount1 = 0
    for _ in mydata1.create_dict_iterator(num_epochs=1):
        mycount1 += 1
    assert mycount1 == 624

    ds.config.set_enable_shared_mem(mem_original)


def test_pyfunc_multiproc_max_rowsize_small():
    """
    Feature: Python Multiprocessing
    Description: Test Map op with python_multiprocessing=True and max_rowsize=1 (less than default of 16)
    Expectation: Number of return data rows is correct
    """
    # Reduce memory needed by reducing queue size
    # and disabling the shared memory optimization
    prefetch_original = ds.config.get_prefetch_size()
    ds.config.set_prefetch_size(1)
    mem_original = ds.config.get_enable_shared_mem()
    ds.config.set_enable_shared_mem(False)

    mydata1 = create_dataset_pyop_multiproc(num_parallel_workers=2, max_rowsize=1, num_samples=500)
    mycount1 = 0
    for _ in mydata1.create_dict_iterator(num_epochs=1):
        mycount1 += 1
    assert mycount1 == 15

    ds.config.set_prefetch_size(prefetch_original)
    ds.config.set_enable_shared_mem(mem_original)


def test_pyfunc_multiproc_max_rowsize_large():
    """
    Feature: Python Multiprocessing
    Description: Test Map op with python_multiprocessing=True and max_rowsize=20 (more than default of 16)
    Expectation: Number of return data rows is correct
    """
    # Reduce memory required by disabling the shared memory optimization
    mem_original = ds.config.get_enable_shared_mem()
    ds.config.set_enable_shared_mem(False)

    mydata1 = create_dataset_pyop_multiproc(num_parallel_workers=4, max_rowsize=20, num_samples=500)
    mycount1 = 0
    for _ in mydata1.create_dict_iterator(num_epochs=1):
        mycount1 += 1
    assert mycount1 == 15

    ds.config.set_enable_shared_mem(mem_original)


def test_pyfunc_multiproc_basic_pipeline(plot=False):
    """
    Feature: Python Multiprocessing
    Description: Test Map op with python_multiprocessing=True in a basic pipeline with Py ops
    Expectation: Images in plots from the 2 pipelines are visually fine
    """
    # Reduce memory required by disabling the shared memory optimization
    mem_original = ds.config.get_enable_shared_mem()
    ds.config.set_enable_shared_mem(False)

    # Define map operations
    transforms_list = [py_vision.CenterCrop(64), py_vision.RandomRotation(30)]
    transforms1 = [
        py_vision.Decode(),
        py_transforms.RandomChoice(transforms_list),
        py_vision.ToTensor()
    ]
    transform1 = py_transforms.Compose(transforms1)

    transforms2 = [
        py_vision.Decode(),
        py_vision.ToTensor()
    ]
    transform2 = py_transforms.Compose(transforms2)

    # First dataset
    data1 = ds.TFRecordDataset(TF_DATA_DIR, TF_SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data1 = data1.map(operations=transform1, input_columns=["image"], num_parallel_workers=2,
                      python_multiprocessing=True, max_rowsize=1)
    # Second dataset
    data2 = ds.TFRecordDataset(TF_DATA_DIR, TF_SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data2 = data2.map(operations=transform2, input_columns=["image"])

    image_choice = []
    image_original = []
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        image1 = (item1["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        image2 = (item2["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        image_choice.append(image1)
        image_original.append(image2)
    if plot:
        visualize_list(image_original, image_choice)

    ds.config.set_enable_shared_mem(mem_original)


def test_pyfunc_multiproc_child_exception():
    """
    Feature: Python Multiprocessing
    Description: Test Map op with python_multiprocessing=True with Python op encountering exception
    Expectation: Exception is correctly processed
    """
    # Reduce memory required by disabling the shared memory optimization
    mem_original = ds.config.get_enable_shared_mem()
    ds.config.set_enable_shared_mem(False)

    # Define map operations
    # Note: crop size[5000, 5000] > image size[4032, 2268]
    transforms_list = [py_vision.RandomCrop(5000)]
    transforms = [
        py_vision.Decode(),
        py_transforms.RandomChoice(transforms_list),
        py_vision.ToTensor()
    ]
    transform = py_transforms.Compose(transforms)
    # Generate dataset
    data = ds.TFRecordDataset(TF_DATA_DIR, TF_SCHEMA_DIR, columns_list=["image"], shuffle=False)
    data = data.map(operations=transform, input_columns=["image"], python_multiprocessing=True)
    # Note: Expect error raised with RandomCrop input: crop size greater than image size
    with pytest.raises(RuntimeError) as info:
        data.create_dict_iterator(num_epochs=1).__next__()
    assert "Crop size" in str(info.value)

    ds.config.set_enable_shared_mem(mem_original)


def test_pyfunc_multiproc_mainproc_exception():
    """
    Feature: PyFunc in Map op
    Description: Test python_multiprocessing=True with exception in main process
    Expectation: Exception is received and test ends gracefully
    """
    # Reduce memory required by disabling the shared memory optimization
    mem_original = ds.config.get_enable_shared_mem()
    ds.config.set_enable_shared_mem(False)

    # Apply dataset operations
    data1 = ds.TFRecordDataset(PYFUNCMAP_DATA_DIR, PYFUNCMAP_SCHEMA_DIR, shuffle=False)

    data1 = data1.map(operations=(lambda x: x + x), input_columns="col0", output_columns="out",
                      python_multiprocessing=True)

    with pytest.raises(ZeroDivisionError) as info:
        i = 0
        for _ in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
            i = i + 4
            if i > 8:
                # Cause division by zero error
                _ = i / 0
        assert "division by zero" in str(info.value)

    ds.config.set_enable_shared_mem(mem_original)


if __name__ == '__main__':
    skip_test_pyfunc_multiproc_shrmem()
    test_pyfunc_multiproc_noshrmem()
    test_pyfunc_multiproc_max_rowsize_small()
    test_pyfunc_multiproc_max_rowsize_large()
    test_pyfunc_multiproc_basic_pipeline(plot=True)
    test_pyfunc_multiproc_child_exception()
    test_pyfunc_multiproc_mainproc_exception()
