# Copyright 2019-2022 Huawei Technologies Co., Ltd
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
import mindspore.dataset.engine.iterators as it
from mindspore import log as logger

DATA_DIR = ["../data/dataset/testPyfuncMap/data.data"]
SCHEMA_DIR = "../data/dataset/testPyfuncMap/schema.json"
COLUMNS = ["col0", "col1", "col2"]
GENERATE_GOLDEN = False


def test_case_0():
    """
    Test PyFunc
    """
    logger.info("Test 1-1 PyFunc : lambda x : x + x")

    # apply dataset operations
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, shuffle=False)

    data1 = data1.map(operations=(lambda x: x + x), input_columns="col0", output_columns="out")

    i = 0
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
        # In this test, the dataset is 2x2 sequential tensors
        golden = np.array([[i * 2, (i + 1) * 2], [(i + 2) * 2, (i + 3) * 2]])
        np.testing.assert_array_equal(item["out"], golden)
        i = i + 4


def test_case_1():
    """
    Test PyFunc
    """
    logger.info("Test 1-n PyFunc : lambda x : (x , x + x) ")

    col = "col0"

    # apply dataset operations
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, shuffle=False)
    data1 = data1.map(operations=(lambda x: (x, x + x)), input_columns=col, output_columns=["out0", "out1"],
                      column_order=["out0", "out1"])

    i = 0
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
        # In this test, the dataset is 2x2 sequential tensors
        golden = np.array([[i, i + 1], [i + 2, i + 3]])
        np.testing.assert_array_equal(item["out0"], golden)
        golden = np.array([[i * 2, (i + 1) * 2], [(i + 2) * 2, (i + 3) * 2]])
        np.testing.assert_array_equal(item["out1"], golden)
        i = i + 4


def test_case_2():
    """
    Test PyFunc
    """
    logger.info("Test n-1 PyFunc : lambda x, y : x + y ")

    col = ["col0", "col1"]

    # apply dataset operations
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, shuffle=False)

    data1 = data1.map(operations=(lambda x, y: x + y), input_columns=col, output_columns="out",
                      column_order=["out"])

    i = 0
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
        # In this test, the dataset is 2x2 sequential tensors
        golden = np.array([[i * 2, (i + 1) * 2], [(i + 2) * 2, (i + 3) * 2]])
        np.testing.assert_array_equal(item["out"], golden)
        i = i + 4


def test_case_3():
    """
    Test PyFunc
    """
    logger.info("Test n-m PyFunc : lambda x, y : (x , x + 1, x + y)")

    col = ["col0", "col1"]

    # apply dataset operations
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, shuffle=False)

    data1 = data1.map(operations=(lambda x, y: (x, x + y, x + y + 1)), input_columns=col,
                      output_columns=["out0", "out1", "out2"], column_order=["out0", "out1", "out2"])

    i = 0
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
        # In this test, the dataset is 2x2 sequential tensors
        golden = np.array([[i, i + 1], [i + 2, i + 3]])
        np.testing.assert_array_equal(item["out0"], golden)
        golden = np.array([[i * 2, (i + 1) * 2], [(i + 2) * 2, (i + 3) * 2]])
        np.testing.assert_array_equal(item["out1"], golden)
        golden = np.array([[i * 2 + 1, (i + 1) * 2 + 1], [(i + 2) * 2 + 1, (i + 3) * 2 + 1]])
        np.testing.assert_array_equal(item["out2"], golden)
        i = i + 4


def test_case_4():
    """
    Test PyFunc
    """
    logger.info("Test Parallel n-m PyFunc : lambda x, y : (x , x + 1, x + y)")

    col = ["col0", "col1"]

    # apply dataset operations
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, shuffle=False)

    data1 = data1.map(operations=(lambda x, y: (x, x + y, x + y + 1)), input_columns=col,
                      output_columns=["out0", "out1", "out2"], num_parallel_workers=4,
                      column_order=["out0", "out1", "out2"])

    i = 0
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
        # In this test, the dataset is 2x2 sequential tensors
        golden = np.array([[i, i + 1], [i + 2, i + 3]])
        np.testing.assert_array_equal(item["out0"], golden)
        golden = np.array([[i * 2, (i + 1) * 2], [(i + 2) * 2, (i + 3) * 2]])
        np.testing.assert_array_equal(item["out1"], golden)
        golden = np.array([[i * 2 + 1, (i + 1) * 2 + 1], [(i + 2) * 2 + 1, (i + 3) * 2 + 1]])
        np.testing.assert_array_equal(item["out2"], golden)
        i = i + 4


# The execution of this function will acquire GIL
def func_5(x):
    return np.ones(x.shape, dtype=x.dtype)


def test_case_5():
    """
    Test PyFunc
    """
    logger.info("Test 1-1 PyFunc : lambda x: np.ones(x.shape)")

    # apply dataset operations
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, shuffle=False)

    data1 = data1.map(operations=func_5, input_columns="col0", output_columns="out")

    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
        # In this test, the dataset is 2x2 sequential tensors
        golden = np.array([[1, 1], [1, 1]])
        np.testing.assert_array_equal(item["out"], golden)


def test_case_6():
    """
    Test PyFunc
    """
    logger.info("Test PyFunc Compose : (lambda x : x + x), (lambda x : x + x)")

    # apply dataset operations
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, shuffle=False)

    data1 = data1.map(operations=[(lambda x: x + x), (lambda x: x + x)], input_columns="col0", output_columns="out")

    i = 0
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
        # In this test, the dataset is 2x2 sequential tensors
        golden = np.array([[i * 4, (i + 1) * 4], [(i + 2) * 4, (i + 3) * 4]])
        np.testing.assert_array_equal(item["out"], golden)
        i = i + 4


def test_case_7():
    """
    Test PyFunc
    """
    logger.info("Test 1-1 PyFunc Multiprocess: lambda x : x + x")

    # Reduce memory required by disabling the shared memory optimization
    mem_original = ds.config.get_enable_shared_mem()
    ds.config.set_enable_shared_mem(False)

    # apply dataset operations
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, shuffle=False)

    data1 = data1.map(operations=(lambda x: x + x), input_columns="col0", output_columns="out",
                      num_parallel_workers=4, python_multiprocessing=True)

    i = 0
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
        # In this test, the dataset is 2x2 sequential tensors
        golden = np.array([[i * 2, (i + 1) * 2], [(i + 2) * 2, (i + 3) * 2]])
        np.testing.assert_array_equal(item["out"], golden)
        i = i + 4

    ds.config.set_enable_shared_mem(mem_original)


def test_case_8():
    """
    Test PyFunc
    """
    logger.info("Test Multiprocess n-m PyFunc : lambda x, y : (x , x + 1, x + y)")

    # Reduce memory required by disabling the shared memory optimization
    mem_original = ds.config.get_enable_shared_mem()
    ds.config.set_enable_shared_mem(False)

    col = ["col0", "col1"]

    # apply dataset operations
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, shuffle=False)

    data1 = data1.map(operations=(lambda x, y: (x, x + y, x + y + 1)), input_columns=col,
                      output_columns=["out0", "out1", "out2"], num_parallel_workers=4,
                      column_order=["out0", "out1", "out2"],
                      python_multiprocessing=True)

    i = 0
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
        # In this test, the dataset is 2x2 sequential tensors
        golden = np.array([[i, i + 1], [i + 2, i + 3]])
        np.testing.assert_array_equal(item["out0"], golden)
        golden = np.array([[i * 2, (i + 1) * 2], [(i + 2) * 2, (i + 3) * 2]])
        np.testing.assert_array_equal(item["out1"], golden)
        golden = np.array([[i * 2 + 1, (i + 1) * 2 + 1], [(i + 2) * 2 + 1, (i + 3) * 2 + 1]])
        np.testing.assert_array_equal(item["out2"], golden)
        i = i + 4

    ds.config.set_enable_shared_mem(mem_original)


def test_case_9():
    """
    Test PyFunc
    """
    logger.info("Test multiple 1-1 PyFunc Multiprocess: lambda x : x + x")

    # Reduce memory required by disabling the shared memory optimization
    mem_original = ds.config.get_enable_shared_mem()
    ds.config.set_enable_shared_mem(False)

    # apply dataset operations
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, shuffle=False)

    data1 = data1.map(operations=[(lambda x: x + x), (lambda x: x + 1), (lambda x: x + 2)], input_columns="col0",
                      output_columns="out", num_parallel_workers=4, python_multiprocessing=True)

    i = 0
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
        # In this test, the dataset is 2x2 sequential tensors
        golden = np.array([[i * 2 + 3, (i + 1) * 2 + 3], [(i + 2) * 2 + 3, (i + 3) * 2 + 3]])
        np.testing.assert_array_equal(item["out"], golden)
        i = i + 4

    ds.config.set_enable_shared_mem(mem_original)


def test_case_10():
    """
    Test PyFunc
    """
    logger.info("Test multiple map with multiprocess: lambda x : x + x")

    # Reduce memory required by disabling the shared memory optimization
    mem_original = ds.config.get_enable_shared_mem()
    ds.config.set_enable_shared_mem(False)

    # apply dataset operations
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, shuffle=False)

    data1 = data1.map(operations=[(lambda x: x * 10)], input_columns="col0",
                      output_columns="out", num_parallel_workers=4)
    data1 = data1.map(operations=[(lambda x: x + x), (lambda x: x + 1), (lambda x: x + 2)], input_columns="out",
                      output_columns="out", num_parallel_workers=4, python_multiprocessing=True)

    i = 0
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
        # In this test, the dataset is 2x2 sequential tensors
        golden = np.array([[i * 20 + 3, (i + 1) * 20 + 3], [(i + 2) * 20 + 3, (i + 3) * 20 + 3]])
        np.testing.assert_array_equal(item["out"], golden)
        i = i + 4

    ds.config.set_enable_shared_mem(mem_original)


def test_pyfunc_implicit_compose():
    """
    Test Implicit Compose with pyfunc
    """
    logger.info("Test n-m PyFunc : lambda x, y : (x , x + 1, x + y)")

    col = ["col0", "col1"]

    # apply dataset operations
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, shuffle=False)

    data1 = data1.map(operations=[(lambda x, y: (x, x + y, x + y + 1)), (lambda x, y, z: (x, y, z))], input_columns=col,
                      output_columns=["out0", "out1", "out2"], column_order=["out0", "out1", "out2"])

    i = 0
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
        # In this test, the dataset is 2x2 sequential tensors
        golden = np.array([[i, i + 1], [i + 2, i + 3]])
        np.testing.assert_array_equal(item["out0"], golden)
        golden = np.array([[i * 2, (i + 1) * 2], [(i + 2) * 2, (i + 3) * 2]])
        np.testing.assert_array_equal(item["out1"], golden)
        golden = np.array([[i * 2 + 1, (i + 1) * 2 + 1], [(i + 2) * 2 + 1, (i + 3) * 2 + 1]])
        np.testing.assert_array_equal(item["out2"], golden)
        i = i + 4


def test_pyfunc_exception():
    logger.info("Test PyFunc Exception Throw: lambda x : raise Exception()")

    # Sometimes there are some ITERATORS left in ITERATORS_LIST when run all UTs together,
    # and cause core dump and blocking in this UT. Add cleanup() here to fix it.
    it._cleanup()  # pylint: disable=W0212

    def pyfunc():
        raise Exception("Pyfunc Throw")

    with pytest.raises(RuntimeError) as info:
        # apply dataset operations
        data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, shuffle=False)
        data1 = data1.map(operations=pyfunc, input_columns="col0", output_columns="out",
                          num_parallel_workers=4)
        for _ in data1:
            pass
        assert "Pyfunc Throw" in str(info.value)


def test_pyfunc_exception_multiprocess():
    """
    Feature: PyFunc in Map op
    Description: Test python_multiprocessing=True with exception in child pyfunc process
    Expectation: Exception is received and test ends gracefully
    """
    logger.info("Test Multiprocess PyFunc Exception Throw: lambda x : raise Exception()")

    def pyfunc():
        raise Exception("MP Pyfunc Throw")

    # Reduce memory required by disabling the shared memory optimization
    mem_original = ds.config.get_enable_shared_mem()
    ds.config.set_enable_shared_mem(False)

    with pytest.raises(RuntimeError) as info:
        # apply dataset operations
        data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, shuffle=False)
        data1 = data1.map(operations=pyfunc, input_columns="col0", output_columns="out",
                          num_parallel_workers=4, python_multiprocessing=True)
        for _ in data1:
            pass
        assert "MP Pyfunc Throw" in str(info.value)

    ds.config.set_enable_shared_mem(mem_original)


def test_func_with_yield_manifest_dataset_01():
    def pass_func(_):
        for i in range(10):
            yield (np.array([i]),)

    # Sometimes there are some ITERATORS left in ITERATORS_LIST when run all UTs together,
    # and cause core dump and blocking in this UT. Add cleanup() here to fix it.
    it._cleanup()  # pylint: disable=W0212

    DATA_FILE = "../data/dataset/testManifestData/test.manifest"
    data = ds.ManifestDataset(DATA_FILE)
    data = data.map(operations=pass_func, input_columns=["image"], num_parallel_workers=1, python_multiprocessing=True,
                    max_rowsize=1)
    num_iter = 0
    try:
        for _ in data.create_dict_iterator(num_epochs=1, output_numpy=True):
            num_iter += 1
    except RuntimeError as e:
        assert "Can not pickle <class 'generator'> object, " in str(e)


def test_func_mixed_with_ops():
    """
    Feature: Test adding computing operator into user defined python function
    Description: will decrease num_parallel_worker into 1
    Expectation: success
    """

    def generator_func():
        for i in range(1, 5):
            yield (np.ones(shape=[2, i]),)

    def func(x):
        import mindspore.ops as ops
        import mindspore
        from mindspore import Tensor

        flatten = ops.Flatten()
        output = flatten(Tensor(x, dtype=mindspore.float32))
        return output.asnumpy()

    dataset = ds.GeneratorDataset(generator_func, ["data"])

    dataset = dataset.map(operations=func, input_columns=["data"])
    assert dataset.num_parallel_workers == 1
    for _ in dataset.create_dict_iterator(num_epochs=1):
        pass


if __name__ == "__main__":
    test_case_0()
    test_case_1()
    test_case_2()
    test_case_3()
    test_case_4()
    test_case_5()
    test_case_6()
    test_case_7()
    test_case_8()
    test_case_9()
    test_case_10()
    test_pyfunc_implicit_compose()
    test_pyfunc_exception()
    test_pyfunc_exception_multiprocess()
    test_func_with_yield_manifest_dataset_01()
    test_func_mixed_with_ops()
