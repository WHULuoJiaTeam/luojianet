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
from util import save_and_check_dict, save_and_check_md5

import mindspore.dataset as ds
from mindspore import log as logger

# Dataset in DIR_1 has 5 rows and 5 columns
DATA_DIR_1 = ["../data/dataset/testTFBert5Rows1/5TFDatas.data"]
SCHEMA_DIR_1 = "../data/dataset/testTFBert5Rows1/datasetSchema.json"
# Dataset in DIR_2 has 5 rows and 2 columns
DATA_DIR_2 = ["../data/dataset/testTFBert5Rows2/5TFDatas.data"]
SCHEMA_DIR_2 = "../data/dataset/testTFBert5Rows2/datasetSchema.json"
# Dataset in DIR_3 has 3 rows and 2 columns
DATA_DIR_3 = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR_3 = "../data/dataset/test_tf_file_3_images/datasetSchema.json"
# Dataset in DIR_4 has 5 rows and 7 columns
DATA_DIR_4 = ["../data/dataset/testTFBert5Rows/5TFDatas.data"]
SCHEMA_DIR_4 = "../data/dataset/testTFBert5Rows/datasetSchema.json"

GENERATE_GOLDEN = False


def test_zip_01():
    """
    Test zip: zip 2 datasets, #rows-data1 == #rows-data2, #cols-data1 < #cols-data2
    """
    logger.info("test_zip_01")
    ds.config.set_seed(1)
    data1 = ds.TFRecordDataset(DATA_DIR_2, SCHEMA_DIR_2)
    data2 = ds.TFRecordDataset(DATA_DIR_1, SCHEMA_DIR_1)
    dataz = ds.zip((data1, data2))
    # Note: zipped dataset has 5 rows and 7 columns
    filename = "zip_01_result.npz"
    save_and_check_dict(dataz, filename, generate_golden=GENERATE_GOLDEN)


def test_zip_02():
    """
    Test zip: zip 2 datasets, #rows-data1 < #rows-data2, #cols-data1 == #cols-data2
    """
    logger.info("test_zip_02")
    ds.config.set_seed(1)
    data1 = ds.TFRecordDataset(DATA_DIR_3, SCHEMA_DIR_3)
    data2 = ds.TFRecordDataset(DATA_DIR_2, SCHEMA_DIR_2)
    dataz = ds.zip((data1, data2))
    # Note: zipped dataset has 3 rows and 4 columns
    filename = "zip_02_result.npz"
    save_and_check_md5(dataz, filename, generate_golden=GENERATE_GOLDEN)


def test_zip_03():
    """
    Test zip: zip 2 datasets, #rows-data1 > #rows-data2, #cols-data1 > #cols-data2
    """
    logger.info("test_zip_03")
    ds.config.set_seed(1)
    data1 = ds.TFRecordDataset(DATA_DIR_1, SCHEMA_DIR_1)
    data2 = ds.TFRecordDataset(DATA_DIR_3, SCHEMA_DIR_3)
    dataz = ds.zip((data1, data2))
    # Note: zipped dataset has 3 rows and 7 columns
    filename = "zip_03_result.npz"
    save_and_check_md5(dataz, filename, generate_golden=GENERATE_GOLDEN)


def test_zip_04():
    """
    Test zip: zip >2 datasets
    """
    logger.info("test_zip_04")
    ds.config.set_seed(1)
    data1 = ds.TFRecordDataset(DATA_DIR_1, SCHEMA_DIR_1)
    data2 = ds.TFRecordDataset(DATA_DIR_2, SCHEMA_DIR_2)
    data3 = ds.TFRecordDataset(DATA_DIR_3, SCHEMA_DIR_3)
    dataz = ds.zip((data1, data2, data3))
    # Note: zipped dataset has 3 rows and 9 columns
    filename = "zip_04_result.npz"
    save_and_check_md5(dataz, filename, generate_golden=GENERATE_GOLDEN)


def test_zip_05():
    """
    Test zip: zip dataset with renamed columns
    """
    logger.info("test_zip_05")
    ds.config.set_seed(1)
    data1 = ds.TFRecordDataset(DATA_DIR_4, SCHEMA_DIR_4, shuffle=True)
    data2 = ds.TFRecordDataset(DATA_DIR_2, SCHEMA_DIR_2, shuffle=True)

    data2 = data2.rename(input_columns="input_ids", output_columns="new_input_ids")
    data2 = data2.rename(input_columns="segment_ids", output_columns="new_segment_ids")

    dataz = ds.zip((data1, data2))
    # Note: zipped dataset has 5 rows and 9 columns
    filename = "zip_05_result.npz"
    save_and_check_dict(dataz, filename, generate_golden=GENERATE_GOLDEN)


def test_zip_06():
    """
    Test zip: zip dataset with renamed columns and repeat zipped dataset
    """
    logger.info("test_zip_06")
    ds.config.set_seed(1)
    data1 = ds.TFRecordDataset(DATA_DIR_4, SCHEMA_DIR_4, shuffle=False)
    data2 = ds.TFRecordDataset(DATA_DIR_2, SCHEMA_DIR_2, shuffle=False)

    data2 = data2.rename(input_columns="input_ids", output_columns="new_input_ids")
    data2 = data2.rename(input_columns="segment_ids", output_columns="new_segment_ids")

    dataz = ds.zip((data1, data2))
    dataz = dataz.repeat(2)
    # Note: resultant dataset has 10 rows and 9 columns
    filename = "zip_06_result.npz"
    save_and_check_dict(dataz, filename, generate_golden=GENERATE_GOLDEN)


def test_zip_exception_01():
    """
    Test zip: zip same datasets
    """
    logger.info("test_zip_exception_01")
    data1 = ds.TFRecordDataset(DATA_DIR_1, SCHEMA_DIR_1)

    try:
        dataz = ds.zip((data1, data1))

        num_iter = 0
        for _, item in enumerate(dataz.create_dict_iterator(num_epochs=1, output_numpy=True)):
            logger.info("item[input_mask] is {}".format(item["input_mask"]))
            num_iter += 1
        logger.info("Number of data in zipped dataz: {}".format(num_iter))

    except Exception as e:
        logger.info("Got an exception in DE: {}".format(str(e)))


def test_zip_exception_02():
    """
    Test zip: zip datasets with duplicate column name
    """
    logger.info("test_zip_exception_02")
    data1 = ds.TFRecordDataset(DATA_DIR_1, SCHEMA_DIR_1)
    data2 = ds.TFRecordDataset(DATA_DIR_4, SCHEMA_DIR_4)

    try:
        dataz = ds.zip((data1, data2))

        num_iter = 0
        for _, item in enumerate(dataz.create_dict_iterator(num_epochs=1, output_numpy=True)):
            logger.info("item[input_mask] is {}".format(item["input_mask"]))
            num_iter += 1
        logger.info("Number of data in zipped dataz: {}".format(num_iter))

    except Exception as e:
        logger.info("Got an exception in DE: {}".format(str(e)))


def test_zip_exception_03():
    """
    Test zip: zip with tuple of 1 dataset
    """
    logger.info("test_zip_exception_03")
    data1 = ds.TFRecordDataset(DATA_DIR_1, SCHEMA_DIR_1)

    try:
        dataz = ds.zip((data1))
        dataz = dataz.repeat(2)

        num_iter = 0
        for _, item in enumerate(dataz.create_dict_iterator(num_epochs=1, output_numpy=True)):
            logger.info("item[input_mask] is {}".format(item["input_mask"]))
            num_iter += 1
        logger.info("Number of data in zipped dataz: {}".format(num_iter))

    except Exception as e:
        logger.info("Got an exception in DE: {}".format(str(e)))


def test_zip_exception_04():
    """
    Test zip: zip with empty tuple of datasets
    """
    logger.info("test_zip_exception_04")

    try:
        dataz = ds.zip(())
        dataz = dataz.repeat(2)

        num_iter = 0
        for _, item in enumerate(dataz.create_dict_iterator(num_epochs=1, output_numpy=True)):
            logger.info("item[input_mask] is {}".format(item["input_mask"]))
            num_iter += 1
        logger.info("Number of data in zipped dataz: {}".format(num_iter))

    except Exception as e:
        logger.info("Got an exception in DE: {}".format(str(e)))


def test_zip_exception_05():
    """
    Test zip: zip with non-tuple of 2 datasets
    """
    logger.info("test_zip_exception_05")
    data1 = ds.TFRecordDataset(DATA_DIR_1, SCHEMA_DIR_1)
    data2 = ds.TFRecordDataset(DATA_DIR_2, SCHEMA_DIR_2)

    try:
        dataz = ds.zip(data1, data2)

        num_iter = 0
        for _, item in enumerate(dataz.create_dict_iterator(num_epochs=1, output_numpy=True)):
            logger.info("item[input_mask] is {}".format(item["input_mask"]))
            num_iter += 1
        logger.info("Number of data in zipped dataz: {}".format(num_iter))

    except Exception as e:
        logger.info("Got an exception in DE: {}".format(str(e)))


def test_zip_exception_06():
    """
    Test zip: zip with non-tuple of 1 dataset
    """
    logger.info("test_zip_exception_06")
    data1 = ds.TFRecordDataset(DATA_DIR_1, SCHEMA_DIR_1)

    try:
        dataz = ds.zip(data1)

        num_iter = 0
        for _, item in enumerate(dataz.create_dict_iterator(num_epochs=1, output_numpy=True)):
            logger.info("item[input_mask] is {}".format(item["input_mask"]))
            num_iter += 1
        logger.info("Number of data in zipped dataz: {}".format(num_iter))

    except Exception as e:
        logger.info("Got an exception in DE: {}".format(str(e)))


def test_zip_exception_07():
    """
    Test zip: zip with string as parameter
    """
    logger.info("test_zip_exception_07")

    try:
        dataz = ds.zip(('dataset1', 'dataset2'))

        num_iter = 0
        for _ in dataz.create_dict_iterator(num_epochs=1, output_numpy=True):
            num_iter += 1
        assert False

    except Exception as e:
        logger.info("Got an exception in DE: {}".format(str(e)))

    try:
        data = ds.TFRecordDataset(DATA_DIR_1, SCHEMA_DIR_1)
        dataz = data.zip(('dataset1',))

        num_iter = 0
        for _ in dataz.create_dict_iterator(num_epochs=1, output_numpy=True):
            num_iter += 1
        assert False

    except Exception as e:
        logger.info("Got an exception in DE: {}".format(str(e)))

if __name__ == '__main__':
    test_zip_01()
    test_zip_02()
    test_zip_03()
    test_zip_04()
    test_zip_05()
    test_zip_06()
    test_zip_exception_01()
    test_zip_exception_02()
    test_zip_exception_03()
    test_zip_exception_04()
    test_zip_exception_05()
    test_zip_exception_06()
    test_zip_exception_07()
