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
import mindspore.common.dtype as mstype
import mindspore.dataset as ds
from mindspore import log as logger

# just a basic test with parallel random data op
def test_randomdataset_basic1():
    logger.info("Test randomdataset basic 1")

    schema = ds.Schema()
    schema.add_column('image', de_type=mstype.uint8, shape=[2])
    schema.add_column('label', de_type=mstype.uint8, shape=[1])

    # apply dataset operations
    ds1 = ds.RandomDataset(schema=schema, total_rows=50, num_parallel_workers=4)
    ds1 = ds1.repeat(4)

    num_iter = 0
    for data in ds1.create_dict_iterator(num_epochs=1):  # each data is a dictionary
        # in this example, each dictionary has keys "image" and "label"
        logger.info("{} image: {}".format(num_iter, data["image"]))
        logger.info("{} label: {}".format(num_iter, data["label"]))
        num_iter += 1

    logger.info("Number of data in ds1: {}".format(num_iter))
    assert num_iter == 200
    logger.info("Test randomdataset basic 1 complete")


# Another simple test
def test_randomdataset_basic2():
    logger.info("Test randomdataset basic 2")

    schema = ds.Schema()
    schema.add_column('image', de_type=mstype.uint8,
                      shape=[640, 480, 3])  # 921600 bytes (a bit less than 1 MB per image)
    schema.add_column('label', de_type=mstype.uint8, shape=[1])

    # Make up 10 rows
    ds1 = ds.RandomDataset(schema=schema, total_rows=10, num_parallel_workers=1)
    ds1 = ds1.repeat(4)

    num_iter = 0
    for data in ds1.create_dict_iterator(num_epochs=1):  # each data is a dictionary
        # in this example, each dictionary has keys "image" and "label"
        # logger.info(data["image"])
        logger.info("printing the label: {}".format(data["label"]))
        num_iter += 1

    logger.info("Number of data in ds1: {}".format(num_iter))
    assert num_iter == 40
    logger.info("Test randomdataset basic 2 complete")


# Another simple test
def test_randomdataset_basic3():
    logger.info("Test randomdataset basic 3")

    # Make up 10 samples, but here even the schema is randomly created
    # The columns are named like this "c0", "c1", "c2" etc
    # But, we will use a tuple iterator instead of dict iterator so the column names
    # are not needed to iterate
    ds1 = ds.RandomDataset(total_rows=10, num_parallel_workers=1)
    ds1 = ds1.repeat(2)

    num_iter = 0
    for _ in ds1.create_tuple_iterator(num_epochs=1):
        num_iter += 1

    logger.info("Number of data in ds1: {}".format(num_iter))
    assert num_iter == 20
    logger.info("Test randomdataset basic 3 Complete")

if __name__ == '__main__':
    test_randomdataset_basic1()
    test_randomdataset_basic2()
    test_randomdataset_basic3()
