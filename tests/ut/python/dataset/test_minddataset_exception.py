#!/usr/bin/env python
# Copyright 2019-2021 Huawei Technologies Co., Ltd
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

import os
import pytest

import mindspore.dataset as ds
from mindspore.mindrecord import FileWriter


def create_cv_mindrecord(file_name, files_num):
    """tutorial for cv dataset writer."""
    if os.path.exists(file_name):
        os.remove(file_name)
    if os.path.exists("{}.db".format(file_name)):
        os.remove("{}.db".format(file_name))
    writer = FileWriter(file_name, files_num)
    cv_schema_json = {"file_name": {"type": "string"},
                      "label": {"type": "int32"}, "data": {"type": "bytes"}}
    data = [{"file_name": "001.jpg", "label": 43,
             "data": bytes('0xffsafdafda', encoding='utf-8')}]
    writer.add_schema(cv_schema_json, "img_schema")
    writer.add_index(["file_name", "label"])
    writer.write_raw_data(data)
    writer.commit()


def create_diff_schema_cv_mindrecord(file_name, files_num):
    """tutorial for cv dataset writer."""
    if os.path.exists(file_name):
        os.remove(file_name)
    if os.path.exists("{}.db".format(file_name)):
        os.remove("{}.db".format(file_name))
    writer = FileWriter(file_name, files_num)
    cv_schema_json = {"file_name_1": {"type": "string"},
                      "label": {"type": "int32"}, "data": {"type": "bytes"}}
    data = [{"file_name_1": "001.jpg", "label": 43,
             "data": bytes('0xffsafdafda', encoding='utf-8')}]
    writer.add_schema(cv_schema_json, "img_schema")
    writer.add_index(["file_name_1", "label"])
    writer.write_raw_data(data)
    writer.commit()


def create_diff_page_size_cv_mindrecord(file_name, files_num):
    """tutorial for cv dataset writer."""
    if os.path.exists(file_name):
        os.remove(file_name)
    if os.path.exists("{}.db".format(file_name)):
        os.remove("{}.db".format(file_name))
    writer = FileWriter(file_name, files_num)
    writer.set_page_size(1 << 26)  # 64MB
    cv_schema_json = {"file_name": {"type": "string"},
                      "label": {"type": "int32"}, "data": {"type": "bytes"}}
    data = [{"file_name": "001.jpg", "label": 43,
             "data": bytes('0xffsafdafda', encoding='utf-8')}]
    writer.add_schema(cv_schema_json, "img_schema")
    writer.add_index(["file_name", "label"])
    writer.write_raw_data(data)
    writer.commit()


def test_cv_lack_json():
    """tutorial for cv minderdataset."""
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    create_cv_mindrecord(file_name, 1)
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    with pytest.raises(Exception):
        ds.MindDataset(file_name, "no_exist.json",
                       columns_list, num_readers)
    os.remove(file_name)
    os.remove("{}.db".format(file_name))


def test_cv_lack_mindrecord():
    """tutorial for cv minderdataset."""
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    with pytest.raises(Exception, match="does not exist or permission denied"):
        _ = ds.MindDataset("no_exist.mindrecord", columns_list, num_readers)


def test_invalid_mindrecord():
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    with open(file_name, 'w') as f:
        f.write('just for test')
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    with pytest.raises(RuntimeError, match="Unexpected error. Invalid file, the size of mindrecord file header "
                                           "is larger than the upper limit."):
        data_set = ds.MindDataset(file_name, columns_list, num_readers)
        for _ in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
            pass
    os.remove(file_name)


def test_minddataset_lack_db():
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    create_cv_mindrecord(file_name, 1)
    os.remove("{}.db".format(file_name))
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    with pytest.raises(RuntimeError, match="Invalid file, failed to open mindrecord meta files "
                                           "while verifying meta file. Please check the meta file:"):
        data_set = ds.MindDataset(file_name, columns_list, num_readers)
        num_iter = 0
        for _ in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
            num_iter += 1
    os.remove(file_name)


def test_cv_minddataset_pk_sample_error_class_column():
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    create_cv_mindrecord(file_name, 1)
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    sampler = ds.PKSampler(5, None, True, 'no_exist_column')
    with pytest.raises(RuntimeError, match="Invalid data, 'class_column': no_exist_column can not found "
                                           "in fields of mindrecord files. Please check 'class_column' in PKSampler"):
        data_set = ds.MindDataset(
            file_name, columns_list, num_readers, sampler=sampler)
        num_iter = 0
        for _ in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
            num_iter += 1
    os.remove(file_name)
    os.remove("{}.db".format(file_name))


def test_cv_minddataset_pk_sample_exclusive_shuffle():
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    create_cv_mindrecord(file_name, 1)
    columns_list = ["data", "file_name", "label"]
    num_readers = 4
    sampler = ds.PKSampler(2)
    with pytest.raises(Exception, match="sampler and shuffle cannot be specified at the same time."):
        data_set = ds.MindDataset(file_name, columns_list, num_readers,
                                  sampler=sampler, shuffle=False)
        num_iter = 0
        for _ in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
            num_iter += 1
    os.remove(file_name)
    os.remove("{}.db".format(file_name))


def test_cv_minddataset_reader_different_schema():
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    file_name_1 = file_name + '_1'
    create_cv_mindrecord(file_name, 1)
    create_diff_schema_cv_mindrecord(file_name_1, 1)
    columns_list = ["data", "label"]
    num_readers = 4
    with pytest.raises(RuntimeError, match="Invalid file, the metadata of mindrecord file: "
                                           "test_cv_minddataset_reader_different_schema_1 is different from others, "
                                           "please make sure all the mindrecord files generated by the same script."):
        data_set = ds.MindDataset([file_name, file_name_1], columns_list,
                                  num_readers)
        num_iter = 0
        for _ in data_set.create_dict_iterator(num_epochs=1):
            num_iter += 1
    os.remove(file_name)
    os.remove("{}.db".format(file_name))
    os.remove(file_name_1)
    os.remove("{}.db".format(file_name_1))


def test_cv_minddataset_reader_different_page_size():
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    file_name_1 = file_name + '_1'
    create_cv_mindrecord(file_name, 1)
    create_diff_page_size_cv_mindrecord(file_name_1, 1)
    columns_list = ["data", "label"]
    num_readers = 4
    with pytest.raises(RuntimeError, match="Invalid file, the metadata of mindrecord file: " \
                                           "test_cv_minddataset_reader_different_page_size_1 is different " \
                                           "from others, please make sure all " \
                                           "the mindrecord files generated by the same script."):
        data_set = ds.MindDataset([file_name, file_name_1], columns_list,
                                  num_readers)
        num_iter = 0
        for _ in data_set.create_dict_iterator(num_epochs=1):
            num_iter += 1
    os.remove(file_name)
    os.remove("{}.db".format(file_name))
    os.remove(file_name_1)
    os.remove("{}.db".format(file_name_1))


def test_minddataset_invalidate_num_shards():
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    create_cv_mindrecord(file_name, 1)
    columns_list = ["data", "label"]
    num_readers = 4
    with pytest.raises(Exception) as error_info:
        data_set = ds.MindDataset(
            file_name, columns_list, num_readers, True, 1, 2)
        num_iter = 0
        for _ in data_set.create_dict_iterator(num_epochs=1):
            num_iter += 1
    try:
        assert 'Input shard_id is not within the required interval of [0, 0].' in str(
            error_info.value)
    except Exception as error:
        os.remove(file_name)
        os.remove("{}.db".format(file_name))
        raise error
    else:
        os.remove(file_name)
        os.remove("{}.db".format(file_name))


def test_minddataset_invalidate_shard_id():
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    create_cv_mindrecord(file_name, 1)
    columns_list = ["data", "label"]
    num_readers = 4
    with pytest.raises(Exception) as error_info:
        data_set = ds.MindDataset(
            file_name, columns_list, num_readers, True, 1, -1)
        num_iter = 0
        for _ in data_set.create_dict_iterator(num_epochs=1):
            num_iter += 1
    try:
        assert 'Input shard_id is not within the required interval of [0, 0].' in str(
            error_info.value)
    except Exception as error:
        os.remove(file_name)
        os.remove("{}.db".format(file_name))
        raise error
    else:
        os.remove(file_name)
        os.remove("{}.db".format(file_name))


def test_minddataset_shard_id_bigger_than_num_shard():
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    create_cv_mindrecord(file_name, 1)
    columns_list = ["data", "label"]
    num_readers = 4
    with pytest.raises(Exception) as error_info:
        data_set = ds.MindDataset(
            file_name, columns_list, num_readers, True, 2, 2)
        num_iter = 0
        for _ in data_set.create_dict_iterator(num_epochs=1):
            num_iter += 1
    try:
        assert 'Input shard_id is not within the required interval of [0, 1].' in str(
            error_info.value)
    except Exception as error:
        os.remove(file_name)
        os.remove("{}.db".format(file_name))
        raise error

    with pytest.raises(Exception) as error_info:
        data_set = ds.MindDataset(
            file_name, columns_list, num_readers, True, 2, 5)
        num_iter = 0
        for _ in data_set.create_dict_iterator(num_epochs=1):
            num_iter += 1
    try:
        assert 'Input shard_id is not within the required interval of [0, 1].' in str(
            error_info.value)
    except Exception as error:
        os.remove(file_name)
        os.remove("{}.db".format(file_name))
        raise error
    else:
        os.remove(file_name)
        os.remove("{}.db".format(file_name))


def test_cv_minddataset_partition_num_samples_equals_0():
    """tutorial for cv minddataset."""
    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    create_cv_mindrecord(file_name, 1)
    columns_list = ["data", "label"]
    num_readers = 4

    def partitions(num_shards):
        for partition_id in range(num_shards):
            data_set = ds.MindDataset(file_name, columns_list, num_readers,
                                      num_shards=num_shards,
                                      shard_id=partition_id, num_samples=-1)
            num_iter = 0
            for _ in data_set.create_dict_iterator(num_epochs=1):
                num_iter += 1

    with pytest.raises(ValueError) as error_info:
        partitions(5)
    try:
        assert 'num_samples exceeds the boundary between 0 and 9223372036854775807(INT64_MAX)' in str(
            error_info.value)
    except Exception as error:
        os.remove(file_name)
        os.remove("{}.db".format(file_name))
        raise error
    else:
        os.remove(file_name)
        os.remove("{}.db".format(file_name))


def test_mindrecord_exception():
    """tutorial for exception scenario of minderdataset + map would print error info."""

    def exception_func(item):
        raise Exception("Error occur!")

    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    create_cv_mindrecord(file_name, 1)
    columns_list = ["data", "file_name", "label"]
    with pytest.raises(RuntimeError, match="The corresponding data files"):
        data_set = ds.MindDataset(file_name, columns_list, shuffle=False)
        data_set = data_set.map(operations=exception_func, input_columns=["data"],
                                num_parallel_workers=1)
        num_iter = 0
        for _ in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
            num_iter += 1
    with pytest.raises(RuntimeError, match="The corresponding data files"):
        data_set = ds.MindDataset(file_name, columns_list, shuffle=False)
        data_set = data_set.map(operations=exception_func, input_columns=["file_name"],
                                num_parallel_workers=1)
        num_iter = 0
        for _ in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
            num_iter += 1
    with pytest.raises(RuntimeError, match="The corresponding data files"):
        data_set = ds.MindDataset(file_name, columns_list, shuffle=False)
        data_set = data_set.map(operations=exception_func, input_columns=["label"],
                                num_parallel_workers=1)
        num_iter = 0
        for _ in data_set.create_dict_iterator(num_epochs=1, output_numpy=True):
            num_iter += 1
    os.remove(file_name)
    os.remove("{}.db".format(file_name))

def test_shuffle_with_num_samples_exception():
    """
    Feature: shuffle files or shuffle samples of each file
    Description: set Shuffle.FILES or Shuffle.INFILE and num_samples
    Expectation: exception occurred
    """
    MIND_DIR = "../data/mindrecord/testMindDataSet/testImageNetData/imagenet.mindrecord0"
    with pytest.raises(ValueError, match="'Shuffle.FILES' or 'Shuffle.INFILE' and 'num_samples' "
                                         "cannot be specified at the same time."):
        _ = ds.MindDataset(MIND_DIR, shuffle=ds.Shuffle.FILES, num_samples=5)

    with pytest.raises(ValueError, match="'Shuffle.FILES' or 'Shuffle.INFILE' and 'num_samples' "
                                         "cannot be specified at the same time."):
        _ = ds.MindDataset(MIND_DIR, shuffle=ds.Shuffle.INFILE, num_samples=5)

if __name__ == '__main__':
    test_cv_lack_json()
    test_cv_lack_mindrecord()
    test_invalid_mindrecord()
    test_minddataset_lack_db()
    test_cv_minddataset_pk_sample_error_class_column()
    test_cv_minddataset_pk_sample_exclusive_shuffle()
    test_cv_minddataset_reader_different_schema()
    test_cv_minddataset_reader_different_page_size()
    test_minddataset_invalidate_num_shards()
    test_minddataset_invalidate_shard_id()
    test_minddataset_shard_id_bigger_than_num_shard()
    test_cv_minddataset_partition_num_samples_equals_0()
    test_mindrecord_exception()
