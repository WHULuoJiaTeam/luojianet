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
# See the License foNtest_resr the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Built-in validators.
"""
import inspect as ins
import os
import re
from functools import wraps

import numpy as np
from mindspore import log
from mindspore._c_expression import typing
from ..core.validator_helpers import parse_user_args, type_check, type_check_list, check_value, \
    INT32_MAX, check_valid_detype, check_dir, check_file, check_sampler_shuffle_shard_options, \
    validate_dataset_param_value, check_padding_options, check_gnn_list_or_ndarray, check_gnn_list_of_pair_or_ndarray, \
    check_num_parallel_workers, check_columns, check_pos_int32, check_valid_str, check_dataset_num_shards_shard_id, \
    check_valid_list_tuple

from . import datasets
from . import samplers
from . import cache_client


def check_cmu_arctic_dataset(method):
    """A wrapper that wraps a parameter checker around the original CMUArcticDataset."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        _, param_dict = parse_user_args(method, *args, **kwargs)

        nreq_param_int = ['num_samples', 'num_parallel_workers', 'num_shards', 'shard_id']
        nreq_param_bool = ['shuffle']

        dataset_dir = param_dict.get('dataset_dir')
        check_dir(dataset_dir)

        name = param_dict.get('name')
        if name is not None:
            check_valid_str(name, ['aew', 'ahw', 'aup', 'awb', 'axb', 'bdl', 'clb', 'eey',
                                   'fem', 'gka', 'jmk', 'ksp', 'ljm', 'lnh', 'rms', 'rxr', 'slp', 'slt'], "name")

        validate_dataset_param_value(nreq_param_int, param_dict, int)
        validate_dataset_param_value(nreq_param_bool, param_dict, bool)

        check_sampler_shuffle_shard_options(param_dict)

        cache = param_dict.get('cache')
        check_cache_option(cache)

        return method(self, *args, **kwargs)

    return new_method


def check_gtzan_dataset(method):
    """A wrapper that wraps a parameter checker around the original GTZANDataset."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        _, param_dict = parse_user_args(method, *args, **kwargs)

        nreq_param_int = ['num_samples', 'num_parallel_workers', 'num_shards', 'shard_id']
        nreq_param_bool = ['shuffle']

        dataset_dir = param_dict.get('dataset_dir')
        check_dir(dataset_dir)

        usage = param_dict.get('usage')
        if usage is not None:
            check_valid_str(usage, ['train', 'valid', 'test', 'all'], "usage")

        validate_dataset_param_value(nreq_param_int, param_dict, int)
        validate_dataset_param_value(nreq_param_bool, param_dict, bool)

        check_sampler_shuffle_shard_options(param_dict)

        cache = param_dict.get('cache')
        check_cache_option(cache)

        return method(self, *args, **kwargs)

    return new_method


def check_imagefolderdataset(method):
    """A wrapper that wraps a parameter checker around the original Dataset(ImageFolderDataset)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        _, param_dict = parse_user_args(method, *args, **kwargs)

        nreq_param_int = ['num_samples', 'num_parallel_workers', 'num_shards', 'shard_id']
        nreq_param_bool = ['shuffle', 'decode']
        nreq_param_list = ['extensions']
        nreq_param_dict = ['class_indexing']

        dataset_dir = param_dict.get('dataset_dir')
        check_dir(dataset_dir)

        validate_dataset_param_value(nreq_param_int, param_dict, int)
        validate_dataset_param_value(nreq_param_bool, param_dict, bool)
        validate_dataset_param_value(nreq_param_list, param_dict, list)
        validate_dataset_param_value(nreq_param_dict, param_dict, dict)
        check_sampler_shuffle_shard_options(param_dict)

        cache = param_dict.get('cache')
        check_cache_option(cache)

        return method(self, *args, **kwargs)

    return new_method


def check_imdb_dataset(method):
    """A wrapper that wraps a parameter checker around the original IMDBDataset."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        _, param_dict = parse_user_args(method, *args, **kwargs)

        nreq_param_int = ['num_samples', 'num_parallel_workers', 'num_shards', 'shard_id']
        nreq_param_bool = ['shuffle']

        dataset_dir = param_dict.get('dataset_dir')
        check_dir(dataset_dir)

        validate_dataset_param_value(nreq_param_int, param_dict, int)
        validate_dataset_param_value(nreq_param_bool, param_dict, bool)
        check_sampler_shuffle_shard_options(param_dict)

        cache = param_dict.get('cache')
        check_cache_option(cache)

        usage = param_dict.get('usage')
        if usage is not None:
            check_valid_str(usage, ["train", "test", "all"], "usage")

        return method(self, *args, **kwargs)

    return new_method


def check_iwslt2016_dataset(method):
    """A wrapper that wraps a parameter checker around the original Dataset(IWSLT2016dataset)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        _, param_dict = parse_user_args(method, *args, **kwargs)

        nreq_param_int = ['num_samples', 'num_parallel_workers', 'num_shards', 'shard_id']

        dataset_dir = param_dict.get('dataset_dir')
        check_dir(dataset_dir)

        # check usage
        usage = param_dict.get('usage')
        if usage is not None:
            check_valid_str(usage, ["train", "test", "valid", "all"], "usage")

        support_language_pair = [
            ['en', 'ar'], ['en', 'ar'], ['en', 'de'], ['en', 'fr'], ['en', 'cs'], ['ar', 'en'], ['fr', 'en'],
            ['de', 'en'], ['cs', 'en']
        ]
        support_language_pair_tuple = (
            ('en', 'ar'), ('en', 'ar'), ('en', 'de'), ('en', 'fr'), ('en', 'cs'), ('ar', 'en'), ('fr', 'en'),
            ('de', 'en'), ('cs', 'en')
        )
        support_set_type = ["dev2010", "tst2010", "tst2011", "tst2012", "tst2013", "tst2014"]
        # check language_pair
        language_pair = param_dict.get('language_pair')
        if language_pair is not None:
            if isinstance(language_pair, (list,)):
                check_valid_list_tuple(language_pair, support_language_pair, (str,), "language_pair")
            elif isinstance(language_pair, (tuple,)):
                check_valid_list_tuple(language_pair, support_language_pair_tuple, (str,), "language_pair")
            else:
                raise TypeError("language_pair should be a type list or tuple of length 2.")

        # check valid_set
        valid_set = param_dict.get('valid_set')
        if valid_set is not None:
            check_valid_str(valid_set, support_set_type, "valid_set")

        # check test_set
        test_set = param_dict.get('test_set')
        if test_set is not None:
            check_valid_str(test_set, support_set_type, "test_set")

        validate_dataset_param_value(nreq_param_int, param_dict, int)
        check_sampler_shuffle_shard_options(param_dict)

        cache = param_dict.get('cache')
        check_cache_option(cache)

        return method(self, *args, **kwargs)

    return new_method


def check_iwslt2017_dataset(method):
    """A wrapper that wraps a parameter checker around the original Dataset(IWSLT2017dataset)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        _, param_dict = parse_user_args(method, *args, **kwargs)

        nreq_param_int = ['num_samples', 'num_parallel_workers', 'num_shards', 'shard_id']

        dataset_dir = param_dict.get('dataset_dir')
        check_dir(dataset_dir)

        # check usage
        usage = param_dict.get('usage')
        if usage is not None:
            check_valid_str(usage, ["train", "test", "valid", "all"], "usage")

        support_language_pair = [
            ['en', 'nl'], ['en', 'de'], ['en', 'it'], ['en', 'ro'], ['ro', 'de'], ['ro', 'en'], ['ro', 'nl'],
            ['ro', 'it'], ['de', 'ro'], ['de', 'en'], ['de', 'nl'], ['de', 'it'], ['it', 'en'], ['it', 'nl'],
            ['it', 'de'], ['it', 'ro'], ['nl', 'de'], ['nl', 'en'], ['nl', 'it'], ['nl', 'ro']
        ]
        support_language_pair_tuple = (
            ('en', 'nl'), ('en', 'de'), ('en', 'it'), ('en', 'ro'), ('ro', 'de'), ('ro', 'en'), ('ro', 'nl'),
            ('ro', 'it'), ('de', 'ro'), ('de', 'en'), ('de', 'nl'), ('de', 'it'), ('it', 'en'), ('it', 'nl'),
            ('it', 'de'), ('it', 'ro'), ('nl', 'de'), ('nl', 'en'), ('nl', 'it'), ('nl', 'ro')
        )
        # check language_pair
        language_pair = param_dict.get('language_pair')
        if language_pair is not None:
            if isinstance(language_pair, (list,)):
                check_valid_list_tuple(language_pair, support_language_pair, (str,), "language_pair")
            elif isinstance(language_pair, (tuple,)):
                check_valid_list_tuple(language_pair, support_language_pair_tuple, (str,), "language_pair")
            else:
                raise TypeError("language_pair should be a type list or tuple of length 2.")

        validate_dataset_param_value(nreq_param_int, param_dict, int)
        check_sampler_shuffle_shard_options(param_dict)

        cache = param_dict.get('cache')
        check_cache_option(cache)

        return method(self, *args, **kwargs)

    return new_method


def check_kittidataset(method):
    """A wrapper that wraps a parameter checker around the original Dataset(KITTIDataset)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        _, param_dict = parse_user_args(method, *args, **kwargs)

        nreq_param_int = ['num_samples', 'num_parallel_workers', 'num_shards', 'shard_id']
        nreq_param_bool = ['shuffle', 'decode']

        dataset_dir = param_dict.get('dataset_dir')
        check_dir(dataset_dir)

        usage = param_dict.get('usage')
        if usage is not None:
            check_valid_str(usage, ["train", "test"], "usage")

        validate_dataset_param_value(nreq_param_int, param_dict, int)
        validate_dataset_param_value(nreq_param_bool, param_dict, bool)
        check_sampler_shuffle_shard_options(param_dict)

        cache = param_dict.get('cache')
        check_cache_option(cache)

        return method(self, *args, **kwargs)

    return new_method


def check_lsun_dataset(method):
    """A wrapper that wraps a parameter checker around the original Dataset(LSUNDataset)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        _, param_dict = parse_user_args(method, *args, **kwargs)

        nreq_param_int = ['num_samples', 'num_parallel_workers', 'num_shards', 'shard_id']
        nreq_param_bool = ['shuffle', 'decode']
        nreq_param_list = ['classes']

        dataset_dir = param_dict.get('dataset_dir')
        check_dir(dataset_dir)

        usage = param_dict.get('usage')
        if usage is not None:
            check_valid_str(usage, ["train", "test", "valid", "all"], "usage")

        validate_dataset_param_value(nreq_param_int, param_dict, int)
        validate_dataset_param_value(nreq_param_bool, param_dict, bool)
        validate_dataset_param_value(nreq_param_list, param_dict, list)

        categories = [
            'bedroom', 'bridge', 'church_outdoor', 'classroom', 'conference_room', 'dining_room', 'kitchen',
            'living_room', 'restaurant', 'tower'
        ]
        classes = param_dict.get('classes')
        if classes is not None:
            for class_name in classes:
                check_valid_str(class_name, categories, "classes")

        check_sampler_shuffle_shard_options(param_dict)

        cache = param_dict.get('cache')
        check_cache_option(cache)

        return method(self, *args, **kwargs)

    return new_method


def check_mnist_cifar_dataset(method):
    """A wrapper that wraps a parameter checker around the original Dataset(ManifestDataset, Cifar10/100Dataset)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        _, param_dict = parse_user_args(method, *args, **kwargs)

        nreq_param_int = ['num_samples', 'num_parallel_workers', 'num_shards', 'shard_id']
        nreq_param_bool = ['shuffle']

        dataset_dir = param_dict.get('dataset_dir')
        check_dir(dataset_dir)

        usage = param_dict.get('usage')
        if usage is not None:
            check_valid_str(usage, ["train", "test", "all"], "usage")

        validate_dataset_param_value(nreq_param_int, param_dict, int)
        validate_dataset_param_value(nreq_param_bool, param_dict, bool)

        check_sampler_shuffle_shard_options(param_dict)

        cache = param_dict.get('cache')
        check_cache_option(cache)

        return method(self, *args, **kwargs)

    return new_method


def check_omniglotdataset(method):
    """A wrapper that wraps a parameter checker around the original Dataset(OmniglotDataset)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        _, param_dict = parse_user_args(method, *args, **kwargs)

        nreq_param_int = ['num_samples', 'num_parallel_workers', 'num_shards', 'shard_id']
        nreq_param_bool = ['shuffle', 'background', 'decode']
        dataset_dir = param_dict.get('dataset_dir')
        check_dir(dataset_dir)

        validate_dataset_param_value(nreq_param_int, param_dict, int)
        validate_dataset_param_value(nreq_param_bool, param_dict, bool)
        check_sampler_shuffle_shard_options(param_dict)

        cache = param_dict.get('cache')
        check_cache_option(cache)

        return method(self, *args, **kwargs)

    return new_method


def check_photo_tour_dataset(method):
    """A wrapper that wraps a parameter checker around the original Dataset(PhotoTourDataset)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        _, param_dict = parse_user_args(method, *args, **kwargs)

        nreq_param_int = ['num_samples', 'num_parallel_workers', 'num_shards', 'shard_id']
        nreq_param_bool = ['shuffle']

        dataset_dir = param_dict.get('dataset_dir')
        check_dir(dataset_dir)

        usage = param_dict.get('usage')
        if usage is not None:
            check_valid_str(usage, ["train", "test"], "usage")
        name = param_dict.get('name')
        check_valid_str(name, ["notredame", "yosemite", "liberty", "notredame_harris",
                               "yosemite_harris", "liberty_harris"], "name")
        validate_dataset_param_value(nreq_param_int, param_dict, int)
        validate_dataset_param_value(nreq_param_bool, param_dict, bool)

        check_sampler_shuffle_shard_options(param_dict)
        cache = param_dict.get('cache')
        check_cache_option(cache)
        return method(self, *args, **kwargs)

    return new_method


def check_places365_dataset(method):
    """A wrapper that wraps a parameter checker around the original Dataset(Places365Dataset)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        _, param_dict = parse_user_args(method, *args, **kwargs)

        nreq_param_int = ['num_samples', 'num_parallel_workers', 'num_shards', 'shard_id']
        nreq_param_bool = ['shuffle', 'small', 'decode']

        dataset_dir = param_dict.get('dataset_dir')
        check_dir(dataset_dir)

        usage = param_dict.get('usage')
        if usage is not None:
            check_valid_str(usage, ["train-standard", "train-challenge", "val"], "usage")

        validate_dataset_param_value(nreq_param_int, param_dict, int)
        validate_dataset_param_value(nreq_param_bool, param_dict, bool)

        check_sampler_shuffle_shard_options(param_dict)

        cache = param_dict.get('cache')
        check_cache_option(cache)

        return method(self, *args, **kwargs)

    return new_method


def check_qmnist_dataset(method):
    """A wrapper that wraps a parameter checker around the original Dataset(QMnistDataset)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        _, param_dict = parse_user_args(method, *args, **kwargs)

        nreq_param_int = ['num_samples', 'num_parallel_workers', 'num_shards', 'shard_id']
        nreq_param_bool = ['shuffle', 'compat']

        dataset_dir = param_dict.get('dataset_dir')
        check_dir(dataset_dir)

        usage = param_dict.get('usage')
        if usage is not None:
            check_valid_str(usage, ["train", "test", "test10k", "test50k", "nist", "all"], "usage")

        validate_dataset_param_value(nreq_param_int, param_dict, int)
        validate_dataset_param_value(nreq_param_bool, param_dict, bool)

        check_sampler_shuffle_shard_options(param_dict)

        cache = param_dict.get('cache')
        check_cache_option(cache)

        return method(self, *args, **kwargs)

    return new_method


def check_manifestdataset(method):
    """A wrapper that wraps a parameter checker around the original Dataset(ManifestDataset)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        _, param_dict = parse_user_args(method, *args, **kwargs)

        nreq_param_int = ['num_samples', 'num_parallel_workers', 'num_shards', 'shard_id']
        nreq_param_bool = ['shuffle', 'decode']
        nreq_param_str = ['usage']
        nreq_param_dict = ['class_indexing']

        dataset_file = param_dict.get('dataset_file')
        check_file(dataset_file)

        validate_dataset_param_value(nreq_param_int, param_dict, int)
        validate_dataset_param_value(nreq_param_bool, param_dict, bool)
        validate_dataset_param_value(nreq_param_str, param_dict, str)
        validate_dataset_param_value(nreq_param_dict, param_dict, dict)

        check_sampler_shuffle_shard_options(param_dict)

        cache = param_dict.get('cache')
        check_cache_option(cache)

        return method(self, *args, **kwargs)

    return new_method


def check_sbu_dataset(method):
    """A wrapper that wraps a parameter checker around the original Dataset(SBUDataset)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        _, param_dict = parse_user_args(method, *args, **kwargs)

        nreq_param_int = ['num_samples', 'num_parallel_workers', 'num_shards', 'shard_id']
        nreq_param_bool = ['shuffle', 'decode']

        dataset_dir = param_dict.get('dataset_dir')
        check_dir(dataset_dir)

        check_file(os.path.join(dataset_dir, "SBU_captioned_photo_dataset_urls.txt"))
        check_file(os.path.join(dataset_dir, "SBU_captioned_photo_dataset_captions.txt"))
        check_dir(os.path.join(dataset_dir, "sbu_images"))

        validate_dataset_param_value(nreq_param_int, param_dict, int)
        validate_dataset_param_value(nreq_param_bool, param_dict, bool)

        check_sampler_shuffle_shard_options(param_dict)

        cache = param_dict.get('cache')
        check_cache_option(cache)

        return method(self, *args, **kwargs)

    return new_method


def check_sogou_news_dataset(method):
    """A wrapper that wraps a parameter checker around the original Dataset(SogouNewsDataset)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        _, param_dict = parse_user_args(method, *args, **kwargs)

        nreq_param_int = ['num_samples', 'num_parallel_workers', 'num_shards', 'shard_id']

        dataset_dir = param_dict.get('dataset_dir')
        check_dir(dataset_dir)

        usage = param_dict.get('usage')
        if usage is not None:
            check_valid_str(usage, ["train", "test", "all"], "usage")

        validate_dataset_param_value(nreq_param_int, param_dict, int)
        check_sampler_shuffle_shard_options(param_dict)

        cache = param_dict.get('cache')
        check_cache_option(cache)

        return method(self, *args, **kwargs)

    return new_method


def check_tfrecorddataset(method):
    """A wrapper that wraps a parameter checker around the original Dataset(TFRecordDataset)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        _, param_dict = parse_user_args(method, *args, **kwargs)

        nreq_param_int = ['num_samples', 'num_parallel_workers', 'num_shards', 'shard_id']
        nreq_param_list = ['columns_list']
        nreq_param_bool = ['shard_equal_rows']

        dataset_files = param_dict.get('dataset_files')
        if not isinstance(dataset_files, (str, list)):
            raise TypeError("dataset_files should be type str or a list of strings.")

        validate_dataset_param_value(nreq_param_int, param_dict, int)
        validate_dataset_param_value(nreq_param_list, param_dict, list)
        validate_dataset_param_value(nreq_param_bool, param_dict, bool)

        check_sampler_shuffle_shard_options(param_dict)

        cache = param_dict.get('cache')
        check_cache_option(cache)

        return method(self, *args, **kwargs)

    return new_method


def check_udpos_dataset(method):
    """A wrapper that wraps a parameter checker around the original Dataset(UDPOSDataset)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        _, param_dict = parse_user_args(method, *args, **kwargs)

        nreq_param_int = ['num_samples', 'num_parallel_workers', 'num_shards', 'shard_id']

        # check dataset_dir; required argument
        dataset_dir = param_dict.get('dataset_dir')
        check_dir(dataset_dir)

        # check usage
        usage = param_dict.get('usage')
        if usage is not None:
            check_valid_str(usage, ["train", "valid", "test", "all"], "usage")

        validate_dataset_param_value(nreq_param_int, param_dict, int)
        check_sampler_shuffle_shard_options(param_dict)

        cache = param_dict.get('cache')
        check_cache_option(cache)

        return method(self, *args, **kwargs)

    return new_method


def check_usps_dataset(method):
    """A wrapper that wraps a parameter checker around the original Dataset(USPSDataset)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        _, param_dict = parse_user_args(method, *args, **kwargs)

        nreq_param_int = ['num_samples', 'num_parallel_workers', 'num_shards', 'shard_id']

        dataset_dir = param_dict.get('dataset_dir')
        check_dir(dataset_dir)

        usage = param_dict.get('usage')
        if usage is not None:
            check_valid_str(usage, ["train", "test", "all"], "usage")

        validate_dataset_param_value(nreq_param_int, param_dict, int)
        check_sampler_shuffle_shard_options(param_dict)

        cache = param_dict.get('cache')
        check_cache_option(cache)

        return method(self, *args, **kwargs)

    return new_method


def check_caltech101_dataset(method):
    """A wrapper that wraps a parameter checker around the original Dataset(Caltech101Dataset)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        _, param_dict = parse_user_args(method, *args, **kwargs)

        nreq_param_int = ['num_samples', 'num_parallel_workers', 'num_shards', 'shard_id']
        nreq_param_bool = ['shuffle', 'decode']
        nreq_param_str = ['target_type']

        dataset_dir = param_dict.get('dataset_dir')
        check_dir(dataset_dir)

        target_type = param_dict.get('target_type')
        if target_type is not None:
            check_valid_str(target_type, ["category", "annotation", "all"], "target_type")

        validate_dataset_param_value(nreq_param_int, param_dict, int)
        validate_dataset_param_value(nreq_param_bool, param_dict, bool)
        validate_dataset_param_value(nreq_param_str, param_dict, str)
        check_sampler_shuffle_shard_options(param_dict)

        cache = param_dict.get('cache')
        check_cache_option(cache)

        return method(self, *args, **kwargs)

    return new_method


def check_caltech256_dataset(method):
    """A wrapper that wraps a parameter checker around the original Dataset(Caltech256Dataset)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        _, param_dict = parse_user_args(method, *args, **kwargs)

        nreq_param_int = ['num_samples', 'num_parallel_workers', 'num_shards', 'shard_id']
        nreq_param_bool = ['shuffle', 'decode']

        dataset_dir = param_dict.get('dataset_dir')
        check_dir(dataset_dir)

        validate_dataset_param_value(nreq_param_int, param_dict, int)
        validate_dataset_param_value(nreq_param_bool, param_dict, bool)
        check_sampler_shuffle_shard_options(param_dict)

        cache = param_dict.get('cache')
        check_cache_option(cache)

        return method(self, *args, **kwargs)

    return new_method


def check_vocdataset(method):
    """A wrapper that wraps a parameter checker around the original Dataset(VOCDataset)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        _, param_dict = parse_user_args(method, *args, **kwargs)

        nreq_param_int = ['num_samples', 'num_parallel_workers', 'num_shards', 'shard_id']
        nreq_param_bool = ['shuffle', 'decode']
        nreq_param_dict = ['class_indexing']

        dataset_dir = param_dict.get('dataset_dir')
        check_dir(dataset_dir)

        task = param_dict.get('task')
        type_check(task, (str,), "task")

        usage = param_dict.get('usage')
        type_check(usage, (str,), "usage")
        dataset_dir = os.path.realpath(dataset_dir)

        if task == "Segmentation":
            imagesets_file = os.path.join(dataset_dir, "ImageSets", "Segmentation", usage + ".txt")
            if param_dict.get('class_indexing') is not None:
                raise ValueError("class_indexing is not supported in Segmentation task.")
        elif task == "Detection":
            imagesets_file = os.path.join(dataset_dir, "ImageSets", "Main", usage + ".txt")
        else:
            raise ValueError("Invalid task : " + task + ".")

        check_file(imagesets_file)

        validate_dataset_param_value(nreq_param_int, param_dict, int)
        validate_dataset_param_value(nreq_param_bool, param_dict, bool)
        validate_dataset_param_value(nreq_param_dict, param_dict, dict)
        check_sampler_shuffle_shard_options(param_dict)

        cache = param_dict.get('cache')
        check_cache_option(cache)

        return method(self, *args, **kwargs)

    return new_method


def check_cocodataset(method):
    """A wrapper that wraps a parameter checker around the original Dataset(CocoDataset)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        _, param_dict = parse_user_args(method, *args, **kwargs)

        nreq_param_int = ['num_samples', 'num_parallel_workers', 'num_shards', 'shard_id']
        nreq_param_bool = ['shuffle', 'decode']

        dataset_dir = param_dict.get('dataset_dir')
        check_dir(dataset_dir)

        annotation_file = param_dict.get('annotation_file')
        check_file(annotation_file)

        task = param_dict.get('task')
        type_check(task, (str,), "task")

        if task not in {'Detection', 'Stuff', 'Panoptic', 'Keypoint', 'Captioning'}:
            raise ValueError("Invalid task type: " + task + ".")

        validate_dataset_param_value(nreq_param_int, param_dict, int)

        validate_dataset_param_value(nreq_param_bool, param_dict, bool)

        sampler = param_dict.get('sampler')
        if sampler is not None and isinstance(sampler, samplers.PKSampler):
            raise ValueError("CocoDataset doesn't support PKSampler.")
        check_sampler_shuffle_shard_options(param_dict)

        cache = param_dict.get('cache')
        check_cache_option(cache)

        return method(self, *args, **kwargs)

    return new_method


def check_celebadataset(method):
    """A wrapper that wraps a parameter checker around the original Dataset(CelebADataset)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        _, param_dict = parse_user_args(method, *args, **kwargs)

        nreq_param_int = ['num_samples', 'num_parallel_workers', 'num_shards', 'shard_id']
        nreq_param_bool = ['shuffle', 'decode']
        nreq_param_list = ['extensions']
        nreq_param_str = ['dataset_type']

        dataset_dir = param_dict.get('dataset_dir')

        check_dir(dataset_dir)

        validate_dataset_param_value(nreq_param_int, param_dict, int)
        validate_dataset_param_value(nreq_param_bool, param_dict, bool)
        validate_dataset_param_value(nreq_param_list, param_dict, list)
        validate_dataset_param_value(nreq_param_str, param_dict, str)

        usage = param_dict.get('usage')
        if usage is not None and usage not in ('all', 'train', 'valid', 'test'):
            raise ValueError("usage should be 'all', 'train', 'valid' or 'test'.")

        check_sampler_shuffle_shard_options(param_dict)

        sampler = param_dict.get('sampler')
        if sampler is not None and isinstance(sampler, samplers.PKSampler):
            raise ValueError("CelebADataset doesn't support PKSampler.")

        cache = param_dict.get('cache')
        check_cache_option(cache)

        return method(self, *args, **kwargs)

    return new_method


def check_libri_tts_dataset(method):
    """A wrapper that wraps a parameter checker around the original Dataset(LibriTTSDataset)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        _, param_dict = parse_user_args(method, *args, **kwargs)

        nreq_param_int = ['num_samples', 'num_parallel_workers', 'num_shards', 'shard_id']
        nreq_param_bool = ['shuffle']

        dataset_dir = param_dict.get('dataset_dir')
        check_dir(dataset_dir)

        usage = param_dict.get('usage')
        if usage is not None:
            check_valid_str(usage, ["dev-clean", "dev-other", "test-clean", "test-other", "train-clean-100",
                                    "train-clean-360", "train-other-500", "all"], "usage")
        validate_dataset_param_value(nreq_param_int, param_dict, int)
        validate_dataset_param_value(nreq_param_bool, param_dict, bool)

        check_sampler_shuffle_shard_options(param_dict)
        cache = param_dict.get('cache')
        check_cache_option(cache)

        return method(self, *args, **kwargs)

    return new_method


def check_lj_speech_dataset(method):
    """A wrapper that wraps a parameter checker around the original Dataset(LJSpeechDataset)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        _, param_dict = parse_user_args(method, *args, **kwargs)

        nreq_param_int = ['num_samples', 'num_parallel_workers', 'num_shards', 'shard_id']
        nreq_param_bool = ['shuffle']

        dataset_dir = param_dict.get('dataset_dir')
        check_dir(dataset_dir)

        validate_dataset_param_value(nreq_param_int, param_dict, int)
        validate_dataset_param_value(nreq_param_bool, param_dict, bool)

        check_sampler_shuffle_shard_options(param_dict)

        cache = param_dict.get('cache')
        check_cache_option(cache)

        return method(self, *args, **kwargs)

    return new_method


def check_lfw_dataset(method):
    """A wrapper that wraps a parameter checker around the original Dataset(LFWDataset)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        _, param_dict = parse_user_args(method, *args, **kwargs)

        nreq_param_int = ['num_samples', 'num_parallel_workers', 'num_shards', 'shard_id']
        nreq_param_bool = ['shuffle', 'decode']

        dataset_dir = param_dict.get('dataset_dir')
        check_dir(dataset_dir)

        task = param_dict.get('task')
        if task is not None:
            check_valid_str(task, ["people", "pairs"], "task")

        usage = param_dict.get('usage')
        if usage is not None:
            check_valid_str(usage, ["10fold", "train", "test", "all"], "usage")

        image_set = param_dict.get('image_set')
        if image_set is not None:
            check_valid_str(image_set, ["original", "funneled", "deepfunneled"], "image_set")

        validate_dataset_param_value(nreq_param_int, param_dict, int)
        validate_dataset_param_value(nreq_param_bool, param_dict, bool)
        check_sampler_shuffle_shard_options(param_dict)

        cache = param_dict.get('cache')
        check_cache_option(cache)

        return method(self, *args, **kwargs)

    return new_method


def check_save(method):
    """A wrapper that wraps a parameter checker around the saved operator."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        _, param_dict = parse_user_args(method, *args, **kwargs)

        nreq_param_int = ['num_files']
        nreq_param_str = ['file_name', 'file_type']
        validate_dataset_param_value(nreq_param_int, param_dict, int)
        if (param_dict.get('num_files') <= 0 or param_dict.get('num_files') > 1000):
            raise ValueError("num_files should between 0 and 1000.")
        validate_dataset_param_value(nreq_param_str, param_dict, str)
        if param_dict.get('file_type') != 'mindrecord':
            raise ValueError("{} dataset format is not supported.".format(param_dict.get('file_type')))
        return method(self, *args, **kwargs)

    return new_method


def check_tuple_iterator(method):
    """A wrapper that wraps a parameter checker around the original create_tuple_iterator and create_dict_iterator."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [columns, num_epochs, _, _], param_dict = parse_user_args(method, *args, **kwargs)
        nreq_param_bool = ['output_numpy']
        validate_dataset_param_value(nreq_param_bool, param_dict, bool)
        if num_epochs is not None:
            type_check(num_epochs, (int,), "num_epochs")
            check_value(num_epochs, [-1, INT32_MAX], "num_epochs")

        if columns is not None:
            check_columns(columns, "column_names")

        return method(self, *args, **kwargs)

    return new_method


def check_dict_iterator(method):
    """A wrapper that wraps a parameter checker around the original create_tuple_iterator and create_dict_iterator."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [num_epochs, _], param_dict = parse_user_args(method, *args, **kwargs)
        nreq_param_bool = ['output_numpy']
        validate_dataset_param_value(nreq_param_bool, param_dict, bool)
        if num_epochs is not None:
            type_check(num_epochs, (int,), "num_epochs")
            check_value(num_epochs, [-1, INT32_MAX], "num_epochs")

        return method(self, *args, **kwargs)

    return new_method


def check_minddataset(method):
    """A wrapper that wraps a parameter checker around the original Dataset(MindDataset)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        _, param_dict = parse_user_args(method, *args, **kwargs)

        nreq_param_int = ['num_samples', 'num_parallel_workers', 'seed', 'num_shards', 'shard_id', 'num_padded']
        nreq_param_list = ['columns_list']
        nreq_param_dict = ['padded_sample']

        dataset_file = param_dict.get('dataset_files')
        if isinstance(dataset_file, list):
            if len(dataset_file) > 4096:
                raise ValueError("length of dataset_file should be less than or equal to {}.".format(4096))
            for f in dataset_file:
                check_file(f)
        else:
            check_file(dataset_file)

        validate_dataset_param_value(nreq_param_int, param_dict, int)
        validate_dataset_param_value(nreq_param_list, param_dict, list)
        validate_dataset_param_value(nreq_param_dict, param_dict, dict)

        check_sampler_shuffle_shard_options(param_dict)

        check_padding_options(param_dict)
        return method(self, *args, **kwargs)

    return new_method


def check_source_function(source):
    """Get used variable and source document in given function."""
    # check whether source is an instanced object of user defined class
    from types import FunctionType
    var = tuple()
    source_doc = ""
    if isinstance(source, FunctionType):
        try:
            var = ins.getclosurevars(source)
            source_doc = ins.getsource(source)
        except OSError:
            return ""
    else:
        try:
            source_attr = source.__class__.__dict__.keys()
            if '__init__' in source_attr:
                var = var + ins.getclosurevars(source.__class__.__init__)
                source_doc = source_doc + ins.getsource(source.__class__.__init__)
            if '__getitem__' in source_attr:
                var = var + ins.getclosurevars(source.__class__.__getitem__)
                source_doc = source_doc + ins.getsource(source.__class__.__getitem__)
            elif '__next__' in source_attr:
                var = var + ins.getclosurevars(source.__class__.__next__)
                source_doc = source_doc + ins.getsource(source.__class__.__next__)
        except TypeError:
            # case: like input is LambdaType or GeneratorType, it will go to else branch, and unable to run normally
            pass
    return str(var) + source_doc


def check_generatordataset(method):
    """A wrapper that wraps a parameter checker around the original Dataset(GeneratorDataset)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        _, param_dict = parse_user_args(method, *args, **kwargs)

        source = param_dict.get('source')

        if not callable(source):
            try:
                iter(source)
            except TypeError:
                raise TypeError("Input `source` function of GeneratorDataset should be callable, iterable or random"
                                " accessible, commonly it should implement one of the method like yield, __getitem__ or"
                                " __next__(__iter__).")

        # check used variable and function document whether contain computing operator
        check_doc = check_source_function(source)
        check_list = ['mindspore.nn', 'mindspore.ops', 'mindspore.numpy', 'mindspore.compression']
        for item in check_list:
            if item in check_doc:
                setattr(self, 'operator_mixed', True)
                break

        column_names = param_dict.get('column_names')
        if column_names is not None:
            check_columns(column_names, "column_names")
        schema = param_dict.get('schema')
        if column_names is None and schema is None:
            raise ValueError("Neither columns_names nor schema are provided.")

        if schema is not None:
            if not isinstance(schema, (datasets.Schema, str)):
                raise ValueError("schema should be a path to schema file or a schema object.")

        # check optional argument
        nreq_param_int = ["max_rowsize", "num_samples", "num_parallel_workers", "num_shards", "shard_id"]
        validate_dataset_param_value(nreq_param_int, param_dict, int)
        nreq_param_list = ["column_types"]
        validate_dataset_param_value(nreq_param_list, param_dict, list)
        nreq_param_bool = ["shuffle"]
        validate_dataset_param_value(nreq_param_bool, param_dict, bool)

        num_shards = param_dict.get("num_shards")
        shard_id = param_dict.get("shard_id")
        check_dataset_num_shards_shard_id(num_shards, shard_id)

        sampler = param_dict.get("sampler")
        if sampler is not None:
            if isinstance(sampler, samplers.PKSampler):
                raise ValueError("GeneratorDataset doesn't support PKSampler.")
            if not isinstance(sampler, samplers.BuiltinSampler):
                try:
                    iter(sampler)
                except TypeError:
                    raise TypeError("sampler should be either iterable or from mindspore.dataset.samplers.")

        if sampler is not None and not hasattr(source, "__getitem__"):
            raise ValueError("sampler is not supported if source does not have attribute '__getitem__'.")
        if num_shards is not None and not hasattr(source, "__getitem__"):
            raise ValueError("num_shards is not supported if source does not have attribute '__getitem__'.")

        return method(self, *args, **kwargs)

    return new_method


def check_random_dataset(method):
    """A wrapper that wraps a parameter checker around the original Dataset(RandomDataset)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        _, param_dict = parse_user_args(method, *args, **kwargs)

        nreq_param_int = ['num_samples', 'num_parallel_workers', 'num_shards', 'shard_id', 'total_rows']
        nreq_param_bool = ['shuffle']
        nreq_param_list = ['columns_list']

        validate_dataset_param_value(nreq_param_int, param_dict, int)
        validate_dataset_param_value(nreq_param_bool, param_dict, bool)
        validate_dataset_param_value(nreq_param_list, param_dict, list)

        check_sampler_shuffle_shard_options(param_dict)

        cache = param_dict.get('cache')
        check_cache_option(cache)

        return method(self, *args, **kwargs)

    return new_method


def check_pad_info(key, val):
    """check the key and value pair of pad_info in batch"""
    type_check(key, (str,), "key in pad_info")

    if val is not None:
        if len(val) != 2:
            raise ValueError("value of pad_info should be a tuple of size 2.")
        type_check(val, (tuple,), "value in pad_info")

        if val[0] is not None:
            type_check(val[0], (list,), "shape in pad_info")

            for dim in val[0]:
                if dim is not None:
                    check_pos_int32(dim, "dim of shape in pad_info")
        if val[1] is not None:
            type_check(val[1], (int, float, str, bytes), "pad_value")


def check_bucket_batch_by_length(method):
    """check the input arguments of bucket_batch_by_length."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [column_names, bucket_boundaries, bucket_batch_sizes, element_length_function, pad_info,
         pad_to_bucket_boundary, drop_remainder], _ = parse_user_args(method, *args, **kwargs)

        nreq_param_list = ['column_names', 'bucket_boundaries', 'bucket_batch_sizes']

        type_check_list([column_names, bucket_boundaries, bucket_batch_sizes], (list,), nreq_param_list)

        nbool_param_list = ['pad_to_bucket_boundary', 'drop_remainder']
        type_check_list([pad_to_bucket_boundary, drop_remainder], (bool,), nbool_param_list)

        # check column_names: must be list of string.
        check_columns(column_names, "column_names")

        if element_length_function is None and len(column_names) != 1:
            raise ValueError("If element_length_function is not specified, exactly one column name should be passed.")

        if element_length_function is not None and not callable(element_length_function):
            raise TypeError("element_length_function object is not callable.")

        # check bucket_boundaries: must be list of int, positive and strictly increasing
        if not bucket_boundaries:
            raise ValueError("bucket_boundaries cannot be empty.")

        all_int = all(isinstance(item, int) for item in bucket_boundaries)
        if not all_int:
            raise TypeError("bucket_boundaries should be a list of int.")

        all_non_negative = all(item > 0 for item in bucket_boundaries)
        if not all_non_negative:
            raise ValueError("bucket_boundaries must only contain positive numbers.")

        for i in range(len(bucket_boundaries) - 1):
            if not bucket_boundaries[i + 1] > bucket_boundaries[i]:
                raise ValueError("bucket_boundaries should be strictly increasing.")

        # check bucket_batch_sizes: must be list of int and positive
        if len(bucket_batch_sizes) != len(bucket_boundaries) + 1:
            raise ValueError("bucket_batch_sizes must contain one element more than bucket_boundaries.")

        all_int = all(isinstance(item, int) for item in bucket_batch_sizes)
        if not all_int:
            raise TypeError("bucket_batch_sizes should be a list of int.")

        all_non_negative = all(item > 0 for item in bucket_batch_sizes)
        if not all_non_negative:
            raise ValueError("bucket_batch_sizes should be a list of positive numbers.")

        if pad_info is not None:
            type_check(pad_info, (dict,), "pad_info")

            for k, v in pad_info.items():
                check_pad_info(k, v)

        return method(self, *args, **kwargs)

    return new_method


def check_batch(method):
    """check the input arguments of batch."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [batch_size, drop_remainder, num_parallel_workers, per_batch_map,
         input_columns, output_columns, column_order, pad_info,
         python_multiprocessing, max_rowsize], param_dict = parse_user_args(method, *args, **kwargs)

        if not (isinstance(batch_size, int) or (callable(batch_size))):
            raise TypeError("batch_size should either be an int or a callable.")

        if callable(batch_size):
            sig = ins.signature(batch_size)
            if len(sig.parameters) != 1:
                raise ValueError("callable batch_size should take one parameter (BatchInfo).")
        else:
            check_pos_int32(int(batch_size), "batch_size")

        if num_parallel_workers is not None:
            check_num_parallel_workers(num_parallel_workers)
        type_check(drop_remainder, (bool,), "drop_remainder")
        type_check(max_rowsize, (int,), "max_rowsize")

        if (pad_info is not None) and (per_batch_map is not None):
            raise ValueError("pad_info and per_batch_map can't both be set.")

        if pad_info is not None:
            type_check(param_dict["pad_info"], (dict,), "pad_info")
            for k, v in param_dict.get('pad_info').items():
                check_pad_info(k, v)

        if (per_batch_map is None) != (input_columns is None):
            # These two parameters appear together.
            raise ValueError("per_batch_map and input_columns need to be passed in together.")

        if input_columns is not None:
            check_columns(input_columns, "input_columns")
            if len(input_columns) != (len(ins.signature(per_batch_map).parameters) - 1):
                raise ValueError("The signature of per_batch_map should match with input columns.")

        if output_columns is not None:
            check_columns(output_columns, "output_columns")

        if column_order is not None:
            check_columns(column_order, "column_order")

        if python_multiprocessing is not None:
            type_check(python_multiprocessing, (bool,), "python_multiprocessing")

        return method(self, *args, **kwargs)

    return new_method


def check_sync_wait(method):
    """check the input arguments of sync_wait."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [condition_name, num_batch, _], _ = parse_user_args(method, *args, **kwargs)

        type_check(condition_name, (str,), "condition_name")
        type_check(num_batch, (int,), "num_batch")

        return method(self, *args, **kwargs)

    return new_method


def check_shuffle(method):
    """check the input arguments of shuffle."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [buffer_size], _ = parse_user_args(method, *args, **kwargs)

        type_check(buffer_size, (int,), "buffer_size")

        check_value(buffer_size, [2, INT32_MAX], "buffer_size")

        return method(self, *args, **kwargs)

    return new_method


def check_map(method):
    """check the input arguments of map."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        from mindspore.dataset.callback import DSCallback
        [operations, input_columns, output_columns, column_order, num_parallel_workers, python_multiprocessing, cache,
         callbacks, max_rowsize, offload], _ = \
            parse_user_args(method, *args, **kwargs)

        # check whether network computing operator exist in input operations(python function)
        # check used variable and function document whether contain computing operator
        from types import FunctionType
        if isinstance(operations, FunctionType):
            try:
                var = ins.getclosurevars(operations)
                operations_doc = ins.getsource(operations)
                check_list = ['mindspore.nn', 'mindspore.ops', 'mindspore.numpy', 'mindspore.compression']
                check_doc = str(var) + operations_doc
                for item in check_list:
                    if item in check_doc:
                        setattr(self, 'operator_mixed', True)
                        break
            except OSError:
                pass

        nreq_param_columns = ['input_columns', 'output_columns', 'column_order']

        if column_order is not None:
            type_check(column_order, (list,), "column_order")
        if num_parallel_workers is not None:
            check_num_parallel_workers(num_parallel_workers)
        type_check(python_multiprocessing, (bool,), "python_multiprocessing")
        check_cache_option(cache)
        type_check(max_rowsize, (int,), "max_rowsize")
        if offload is not None:
            type_check(offload, (bool,), "offload")

        if callbacks is not None:
            if isinstance(callbacks, (list, tuple)):
                type_check_list(callbacks, (DSCallback,), "callbacks")
            else:
                type_check(callbacks, (DSCallback,), "callbacks")

        for param_name, param in zip(nreq_param_columns, [input_columns, output_columns, column_order]):
            if param is not None:
                check_columns(param, param_name)
        if callbacks is not None:
            type_check(callbacks, (list, DSCallback), "callbacks")

        return method(self, *args, **kwargs)

    return new_method


def check_filter(method):
    """"check the input arguments of filter."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [predicate, input_columns, num_parallel_workers], _ = parse_user_args(method, *args, **kwargs)
        if not callable(predicate):
            raise TypeError("Predicate should be a Python function or a callable Python object.")

        if num_parallel_workers is not None:
            check_num_parallel_workers(num_parallel_workers)

        if input_columns is not None:
            check_columns(input_columns, "input_columns")

        return method(self, *args, **kwargs)

    return new_method


def check_repeat(method):
    """check the input arguments of repeat."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [count], _ = parse_user_args(method, *args, **kwargs)

        type_check(count, (int, type(None)), "repeat")
        if isinstance(count, int):
            if (count <= 0 and count != -1) or count > INT32_MAX:
                raise ValueError("count should be either -1 or positive integer, range[1, INT32_MAX].")
        return method(self, *args, **kwargs)

    return new_method


def check_skip(method):
    """check the input arguments of skip."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [count], _ = parse_user_args(method, *args, **kwargs)

        type_check(count, (int,), "count")
        check_value(count, (0, INT32_MAX), "count")

        return method(self, *args, **kwargs)

    return new_method


def check_take(method):
    """check the input arguments of take."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [count], _ = parse_user_args(method, *args, **kwargs)
        type_check(count, (int,), "count")
        if (count <= 0 and count != -1) or count > INT32_MAX:
            raise ValueError("count should be either -1 or within the required interval of ({}, {}], got {}."
                             .format(0, INT32_MAX, count))

        return method(self, *args, **kwargs)

    return new_method


def check_positive_int32(method):
    """check whether the input argument is positive and int, only works for functions with one input."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [count], param_dict = parse_user_args(method, *args, **kwargs)
        para_name = None
        for key in list(param_dict.keys()):
            if key not in ['self', 'cls']:
                para_name = key
        # Need to get default value of param
        if count is not None:
            check_pos_int32(count, para_name)

        return method(self, *args, **kwargs)

    return new_method


def check_device_send(method):
    """check the input argument for to_device and device_que."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [send_epoch_end, create_data_info_queue], _ = parse_user_args(method, *args, **kwargs)
        type_check(send_epoch_end, (bool,), "send_epoch_end")
        type_check(create_data_info_queue, (bool,), "create_data_info_queue")

        return method(self, *args, **kwargs)

    return new_method


def check_zip(method):
    """check the input arguments of zip."""

    @wraps(method)
    def new_method(*args, **kwargs):
        [ds], _ = parse_user_args(method, *args, **kwargs)
        type_check(ds, (tuple,), "datasets")

        return method(*args, **kwargs)

    return new_method


def check_zip_dataset(method):
    """check the input arguments of zip method in `Dataset`."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [ds], _ = parse_user_args(method, *args, **kwargs)
        type_check(ds, (tuple, datasets.Dataset), "datasets")

        return method(self, *args, **kwargs)

    return new_method


def check_concat(method):
    """check the input arguments of concat method in `Dataset`."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [ds], _ = parse_user_args(method, *args, **kwargs)
        type_check(ds, (list, datasets.Dataset), "datasets")
        if isinstance(ds, list):
            type_check_list(ds, (datasets.Dataset,), "dataset")
        return method(self, *args, **kwargs)

    return new_method


def check_rename(method):
    """check the input arguments of rename."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        values, _ = parse_user_args(method, *args, **kwargs)

        req_param_columns = ['input_columns', 'output_columns']
        for param_name, param in zip(req_param_columns, values):
            check_columns(param, param_name)

        input_size, output_size = 1, 1
        input_columns, output_columns = values
        if isinstance(input_columns, list):
            input_size = len(input_columns)
        if isinstance(output_columns, list):
            output_size = len(output_columns)
        if input_size != output_size:
            raise ValueError("Number of column in input_columns and output_columns is not equal.")

        return method(self, *args, **kwargs)

    return new_method


def check_project(method):
    """check the input arguments of project."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [columns], _ = parse_user_args(method, *args, **kwargs)
        check_columns(columns, 'columns')

        return method(self, *args, **kwargs)

    return new_method


def check_schema(method):
    """check the input arguments of Schema.__init__."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [schema_file], _ = parse_user_args(method, *args, **kwargs)

        if schema_file is not None:
            check_file(schema_file)

        return method(self, *args, **kwargs)

    return new_method


def check_add_column(method):
    """check the input arguments of add_column."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [name, de_type, shape], _ = parse_user_args(method, *args, **kwargs)

        type_check(name, (str,), "name")

        if not name:
            raise TypeError("Expected non-empty string for column name.")

        if de_type is not None:
            if not isinstance(de_type, typing.Type) and not check_valid_detype(de_type):
                raise TypeError("Unknown column type: {}.".format(de_type))
        else:
            raise TypeError("Expected non-empty string for de_type.")

        if shape is not None:
            type_check(shape, (list,), "shape")
            type_check_list(shape, (int,), "shape")

        return method(self, *args, **kwargs)

    return new_method


def check_cluedataset(method):
    """A wrapper that wraps a parameter checker around the original Dataset(CLUEDataset)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        _, param_dict = parse_user_args(method, *args, **kwargs)

        nreq_param_int = ['num_samples', 'num_parallel_workers', 'num_shards', 'shard_id']

        dataset_files = param_dict.get('dataset_files')
        type_check(dataset_files, (str, list), "dataset files")

        # check task
        task_param = param_dict.get('task')
        if task_param not in ['AFQMC', 'TNEWS', 'IFLYTEK', 'CMNLI', 'WSC', 'CSL']:
            raise ValueError("task should be 'AFQMC', 'TNEWS', 'IFLYTEK', 'CMNLI', 'WSC' or 'CSL'.")

        # check usage
        usage_param = param_dict.get('usage')
        if usage_param not in ['train', 'test', 'eval']:
            raise ValueError("usage should be 'train', 'test' or 'eval'.")

        validate_dataset_param_value(nreq_param_int, param_dict, int)
        check_sampler_shuffle_shard_options(param_dict)

        cache = param_dict.get('cache')
        check_cache_option(cache)

        return method(self, *args, **kwargs)

    return new_method


def check_csvdataset(method):
    """A wrapper that wraps a parameter checker around the original Dataset(CSVDataset)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        _, param_dict = parse_user_args(method, *args, **kwargs)

        nreq_param_int = ['num_samples', 'num_parallel_workers', 'num_shards', 'shard_id']

        # check dataset_files; required argument
        dataset_files = param_dict.get('dataset_files')
        type_check(dataset_files, (str, list), "dataset files")

        # check field_delim
        field_delim = param_dict.get('field_delim')
        if field_delim is not None:
            type_check(field_delim, (str,), 'field delim')
            if field_delim in ['"', '\r', '\n'] or len(field_delim) > 1:
                raise ValueError("field_delim is invalid.")

        # check column_defaults
        column_defaults = param_dict.get('column_defaults')
        if column_defaults is not None:
            if not isinstance(column_defaults, list):
                raise TypeError("column_defaults should be type of list.")
            for item in column_defaults:
                if not isinstance(item, (str, int, float)):
                    raise TypeError("column type in column_defaults is invalid.")

        # check column_names: must be list of string.
        column_names = param_dict.get("column_names")
        if column_names is not None:
            all_string = all(isinstance(item, str) for item in column_names)
            if not all_string:
                raise TypeError("column_names should be a list of str.")

        validate_dataset_param_value(nreq_param_int, param_dict, int)
        check_sampler_shuffle_shard_options(param_dict)

        cache = param_dict.get('cache')
        check_cache_option(cache)

        return method(self, *args, **kwargs)

    return new_method


def check_flowers102dataset(method):
    """A wrapper that wraps a parameter checker around the original Dataset(Flowers102Dataset)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        _, param_dict = parse_user_args(method, *args, **kwargs)

        nreq_param_int = ['num_samples', 'num_parallel_workers', 'num_shards', 'shard_id']
        nreq_param_bool = ['shuffle', 'decode']

        dataset_dir = param_dict.get('dataset_dir')
        check_dir(dataset_dir)

        check_dir(os.path.join(dataset_dir, "jpg"))

        check_file(os.path.join(dataset_dir, "imagelabels.mat"))
        check_file(os.path.join(dataset_dir, "setid.mat"))

        usage = param_dict.get('usage')
        if usage is not None:
            check_valid_str(usage, ["train", "valid", "test", "all"], "usage")

        task = param_dict.get('task')
        if task is not None:
            check_valid_str(task, ["Classification", "Segmentation"], "task")
        if task == "Segmentation":
            check_dir(os.path.join(dataset_dir, "segmim"))

        validate_dataset_param_value(nreq_param_int, param_dict, int)
        validate_dataset_param_value(nreq_param_bool, param_dict, bool)

        check_sampler_shuffle_shard_options(param_dict)

        return method(self, *args, **kwargs)

    return new_method


def check_textfiledataset(method):
    """A wrapper that wraps a parameter checker around the original Dataset(TextFileDataset)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        _, param_dict = parse_user_args(method, *args, **kwargs)

        nreq_param_int = ['num_samples', 'num_parallel_workers', 'num_shards', 'shard_id']

        dataset_files = param_dict.get('dataset_files')
        type_check(dataset_files, (str, list), "dataset files")
        validate_dataset_param_value(nreq_param_int, param_dict, int)
        check_sampler_shuffle_shard_options(param_dict)

        cache = param_dict.get('cache')
        check_cache_option(cache)

        return method(self, *args, **kwargs)

    return new_method


def check_penn_treebank_dataset(method):
    """A wrapper that wraps a parameter checker around the original Dataset(PennTreebankDataset)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        _, param_dict = parse_user_args(method, *args, **kwargs)

        nreq_param_int = ['num_samples', 'num_parallel_workers', 'num_shards', 'shard_id']

        # check dataset_dir; required argument
        dataset_dir = param_dict.get('dataset_dir')
        check_dir(dataset_dir)

        # check usage
        usage = param_dict.get('usage')
        if usage is not None:
            check_valid_str(usage, ["train", "valid", "test", "all"], "usage")

        validate_dataset_param_value(nreq_param_int, param_dict, int)
        check_sampler_shuffle_shard_options(param_dict)

        cache = param_dict.get('cache')
        check_cache_option(cache)

        return method(self, *args, **kwargs)

    return new_method


def check_split(method):
    """check the input arguments of split."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [sizes, randomize], _ = parse_user_args(method, *args, **kwargs)

        type_check(sizes, (list,), "sizes")
        type_check(randomize, (bool,), "randomize")

        # check sizes: must be list of float or list of int
        if not sizes:
            raise ValueError("sizes cannot be empty.")

        all_int = all(isinstance(item, int) for item in sizes)
        all_float = all(isinstance(item, float) for item in sizes)

        if not (all_int or all_float):
            raise ValueError("sizes should be list of int or list of float.")

        if all_int:
            all_positive = all(item > 0 for item in sizes)
            if not all_positive:
                raise ValueError("sizes is a list of int, but there should be no negative or zero numbers.")

        if all_float:
            all_valid_percentages = all(0 < item <= 1 for item in sizes)
            if not all_valid_percentages:
                raise ValueError("sizes is a list of float, but there should be no numbers outside the range (0, 1].")

            epsilon = 0.00001
            if not abs(sum(sizes) - 1) < epsilon:
                raise ValueError("sizes is a list of float, but the percentages do not sum up to 1.")

        return method(self, *args, **kwargs)

    return new_method


def check_hostname(hostname):
    if not hostname or len(hostname) > 255:
        return False
    if hostname[-1] == ".":
        hostname = hostname[:-1]  # strip exactly one dot from the right, if present
    allowed = re.compile("(?!-)[A-Z\\d-]{1,63}(?<!-)$", re.IGNORECASE)
    return all(allowed.match(x) for x in hostname.split("."))


def check_gnn_graphdata(method):
    """check the input arguments of graphdata."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [dataset_file, num_parallel_workers, working_mode, hostname,
         port, num_client, auto_shutdown], _ = parse_user_args(method, *args, **kwargs)
        check_file(dataset_file)
        if num_parallel_workers is not None:
            check_num_parallel_workers(num_parallel_workers)
        type_check(hostname, (str,), "hostname")
        if check_hostname(hostname) is False:
            raise ValueError("The hostname is illegal")
        type_check(working_mode, (str,), "working_mode")
        if working_mode not in {'local', 'client', 'server'}:
            raise ValueError("Invalid working mode, please enter 'local', 'client' or 'server'.")
        type_check(port, (int,), "port")
        check_value(port, (1024, 65535), "port")
        type_check(num_client, (int,), "num_client")
        check_value(num_client, (1, 255), "num_client")
        type_check(auto_shutdown, (bool,), "auto_shutdown")
        return method(self, *args, **kwargs)

    return new_method


def check_gnn_get_all_nodes(method):
    """A wrapper that wraps a parameter checker around the GNN `get_all_nodes` function."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [node_type], _ = parse_user_args(method, *args, **kwargs)
        type_check(node_type, (int,), "node_type")

        return method(self, *args, **kwargs)

    return new_method


def check_gnn_get_all_edges(method):
    """A wrapper that wraps a parameter checker around the GNN `get_all_edges` function."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [edge_type], _ = parse_user_args(method, *args, **kwargs)
        type_check(edge_type, (int,), "edge_type")

        return method(self, *args, **kwargs)

    return new_method


def check_gnn_get_nodes_from_edges(method):
    """A wrapper that wraps a parameter checker around the GNN `get_nodes_from_edges` function."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [edge_list], _ = parse_user_args(method, *args, **kwargs)
        check_gnn_list_or_ndarray(edge_list, "edge_list")

        return method(self, *args, **kwargs)

    return new_method


def check_gnn_get_edges_from_nodes(method):
    """A wrapper that wraps a parameter checker around the GNN `get_edges_from_nodes` function."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [node_list], _ = parse_user_args(method, *args, **kwargs)
        check_gnn_list_of_pair_or_ndarray(node_list, "node_list")

        return method(self, *args, **kwargs)

    return new_method


def check_gnn_get_all_neighbors(method):
    """A wrapper that wraps a parameter checker around the GNN `get_all_neighbors` function."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [node_list, neighbour_type, _], _ = parse_user_args(method, *args, **kwargs)

        check_gnn_list_or_ndarray(node_list, 'node_list')
        type_check(neighbour_type, (int,), "neighbour_type")

        return method(self, *args, **kwargs)

    return new_method


def check_gnn_get_sampled_neighbors(method):
    """A wrapper that wraps a parameter checker around the GNN `get_sampled_neighbors` function."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [node_list, neighbor_nums, neighbor_types, _], _ = parse_user_args(method, *args, **kwargs)

        check_gnn_list_or_ndarray(node_list, 'node_list')

        check_gnn_list_or_ndarray(neighbor_nums, 'neighbor_nums')
        if not neighbor_nums or len(neighbor_nums) > 6:
            raise ValueError("Wrong number of input members for {0}, should be between 1 and 6, got {1}.".format(
                'neighbor_nums', len(neighbor_nums)))

        check_gnn_list_or_ndarray(neighbor_types, 'neighbor_types')
        if not neighbor_types or len(neighbor_types) > 6:
            raise ValueError("Wrong number of input members for {0}, should be between 1 and 6, got {1}.".format(
                'neighbor_types', len(neighbor_types)))

        if len(neighbor_nums) != len(neighbor_types):
            raise ValueError(
                "The number of members of neighbor_nums and neighbor_types is inconsistent.")

        return method(self, *args, **kwargs)

    return new_method


def check_gnn_get_neg_sampled_neighbors(method):
    """A wrapper that wraps a parameter checker around the GNN `get_neg_sampled_neighbors` function."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [node_list, neg_neighbor_num, neg_neighbor_type], _ = parse_user_args(method, *args, **kwargs)

        check_gnn_list_or_ndarray(node_list, 'node_list')
        type_check(neg_neighbor_num, (int,), "neg_neighbor_num")
        type_check(neg_neighbor_type, (int,), "neg_neighbor_type")

        return method(self, *args, **kwargs)

    return new_method


def check_gnn_random_walk(method):
    """A wrapper that wraps a parameter checker around the GNN `random_walk` function."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [target_nodes, meta_path, step_home_param, step_away_param, default_node], _ = parse_user_args(method, *args,
                                                                                                       **kwargs)
        check_gnn_list_or_ndarray(target_nodes, 'target_nodes')
        check_gnn_list_or_ndarray(meta_path, 'meta_path')
        type_check(step_home_param, (float,), "step_home_param")
        type_check(step_away_param, (float,), "step_away_param")
        type_check(default_node, (int,), "default_node")
        check_value(default_node, (-1, INT32_MAX), "default_node")

        return method(self, *args, **kwargs)

    return new_method


def check_aligned_list(param, param_name, member_type):
    """Check whether the structure of each member of the list is the same."""

    type_check(param, (list,), "param")
    if not param:
        raise TypeError(
            "Parameter {0} or its members are empty".format(param_name))
    member_have_list = None
    list_len = None
    for member in param:
        if isinstance(member, list):
            check_aligned_list(member, param_name, member_type)

            if member_have_list not in (None, True):
                raise TypeError("The type of each member of the parameter {0} is inconsistent.".format(
                    param_name))
            if list_len is not None and len(member) != list_len:
                raise TypeError("The size of each member of parameter {0} is inconsistent.".format(
                    param_name))
            member_have_list = True
            list_len = len(member)
        else:
            type_check(member, (member_type,), param_name)
            if member_have_list not in (None, False):
                raise TypeError("The type of each member of the parameter {0} is inconsistent.".format(
                    param_name))
            member_have_list = False


def check_gnn_get_node_feature(method):
    """A wrapper that wraps a parameter checker around the GNN `get_node_feature` function."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [node_list, feature_types], _ = parse_user_args(method, *args, **kwargs)

        type_check(node_list, (list, np.ndarray), "node_list")
        if isinstance(node_list, list):
            check_aligned_list(node_list, 'node_list', int)
        elif isinstance(node_list, np.ndarray):
            if not node_list.dtype == np.int32:
                raise TypeError("Each member in {0} should be of type int32. Got {1}.".format(
                    node_list, node_list.dtype))

        check_gnn_list_or_ndarray(feature_types, 'feature_types')

        return method(self, *args, **kwargs)

    return new_method


def check_gnn_get_edge_feature(method):
    """A wrapper that wraps a parameter checker around the GNN `get_edge_feature` function."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [edge_list, feature_types], _ = parse_user_args(method, *args, **kwargs)

        type_check(edge_list, (list, np.ndarray), "edge_list")
        if isinstance(edge_list, list):
            check_aligned_list(edge_list, 'edge_list', int)
        elif isinstance(edge_list, np.ndarray):
            if not edge_list.dtype == np.int32:
                raise TypeError("Each member in {0} should be of type int32. Got {1}.".format(
                    edge_list, edge_list.dtype))

        check_gnn_list_or_ndarray(feature_types, 'feature_types')

        return method(self, *args, **kwargs)

    return new_method


def check_numpyslicesdataset(method):
    """A wrapper that wraps a parameter checker around the original Dataset(NumpySlicesDataset)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        _, param_dict = parse_user_args(method, *args, **kwargs)

        data = param_dict.get("data")
        column_names = param_dict.get("column_names")
        type_check(data, (list, tuple, dict, np.ndarray), "data")
        if data is None or len(data) == 0:  # pylint: disable=len-as-condition
            raise ValueError("Argument data cannot be empty")
        if isinstance(data, tuple):
            type_check(data[0], (list, np.ndarray), "data[0]")

        # check column_names
        if column_names is not None:
            check_columns(column_names, "column_names")

            # check num of input column in column_names
            column_num = 1 if isinstance(column_names, str) else len(column_names)
            if isinstance(data, dict):
                data_column = len(list(data.keys()))
                if column_num != data_column:
                    raise ValueError("Num of input column names is {0}, but required is {1}."
                                     .format(column_num, data_column))

            elif isinstance(data, tuple):
                if column_num != len(data):
                    raise ValueError("Num of input column names is {0}, but required is {1}."
                                     .format(column_num, len(data)))
            else:
                if column_num != 1:
                    raise ValueError("Num of input column names is {0}, but required is {1} as data is list."
                                     .format(column_num, 1))

        return method(self, *args, **kwargs)

    return new_method


def check_paddeddataset(method):
    """A wrapper that wraps a parameter checker around the original Dataset(PaddedDataset)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        _, param_dict = parse_user_args(method, *args, **kwargs)

        padded_samples = param_dict.get("padded_samples")
        if not padded_samples:
            raise ValueError("padded_samples cannot be empty.")
        type_check(padded_samples, (list,), "padded_samples")
        type_check(padded_samples[0], (dict,), "padded_element")
        return method(self, *args, **kwargs)

    return new_method


def check_cache_option(cache):
    """Sanity check for cache parameter"""
    if cache is not None:
        type_check(cache, (cache_client.DatasetCache,), "cache")


def check_to_device_send(method):
    """Check the input arguments of send function for TransferDataset."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        [num_epochs], _ = parse_user_args(method, *args, **kwargs)

        if num_epochs is not None:
            type_check(num_epochs, (int,), "num_epochs")
            check_value(num_epochs, [-1, INT32_MAX], "num_epochs")

        return method(self, *args, **kwargs)

    return new_method


def check_emnist_dataset(method):
    """A wrapper that wraps a parameter checker emnist dataset"""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        _, param_dict = parse_user_args(method, *args, **kwargs)

        nreq_param_int = ['num_samples', 'num_parallel_workers', 'num_shards', 'shard_id']
        nreq_param_bool = ['shuffle']

        validate_dataset_param_value(nreq_param_int, param_dict, int)
        validate_dataset_param_value(nreq_param_bool, param_dict, bool)

        dataset_dir = param_dict.get('dataset_dir')
        check_dir(dataset_dir)

        name = param_dict.get('name')
        check_valid_str(name, ["byclass", "bymerge", "balanced", "letters", "digits", "mnist"], "name")

        usage = param_dict.get('usage')
        if usage is not None:
            check_valid_str(usage, ["train", "test", "all"], "usage")

        check_sampler_shuffle_shard_options(param_dict)

        cache = param_dict.get('cache')
        check_cache_option(cache)

        return method(self, *args, **kwargs)

    return new_method


def check_flickr_dataset(method):
    """A wrapper that wraps a parameter checker around the original Dataset(Flickr8k, Flickr30k)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        _, param_dict = parse_user_args(method, *args, **kwargs)

        nreq_param_int = ['num_samples', 'num_parallel_workers', 'num_shards', 'shard_id']
        nreq_param_bool = ['shuffle', 'decode']

        dataset_dir = param_dict.get('dataset_dir')
        annotation_file = param_dict.get('annotation_file')
        check_dir(dataset_dir)
        check_file(annotation_file)

        validate_dataset_param_value(nreq_param_int, param_dict, int)
        validate_dataset_param_value(nreq_param_bool, param_dict, bool)

        check_sampler_shuffle_shard_options(param_dict)

        cache = param_dict.get('cache')
        check_cache_option(cache)

        return method(self, *args, **kwargs)

    return new_method


def check_sb_dataset(method):
    """A wrapper that wraps a parameter checker around the original Semantic Boundaries Dataset."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        _, param_dict = parse_user_args(method, *args, **kwargs)

        nreq_param_int = ['num_samples', 'num_parallel_workers', 'num_shards', 'shard_id']
        nreq_param_bool = ['shuffle', 'decode']

        dataset_dir = param_dict.get('dataset_dir')
        check_dir(dataset_dir)

        usage = param_dict.get('usage')
        if usage is not None:
            check_valid_str(usage, ["train", "val", "train_noval", "all"], "usage")

        task = param_dict.get('task')
        if task is not None:
            check_valid_str(task, ["Boundaries", "Segmentation"], "task")

        validate_dataset_param_value(nreq_param_int, param_dict, int)
        validate_dataset_param_value(nreq_param_bool, param_dict, bool)

        check_sampler_shuffle_shard_options(param_dict)

        return method(self, *args, **kwargs)

    return new_method


def check_speech_commands_dataset(method):
    """A wrapper that wraps a parameter checker around the original Dataset(SpeechCommandsDataset)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        _, param_dict = parse_user_args(method, *args, **kwargs)

        nreq_param_int = ['num_samples', 'num_parallel_workers', 'num_shards', 'shard_id']
        nreq_param_bool = ['shuffle']

        dataset_dir = param_dict.get('dataset_dir')
        check_dir(dataset_dir)

        usage = param_dict.get('usage')
        if usage is not None:
            check_valid_str(usage, ["train", "test", "valid", "all"], "usage")

        validate_dataset_param_value(nreq_param_int, param_dict, int)
        validate_dataset_param_value(nreq_param_bool, param_dict, bool)

        check_sampler_shuffle_shard_options(param_dict)

        cache = param_dict.get('cache')
        check_cache_option(cache)

        return method(self, *args, **kwargs)

    return new_method


def check_squad_dataset(method):
    """A wrapper that wraps a parameter checker around the original Dataset(SQuADDataset)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        _, param_dict = parse_user_args(method, *args, **kwargs)

        nreq_param_int = ['num_samples', 'num_parallel_workers', 'num_shards', 'shard_id']

        dataset_dir = param_dict.get('dataset_dir')
        check_dir(dataset_dir)

        # check usage
        usage = param_dict.get('usage')
        if usage is not None:
            check_valid_str(usage, ['train', 'dev', 'all'], "usage")

        validate_dataset_param_value(nreq_param_int, param_dict, int)
        check_sampler_shuffle_shard_options(param_dict)

        cache = param_dict.get('cache')
        check_cache_option(cache)

        return method(self, *args, **kwargs)

    return new_method


def check_cityscapes_dataset(method):
    """A wrapper that wraps a parameter checker around the original CityScapesDataset."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        _, param_dict = parse_user_args(method, *args, **kwargs)

        nreq_param_int = ['num_samples', 'num_parallel_workers', 'num_shards', 'shard_id']
        nreq_param_bool = ['shuffle', 'decode']

        dataset_dir = param_dict.get('dataset_dir')
        check_dir(dataset_dir)

        task = param_dict.get('task')
        check_valid_str(task, ["instance", "semantic", "polygon", "color"], "task")

        quality_mode = param_dict.get('quality_mode')
        check_valid_str(quality_mode, ["fine", "coarse"], "quality_mode")

        usage = param_dict.get('usage')
        if quality_mode == "fine":
            valid_strings = ["train", "test", "val", "all"]
        else:
            valid_strings = ["train", "train_extra", "val", "all"]
        check_valid_str(usage, valid_strings, "usage")

        validate_dataset_param_value(nreq_param_int, param_dict, int)
        validate_dataset_param_value(nreq_param_bool, param_dict, bool)

        check_sampler_shuffle_shard_options(param_dict)

        return method(self, *args, **kwargs)

    return new_method


def check_div2k_dataset(method):
    """A wrapper that wraps a parameter checker around the original DIV2KDataset."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        _, param_dict = parse_user_args(method, *args, **kwargs)

        nreq_param_int = ['num_samples', 'num_parallel_workers', 'num_shards', 'shard_id']
        nreq_param_bool = ['shuffle', 'decode']

        dataset_dir = param_dict.get('dataset_dir')
        check_dir(dataset_dir)

        usage = param_dict.get('usage')
        check_valid_str(usage, ['train', 'valid', 'all'], "usage")

        downgrade = param_dict.get('downgrade')
        check_valid_str(downgrade, ['bicubic', 'unknown', 'mild', 'difficult', 'wild'], 'downgrade')

        validate_dataset_param_value(['scale'], param_dict, int)
        scale = param_dict.get('scale')
        scale_values = [2, 3, 4, 8]
        if scale not in scale_values:
            raise ValueError("Input scale is not within the valid set of {0}.".format(str(scale_values)))

        if scale == 8 and downgrade != "bicubic":
            raise ValueError("DIV2KNode: scale equal to 8 is allowed only in bicubic downgrade.")

        downgrade_2018 = ["mild", "difficult", "wild"]
        if downgrade in downgrade_2018 and scale != 4:
            raise ValueError("DIV2KNode: {0} downgrade requires scale equal to 4.".format(downgrade))

        validate_dataset_param_value(nreq_param_int, param_dict, int)
        validate_dataset_param_value(nreq_param_bool, param_dict, bool)

        check_sampler_shuffle_shard_options(param_dict)

        return method(self, *args, **kwargs)

    return new_method


def check_fake_image_dataset(method):
    """A wrapper that wraps a parameter checker around the original Dataset(FakeImageDataset)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        _, param_dict = parse_user_args(method, *args, **kwargs)

        nreq_param_int = ['num_images', 'num_classes', 'base_seed', 'num_samples',
                          'num_parallel_workers', 'num_shards', 'shard_id']
        nreq_param_bool = ['shuffle']

        validate_dataset_param_value(nreq_param_int, param_dict, int)
        validate_dataset_param_value(nreq_param_bool, param_dict, bool)

        num_images = param_dict.get("num_images")
        check_pos_int32(num_images, "num_images")

        image_size = param_dict.get("image_size")
        type_check(image_size, (list, tuple), "image_size")
        if len(image_size) != 3:
            raise ValueError("image_size should be a list or tuple of length 3, but got {0}".format(len(image_size)))
        for i, value in enumerate(image_size):
            check_pos_int32(value, "image_size[{0}]".format(i))

        num_classes = param_dict.get("num_classes")
        check_pos_int32(num_classes, "num_classes")

        check_sampler_shuffle_shard_options(param_dict)

        cache = param_dict.get('cache')
        check_cache_option(cache)

        return method(self, *args, **kwargs)

    return new_method


def check_ag_news_dataset(method):
    """A wrapper that wraps a parameter checker around the original Dataset(AGNewsDataset)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        _, param_dict = parse_user_args(method, *args, **kwargs)

        nreq_param_int = ['num_samples', 'num_parallel_workers', 'num_shards', 'shard_id']

        # check dataset_files; required argument
        dataset_dir = param_dict.get('dataset_dir')
        check_dir(dataset_dir)

        # check usage
        usage = param_dict.get('usage')
        if usage is not None:
            check_valid_str(usage, ["train", "test", "all"], "usage")

        validate_dataset_param_value(nreq_param_int, param_dict, int)
        check_sampler_shuffle_shard_options(param_dict)

        cache = param_dict.get('cache')
        check_cache_option(cache)

        return method(self, *args, **kwargs)

    return new_method


def check_dbpedia_dataset(method):
    """A wrapper that wraps a parameter checker around the original DBpediaDataset."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        _, param_dict = parse_user_args(method, *args, **kwargs)

        nreq_param_int = ['num_samples', 'num_parallel_workers', 'num_shards', 'shard_id']

        dataset_dir = param_dict.get('dataset_dir')
        check_dir(dataset_dir)

        usage = param_dict.get('usage')
        if usage is not None:
            check_valid_str(usage, ["train", "test", "all"], "usage")

        validate_dataset_param_value(nreq_param_int, param_dict, int)

        check_sampler_shuffle_shard_options(param_dict)

        cache = param_dict.get('cache')
        check_cache_option(cache)

        return method(self, *args, **kwargs)

    return new_method


def check_wider_face_dataset(method):
    """A wrapper that wraps a parameter checker around the WIDERFaceDataset."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        _, param_dict = parse_user_args(method, *args, **kwargs)

        nreq_param_int = ['num_samples', 'num_parallel_workers', 'num_shards', 'shard_id']
        nreq_param_bool = ['decode', 'shuffle']

        dataset_dir = param_dict.get('dataset_dir')
        check_dir(dataset_dir)

        usage = param_dict.get('usage')
        if usage is not None:
            check_valid_str(usage, ["train", "test", "valid", "all"], "usage")

        validate_dataset_param_value(nreq_param_int, param_dict, int)
        validate_dataset_param_value(nreq_param_bool, param_dict, bool)

        check_sampler_shuffle_shard_options(param_dict)

        cache = param_dict.get('cache')
        check_cache_option(cache)

        return method(self, *args, **kwargs)

    return new_method


def check_yelp_review_dataset(method):
    """A wrapper that wraps a parameter checker around the original Dataset(YelpReviewDataset)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        _, param_dict = parse_user_args(method, *args, **kwargs)

        nreq_param_int = ['num_samples', 'num_parallel_workers', 'num_shards', 'shard_id']

        dataset_dir = param_dict.get('dataset_dir')
        check_dir(dataset_dir)

        # check usage
        usage = param_dict.get('usage')
        if usage is not None:
            check_valid_str(usage, ["train", "test", "all"], "usage")

        validate_dataset_param_value(nreq_param_int, param_dict, int)
        check_sampler_shuffle_shard_options(param_dict)

        cache = param_dict.get('cache')
        check_cache_option(cache)

        return method(self, *args, **kwargs)

    return new_method


def check_yes_no_dataset(method):
    """A wrapper that wraps a parameter checker around the original Dataset(YesNoDataset)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        _, param_dict = parse_user_args(method, *args, **kwargs)

        nreq_param_int = ['num_samples', 'num_parallel_workers', 'num_shards', 'shard_id']
        nreq_param_bool = ['shuffle']

        dataset_dir = param_dict.get('dataset_dir')
        check_dir(dataset_dir)

        validate_dataset_param_value(nreq_param_int, param_dict, int)
        validate_dataset_param_value(nreq_param_bool, param_dict, bool)

        check_sampler_shuffle_shard_options(param_dict)

        cache = param_dict.get('cache')
        check_cache_option(cache)

        return method(self, *args, **kwargs)

    return new_method


def check_tedlium_dataset(method):
    """Wrapper method to check the parameters of TedliumDataset."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        _, param_dict = parse_user_args(method, *args, **kwargs)

        nreq_param_int = ['num_samples', 'num_parallel_workers', 'num_shards', 'shard_id']
        nreq_param_bool = ['shuffle']

        release = param_dict.get('release')
        check_valid_str(release, ["release1", "release2", "release3"], "release")

        dataset_dir = param_dict.get('dataset_dir')
        check_dir(dataset_dir)

        usage = param_dict.get('usage')
        if usage is not None:
            if release in ["release1", "release2"]:
                check_valid_str(usage, ["train", "test", "dev", "all"], "usage")
            else:
                check_valid_str(usage, ["all"], "usage")

        validate_dataset_param_value(nreq_param_int, param_dict, int)
        validate_dataset_param_value(nreq_param_bool, param_dict, bool)

        check_sampler_shuffle_shard_options(param_dict)

        cache = param_dict.get('cache')
        check_cache_option(cache)

        return method(self, *args, **kwargs)

    return new_method


def check_svhn_dataset(method):
    """A wrapper that wraps a parameter checker around the original Dataset(SVHNDataset)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        _, param_dict = parse_user_args(method, *args, **kwargs)
        dataset_dir = param_dict.get('dataset_dir')
        check_dir(dataset_dir)

        nreq_param_int = ['num_samples', 'num_parallel_workers', 'num_shards', 'shard_id']
        nreq_param_bool = ['shuffle']

        usage = param_dict.get('usage')
        if usage is not None:
            check_valid_str(usage, ["train", "test", "extra", "all"], "usage")
            if usage == "all":
                for _usage in ["train", "test", "extra"]:
                    check_file(os.path.join(dataset_dir, _usage + "_32x32.mat"))
            else:
                check_file(os.path.join(dataset_dir, usage + "_32x32.mat"))

        validate_dataset_param_value(nreq_param_int, param_dict, int)
        validate_dataset_param_value(nreq_param_bool, param_dict, bool)

        check_sampler_shuffle_shard_options(param_dict)

        return method(self, *args, **kwargs)

    return new_method


def check_stl10_dataset(method):
    """A wrapper that wraps a parameter checker around the original Dataset(STL10Dataset)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        _, param_dict = parse_user_args(method, *args, **kwargs)

        nreq_param_int = ['num_samples', 'num_parallel_workers', 'num_shards', 'shard_id']
        nreq_param_bool = ['shuffle']

        dataset_dir = param_dict.get('dataset_dir')
        check_dir(dataset_dir)

        usage = param_dict.get('usage')
        if usage is not None:
            check_valid_str(usage, ["train", "test", "unlabeled", "train+unlabeled", "all"], "usage")
            if usage == "all":
                for _usage in ["train", "test", "unlabeled"]:
                    check_file(os.path.join(dataset_dir, _usage + "_X.bin"))
                    if _usage == "unlabeled":
                        continue
                    else:
                        check_file(os.path.join(dataset_dir, _usage + "_y.bin"))
            elif usage == "train+unlabeled":
                check_file(os.path.join(dataset_dir, "train_X.bin"))
                check_file(os.path.join(dataset_dir, "train_y.bin"))
                check_file(os.path.join(dataset_dir, "unlabeled_X.bin"))
            elif usage == "unlabeled":
                check_file(os.path.join(dataset_dir, "unlabeled_X.bin"))
            else:
                check_file(os.path.join(dataset_dir, usage + "_X.bin"))
                check_file(os.path.join(dataset_dir, usage + "_y.bin"))

        validate_dataset_param_value(nreq_param_int, param_dict, int)
        validate_dataset_param_value(nreq_param_bool, param_dict, bool)

        check_sampler_shuffle_shard_options(param_dict)

        cache = param_dict.get('cache')
        check_cache_option(cache)

        return method(self, *args, **kwargs)

    return new_method


def check_yahoo_answers_dataset(method):
    """A wrapper that wraps a parameter checker around the original YahooAnswers Dataset."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        _, param_dict = parse_user_args(method, *args, **kwargs)

        nreq_param_int = ['num_samples', 'num_parallel_workers', 'num_shards', 'shard_id']

        dataset_dir = param_dict.get('dataset_dir')
        check_dir(dataset_dir)

        usage = param_dict.get('usage')
        if usage is not None:
            check_valid_str(usage, ["train", "test", "all"], "usage")

        validate_dataset_param_value(nreq_param_int, param_dict, int)

        check_sampler_shuffle_shard_options(param_dict)

        cache = param_dict.get('cache')
        check_cache_option(cache)

        return method(self, *args, **kwargs)

    return new_method


def check_conll2000_dataset(method):
    """ A wrapper that wraps a parameter checker around the original Dataset(CoNLL2000Dataset)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        _, param_dict = parse_user_args(method, *args, **kwargs)

        nreq_param_int = ['num_samples', 'num_parallel_workers', 'num_shards', 'shard_id']

        # check dataset_dir
        dataset_dir = param_dict.get('dataset_dir')
        check_dir(dataset_dir)

        # check usage
        usage = param_dict.get('usage')
        if usage is not None:
            check_valid_str(usage, ["train", "test", "all"], "usage")

        validate_dataset_param_value(nreq_param_int, param_dict, int)
        check_sampler_shuffle_shard_options(param_dict)

        cache = param_dict.get('cache')
        check_cache_option(cache)

        return method(self, *args, **kwargs)

    return new_method


def check_amazon_review_dataset(method):
    """A wrapper that wraps a parameter checker around the original Dataset(AmazonReviewDataset)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        _, param_dict = parse_user_args(method, *args, **kwargs)

        nreq_param_int = ['num_samples', 'num_parallel_workers', 'num_shards', 'shard_id']

        # check dataset_files
        dataset_dir = param_dict.get('dataset_dir')
        check_dir(dataset_dir)

        # check usage
        usage = param_dict.get('usage')
        if usage is not None:
            check_valid_str(usage, ["train", "test", "all"], "usage")

        validate_dataset_param_value(nreq_param_int, param_dict, int)
        check_sampler_shuffle_shard_options(param_dict)

        cache = param_dict.get('cache')
        check_cache_option(cache)

        return method(self, *args, **kwargs)

    return new_method


def check_semeion_dataset(method):
    """Wrapper method to check the parameters of SemeionDataset."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        _, param_dict = parse_user_args(method, *args, **kwargs)

        nreq_param_int = ['num_samples', 'num_parallel_workers', 'num_shards', 'shard_id']
        nreq_param_bool = ['shuffle']

        dataset_dir = param_dict.get('dataset_dir')
        check_dir(dataset_dir)

        validate_dataset_param_value(nreq_param_int, param_dict, int)
        validate_dataset_param_value(nreq_param_bool, param_dict, bool)

        check_sampler_shuffle_shard_options(param_dict)

        cache = param_dict.get('cache')
        check_cache_option(cache)

        return method(self, *args, **kwargs)

    return new_method


def check_wiki_text_dataset(method):
    """A wrapper that wraps a parameter checker around the original Dataset(WikiTextDataset)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        _, param_dict = parse_user_args(method, *args, **kwargs)

        nreq_param_int = ['num_samples', 'num_parallel_workers', 'num_shards', 'shard_id']

        # check dataset_dir
        dataset_dir = param_dict.get('dataset_dir')
        check_dir(dataset_dir)

        # check usage
        usage = param_dict.get('usage')
        if usage is not None:
            check_valid_str(usage, ["train", "valid", "test", "all"], "usage")

        validate_dataset_param_value(nreq_param_int, param_dict, int)
        check_sampler_shuffle_shard_options(param_dict)

        cache = param_dict.get('cache')
        check_cache_option(cache)

        return method(self, *args, **kwargs)

    return new_method


def check_en_wik9_dataset(method):
    """Wrapper method to check the parameters of EnWik9 dataset."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        _, param_dict = parse_user_args(method, *args, **kwargs)

        nreq_param_int = ['num_samples', 'num_parallel_workers', 'num_shards', 'shard_id']
        dataset_dir = param_dict.get('dataset_dir')
        check_dir(dataset_dir)

        validate_dataset_param_value(nreq_param_int, param_dict, int)
        check_sampler_shuffle_shard_options(param_dict)

        cache = param_dict.get('cache')
        check_cache_option(cache)

        return method(self, *args, **kwargs)

    return new_method


def check_multi30k_dataset(method):
    """A wrapper that wraps a parameter checker around the original Dataset (Multi30kDataset)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        _, param_dict = parse_user_args(method, *args, **kwargs)

        nreq_param_int = ['num_samples', 'num_parallel_workers', 'num_shards', 'shard_id']
        nreq_param_bool = ['shuffle', 'decode']

        dataset_dir = param_dict.get('dataset_dir')
        check_dir(dataset_dir)

        usage = param_dict.get('usage')
        if usage is not None:
            check_valid_str(usage, ["train", "test", "valid", "all"], "usage")

        language_pair = param_dict.get('language_pair')
        support_language_pair = [['en', 'de'], ['de', 'en'], ('en', 'de'), ('de', 'en')]
        if language_pair is not None:
            type_check(language_pair, (list, tuple), "language_pair")
            if len(language_pair) != 2:
                raise ValueError(
                    "language_pair should be a list or tuple of length 2, but got {0}".format(len(language_pair)))
            if language_pair not in support_language_pair:
                raise ValueError(
                    "language_pair can only be ['en', 'de'] or ['en', 'de'], but got {0}".format(language_pair))

        validate_dataset_param_value(nreq_param_int, param_dict, int)
        validate_dataset_param_value(nreq_param_bool, param_dict, bool)

        check_sampler_shuffle_shard_options(param_dict)

        return method(self, *args, **kwargs)

    return new_method


def deprecated(version, substitute=None):
    """deprecated warning

    Args:
        version (str): version that the operator or function is deprecated.
        substitute (str): the substitute name for deprecated operator or function.
    """

    def decorate(func):
        def wrapper(*args, **kwargs):
            name = func.__name__
            message = f"'{name}' is deprecated from version {version} and will be removed in a future version. "
            if substitute:
                message += f"Use '{substitute}' instead."
            log.warning(message)
            ret = func(*args, **kwargs)
            return ret

        return wrapper

    return decorate


def check_obsminddataset(method):
    """A wrapper that wraps a parameter checker around the original Dataset(OBSMindDataset)."""

    @wraps(method)
    def new_method(self, *args, **kwargs):
        _, param_dict = parse_user_args(method, *args, **kwargs)

        nreq_param_int = ['num_shards', 'shard_id']
        nreq_param_list = ['columns_list']
        nreq_param_bool = ['shard_equal_rows']
        nreq_param_str = ['server', 'ak', 'sk', 'sync_obs_path']

        dataset_files = param_dict.get('dataset_files')
        type_check(dataset_files, (list,), "dataset_files")
        for dataset_file in dataset_files:
            if not isinstance(dataset_file, str):
                raise TypeError("Item of dataset files is not of type [{}], but got {}.".format(type(''),
                                                                                                type(dataset_file)))
        validate_dataset_param_value(nreq_param_int, param_dict, int)
        validate_dataset_param_value(nreq_param_list, param_dict, list)
        validate_dataset_param_value(nreq_param_bool, param_dict, bool)
        validate_dataset_param_value(nreq_param_str, param_dict, str)

        server = param_dict.get('server')
        if not server.startswith(('http://', 'https://')):
            raise ValueError("server should be a str that starts with http:// or https://, but got {}.".format(server))

        check_sampler_shuffle_shard_options(param_dict)

        return method(self, *args, **kwargs)

    return new_method
