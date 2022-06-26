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
1. This file is an abstraction of the dataset loading class. It contains
some basic dataset operations(skip, filter, map, batch, ...).
2. Specific dataset loading classes can be found in datasets_vision.py, datasets_text.py,
datasets_audio.py, datasets_standard_format.py and dataets_user_defined.py files.
    datasets_vision.py: contains vision dataset loading classes.
    datasets_text.py: contains text dataset loading classes.
    datasets_audio.py: contains audio dataset loading classes.
    datasets_standard_format.py: contains standard format loading classes which
                                 any other kinds of datasets can be converted to.
    dataets_user_defined.py: contains basic classes that help users to define
                             flexible ways to load dataset.
"""
import atexit
import glob
import json
import os
import signal
import stat

import gc
import time
import uuid
import multiprocessing
from multiprocessing.pool import RUN, TERMINATE
from enum import Enum
from importlib import import_module
import sys
import threading

import copy
import weakref
import platform
import psutil
import numpy as np

import mindspore._c_dataengine as cde
from mindspore._c_expression import typing

from mindspore import log as logger
from mindspore.parallel._ps_context import _is_role_pserver, _is_role_sched
from mindspore.dataset.engine.offload import GetOffloadModel

import mindspore.dataset.transforms.c_transforms as c_transforms
import mindspore.dataset.transforms.py_transforms as py_transforms
from mindspore.dataset.text.utils import SentencePieceModel, DE_C_INTER_SENTENCEPIECE_MODE
from mindspore.parallel._utils import _get_device_num

from . import samplers
from .iterators import DictIterator, TupleIterator, DummyIterator, check_iterator_cleanup, _set_iterator_cleanup, \
    ITERATORS_LIST, _unset_iterator_cleanup
from .queue import _SharedQueue
from .validators import check_batch, check_shuffle, check_map, check_filter, check_repeat, check_skip, check_zip, \
    check_rename, check_device_send, check_take, check_project, \
    check_sync_wait, check_zip_dataset, check_add_column, check_concat, check_split, check_bucket_batch_by_length, \
    check_save, check_tuple_iterator, check_dict_iterator, check_schema, check_to_device_send, deprecated
from ..core.config import get_callback_timeout, _init_device_info, get_enable_shared_mem, get_num_parallel_workers, \
    get_enable_watchdog
from ..core.datatypes import mstype_to_detype
from ..core.validator_helpers import replace_none
from ..core.py_util_helpers import ExceptionHandler
from ..transforms.py_transforms_util import FuncWrapper

try:
    context = import_module("mindspore.context")
except ModuleNotFoundError:
    context = None

if platform.system().lower() == "darwin" and multiprocessing.get_start_method() != "fork":
    multiprocessing.set_start_method("fork", True)

OffloadToManualOffloadMode = {
    None: cde.ManualOffloadMode.UNSPECIFIED,
    False: cde.ManualOffloadMode.DISABLED,
    True: cde.ManualOffloadMode.ENABLED
}

_train_dataset = None


def _set_training_dataset(dataset):
    """
    Set the dataset to be used when training recovery has occurred.

    Args:
        dataset: the training dataset or iterator
    """
    global _train_dataset
    _train_dataset = dataset


def _get_training_dataset():
    """
    Get the dataset to be used when training recovery has occurred.

    Returns:
        training dataset/iterator
    """
    return _train_dataset


def _reset_training_dataset(step):
    """
    Reset the training dataset to the given step number.

    Args:
        step (int): Global step number.
    """
    dataset = _get_training_dataset()
    if dataset is not None:
        dataset._reset(step)  # pylint: disable=W0212
    else:
        raise RuntimeError("Training dataset is not set.")


class Shuffle(str, Enum):
    """Specify the shuffle mode.

    - Shuffle.GLOBAL: Shuffle both the files and samples.
    - Shuffle.FILES: Shuffle files only.
    - Shuffle.INFILE: Shuffle data within each file.
    """
    GLOBAL: str = "global"
    FILES: str = "files"
    INFILE: str = "infile"


ShuffleToShuffleMode = {Shuffle.FILES: cde.ShuffleMode.FILES,
                        Shuffle.GLOBAL: cde.ShuffleMode.GLOBAL,
                        Shuffle.INFILE: cde.ShuffleMode.INFILE}


def shuffle_to_shuffle_mode(shuffle):
    """
    Shuffle Enum to Shuffle Mode

    Args:
        shuffle (Shuffle): shuffle flag to shuffle mode in C layer

    Returns:
        ShuffleMode, shuffle mode
    """
    shuffle_mode = cde.ShuffleMode.GLOBAL  # Global shuffle
    if not isinstance(shuffle, Shuffle):
        if shuffle is None or shuffle:
            shuffle_mode = cde.ShuffleMode.GLOBAL  # Global shuffle
        else:
            shuffle_mode = cde.ShuffleMode.FALSE  # No shuffle
    else:
        shuffle_mode = ShuffleToShuffleMode[shuffle]
    return shuffle_mode


def shuffle_to_bool(shuffle):
    """
    Shuffle Enum to bool

    Args:
        shuffle (Shuffle): shuffle flag to bool

    Returns:
        bool, True / False
    """
    if shuffle is not None and not isinstance(shuffle, (bool, Shuffle)):
        raise TypeError("shuffle must be of boolean or enum of 'Shuffle' values like 'Shuffle.GLOBAL' or "
                        "'Shuffle.FILES' or 'Shuffle.INFILE'.")

    shuffle_bool = True
    if not isinstance(shuffle, Shuffle):
        if shuffle is None:
            shuffle_bool = None
        elif shuffle:
            shuffle_bool = True
        else:
            shuffle_bool = False
    else:
        shuffle_bool = True
    return shuffle_bool


@check_zip
def zip(datasets):
    """
    Zip the datasets in the input tuple of datasets.

    Args:
        datasets (tuple[Dataset]): A tuple of datasets to be zipped together.
            The number of datasets must be more than 1.

    Returns:
        Dataset, dataset zipped.

    Raises:
        ValueError: If the number of datasets is 1.
        TypeError: If datasets is not a tuple.

    Examples:
            >>> # Create a dataset which is the combination of dataset_1 and dataset_2
            >>> dataset = ds.zip((dataset_1, dataset_2))
    """
    if len(datasets) <= 1:
        raise ValueError(
            "Can't zip empty or just one dataset!")
    for dataset in datasets:
        if not isinstance(dataset, Dataset):
            raise TypeError("Invalid dataset, expected Dataset object, but got %s!" % type(dataset))
    return ZipDataset(datasets)


def _get_operator_process():
    """
    Inner implemented method, mainly for passing sub-process id in C layer

    Returns:
         dict, mapping dict of operator id and corresponding process id.
    """
    global _OP_PROCESS
    process_info = _OP_PROCESS
    op_process = dict()
    keys = process_info.keys()
    fetched_all = True
    for key in keys:
        try:
            op_process[key] = list(process_info[key][1])
            item_full = (len(process_info[key][1]) == process_info[key][0])
        except KeyError as err:
            raise err
        fetched_all = fetched_all and item_full
    return op_process, fetched_all


def _set_dataset_permissions(file_name, num_files):
    """
    set saved dataset files' permissions to 600
    the rule of dataset filenames should be the same as those in C++.
    """
    num_digits = len(str(num_files - 1))
    if num_files == 1:
        paths = [file_name]
    else:
        paths = ["{}{}".format(file_name, str(x).rjust(num_digits, '0')) for x in range(num_files)]

    for item in paths:
        if os.path.exists(item):
            os.chmod(item, stat.S_IRUSR | stat.S_IWUSR)
            index_file = item + ".db"
            if os.path.exists(index_file):
                os.chmod(index_file, stat.S_IRUSR | stat.S_IWUSR)


class Dataset:
    """
    Abstract class to represent a dataset in DataEngine's data pipeline.

    This class is the base class of SourceDataset and Dataset, and represents
    a node in the data flow graph.
                                     Dataset
           -----------------------------------------------------------
           |                  |                   |                  |
    VisionBaseDataset    TextBaseDataset    AudioBaseDataset         |
           -                  -                   -                  |
           |                  |                   |                  |
           ----------------------------------------                  |
                      UnionBaseDataset                               |
                                                                     |
                                                               SourceDataset
                                                                     -
                                                                     |
                                                              MappableDataset

    DatasetOperator: MapDataset(UnionBaseDataset)
                     BatchDataset(UnionBaseDataset)
                     BucketBatchByLengthDataset(UnionBaseDataset)
                     ShuffleDataset(UnionBaseDataset)
                     FilterDataset(UnionBaseDataset)
                     RepeatDataset(UnionBaseDataset)
                     SkipDataset(UnionBaseDataset)
                     TakeDataset(UnionBaseDataset)
                     ZipDataset(UnionBaseDataset)
                     ConcatDataset(UnionBaseDataset)
                     RenameDataset(UnionBaseDataset)
                     ProjectDataset(UnionBaseDataset)
                     SyncWaitDataset(UnionBaseDataset)

    Impl Dataset - vision:       ImageFolderDataset(MappableDataset, VisionBaseDataset)
                                 USPSDataset(SourceDataset, VisionBaseDataset)
    Impl Dataset - text:         TextFileDataset(SourceDataset, TextBaseDataset)
                                 YahooAnswersDataset(SourceDataset, TextBaseDataset)
    Impl Dataset - audio:        LJSpeechDataset(MappableDataset, AudioBaseDataset)
                                 TedliumDataset(MappableDataset, AudioBaseDataset)
    Impl Dataset - standard:     MindDataset(MappableDataset, UnionBaseDataset)
                                 TFRecordDataset(SourceDataset, UnionBaseDataset)
    Impl Dataset - user defined: GeneratorDataset(MappableDataset, UnionBaseDataset)
                                 NumpySlicesDataset(GeneratorDataset)

    Args:
        num_parallel_workers (int, optional): Number of workers to process the dataset in parallel
            (default=None).
    """

    def __init__(self, children=None, num_parallel_workers=None, cache=None):
        # Note: children and parent are internal variables, not recommended for external using.
        self.children = replace_none(children, [])
        if isinstance(self.children, tuple):
            self.children = list(self.children)
        if not isinstance(self.children, list):
            self.children = [self.children]

        self.parent = []
        for child in self.children:
            child.parent.append(weakref.ref(self))
        self.num_parallel_workers = num_parallel_workers
        self.cache = cache

        self._device_iter = 0
        self._input_indexs = ()
        self.saved_output_types = None
        self.saved_output_shapes = None
        self.runtime_context = None
        self.dynamic_setting = [False, None]
        self.saved_min_shapes = None
        self.saved_max_shapes = None
        self._col_names = None
        self.dataset_size = None
        self._batch_size = None
        self._num_classes = None
        self._repeat_count = None
        self._class_indexing = None
        self._sync = False

    def create_ir_tree(self):
        """
        Internal method to build an IR tree.

        Returns:
            DatasetNode, the root node of the IR tree.
            Dataset, the root dataset of the IR tree.
        """
        parent = self.parent
        self.parent = []
        dataset = copy.deepcopy(self)
        global _OP_NAME
        _OP_NAME = Dataset._get_operator_id(dataset)
        ir_tree = dataset.parse_tree()
        self.parent = parent
        _init_device_info()
        return ir_tree, dataset

    def close_pool(self):
        """
        Close multiprocessing pool in dataset. If you are familiar with multiprocessing library, you can regard this
        as a destructor for a processingPool object.
        """
        # del all the SharedQueue when close the pool
        if hasattr(self, 'process_pool') and self.process_pool is not None:
            self.process_pool.close_pool()
            self.process_pool.delete_shared_memory()
        for child in self.children:
            child.close_pool()

    def notify_watchdog(self):
        """
        Close watchdog thread in dataset. Now GeneratorDataset/map/batch will use a thread named watch_dog to monitor
        multiprocess, for get_dataset_size/output_shapes/output_types/get_col_name/num_classes, we need notify_watchdog
        to close watch_dog thread manually.
        """
        if hasattr(self, 'sample_fn') and self.sample_fn is not None:
            if self.sample_fn.multi_process:
                self.sample_fn._abort_watchdog()  # pylint: disable=W0212
        if hasattr(self, 'process_pool') and self.process_pool is not None:
            self.process_pool.abort_watchdog()
        for child in self.children:
            child.notify_watchdog()

    @staticmethod
    def _get_operator_id(dataset):
        """
        Internal method to iterate the tree and obtain op_id of each operator.

        Returns:
            Dataset, the root dataset of the tree.
        """
        op_name = dict()
        generator_process = dict()
        op_name[str(dataset)] = 0
        op_id = 1

        def process_name(datasets, operator_id):
            if not datasets:
                return 0
            temp = []
            for item in datasets:
                for d in item.children:
                    temp.append(d)
                    op_name[str(d)] = operator_id

                    from mindspore.dataset.engine.datasets_user_defined import GeneratorDataset
                    if isinstance(d, GeneratorDataset) and d.sample_fn and d.sample_fn.pids:
                        generator_process[operator_id] = [d.num_parallel_workers, set(d.sample_fn.pids)]

            operator_id = operator_id + 1
            return process_name(temp, operator_id)

        process_name([dataset], op_id)
        if generator_process:
            global _OP_PROCESS
            _OP_PROCESS.update(generator_process)
        return op_name

    def parse_tree(self):
        """
        Internal method to parse the API tree into an IR tree.

        Returns:
            DatasetNode, the root node of the IR tree.
        """
        if len(self.parent) > 1:
            raise ValueError("The data pipeline is not a tree (i.e., one node has 2 consumers)")
        ir_children = [d.parse_tree() for d in self.children]
        # Bootstrap can only be performed on a copy of the original dataset node.
        # Bootstrap on original dataset node will make all iterators share the same process pool
        self.iterator_bootstrap()
        ir_node = self.parse(ir_children)
        ir_node = self.post_parse(ir_node)
        return ir_node

    def __safe_deepcopy__(self, memodict, exclude=()):
        if id(self) in memodict:
            return memodict[id(self)]
        cls = self.__class__
        new_op = cls.__new__(cls)
        memodict[id(self)] = new_op
        for arg, value in self.__dict__.items():
            if arg in exclude:
                setattr(new_op, arg, value)
            else:
                try:
                    setattr(new_op, arg, copy.deepcopy(value, memodict))
                except TypeError:
                    setattr(new_op, arg, value)
        return new_op

    @staticmethod
    def _noop_mode():
        if _is_role_sched() or _is_role_pserver():
            return True
        return False

    def iterator_bootstrap(self):
        pass

    def __add__(self, datasets):
        return self.concat(datasets)

    def to_json(self, filename=""):
        """
        Serialize a pipeline into JSON string and dump into file if filename is provided.

        Args:
            filename (str): filename of JSON file to be saved as (default="").

        Returns:
            str, JSON string of the pipeline.
        """
        ir_tree, _ = self.create_ir_tree()
        return json.loads(ir_tree.to_json(filename))

    @check_bucket_batch_by_length
    def bucket_batch_by_length(self, column_names, bucket_boundaries, bucket_batch_sizes, element_length_function=None,
                               pad_info=None, pad_to_bucket_boundary=False, drop_remainder=False):
        """
        Bucket elements according to their lengths. Each bucket will be padded and batched when
        they are full.

        A length function is called on each row in the dataset. The row is then
        bucketed based on its length and bucket boundaries. When a bucket reaches its
        corresponding size specified in bucket_batch_sizes, the entire bucket will be
        padded according to pad_info, and then form a batch.
        Each batch will be full, except one special case: the last batch for each bucket may not be full.

        Args:
            column_names (list[str]): Columns passed to element_length_function.
            bucket_boundaries (list[int]): A list consisting of the upper boundaries
                of the buckets. Must be strictly increasing. If there are n boundaries,
                n+1 buckets are created: One bucket for [0, bucket_boundaries[0]), one
                bucket for [bucket_boundaries[i], bucket_boundaries[i+1]) for each
                0<i<n-1, and the last bucket for [bucket_boundaries[n-1], inf).
            bucket_batch_sizes (list[int]): A list consisting of the batch sizes for
                each bucket. Must contain len(bucket_boundaries)+1 elements.
            element_length_function (Callable, optional): A function that takes in
                M arguments where M = len(column_names) and returns an integer. If no value
                provided, parameter M the len(column_names) must be 1, and the size of the first
                dimension of that column will be taken as the length (default=None).
            pad_info (dict, optional): The information about how to batch each column. The key
                corresponds to the column name, and the value must be a tuple of 2 elements.
                The first element corresponds to the shape to pad to, and the second
                element corresponds to the value to pad with. If a column is not
                specified, then that column will be padded to the longest in the current
                batch, and 0 will be used as the padding value. Any None dimensions will
                be padded to the longest in the current batch, unless if
                pad_to_bucket_boundary is True. If no padding is wanted, set pad_info
                to None (default=None).
            pad_to_bucket_boundary (bool, optional): If True, will pad each None
                dimension in pad_info to the bucket_boundary minus 1. If there are any
                elements that fall into the last bucket, an error will occur
                (default=False).
            drop_remainder (bool, optional): If True, will drop the last batch for each
                bucket if it is not a full batch (default=False).

        Returns:
            Dataset, dataset bucketed and batched by length.

        Examples:
            >>> # Create a dataset where certain counts rows are combined into a batch
            >>> # and drops the last incomplete batch if there is one.
            >>> import numpy as np
            >>> def generate_2_columns(n):
            ...     for i in range(n):
            ...         yield (np.array([i]), np.array([j for j in range(i + 1)]))
            >>>
            >>> column_names = ["col1", "col2"]
            >>> dataset = ds.GeneratorDataset(generate_2_columns(8), column_names)
            >>> bucket_boundaries = [5, 10]
            >>> bucket_batch_sizes = [2, 1, 1]
            >>> element_length_function = (lambda col1, col2: max(len(col1), len(col2)))
            >>> # Will pad col2 to shape [bucket_boundaries[i]] where i is the
            >>> # index of the bucket that is currently being batched.
            >>> pad_info = {"col2": ([None], -1)}
            >>> pad_to_bucket_boundary = True
            >>> dataset = dataset.bucket_batch_by_length(column_names, bucket_boundaries,
            ...                                          bucket_batch_sizes,
            ...                                          element_length_function, pad_info,
            ...                                          pad_to_bucket_boundary)
        """
        return BucketBatchByLengthDataset(self, column_names, bucket_boundaries, bucket_batch_sizes,
                                          element_length_function, pad_info, pad_to_bucket_boundary, drop_remainder)

    @check_batch
    def batch(self, batch_size, drop_remainder=False, num_parallel_workers=None, per_batch_map=None,
              input_columns=None, output_columns=None, column_order=None, pad_info=None,
              python_multiprocessing=False, max_rowsize=16):
        """
        Combine batch_size number of consecutive rows into batches.

        For any child node, a batch is treated as a single row.
        For any column, all the elements within that column must have the same shape.
        If a per_batch_map callable is provided, it will be applied to the batches of tensors.

        Note:
            The order of using repeat and batch reflects the number of batches and per_batch_map.
            It is recommended that the repeat operation applied after the batch operation finished.

        Args:
            batch_size (int or function): The number of rows each batch is created with. An
                int or callable object which takes exactly 1 parameter, BatchInfo.
            drop_remainder (bool, optional): Determines whether or not to drop the last block
                whose data row number is less than batch size (default=False). If True, and if there are less
                than batch_size rows available to make the last batch, then those rows will
                be dropped and not propagated to the child node.
            num_parallel_workers (int, optional): Number of workers(threads) to process the dataset in parallel
                (default=None).
            per_batch_map (callable, optional): Per batch map callable (default=None). A callable which takes
                (list[Tensor], list[Tensor], ..., BatchInfo) as input parameters. Each list[Tensor] represents a batch
                of Tensors on a given column. The number of lists should match with the number of entries in
                input_columns. The last parameter of the callable should always be a BatchInfo object. Per_batch_map
                should return (list[Tensor], list[Tensor], ...). The length of each list in output should be the same as
                the input. output_columns is required if the number of output lists is different from input.
            input_columns (Union[str, list[str]], optional): List of names of the input columns. The size of the list
                should match with signature of per_batch_map callable (default=None).
            output_columns (Union[str, list[str]], optional): List of names assigned to the columns
                outputted by the last operation. This parameter is mandatory if len(input_columns) !=
                len(output_columns). The size of this list must match the number of output
                columns of the last operation. (default=None, output columns will have the same
                name as the input columns, i.e., the columns will be replaced).
            column_order (Union[str, list[str]], optional): Specifies the list of all the columns you need in the whole
                dataset (default=None). The parameter is required when len(input_column) != len(output_column).
                Caution: the list here is not just the columns specified in parameter input_columns and output_columns.
            pad_info (dict, optional): Whether to perform padding on selected columns. pad_info={"col1":([224,224],0)}
                would pad column with name "col1" to a tensor of size [224,224] and fill the missing with 0
                (default=None).
            python_multiprocessing (bool, optional): Parallelize Python function per_batch_map with multi-processing.
                This option could be beneficial if the function is computational heavy (default=False).
            max_rowsize(int, optional): Maximum size of row in MB that is used for shared memory allocation to copy
               data between processes.  This is only used if python_multiprocessing is set to True (default=16).

        Returns:
            BatchDataset, dataset batched.

        Examples:
            >>> # 1) Create a dataset where every 100 rows are combined into a batch
            >>> # and drops the last incomplete batch if there is one.
            >>> dataset = dataset.batch(100, True)
            >>>
            >>> # 2）resize image according to its batch number, if it's 5-th batch, resize to (5^2, 5^2) = (25, 25)
            >>> def np_resize(col, BatchInfo):
            ...     output = col.copy()
            ...     s = (BatchInfo.get_batch_num() + 1) ** 2
            ...     index = 0
            ...     for c in col:
            ...         img = Image.fromarray(c.astype('uint8')).convert('RGB')
            ...         img = img.resize((s, s), Image.ANTIALIAS)
            ...         output[index] = np.array(img)
            ...         index += 1
            ...     return (output,)
            >>> dataset = dataset.batch(batch_size=8, input_columns=["image"], per_batch_map=np_resize)
            >>>
            >>> # 3）Create a dataset where its batch size is dynamic
            >>> # Define a callable batch size function and let batch size increase 1 each time.
            >>> def add_one(BatchInfo):
            ...     return BatchInfo.get_batch_num() + 1
            >>> dataset = dataset.batch(batch_size=add_one, drop_remainder=True)
            >>>
            >>> # 4）Create a dataset with batch, then specify the column order.
            >>> # Assume that the original coulmn order is ["image", "label"] and change to ["label", "image"].
            >>> dataset = dataset.batch(32, column_order=["label", "image"])
        """
        return BatchDataset(self, batch_size, drop_remainder, num_parallel_workers, per_batch_map, input_columns,
                            output_columns, column_order, pad_info, python_multiprocessing, max_rowsize)

    @check_sync_wait
    def sync_wait(self, condition_name, num_batch=1, callback=None):
        """
        Add a blocking condition to the input Dataset. A synchronize action will be applied.

        Args:
            condition_name (str): The condition name that is used to toggle sending next row.
            num_batch (int): the number of batches without blocking at the start of each epoch (default=1).
            callback (function): The callback function that will be invoked when sync_update is called (default=None).

        Returns:
            SyncWaitDataset, dataset added a blocking condition.

        Raises:
            RuntimeError: If condition name already exists.

        Examples:
            >>> import numpy as np
            >>> def gen():
            ...     for i in range(100):
            ...         yield (np.array(i),)
            >>>
            >>> class Augment:
            ...     def __init__(self, loss):
            ...         self.loss = loss
            ...
            ...     def preprocess(self, input_):
            ...         return input_
            ...
            ...     def update(self, data):
            ...         self.loss = data["loss"]
            >>>
            >>> batch_size = 4
            >>> dataset = ds.GeneratorDataset(gen, column_names=["input"])
            >>>
            >>> aug = Augment(0)
            >>> dataset = dataset.sync_wait(condition_name="policy", callback=aug.update)
            >>> dataset = dataset.map(operations=[aug.preprocess], input_columns=["input"])
            >>> dataset = dataset.batch(batch_size)
            >>> count = 0
            >>> for data in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
            ...     assert data["input"][0] == count
            ...     count += batch_size
            ...     data = {"loss": count}
            ...     dataset.sync_update(condition_name="policy", data=data)
        """
        return SyncWaitDataset(self, condition_name, num_batch, callback)

    @check_shuffle
    def shuffle(self, buffer_size):
        """
        Randomly shuffles the rows of this dataset using the following policy:

        1. Make a shuffle buffer that contains the first buffer_size rows.
        2. Randomly select an element from the shuffle buffer to be the next row
           propagated to the child node.
        3. Get the next row (if any) from the parent node and put it in the shuffle buffer.
        4. Repeat steps 2 and 3 until there are no more rows left in the shuffle buffer.

        A random seed can be provided to be used on the first epoch. In every subsequent
        epoch, the seed is changed to a new one, randomly generated value.

        Args:
            buffer_size (int): The size of the buffer (must be larger than 1) for
                shuffling. Setting buffer_size equal to the number of rows in the entire
                dataset will result in a global shuffle.

        Returns:
            Dataset, dataset shuffled.

        Raises:
            RuntimeError: If exist sync operators before shuffle.

        Examples:
            >>> # dataset is an instance object of Dataset
            >>> # Optionally set the seed for the first epoch
            >>> ds.config.set_seed(58)
            >>> # Create a shuffled dataset using a shuffle buffer of size 4
            >>> dataset = dataset.shuffle(4)
        """
        return ShuffleDataset(self, buffer_size)

    def flat_map(self, func):
        """
        Map `func` to each row in dataset and flatten the result.

        The specified `func` is a function that must take one 'Ndarray' as input
        and return a 'Dataset'.

        Args:
            func (function): A function that must take one 'Ndarray' as an argument and
                return a 'Dataset'.

        Returns:
            Dataset, dataset applied by the function.

        Examples:
            >>> # use NumpySlicesDataset as an example
            >>> dataset = ds.NumpySlicesDataset([[0, 1], [2, 3]])
            >>>
            >>> def flat_map_func(array):
            ...     # create a NumpySlicesDataset with the array
            ...     dataset = ds.NumpySlicesDataset(array)
            ...     # repeat the dataset twice
            ...     dataset = dataset.repeat(2)
            ...     return dataset
            >>>
            >>> dataset = dataset.flat_map(flat_map_func)
            >>> # [[0, 1], [0, 1], [2, 3], [2, 3]]

        Raises:
            TypeError: If `func` is not a function.
            TypeError: If `func` doesn't return a Dataset.
        """
        dataset = None
        if not hasattr(func, '__call__'):
            logger.critical("func must be a function.")
            raise TypeError("func must be a function.")

        for row_data in self.create_tuple_iterator(output_numpy=True):
            if dataset is None:
                dataset = func(row_data)
            else:
                dataset += func(row_data)

        if not isinstance(dataset, Dataset):
            logger.critical("flat_map must return a Dataset object.")
            raise TypeError("flat_map must return a Dataset object.")
        return dataset

    @check_map
    def map(self, operations, input_columns=None, output_columns=None, column_order=None,
            num_parallel_workers=None, python_multiprocessing=False, cache=None, callbacks=None,
            max_rowsize=16, offload=None):
        """
        Apply each operation in operations to this dataset.

        The order of operations is determined by the position of each operation in the operations parameter.
        operations[0] will be applied first, then operations[1], then operations[2], etc.

        Each operation will be passed one or more columns from the dataset as input, and zero or
        more columns will be outputted. The first operation will be passed the columns specified
        in input_columns as input. If there is more than one operator in operations, the outputted
        columns of the previous operation are used as the input columns for the next operation.
        The columns outputted by the very last operation will be assigned names specified by
        output_columns.

        Only the columns specified in column_order will be propagated to the child node. These
        columns will be in the same order as specified in column_order.

        Args:
            operations (Union[list[TensorOperation], list[functions]]): List of operations to be
                applied on the dataset. Operations are applied in the order they appear in this list.
            input_columns (Union[str, list[str]], optional): List of the names of the columns that will be passed to
                the first operation as input. The size of this list must match the number of
                input columns expected by the first operator. (default=None, the first
                operation will be passed however many columns that are required, starting from
                the first column).
            output_columns (Union[str, list[str]], optional): List of names assigned to the columns outputted by
                the last operation. This parameter is mandatory if len(input_columns) !=
                len(output_columns). The size of this list must match the number of output
                columns of the last operation. (default=None, output columns will have the same
                name as the input columns, i.e., the columns will be replaced).
            column_order (list[str], optional): Specifies the list of all the columns you need in the whole
                dataset (default=None). The parameter is required when len(input_column) != len(output_column).
                Caution: the list here is not just the columns specified in parameter input_columns and output_columns.
            num_parallel_workers (int, optional): Number of threads used to process the dataset in
                parallel (default=None, the value from the configuration will be used).
            python_multiprocessing (bool, optional): Parallelize Python operations with multiple worker processes. This
                option could be beneficial if the Python operation is computational heavy (default=False).
            cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing.
                (default=None, which means no cache is used).
            callbacks (DSCallback, list[DSCallback], optional): List of Dataset callbacks to be called (Default=None).
            max_rowsize (int, optional): Maximum size of row in MB that is used for shared memory allocation to copy
               data between processes.  This is only used if python_multiprocessing is set to True (Default=16).
            offload (bool, optional): Flag to indicate whether offload is used (Default=None).

        Note:
            - Input `operations` mainly accept c_transforms, py_transforms operator in mindspore.dataset part, plus user
              defined Python function(PyFuncs).
            - Do not add network computing operators from mindspore.nn and mindspore.ops or others into this
              `operations`.

        Returns:
            Dataset, dataset after mapping operation.

        Examples:
            >>> # dataset is an instance of Dataset which has 2 columns, "image" and "label".
            >>>
            >>> # Define two operations, where each operation accepts 1 input column and outputs 1 column.
            >>> decode_op = c_vision.Decode(rgb=True)
            >>> random_jitter_op = c_vision.RandomColorAdjust(brightness=(0.8, 0.8), contrast=(1, 1),
            ...                                               saturation=(1, 1), hue=(0, 0))
            >>>
            >>> # 1) Simple map example.
            >>>
            >>> # Apply decode_op on column "image". This column will be replaced by the outputted
            >>> # column of decode_op. Since column_order is not provided, both columns "image"
            >>> # and "label" will be propagated to the child node in their original order.
            >>> dataset = dataset.map(operations=[decode_op], input_columns=["image"])
            >>>
            >>> # Decode and rename column "image" to "decoded_image".
            >>> dataset = dataset.map(operations=[decode_op], input_columns=["image"], output_columns=["decoded_image"])
            >>>
            >>> # Specify the order of the output columns.
            >>> dataset = dataset.map(operations=[decode_op], input_columns=["image"],
            ...                       output_columns=None, column_order=["label", "image"])
            >>>
            >>> # Rename column "image" to "decoded_image" and also specify the order of the output columns.
            >>> dataset = dataset.map(operations=[decode_op], input_columns=["image"],
            ...                       output_columns=["decoded_image"], column_order=["label", "decoded_image"])
            >>>
            >>> # Rename column "image" to "decoded_image" and keep only this column.
            >>> dataset = dataset.map(operations=[decode_op], input_columns=["image"],
            ...                       output_columns=["decoded_image"], column_order=["decoded_image"])
            >>>
            >>> # A simple example for mapping pyfunc. Renaming columns and specifying column order
            >>> # work in the same way as the previous examples.
            >>> dataset = ds.NumpySlicesDataset(data=[[0, 1, 2]], column_names=["data"])
            >>> dataset = dataset.map(operations=[(lambda x: x + 1)], input_columns=["data"])
            >>>
            >>> # 2) Map example with more than one operation.
            >>>
            >>> # Create a dataset where the images are decoded, then randomly color jittered.
            >>> # decode_op takes column "image" as input and outputs one column. The column
            >>> # outputted by decode_op is passed as input to random_jitter_op.
            >>> # random_jitter_op will output one column. Column "image" will be replaced by
            >>> # the column outputted by random_jitter_op (the very last operation). All other
            >>> # columns are unchanged. Since column_order is not specified, the order of the
            >>> # columns will remain the same.
            >>> dataset = dataset.map(operations=[decode_op, random_jitter_op], input_columns=["image"])
            >>>
            >>> # Rename the column outputted by random_jitter_op to "image_mapped".
            >>> # Specifying column order works in the same way as examples in 1).
            >>> dataset = dataset.map(operations=[decode_op, random_jitter_op], input_columns=["image"],
            ...                       output_columns=["image_mapped"])
            >>>
            >>> # Map with multiple operations using pyfunc. Renaming columns and specifying column order
            >>> # work in the same way as examples in 1).
            >>> dataset = ds.NumpySlicesDataset(data=[[0, 1, 2]], column_names=["data"])
            >>> dataset = dataset.map(operations=[(lambda x: x * x), (lambda x: x - 1)], input_columns=["data"],
            ...                                   output_columns=["data_mapped"])
            >>>
            >>> # 3) Example where number of input columns is not equal to number of output columns.
            >>>
            >>> # operations[0] is a lambda that takes 2 columns as input and outputs 3 columns.
            >>> # operations[1] is a lambda that takes 3 columns as input and outputs 1 column.
            >>> # operations[2] is a lambda that takes 1 column as input and outputs 4 columns.
            >>> #
            >>> # Note: The number of output columns of operation[i] must equal the number of
            >>> # input columns of operation[i+1]. Otherwise, this map call will also result
            >>> # in an error.
            >>> operations = [(lambda x, y: (x, x + y, x + y + 1)),
            ...               (lambda x, y, z: x * y * z),
            ...               (lambda x: (x % 2, x % 3, x % 5, x % 7))]
            >>>
            >>> # Note: Since the number of input columns is not the same as the number of
            >>> # output columns, the output_columns and column_order parameters must be
            >>> # specified. Otherwise, this map call will also result in an error.
            >>>
            >>> dataset = ds.NumpySlicesDataset(data=([[0, 1, 2]], [[3, 4, 5]]), column_names=["x", "y"])
            >>>
            >>> # Propagate all columns to the child node in this order:
            >>> dataset = dataset.map(operations, input_columns=["x", "y"],
            ...                       output_columns=["mod2", "mod3", "mod5", "mod7"],
            ...                       column_order=["mod2", "mod3", "mod5", "mod7"])
            >>>
            >>> # Propagate some columns to the child node in this order:
            >>> dataset = dataset.map(operations, input_columns=["x", "y"],
            ...                       output_columns=["mod2", "mod3", "mod5", "mod7"],
            ...                       column_order=["mod7", "mod3", "col2"])
        """
        if hasattr(self, 'operator_mixed') and getattr(self, 'operator_mixed') is True:
            num_parallel_workers = 1
            logger.warning(
                "Input 'operations' of 'map' includes network computing operators like in mindspore.nn, mindspore.ops, "
                "mindspore.numpy module and etc, which do not support multi-thread compiling, recommend to replace it "
                "with python implemented operator like numpy etc. Here decrease 'num_parallel_workers' into 1.")

        return MapDataset(self, operations, input_columns, output_columns, column_order, num_parallel_workers,
                          python_multiprocessing, cache, callbacks, max_rowsize, offload)

    @check_filter
    def filter(self, predicate, input_columns=None, num_parallel_workers=None):
        """
        Filter dataset by prediction.

        Args:
            predicate (callable): Python callable which returns a boolean value. If False then filter the element.
            input_columns (Union[str, list[str]], optional): List of names of the input columns. If not provided
                or provided with None, the predicate will be applied on all columns in the dataset (default=None).
            num_parallel_workers (int, optional): Number of workers to process the dataset
                in parallel (default=None).

        Returns:
            Dataset, dataset filtered.

        Examples:
            >>> # generator data(0 ~ 63)
            >>> # filter the data that greater than or equal to 11
            >>> dataset = dataset.filter(predicate=lambda data: data < 11, input_columns = ["data"])
        """
        return FilterDataset(self, predicate, input_columns, num_parallel_workers)

    @check_repeat
    def repeat(self, count=None):
        """
        Repeat this dataset `count` times. Repeat infinitely if the count is None or -1.

        Note:
            The order of using repeat and batch reflects the number of batches. It is recommended that
            the repeat operation is used after the batch operation.

        Args:
            count (int): Number of times the dataset is going to be repeated (default=None).

        Returns:
            Dataset, dataset repeated.

        Examples:
            >>> # dataset is an instance object of Dataset
            >>>
            >>> # Create a dataset where the dataset is repeated for 50 epochs
            >>> dataset = dataset.repeat(50)
            >>>
            >>> # Create a dataset where each epoch is shuffled individually
            >>> dataset = dataset.shuffle(10)
            >>> dataset = dataset.repeat(50)
            >>>
            >>> # Create a dataset where the dataset is first repeated for
            >>> # 50 epochs before shuffling. The shuffle operator will treat
            >>> # the entire 50 epochs as one big dataset.
            >>> dataset = dataset.repeat(50)
            >>> dataset = dataset.shuffle(10)
        """
        return RepeatDataset(self, count)

    @check_skip
    def skip(self, count):
        """
        Skip the first N elements of this dataset.

        Args:
            count (int): Number of elements in the dataset to be skipped.

        Returns:
            Dataset, dataset that containing rows like origin rows subtract skipped rows.

        Examples:
            >>> # dataset is an instance object of Dataset
            >>> # Create a dataset which skips first 3 elements from data
            >>> dataset = dataset.skip(3)
        """
        return SkipDataset(self, count)

    @check_take
    def take(self, count=-1):
        """
        Takes at most given numbers of elements from the dataset.

        Note:
            1. If count is greater than the number of elements in the dataset or equal to -1,
               all the elements in dataset will be taken.
            2. The order of using take and batch matters. If take is before batch operation,
               then take the given number of rows; otherwise take the given number of batches.

        Args:
            count (int, optional): Number of elements to be taken from the dataset (default=-1).

        Returns:
            Dataset, dataset taken.

        Examples:
            >>> # dataset is an instance object of Dataset
            >>> # Create a dataset where the dataset includes 50 elements.
            >>> dataset = dataset.take(50)
        """
        return TakeDataset(self, count)

    def _get_absolute_split_sizes(self, sizes):
        """
        Internal method called by split to calculate absolute split sizes and to
        do some error checking after calculating absolute split sizes.

        Returns:
            int, absolute split sizes of the dataset.
        """
        # Call get_dataset_size here and check input here because
        # don't want to call this once in check_split and another time in
        # here again
        dataset_size = self.get_dataset_size()

        if dataset_size is None or dataset_size <= 0:
            raise RuntimeError("dataset_size is unknown, unable to split.")

        if not isinstance(sizes, list):
            raise RuntimeError("sizes must be a list.")

        all_int = all(isinstance(item, int) for item in sizes)
        if all_int:
            sizes_sum = sum(sizes)
            if sizes_sum != dataset_size:
                raise RuntimeError("Sum of split sizes {} is not equal to dataset size {}."
                                   .format(sizes_sum, dataset_size))
            return sizes

        absolute_sizes = []
        for item in sizes:
            absolute_size = int(round(item * dataset_size))
            if absolute_size == 0:
                raise RuntimeError("Split percentage {} is too small.".format(item))
            absolute_sizes.append(absolute_size)

        absolute_sizes_sum = sum(absolute_sizes)

        # if we still need more rows, give them to the first split.
        # if we have too many rows, remove the extras from the first split that has
        # enough rows.
        size_difference = int(dataset_size - absolute_sizes_sum)
        if size_difference > 0:
            absolute_sizes[0] += size_difference
        else:
            for i, _ in enumerate(absolute_sizes):
                if absolute_sizes[i] + size_difference > 0:
                    absolute_sizes[i] += size_difference
                    break

        if sum(absolute_sizes) != dataset_size:
            raise RuntimeError("Sum of calculated split sizes {} is not equal to dataset size {}."
                               .format(absolute_sizes_sum, dataset_size))

        return absolute_sizes

    @check_split
    def split(self, sizes, randomize=True):
        """
        Split the dataset into smaller, non-overlapping datasets.

        This is a general purpose split function which can be called from any operator in the pipeline.
        There is another, optimized split function, which will be called automatically if ds.split is
        called where ds is a MappableDataset.

        Args:
            sizes (Union[list[int], list[float]]): If a list of integers [s1, s2, …, sn] is
                provided, the dataset will be split into n datasets of size s1, size s2, …, size sn
                respectively. If the sum of all input sizes does not equal the original dataset size, an
                error will throw.
                If a list of floats [f1, f2, …, fn] is provided, all floats must be between 0 and 1
                and must sum to 1, otherwise an error will throw. The dataset will be split into n
                Datasets of size round(f1*K), round(f2*K), …, round(fn*K) where K is the size of the
                original dataset.
                If after rounding:

                - Any size equals 0, an error will occur.
                - The sum of split sizes < K, the difference of K - sigma(round(fi * k)) will be added to the first
                  split.
                - The sum of split sizes > K, the difference of sigma(round(fi * K)) - K will be removed from the first
                  large enough split such that it will have at least 1 row after removing the difference.

            randomize (bool, optional): Determines whether or not to split the data randomly (default=True).
                If True, the data will be randomly split. Otherwise, each split will be created with
                consecutive rows from the dataset.

        Note:
            1. Dataset cannot be sharded if split is going to be called.
            2. It is strongly recommended to not shuffle the dataset, but use randomize=True instead.
               Shuffling the dataset may not be deterministic, which means the data in each split
               will be different in each epoch.

        Raises:
            RuntimeError: If get_dataset_size returns None or is not supported for this dataset.
            RuntimeError: If `sizes` is list of integers and sum of all elements in sizes does not
                equal the dataset size.
            RuntimeError: If `sizes` is list of float and there is a split with size 0 after calculations.
            RuntimeError: If the dataset is sharded prior to calling split.
            ValueError: If `sizes` is list of float and not all floats are between 0 and 1, or if the
                floats don't sum to 1.

        Returns:
            tuple(Dataset), a tuple of datasets that have been split.

        Examples:
            >>> # TextFileDataset is not a mappable dataset, so this non-optimized split will be called.
            >>> # Since many datasets have shuffle on by default, set shuffle to False if split will be called!
            >>> dataset = ds.TextFileDataset(text_file_dataset_dir, shuffle=False)
            >>> train_dataset, test_dataset = dataset.split([0.9, 0.1])
        """
        if self.is_shuffled():
            logger.warning("Dataset is shuffled before split.")

        if self.is_sharded():
            raise RuntimeError("Dataset should not be sharded before split.")

        absolute_sizes = self._get_absolute_split_sizes(sizes)
        splits = []
        rows_to_skip = 0
        for size in absolute_sizes:
            ds = copy.deepcopy(self)
            if randomize:
                # want to shuffle the same way every epoch before split
                # in alter_tree, shuffle buffer is minimum 10000, so use 10000 here
                ds = ds.shuffle(10000)
                ds.reshuffle_each_epoch = False

            if rows_to_skip > 0:
                ds = ds.skip(rows_to_skip)

            ds = ds.take(size)
            splits.append(ds)

            rows_to_skip += size

        return tuple(splits)

    @check_zip_dataset
    def zip(self, datasets):
        """
        Zip the datasets in the sense of input tuple of datasets. Columns in the input datasets must have different
        name.

        Args:
            datasets (Union[tuple, class Dataset]): A tuple of datasets or a single class Dataset
                to be zipped together with this dataset.

        Returns:
            Dataset, dataset zipped.

        Examples:
            >>> # Create a dataset which is the combination of dataset and dataset_1
            >>> dataset = dataset.zip(dataset_1)
        """
        if isinstance(datasets, tuple):
            datasets = (self, *datasets)
        elif isinstance(datasets, Dataset):
            datasets = (self, datasets)
        else:
            raise TypeError("Invalid datasets, expected Dataset object or tuple of Dataset, but got %s!" % datasets)
        return ZipDataset(datasets)

    @check_concat
    def concat(self, datasets):
        """
        Concatenate the dataset objects in the input list.
        Performing "+" operation on dataset objects can achieve the same effect.

        Note:
            The column name, and rank and type of the column data must be the same in the input datasets.

        Args:
            datasets (Union[list, class Dataset]): A list of datasets or a single class Dataset
                to be concatenated together with this dataset.

        Returns:
            Dataset, dataset concatenated.

        Examples:
            >>> # Create a dataset by concatenating dataset_1 and dataset_2 with "+" operator
            >>> dataset = dataset_1 + dataset_2
            >>> # Create a dataset by concatenating dataset_1 and dataset_2 with concat operation
            >>> dataset = dataset_1.concat(dataset_2)
        """
        if isinstance(datasets, Dataset):
            datasets = [self] + [datasets]
        elif isinstance(datasets, list):
            datasets = [self] + datasets
        else:
            raise TypeError("Invalid datasets, expected Dataset object or list of Dataset, but got %s!" % datasets)
        return ConcatDataset(datasets)

    @check_rename
    def rename(self, input_columns, output_columns):
        """
        Rename the columns in input datasets.

        Args:
            input_columns (Union[str, list[str]]): List of names of the input columns.
            output_columns (Union[str, list[str]]): List of names of the output columns.

        Returns:
            Dataset, dataset renamed.

        Examples:
            >>> # dataset is an instance object of Dataset
            >>> input_columns = ["input_col1", "input_col2", "input_col3"]
            >>> output_columns = ["output_col1", "output_col2", "output_col3"]
            >>>
            >>> # Create a dataset where input_col1 is renamed to output_col1, and
            >>> # input_col2 is renamed to output_col2, and input_col3 is renamed
            >>> # to output_col3.
            >>> dataset = dataset.rename(input_columns=input_columns, output_columns=output_columns)
        """

        return RenameDataset(self, input_columns, output_columns)

    @check_project
    def project(self, columns):
        """
        Project certain columns in input dataset.

        The specified columns will be selected from the dataset and passed into
        the pipeline with the order specified. The other columns are discarded.

        Args:
            columns(Union[str, list[str]]): List of names of the columns to project.

        Returns:
            Dataset, dataset projected.

        Examples:
            >>> # dataset is an instance object of Dataset
            >>> columns_to_project = ["column3", "column1", "column2"]
            >>>
            >>> # Create a dataset that consists of column3, column1, column2
            >>> # in that order, regardless of the original order of columns.
            >>> dataset = dataset.project(columns=columns_to_project)
        """

        return ProjectDataset(self, columns)

    def apply(self, apply_func):
        """
        Apply a function in this dataset.

        Args:
            apply_func (function): A function that must take one 'Dataset' as an argument and
                                   return a preprocessed 'Dataset'.

        Returns:
            Dataset, dataset applied by the function.

        Examples:
            >>> # dataset is an instance object of Dataset
            >>>
            >>> # Declare an apply_func function which returns a Dataset object
            >>> def apply_func(data):
            ...     data = data.batch(2)
            ...     return data
            >>>
            >>> # Use apply to call apply_func
            >>> dataset = dataset.apply(apply_func)

        Raises:
            TypeError: If apply_func is not a function.
            TypeError: If apply_func doesn't return a Dataset.
        """

        if not hasattr(apply_func, '__call__'):
            raise TypeError("apply_func must be a function.")

        dataset = apply_func(self)
        if not isinstance(dataset, Dataset):
            raise TypeError("apply_func must return a dataset.")
        return dataset

    @check_device_send
    def device_que(self, send_epoch_end=True, create_data_info_queue=False):
        """
        Return a transferred Dataset that transfers data through a device.

        Args:
            send_epoch_end (bool, optional): Whether to send end of sequence to device or not (default=True).
            create_data_info_queue (bool, optional): Whether to create queue which stores
                types and shapes of data or not(default=False).

        Note:
            If device is Ascend, features of data will be transferred one by one. The limitation
            of data transmission per time is 256M.

        Returns:
            Dataset, dataset for transferring.
        """
        return self.to_device(send_epoch_end=send_epoch_end, create_data_info_queue=create_data_info_queue)

    @check_device_send
    def to_device(self, send_epoch_end=True, create_data_info_queue=False):
        """
        Transfer data from CPU to GPU or Ascend or other devices.

        Args:
            send_epoch_end (bool, optional): Whether to send the end of sequence to device or not (default=True).
            create_data_info_queue (bool, optional): Whether to create queue which stores
                types and shapes of data or not(default=False).

        Note:
            If device is Ascend, features of data will be transferred one by one. The limitation
            of data transmission per second is 256M.

        Returns:
            TransferDataset, dataset for transferring.

        Raises:
            RuntimeError: If distribution file path is given but failed to read.
        """
        return TransferDataset(self, send_epoch_end, create_data_info_queue)

    @check_save
    def save(self, file_name, num_files=1, file_type='mindrecord'):
        """
        Save the dynamic data processed by the dataset pipeline in common dataset format.
        Supported dataset formats: 'mindrecord' only

        Implicit type casting exists when saving data as 'mindrecord'. The transform table shows how to do type casting.

        .. list-table:: Implicit Type Casting when Saving as 'mindrecord'
           :widths: 25 25 50
           :header-rows: 1

           * - Type in 'dataset'
             - Type in 'mindrecord'
             - Details
           * - bool
             - None
             - Not supported
           * - int8
             - int32
             -
           * - uint8
             - bytes(1D uint8)
             - Drop dimension
           * - int16
             - int32
             -
           * - uint16
             - int32
             -
           * - int32
             - int32
             -
           * - uint32
             - int64
             -
           * - int64
             - int64
             -
           * - uint64
             - None
             - Not supported
           * - float16
             - float32
             -
           * - float32
             - float32
             -
           * - float64
             - float64
             -
           * - string
             - string
             - Multi-dimensional string not supported

        Note:
            1. To save the samples in order, set dataset's shuffle to False and num_files to 1.
            2. Before calling the function, do not use batch operator, repeat operator or data augmentation operators
               with random attribute in map operator.
            3. When array dimension is variable, one-dimensional arrays or
               multi-dimensional arrays with variable dimension 0 are supported.
            4. Mindrecord does not support uint64, multi-dimensional uint8(drop dimension) nor
               multi-dimensional string.

        Args:
            file_name (str): Path to dataset file.
            num_files (int, optional): Number of dataset files (default=1).
            file_type (str, optional): Dataset format (default='mindrecord').

        """
        ir_tree, api_tree = self.create_ir_tree()

        runtime_context = cde.PythonRuntimeContext()
        runtime_context.Init()
        consumer = cde.PythonSaveToDisk(file_name, num_files, file_type)
        consumer.Init(ir_tree)
        runtime_context.AssignConsumer(consumer)

        consumer.Save()
        _set_dataset_permissions(file_name, num_files)
        del api_tree

    @check_tuple_iterator
    def create_tuple_iterator(self, columns=None, num_epochs=-1, output_numpy=False, do_copy=True):
        """
        Create an iterator over the dataset. The datatype retrieved back will be a list of ndarrays.

        To specify which columns to list and the order needed, use columns_list. If columns_list
        is not provided, the order of the columns will remain unchanged.

        Args:
            columns (list[str], optional): List of columns to be used to specify the order of columns
                (default=None, means all columns).
            num_epochs (int, optional): Maximum number of epochs that iterator can be iterated.
                (default=-1, iterator can be iterated infinite number of epochs)
            output_numpy (bool, optional): Whether or not to output NumPy datatype.
                If output_numpy=False, iterator will output MSTensor (default=False).
            do_copy (bool, optional): when output data type is mindspore.Tensor,
                use this param to select the conversion method, only take False for better performance (default=True).

        Returns:
            Iterator, tuple iterator over the dataset.

        Examples:
            >>> # dataset is an instance object of Dataset
            >>> iterator = dataset.create_tuple_iterator()
            >>> for item in iterator:
            ...     # item is a list
            ...     print(type(item))
            ...     break
            <class 'list'>
        """
        if output_numpy is None:
            output_numpy = False

        if Dataset._noop_mode():
            return DummyIterator(self, 'tuple')
        return TupleIterator(self, columns, num_epochs, output_numpy, do_copy)

    @check_dict_iterator
    def create_dict_iterator(self, num_epochs=-1, output_numpy=False):
        """
        Create an iterator over the dataset. The data retrieved will be a dictionary datatype.

        The order of the columns in the dictionary may not be the same as the original order.

        Args:
            num_epochs (int, optional): Maximum number of epochs that iterator can be iterated
                (default=-1, iterator can be iterated infinite number of epochs).
            output_numpy (bool, optional): Whether or not to output NumPy datatype,
                if output_numpy=False, iterator will output MSTensor (default=False).

        Returns:
            Iterator, dictionary iterator over the dataset.

        Examples:
            >>> # dataset is an instance object of Dataset
            >>> iterator = dataset.create_dict_iterator()
            >>> for item in iterator:
            ...     # item is a dict
            ...     print(type(item))
            ...     break
            <class 'dict'>
        """
        if output_numpy is None:
            output_numpy = False

        if Dataset._noop_mode():
            return DummyIterator(self, 'dict')
        return DictIterator(self, num_epochs, output_numpy)

    def __iter__(self):
        """Create an iterator over the dataset."""
        return self.create_tuple_iterator(num_epochs=1)

    @property
    def input_indexs(self):
        """
        Get Input Index Information

        Returns:
            int, tuple of the input index information.

        Examples:
            >>> # dataset is an instance object of Dataset
            >>> # set input_indexs
            >>> dataset.input_indexs = 10
            >>> print(dataset.input_indexs)
            10
        """
        if self._input_indexs != ():
            return self._input_indexs

        # find input_indexes of children
        children_input_index = [child.input_indexs for child in self.children]

        # in case of more than one child, return the first input_indexes
        for cix in children_input_index:
            if cix != ():
                return cix

        # if all children's input_indexes are () or the node is a leaf
        return self._input_indexs

    @input_indexs.setter
    def input_indexs(self, value):
        self._input_indexs = value

    def copy_batch_size(self, value):
        self._batch_size = value

    def _init_tree_getters(self):
        """
        Get pipeline information.
        """
        ir_tree, api_tree = self.create_ir_tree()

        runtime_context = cde.PythonRuntimeContext()
        runtime_context.Init()
        getter = cde.TreeGetters()
        getter.Init(ir_tree)
        runtime_context.AssignConsumer(getter)
        return getter, runtime_context, api_tree

    def __init_size_getter(self):
        """
        Get pipeline information.
        """
        ir_tree, api_tree = self.create_ir_tree()

        runtime_context = cde.PythonRuntimeContext()
        runtime_context.Init()
        getter = cde.DatasetSizeGetters()
        getter.Init(ir_tree)
        runtime_context.AssignConsumer(getter)
        return getter, runtime_context, api_tree

    def get_col_names(self):
        """
        Return the names of the columns in dataset.

        Returns:
            list, list of column names in the dataset.

        Examples:
            >>> # dataset is an instance object of Dataset
            >>> col_names = dataset.get_col_names()
        """
        if self._col_names is None:
            runtime_getter = self._init_tree_getters()
            self._col_names = runtime_getter[0].GetColumnNames()
            runtime_getter[2].close_pool()
            runtime_getter[2].notify_watchdog()
        return self._col_names

    def output_shapes(self):
        """
        Get the shapes of output data.

        Returns:
            list, list of shapes of each column.

        Examples:
            >>> # dataset is an instance object of Dataset
            >>> output_shapes = dataset.output_shapes()
        """
        if self.saved_output_shapes is None:
            runtime_getter = self._init_tree_getters()
            # We have a hang problem when two-level pipeline with multiprocessing, we need to extend the life cycle
            # of runtime_context. We found this hang problem only occur on output_types and output_shapes.
            self.runtime_context = runtime_getter[1]
            self.saved_output_shapes = runtime_getter[0].GetOutputShapes()
            self.saved_output_types = runtime_getter[0].GetOutputTypes()
            runtime_getter[2].close_pool()
            runtime_getter[2].notify_watchdog()
            del self.runtime_context
        if self.dynamic_setting[0]:
            self.saved_output_shapes, self.saved_min_shapes, self.saved_max_shapes = self._dynamic_output_shapes()
        return self.saved_output_shapes

    def output_types(self):
        """
        Get the types of output data.

        Returns:
            list, list of data types.

        Examples:
            >>> # dataset is an instance object of Dataset
            >>> output_types = dataset.output_types()
        """
        if self.saved_output_types is None:
            runtime_getter = self._init_tree_getters()
            # We have a hang problem when two-level pipeline with multiprocessing, we need to extend the life cycle
            # of runtime_context. We found this hang problem only occur on output_types and output_shapes.
            self.runtime_context = runtime_getter[1]
            self.saved_output_shapes = runtime_getter[0].GetOutputShapes()
            self.saved_output_types = runtime_getter[0].GetOutputTypes()
            runtime_getter[2].close_pool()
            runtime_getter[2].notify_watchdog()
            del self.runtime_context
        if self.dynamic_setting[0]:
            self.saved_output_shapes, self.saved_min_shapes, self.saved_max_shapes = self._dynamic_output_shapes()
        return self.saved_output_types

    def get_dataset_size(self):
        """
        Return the number of batches in an epoch.

        Returns:
            int, number of batches.

        Examples:
            >>> # dataset is an instance object of Dataset
            >>> dataset_size = dataset.get_dataset_size()
        """
        if self.dataset_size is None:
            runtime_getter = self.__init_size_getter()
            self.dataset_size = runtime_getter[0].GetDatasetSize(False)
            runtime_getter[2].close_pool()
            runtime_getter[2].notify_watchdog()
        return self.dataset_size

    @deprecated("1.5")
    def set_dynamic_columns(self, columns=None):
        """
        Set dynamic shape information of source data, it should be set after the pipeline is defined.

        Args:
            columns (dict): A dict contains shape information of each column in dataset.
                The value of shape[i] is :py:obj:`None` indicates that the data length of shape[i] is dynamic.

        Examples:
            >>> import numpy as np
            >>>
            >>> def generator1():
            ...     for i in range(1, 100):
            ...         yield np.ones((16, i, 83)), np.array(i)
            >>>
            >>> dataset = ds.GeneratorDataset(generator1, ["data1", "data2"])
            >>> dataset.set_dynamic_columns(columns={"data1": [16, None, 83], "data2": []})
        """
        if not isinstance(columns, dict):
            raise TypeError("Pass a dict to set dynamic shape, example: {\"data1\": [16, None, 256]}")
        self.dynamic_setting[0] = True
        self.dynamic_setting[1] = columns

    def dynamic_min_max_shapes(self):
        """
        Get minimum and maximum data length of dynamic source data, for dynamic graph compilation.

        Returns:
            lists, min_shapes, max_shapes of source data.

        Examples:
            >>> import numpy as np
            >>>
            >>> def generator1():
            ...     for i in range(1, 100):
            ...         yield np.ones((16, i, 83)), np.array(i)
            >>>
            >>> dataset = ds.GeneratorDataset(generator1, ["data1", "data2"])
            >>> dataset.set_dynamic_columns(columns={"data1": [16, None, 83], "data2": []})
            >>> min_shapes, max_shapes = dataset.dynamic_min_max_shapes()
        """
        if self.saved_min_shapes is None or self.saved_max_shapes is None:
            self.saved_output_shapes, self.saved_min_shapes, self.saved_max_shapes = self._dynamic_output_shapes()
        return self.saved_min_shapes, self.saved_max_shapes

    @staticmethod
    def __check_dynamic_column_name(dynamic_columns, dataset_columns):
        for column in dynamic_columns:
            if column not in dataset_columns:
                raise RuntimeError("dynamic column [" + column + "] does not match any column in dataset: " +
                                   str(dataset_columns))

    @staticmethod
    def __check_dynamic_column_shape(data, col, dynamic_columns):
        shape_mismatch = "dynamic column [" + col + "] with shape " + str(dynamic_columns[col]) + \
                         " does not match dataset column [" + col + "] with shape " + str(list(data[col].shape))
        if data[col].ndim != len(dynamic_columns[col]):
            raise RuntimeError(shape_mismatch)
        for dim in range(len(dynamic_columns[col])):
            if dynamic_columns[col][dim] is not None and dynamic_columns[col][dim] != data[col].shape[dim]:
                raise RuntimeError(shape_mismatch)

    def _dynamic_output_shapes(self):
        """
        Get dynamic information of source data.

        Returns:
            lists, dynamic_shapes, min_shapes, max_shapes of source data.
        """
        if not self.dynamic_setting[1]:
            raise RuntimeError("dynamic_columns is not set, call set_dynamic_columns() by final Dataset Op.")

        if self.saved_output_shapes is not None and self.saved_min_shapes is not None and \
                self.saved_max_shapes is not None:
            return self.saved_output_shapes, self.saved_min_shapes, self.saved_max_shapes

        logger.warning("Calculating dynamic shape of input data, this will take a few minutes...")
        # Assume data1 shape is dynamic, data2 shape is fix
        dynamic_columns = self.dynamic_setting[1]
        # ["data1", "data2"]
        dataset_columns = self.get_col_names()
        Dataset.__check_dynamic_column_name(dynamic_columns, dataset_columns)

        # Shape[1] of data1 is variable
        # {"data1": {(batch_size, 100, feat_len), (16, 200, 83)}, "data2": {(batch_size, feat_len)}}
        column_shape_set = {col: set() for col in dataset_columns}
        dataset_size_counter = 0
        for data in self.create_dict_iterator(num_epochs=1, output_numpy=True):
            dataset_size_counter += 1
            for col in data.keys():
                if col in dynamic_columns:
                    Dataset.__check_dynamic_column_shape(data, col, dynamic_columns)
                column_shape_set[col].add(tuple(data[col].shape))

        # we get dataset_size after dryrun
        self.dataset_size = dataset_size_counter

        min_shapes, max_shapes, dynamic_shapes = list(), list(), list()
        for col, shape_set in column_shape_set.items():
            if len(shape_set) > 1:
                if col not in dynamic_columns:
                    raise RuntimeError("column [" + col + "] has dynamic shape but not set by set_dynamic_columns()" +
                                       ", shapes of [" + col + "]: " + str(list(shape_set)))
                shape_npy = np.array(list(shape_set))
                max_shape = shape_npy.max(axis=0)
                min_shape = shape_npy.min(axis=0)

                # Set min shape to 1 due to unknown shuffle
                min_shape = np.where(np.equal(dynamic_columns[col], None), 1, min_shape)
                # Set dynamic dim to -1 for ME
                dynamic_shape = np.where(np.equal(dynamic_columns[col], None), -1, dynamic_columns[col])

                max_shapes.append(max_shape.tolist())
                min_shapes.append(min_shape.tolist())
                dynamic_shapes.append(dynamic_shape.tolist())
            else:
                # Also append fix shape to keep order of column shape
                fix_shape = list(list(shape_set)[0])
                max_shapes.append(fix_shape)
                min_shapes.append(fix_shape)
                dynamic_shapes.append(fix_shape)
                if col in dynamic_columns:
                    logger.warning("column [" + col + "] has no dynamic shape but set by set_dynamic_columns()")
                    # Set min shape to 1 due to unknown shuffle
                    min_shapes[-1] = np.where(np.equal(dynamic_columns[col], None), 1, fix_shape).tolist()
                    # Set dynamic dim to -1 for ME
                    dynamic_shapes[-1] = np.where(np.equal(dynamic_columns[col], None), -1, fix_shape).tolist()
        return dynamic_shapes, min_shapes, max_shapes

    def num_classes(self):
        """
        Get the number of classes in a dataset.

        Returns:
            int, number of classes.

        Examples:
            >>> # dataset is an instance object of Dataset
            >>> num_classes = dataset.num_classes()
        """
        if self._num_classes is None:
            runtime_getter = self._init_tree_getters()
            self._num_classes = runtime_getter[0].GetNumClasses()
            runtime_getter[2].close_pool()
            runtime_getter[2].notify_watchdog()
        if self._num_classes == -1:
            return None
        return self._num_classes

    def get_sync_notifiers(self):
        if self.children:
            return self.children[0].get_sync_notifiers()
        return {}

    def disable_sync(self):
        if self.children:
            return self.children[0].disable_sync()
        return {}

    def is_sync(self):
        if self.children:
            return self.children[0].is_sync()
        return False

    def sync_update(self, condition_name, num_batch=None, data=None):
        """
        Release a blocking condition and trigger callback with given data.

        Args:
            condition_name (str): The condition name that is used to toggle sending next row.
            num_batch (Union[int, None]): The number of batches (rows) that are released.
                When num_batch is None, it will default to the number specified by the
                sync_wait operator (default=None).
            data (Any): The data passed to the callback, user defined (default=None).
        """
        if (not isinstance(num_batch, int) and num_batch is not None) or \
                (isinstance(num_batch, int) and num_batch <= 0):
            # throwing exception, disable all sync_wait in pipeline
            self.disable_sync()
            raise RuntimeError("Sync_update batch size can only be positive integer, got : {}.".format(num_batch))
        notifiers_dict = self.get_sync_notifiers()
        if not isinstance(condition_name, str):
            raise TypeError("Argument condition_name with value {} is not of type str, but got {}."
                            .format(condition_name, type(condition_name)))
        if condition_name not in notifiers_dict:
            # throwing exception, disable all sync_wait in pipeline
            self.disable_sync()
            raise RuntimeError("Condition name not found.")
        if num_batch is not None:
            num_batch *= self.get_batch_size()
        notifiers_dict[condition_name](num_batch, data)

    def get_batch_size(self):
        """
        Return the size of batch.

        Returns:
            int, the number of data in a batch.

        Examples:
            >>> # dataset is an instance object of Dataset
            >>> batch_size = dataset.get_batch_size()
        """
        if self._batch_size is None:
            runtime_getter = self._init_tree_getters()
            self._batch_size = runtime_getter[0].GetBatchSize()
        if self._batch_size is None:
            self._batch_size = 1
        return self._batch_size

    def get_repeat_count(self):
        """
        Get the replication times in RepeatDataset (default is 1).

        Returns:
            int, the count of repeat.

        Examples:
            >>> # dataset is an instance object of Dataset
            >>> repeat_count = dataset.get_repeat_count()
        """
        if self._repeat_count is None:
            runtime_getter = self._init_tree_getters()
            self._repeat_count = runtime_getter[0].GetRepeatCount()
        if self._repeat_count is None:
            self._repeat_count = 1
        return self._repeat_count

    def get_class_indexing(self):
        """
        Return the class index.

        Returns:
            dict, a str-to-int mapping from label name to index.
            dict, a str-to-list<int> mapping from label name to index for Coco ONLY. The second number
            in the list is used to indicate the super category.

        Examples:
            >>> # dataset is an instance object of Dataset
            >>> class_indexing = dataset.get_class_indexing()
        """
        if self.children:
            return self.children[0].get_class_indexing()
        return {}

    def reset(self):
        """Reset the dataset for next epoch."""

    def is_shuffled(self):
        """Returns True if the dataset or its children is shuffled."""
        for input_dataset in self.children:
            if input_dataset.is_shuffled():
                return True

        return False

    def is_sharded(self):
        """Returns True if the dataset or its children is sharded."""
        for input_dataset in self.children:
            if input_dataset.is_sharded():
                return True

        return False

    def parse(self, children=None):
        raise NotImplementedError("Dataset has to implement parse method.")

    def post_parse(self, ir_node):
        if self.cache:
            ir_node = ir_node.set_cache_client(self.cache.cache_client)
        if self.num_parallel_workers:
            ir_node = ir_node.set_num_workers(self.num_parallel_workers)

        return ir_node


class VisionBaseDataset(Dataset):
    """
    Abstract class to represent a vision source dataset which produces content to the data pipeline.
    """

    def __init__(self, children=None, num_parallel_workers=None, cache=None):
        super().__init__(children=children, num_parallel_workers=num_parallel_workers, cache=cache)

    def parse(self, children=None):
        raise NotImplementedError("Dataset has to implement parse method.")


class TextBaseDataset(Dataset):
    """
    Abstract class to represent a text source dataset which produces content to the data pipeline.
    """

    def __init__(self, children=None, num_parallel_workers=None, cache=None):
        super().__init__(children=children, num_parallel_workers=num_parallel_workers, cache=cache)

    def parse(self, children=None):
        raise NotImplementedError("Dataset has to implement parse method.")

    def build_vocab(self, columns, freq_range, top_k, special_tokens, special_first):
        """
        Function to create a Vocab from source dataset.
        Desired source dataset is a text type dataset.

        Build a vocab from a dataset. This would collect all the unique words in a dataset and return a vocab
        which contains top_k most frequent words (if top_k is specified)

        Args:

            columns(Union[str, list[str]]): Column names to get words from.
            freq_range(tuple[int]): A tuple of integers (min_frequency, max_frequency). Words within the frequency
                range will be stored.
                Naturally 0 <= min_frequency <= max_frequency <= total_words. min_frequency/max_frequency
                can be set to default, which corresponds to 0/total_words separately.
            top_k(int): Number of words to be built into vocab. top_k most frequent words are
                taken. The top_k is taken after freq_range. If not enough top_k, all words will be taken
            special_tokens(list[str]): A list of strings, each one is a special token.
            special_first(bool): Whether special_tokens will be prepended/appended to vocab, If special_tokens
                is specified and special_first is set to default, special_tokens will be prepended.

        Returns:
            Vocab, vocab built from the dataset.

        Examples:
            >>> import numpy as np
            >>>
            >>> def gen_corpus():
            ...     # key: word, value: number of occurrences, reason for using letters is so their order is apparent
            ...     corpus = {"Z": 4, "Y": 4, "X": 4, "W": 3, "U": 3, "V": 2, "T": 1}
            ...     for k, v in corpus.items():
            ...         yield (np.array([k] * v, dtype='S'),)
            >>> column_names = ["column1"]
            >>> dataset = ds.GeneratorDataset(gen_corpus, column_names)
            >>> dataset = dataset.build_vocab(columns=["column1"],
            ...                               freq_range=(1, 10), top_k=5,
            ...                               special_tokens=["<pad>", "<unk>"],
            ...                               special_first=True)

        """
        vocab = cde.Vocab()
        columns = replace_none(columns, [])
        if not isinstance(columns, list):
            columns = [columns]

        freq_range = replace_none(freq_range, (0, 9223372036854775807))
        if freq_range[0] is None:
            freq_range = (0, freq_range[1])
        if freq_range[1] is None:
            freq_range = (freq_range[0], 9223372036854775807)
        special_tokens = replace_none(special_tokens, [])
        top_k = replace_none(top_k, 9223372036854775807)

        ir_tree, api_tree = self.create_ir_tree()

        # vocab node
        vocab_node = cde.BuildVocabNode(ir_tree, vocab, columns, freq_range, top_k, special_tokens, special_first)

        runtime_context = cde.PythonRuntimeContext()
        runtime_context.Init()

        # build vocab
        consumer = cde.PythonBuildVocabConsumer()
        consumer.Init(vocab_node)
        runtime_context.AssignConsumer(consumer)

        consumer.Start()
        del api_tree

        return vocab

    def build_sentencepiece_vocab(self, columns, vocab_size, character_coverage, model_type, params):
        """
        Function to create a SentencePieceVocab from source dataset.
        Desired source dataset is a text type dataset.

        Args:

            columns(list[str]): Column names to get words from.
            vocab_size(int): Vocabulary size.
            character_coverage(float): Percentage of characters covered by the model, must be between
                0.98 and 1.0 Good defaults are: 0.9995 for languages with rich character sets like
                Japanese or Chinese character sets, and 1.0 for other languages with small character sets
                like English or Latin.
            model_type(SentencePieceModel): Model type. Choose from unigram (default), bpe, char, or word.
                The input sentence must be pretokenized when using word type.
            params(dict): Any extra optional parameters of sentencepiece library according to your raw data

        Returns:
            SentencePieceVocab, vocab built from the dataset.

        Examples:
            >>> from mindspore.dataset.text import SentencePieceModel
            >>>
            >>> # You can construct any text dataset as source, take TextFileDataset as example.
            >>> dataset = ds.TextFileDataset("/path/to/sentence/piece/vocab/file", shuffle=False)
            >>> dataset = dataset.build_sentencepiece_vocab(["text"], 5000, 0.9995, SentencePieceModel.UNIGRAM, {})
        """
        if not isinstance(model_type, SentencePieceModel):
            raise TypeError("Argument model_type with value {0} is not of type SentencePieceModel, but got {1}."\
                            .format(model_type, type(model_type)))
        model_type = DE_C_INTER_SENTENCEPIECE_MODE[model_type]
        vocab = cde.SentencePieceVocab()

        ir_tree, api_tree = self.create_ir_tree()

        # vocab node
        vocab_node = cde.BuildSentenceVocabNode(ir_tree, vocab, columns, vocab_size, character_coverage, model_type,
                                                params)

        runtime_context = cde.PythonRuntimeContext()
        runtime_context.Init()

        # build vocab
        consumer = cde.PythonBuildVocabConsumer()
        consumer.Init(vocab_node)
        runtime_context.AssignConsumer(consumer)

        consumer.Start()
        del api_tree

        return vocab


class AudioBaseDataset(Dataset):
    """
    Abstract class to represent a audio source dataset which produces content to the data pipeline.
    """

    def __init__(self, children=None, num_parallel_workers=None, cache=None):
        super().__init__(children=children, num_parallel_workers=num_parallel_workers, cache=cache)

    def parse(self, children=None):
        raise NotImplementedError("Dataset has to implement parse method.")


class UnionBaseDataset(VisionBaseDataset, TextBaseDataset, AudioBaseDataset):
    """
    Abstract class to represent a union source dataset which produces content to the data pipeline.
    """

    def __init__(self, children=None, num_parallel_workers=None, cache=None):
        super().__init__(children=children, num_parallel_workers=num_parallel_workers, cache=cache)

    def parse(self, children=None):
        raise NotImplementedError("Dataset has to implement parse method.")


class SourceDataset(Dataset):
    """
    Abstract class to represent a source dataset which produces content to the data pipeline.
    """

    def __init__(self, num_parallel_workers=None, num_samples=None, shuffle=True, num_shards=None, shard_id=None,
                 cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, cache=cache)
        self.num_samples = replace_none(num_samples, 0)
        self.num_shards = replace_none(num_shards, 1)
        self.shard_id = replace_none(shard_id, 0)

        if shuffle is not None and not isinstance(shuffle, (bool, Shuffle)):
            raise TypeError("shuffle must be of boolean or enum of 'Shuffle' values like 'Shuffle.GLOBAL' or "
                            "'Shuffle.FILES' or 'Shuffle.INFILE'.")

        self.shuffle_flag = 2  # Global shuffle
        if not isinstance(shuffle, Shuffle):
            if shuffle is None or shuffle:
                self.shuffle_flag = 2  # Global shuffle
            else:
                self.shuffle_flag = 0  # No shuffle
        else:
            if shuffle == Shuffle.GLOBAL:
                self.shuffle_flag = 2  # Global shuffle
            elif shuffle == Shuffle.FILES:
                self.shuffle_flag = 1  # Files shuffle
            elif shuffle == Shuffle.INFILE:
                self.shuffle_flag = 3  # Infile shuffle

    def parse(self, children=None):
        raise NotImplementedError("Dataset has to implement parse method.")

    @staticmethod
    def _find_files(patterns):
        """
        Utility function to search for files with the given glob patterns.

        Args:
            patterns (Union[str, list[str]]): String or list of patterns to be searched.

        Returns:
            list, list of files.
        """

        if not isinstance(patterns, list):
            patterns = [patterns]

        file_list = []
        unmatched_patterns = []
        for pattern in patterns:
            matches = [match for match in glob.glob(pattern, recursive=True) if os.path.isfile(match)]

            if matches:
                file_list.extend(matches)
            else:
                unmatched_patterns.append(pattern)

        if unmatched_patterns:
            raise ValueError("The following patterns did not match any files: {}.".format(unmatched_patterns))

        if file_list:  # not empty
            return file_list
        raise ValueError("The list of path names matching the patterns is empty.")

    def is_shuffled(self):
        return self.shuffle_flag > 0

    def is_sharded(self):
        if self.num_shards is not None:
            return self.num_shards > 1
        return False


class MappableDataset(SourceDataset):
    """
    Abstract class to represent a source dataset which supports use of samplers.
    """

    def parse(self, children=None):
        raise NotImplementedError("Dataset has to implement parse method.")

    def __init__(self, num_parallel_workers=None, sampler=None, num_samples=None, shuffle=None, num_shards=None,
                 shard_id=None, cache=None):
        super().__init__(num_parallel_workers=num_parallel_workers, num_samples=num_samples, shuffle=shuffle,
                         num_shards=num_shards, shard_id=shard_id, cache=cache)
        self.shuffle_flag = replace_none(shuffle, True)
        self.sampler = samplers.select_sampler(num_samples, sampler, shuffle, num_shards, shard_id)

    def add_sampler(self, new_sampler):
        """
        Add a child sampler for the current dataset.

        Args:
            new_sampler (Sampler): The child sampler to be added.

        Examples:
            >>> new_sampler = ds.DistributedSampler(10, 2)
            >>> dataset.add_sampler(new_sampler)  # dataset is an instance of Dataset
        """
        # Note: By adding a sampler, the sampled IDs will flow to the new_sampler
        # after first passing through the current samplers attached to this dataset.
        self.dataset_size = None
        new_sampler.add_child(self.sampler)
        self.sampler = new_sampler

    def use_sampler(self, new_sampler):
        """
        Replace the last child sampler of the current dataset, remaining the parent sampler unchanged.

        Args:
            new_sampler (Sampler): The new sampler to replace with.

        Examples:
            >>> # dataset is an instance object of Dataset
            >>> # use a DistributedSampler instead
            >>> new_sampler = ds.DistributedSampler(10, 2)
            >>> dataset.use_sampler(new_sampler)
        """
        if new_sampler is None:
            raise TypeError("Input sampler can not be None.")
        if not isinstance(new_sampler, (samplers.BuiltinSampler, samplers.Sampler)):
            raise TypeError("Input sampler is not an instance of a sampler.")
        self.dataset_size = None

        self.sampler = self.sampler.child_sampler
        self.add_sampler(new_sampler)

    def is_shuffled(self):
        return self.sampler.is_shuffled()

    def is_sharded(self):
        return self.sampler.is_sharded()

    @check_split
    def split(self, sizes, randomize=True):
        """
        Split the dataset into smaller, non-overlapping datasets.

        Args:
            sizes (Union[list[int], list[float]]): If a list of integers [s1, s2, …, sn] is
                provided, the dataset will be split into n datasets of size s1, size s2, …, size sn
                respectively. If the sum of all sizes does not equal the original dataset size, an
                error will occur.
                If a list of floats [f1, f2, …, fn] is provided, all floats must be between 0 and 1
                and must sum to 1, otherwise an error will occur. The dataset will be split into n
                Datasets of size round(f1*K), round(f2*K), …, round(fn*K) where K is the size of the
                original dataset.
                If after rounding:

                - Any size equals 0, an error will occur.
                - The sum of split sizes < K, the difference will be added to the first split.
                - The sum of split sizes > K, the difference will be removed from the first large
                  enough split such that it will have at least 1 row after removing the difference.

            randomize (bool, optional): Determines whether or not to split the data randomly (default=True).
                If True, the data will be randomly split. Otherwise, each split will be created with
                consecutive rows from the dataset.

        Note:
            1. There is an optimized split function, which will be called automatically when the dataset
               that calls this function is a MappableDataset.
            2. Dataset should not be sharded if split is going to be called. Instead, create a
               DistributedSampler and specify a split to shard after splitting. If the dataset is
               sharded after a split, it is strongly recommended setting the same seed in each instance
               of execution, otherwise each shard may not be part of the same split (see Examples).
            3. It is strongly recommended to not shuffle the dataset, but use randomize=True instead.
               Shuffling the dataset may not be deterministic, which means the data in each split
               will be different in each epoch. Furthermore, if sharding occurs after split, each
               shard may not be part of the same split.

        Raises:
            RuntimeError: If get_dataset_size returns None or is not supported for this dataset.
            RuntimeError: If `sizes` is list of integers and sum of all elements in sizes does not
                equal the dataset size.
            RuntimeError: If `sizes` is list of float and there is a split with size 0 after calculations.
            RuntimeError: If the dataset is sharded prior to calling split.
            ValueError: If `sizes` is list of float and not all floats are between 0 and 1, or if the
                floats don't sum to 1.

        Returns:
            tuple(Dataset), a tuple of datasets that have been split.

        Examples:
            >>> # Since many datasets have shuffle on by default, set shuffle to False if split will be called!
            >>> dataset = ds.ImageFolderDataset(image_folder_dataset_dir, shuffle=False)
            >>>
            >>> # Set the seed, and tell split to use this seed when randomizing.
            >>> # This is needed because sharding will be done later
            >>> ds.config.set_seed(58)
            >>> train_dataset, test_dataset = dataset.split([0.9, 0.1])
            >>>
            >>> # To shard the train dataset, use a DistributedSampler
            >>> train_sampler = ds.DistributedSampler(10, 2)
            >>> train_dataset.use_sampler(train_sampler)
        """
        if self.is_shuffled():
            logger.warning("Dataset is shuffled before split.")

        if self.is_sharded():
            raise RuntimeError("Dataset should not be sharded before split.")

        absolute_sizes = self._get_absolute_split_sizes(sizes)
        splits = []
        current_split_start_index = 0
        for size in absolute_sizes:
            ds = copy.deepcopy(self)
            ds.dataset_size = None
            if randomize:
                # want to shuffle the same way every epoch before split, we are assuming
                # that the user will call set_seed
                random_sampler = samplers.RandomSampler()
                random_sampler.reshuffle_each_epoch = False
                ds.add_sampler(random_sampler)

            subset_sampler = samplers.SequentialSampler(current_split_start_index, size)
            ds.add_sampler(subset_sampler)

            # add sequential sampler, so that if user calls use_sampler, we will
            # get rid of the sequential sampler instead of something we need
            ds.add_sampler(samplers.SequentialSampler())

            splits.append(ds)

            current_split_start_index += size

        return tuple(splits)


class BucketBatchByLengthDataset(UnionBaseDataset):
    """
    The result of applying BucketBatchByLength operator to the input dataset.
    """

    def __init__(self, input_dataset, column_names, bucket_boundaries, bucket_batch_sizes, element_length_function,
                 pad_info, pad_to_bucket_boundary, drop_remainder):
        super().__init__(children=input_dataset)

        self.column_names = to_list(column_names)
        self.bucket_boundaries = replace_none(bucket_boundaries, [])
        self.bucket_batch_sizes = replace_none(bucket_batch_sizes, [])
        self.element_length_function = element_length_function
        self.pad_info = replace_none(pad_info, {})
        self.pad_to_bucket_boundary = replace_none(pad_to_bucket_boundary, False)
        self.drop_remainder = replace_none(drop_remainder, False)

    def parse(self, children=None):
        return cde.BucketBatchByLengthNode(children[0], self.column_names, self.bucket_boundaries,
                                           self.bucket_batch_sizes, self.element_length_function, self.pad_info,
                                           self.pad_to_bucket_boundary, self.drop_remainder)


def _check_shm_usage(num_worker, queue_size, max_rowsize, num_queues=1):
    """
    Check sufficient shared memory is available for shared memory queues
    when training in parallel mode.
    """
    threshold_ratio = 0.8
    if platform.system().lower() not in {"windows", "darwin"}:
        device_num = _get_device_num()
        # In the cluster, _get_device_num indicates the number of the entire cluster. The maximum number of cards
        # on the ascend server is 8.
        if device_num > 1 and context.get_context("device_target") == "Ascend":
            device_num = min(device_num, 8)
        shm_estimate_usage = device_num * num_worker * num_queues * \
                             (queue_size + 2) * max_rowsize * 1024 * 1024
        try:
            shm_available = psutil.disk_usage('/dev/shm').free
            if shm_estimate_usage >= threshold_ratio * shm_available:
                raise RuntimeError(
                    "Insufficient shared memory available. Required: {}, Available: {}. "
                    "The required memory can't exceed 80% of the available shared memory, "
                    "it's recommended to reduce memory usage by following methods:\n"
                    "1. reduce value of parameter max_rowsize or num_parallel_workers.\n"
                    "2. reduce prefetch size by set_prefetch_size().\n"
                    "3. disable shared memory by set_enable_shared_mem()."
                    .format(shm_estimate_usage, shm_available))
        except FileNotFoundError:
            raise RuntimeError("Expected /dev/shm to exist.")


class BatchDataset(UnionBaseDataset):
    """
    The result of applying Batch operator to the input dataset.

    Args:
        input_dataset (Dataset): Input Dataset to be batched.
        batch_size (Union[int, function]): The number of rows each batch is created with. An
            int or callable which takes exactly 1 parameter, BatchInfo.
        drop_remainder (bool, optional): Determines whether or not to drop the last
            possibly incomplete batch (default=False). If True, and if there are less
            than batch_size rows available to make the last batch, then those rows will
            be dropped and not propagated to the child node.
        num_parallel_workers (int, optional): Number of workers to process the dataset in parallel (default=None).
        per_batch_map (callable, optional): Per batch map callable. A callable which takes
            (list[Tensor], list[Tensor], ..., BatchInfo) as input parameters. Each list[Tensor] represents a batch of
            Tensors on a given column. The number of lists should match with number of entries in input_columns. The
            last parameter of the callable must always be a BatchInfo object.
        input_columns (Union[str, list[str]], optional): List of names of the input columns. The size of the list must
            match with signature of per_batch_map callable.
        output_columns (Union[str, list[str]], optional): List of names assigned to the columns outputted by
            the last operation. This parameter is mandatory if len(input_columns) !=
            len(output_columns). The size of this list must match the number of output
            columns of the last operation. (default=None, output columns will have the same
            name as the input columns, i.e., the columns will be replaced).
        column_order (Union[str, list[str]], optional): Specifies the list of all the columns you need in the whole
                dataset. The parameter is required when len(input_column) != len(output_column). Caution: the list here
                is not just the columns specified in parameter input_columns and output_columns.
        pad_info (dict, optional): Whether to perform padding on selected columns. pad_info={"col1":([224,224],0)}
            will pad column with name "col1" to a tensor of size [224,224] and fill the missing with 0.
        max_rowsize(int, optional): Maximum size of row in MB that is used for shared memory allocation to copy
            data between processes.  This is only used if python_multiprocessing is set to True (default=16).

    """

    def __init__(self, input_dataset, batch_size, drop_remainder=False, num_parallel_workers=None, per_batch_map=None,
                 input_columns=None, output_columns=None, column_order=None, pad_info=None,
                 python_multiprocessing=False, max_rowsize=16):
        super().__init__(children=input_dataset, num_parallel_workers=num_parallel_workers)

        if BatchDataset._is_ancestor_of_repeat(input_dataset):
            logger.warning("Repeat is located before batch, data from two epochs can be batched together.")

        BatchDataset._update_batch_size_for_syncwait(input_dataset, batch_size)

        # if batch_size is callable, set batch_size to 1 and batch_size_func to that callable function
        self.batch_size = batch_size if not callable(batch_size) else 1
        self.batch_size_func = None if not callable(batch_size) else batch_size

        self.drop_remainder = replace_none(drop_remainder, False)

        self.per_batch_map = per_batch_map

        self.input_columns = to_list(input_columns)
        self.output_columns = to_list(output_columns)
        self.column_order = to_list(column_order)

        self.pad = bool(pad_info is not None)
        self.pad_info = replace_none(pad_info, dict())

        self.python_multiprocessing = python_multiprocessing
        self.process_pool = None
        self.max_rowsize = max_rowsize

    def parse(self, children=None):
        return cde.BatchNode(children[0], self.batch_size, self.drop_remainder, self.pad, self.input_columns,
                             self.output_columns, self.column_order, self.batch_size_func, self.per_batch_map,
                             self.pad_info, self.process_pool)

    @staticmethod
    def _is_ancestor_of_repeat(dataset):
        """
        Utility function to find the case where repeat is used before batch.

        Args:
             dataset (Dataset): Dataset to be checked.

        Returns:
            bool, whether repeat is used before batch.
        """
        if isinstance(dataset, RepeatDataset):
            return True
        flag = False
        for input_dataset in dataset.children:
            flag = flag | BatchDataset._is_ancestor_of_repeat(input_dataset)
        return flag

    @staticmethod
    def _update_batch_size_for_syncwait(dataset, batch_size):
        """
        Utility function to notify batch size to sync_wait.

        Args:
             dataset (Dataset): Dataset to be checked.
             batch_size (int): batch size to notify.
        """
        if isinstance(dataset, SyncWaitDataset):
            dataset.update_sync_batch_size(batch_size)
        for input_dataset in dataset.children:
            BatchDataset._update_batch_size_for_syncwait(input_dataset, batch_size)

    def __deepcopy__(self, memodict):
        return self.__safe_deepcopy__(memodict, exclude=("per_batch_map", "batch_size_func", "__transfer_dataset__"))

    # Iterator bootstrap will be called on iterator construction.
    # A deep copy of Dataset object is created prior of iterator_bootstrap.
    # This method will create per iterator process pool and bind pyfunc execution to the pool.
    def iterator_bootstrap(self):
        """
        Per iterator bootstrap callback.
        """
        if self.python_multiprocessing:
            if self.per_batch_map is None:
                logger.warning("per_batch_map is None so python_multiprocessing is ignored for batch.")
                return

            # If user didn't specify num_parallel_workers, set it to default
            if self.num_parallel_workers is None:
                self.num_parallel_workers = get_num_parallel_workers()

            self.process_pool = _PythonMultiprocessing(str(self), self.num_parallel_workers, [self.per_batch_map],
                                                       self.max_rowsize * self.batch_size)
            # Wrap per_batch_map into _PythonCallable
            self.per_batch_map = _PythonCallable(self.per_batch_map, 0, self.process_pool)
        else:
            if self.per_batch_map is not None:
                self.per_batch_map = FuncWrapper(self.per_batch_map)


class BatchInfo(cde.CBatchInfo):
    """
    Only the batch size function and per_batch_map of the batch operator can dynamically adjust parameters
    based on the number of batches and epochs during training.
    """

    def get_batch_num(self):
        """
        Return the batch number of the current batch.
        """
        return

    def get_epoch_num(self):
        """
        Return the epoch number of the current batch.
        """
        return


class BlockReleasePair:
    """
    The blocking condition class used by SyncWaitDataset.

    Args:
        init_release_rows (int): Number of lines to allow through the pipeline.
        callback (function): The callback function that will be called when release is called (default=None).
    """

    def __init__(self, init_release_rows, callback=None):
        if isinstance(init_release_rows, int) and init_release_rows <= 0:
            raise ValueError("release_rows need to be greater than 0.")
        self.row_count = -init_release_rows
        self.cv = threading.Condition()
        self.callback = callback
        self.default_rows = init_release_rows
        self.disable = False

    def __deepcopy__(self, memodict):
        return self

    def reset(self):
        with self.cv:
            self.row_count = -self.default_rows
            self.cv.notify_all()

    def update_batched_size(self, batch_size):
        # sanity check
        if isinstance(batch_size, int) and batch_size <= 0:
            raise ValueError("batch_size need to be greater than 0.")

        # should only use before the pipeline creates
        self.row_count *= batch_size
        self.default_rows *= batch_size

    def block_func(self):
        """
        Function for handing blocking condition.

        Returns:
            bool, True.
        """
        with self.cv:
            # if disable is true, the always evaluate to true
            not_time_out = self.cv.wait_for(lambda: (self.row_count < 0 or self.disable),
                                            timeout=get_callback_timeout())
            # time_out will be False if time out occurs
            if not not_time_out:
                logger.warning("Timeout happened in sync_wait, maybe dataset.sync_update(condition=...) "
                               "is not added after dataset.create_dict_iterator(...), now disabling lock.")
                self.disable = True
            self.row_count += 1
        return True

    def release_func(self, pass_rows=None, data=None):
        with self.cv:
            if pass_rows is None:
                pass_rows = self.default_rows
            self.row_count -= pass_rows
            if self.callback is not None:
                self.callback(data)
            self.cv.notify_all()

    def disable_lock(self):
        with self.cv:
            self.disable = True
            self.cv.notify_all()


class SyncWaitDataset(UnionBaseDataset):
    """
    The result of adding a blocking condition to the input Dataset.

    Args:
        input_dataset (Dataset): Input dataset to apply flow control.
        num_batch (int): Number of batches without blocking at the start of each epoch.
        condition_name (str): Condition name that is used to toggle sending next row.
        callback (function): Callback function that will be invoked when sync_update is called (default=None).

    Raises:
        RuntimeError: If condition name already exists.
    """

    def __init__(self, input_dataset, condition_name, num_batch, callback=None):
        super().__init__(children=input_dataset)

        # set to the default value, waiting for the batch to update it
        self._condition_name = condition_name
        if isinstance(num_batch, int) and num_batch <= 0:
            raise ValueError("num_batch need to be greater than 0.")

        self._pair = BlockReleasePair(num_batch, callback)
        if self._condition_name in self.children[0].get_sync_notifiers():
            raise RuntimeError("Condition name is already in use.")
        logger.info("Please remember to add dataset.sync_update(condition=%s), otherwise hanging will result. "
                    "If dataset.sync_update(condition=%s) has already been added, you can ignore the info.",
                    condition_name, condition_name)

    def parse(self, children=None):
        return cde.SyncWaitNode(children[0], self._condition_name, self._pair.block_func)

    def get_sync_notifiers(self):
        return {**self.children[0].get_sync_notifiers(), **{self._condition_name: self._pair.release_func}}

    def is_sync(self):
        return True

    def update_sync_batch_size(self, batch_size):
        if isinstance(batch_size, int) and batch_size <= 0:
            raise ValueError("num_batch need to be greater than 0.")
        self._pair.update_batched_size(batch_size)

    def disable_sync(self):
        logger.info("Disabling Sync")
        self._pair.disable_lock()

    @staticmethod
    def _is_ancestor_of_batch(dataset):
        """
        Utility function to find the case where sync_wait is used before batch.

        Args:
             dataset (Dataset): Dataset to be checked.

        Returns:
            bool, whether sync_wait is used before batch.
        """
        if isinstance(dataset, BatchDataset):
            return True
        flag = False
        for input_dataset in dataset.children:
            flag = flag | SyncWaitDataset._is_ancestor_of_batch(input_dataset)
        return flag

    def iterator_bootstrap(self):
        self._pair.reset()


class ShuffleDataset(UnionBaseDataset):
    """
    The result of applying Shuffle operator to the input Dataset.

    Args:
        input_dataset (Dataset): Input Dataset to be shuffled.
        buffer_size (int): Size of the buffer.

    Raises:
        RuntimeError: If exist sync operators before shuffle.
    """

    def __init__(self, input_dataset, buffer_size):
        super().__init__(children=input_dataset)
        self.buffer_size = buffer_size
        self.reshuffle_each_epoch = True

        if self.is_sync():
            raise RuntimeError("No shuffle after sync operators.")

    def parse(self, children=None):
        return cde.ShuffleNode(children[0], self.buffer_size, self.reshuffle_each_epoch)

    def is_shuffled(self):
        return True


# Pyfunc collection for multiprocess pyfunc
# This global variable will only be used within subprocesses
_GLOBAL_PYFUNC_LIST = []
_ARGS_QUEUE = []
_RET_QUEUE = []
_OP_NAME = dict()
_OP_PROCESS = dict()
_LOCK = threading.Lock()


# Pyfunc worker init function
# Python multiprocessing library forbid sending lambda function through pipe.
# This init function allow us to add all Python function to a global collection and then fork afterwards.
def _pyfunc_worker_init(pyfunc_list, args_queue, ret_queue):
    # Some threads in multiprocess.pool can't process sigint signal,
    # and will occur hang problem, so ctrl+c will pass to parent process.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    global _GLOBAL_PYFUNC_LIST
    global _ARGS_QUEUE
    global _RET_QUEUE
    _GLOBAL_PYFUNC_LIST = pyfunc_list
    _ARGS_QUEUE = args_queue
    _RET_QUEUE = ret_queue


# Pyfunc worker execution function
# All exceptions will be raised to main processes
def _pyfunc_worker_exec(index, qid, *args):
    """
    Internal function for call certain pyfunc in Python process.
    """
    # Some threads in multiprocess.pool can't process sigint signal,
    # and will occur hang problem, so ctrl+c will pass to parent process.
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    if qid != -1:
        # Pass arguments through the Queue instead of directly to remote process
        args = _ARGS_QUEUE[qid].get()
        try:
            r = _GLOBAL_PYFUNC_LIST[index](*args)
        except Exception:
            return ExceptionHandler(where="in map(or batch) worker and execute python function")
        if isinstance(r, tuple):
            _RET_QUEUE[qid].put(r)
        else:
            _RET_QUEUE[qid].put((r,))
        return [qid]
    # not using shared memory for passing arguments, call function directly
    result = None
    try:
        result = _GLOBAL_PYFUNC_LIST[index](*args)
    except Exception:
        result = ExceptionHandler(where="in map(or batch) worker and execute python function")
    return result


# PythonCallable wrapper for multiprocess pyfunc
class _PythonCallable:
    """
    Internal Python function wrapper for multiprocessing pyfunc.
    """

    def __init__(self, py_callable, idx, pool=None):
        # Original Python callable from user.
        self.py_callable = py_callable
        # Process pool created for current iterator.
        self.pool = pool
        # Python callable index for subprocess _GLOBAL_PYFUNC_LIST
        self.idx = idx

    def __call__(self, *args):
        if self.pool.is_running() and check_iterator_cleanup() is False:
            try:
                return self.pool.execute(self.py_callable, self.idx, *args)
            except multiprocessing.TimeoutError:
                return self.py_callable(*args)
        # Invoke original Python callable in master process in case the pool is gone.
        return self.py_callable(*args)

    def to_json(self):
        return self.py_callable.to_json()


class _PythonMultiprocessing(cde.PythonMultiprocessingRuntime):
    """
    A wrapper to multiprocessing.pool that performs cleanup and ensure proper termination of forked processes.
    """

    class _ExceptHookHandler:
        """
        Internal class ExceptionHandler
        """

        def __init__(self):
            sys.excepthook = self.__handler_exception

        @staticmethod
        def mp_pool_exit_preprocess():
            if check_iterator_cleanup() is False:
                # Set the iterator_cleanup flag to True before exiting, and wait 3s for all apply_async
                # applied to the multiprocessing task to prevent multiprocessing from hang when exiting
                _set_iterator_cleanup()
                time.sleep(3)

        def __handler_exception(self, ex_type, value, tb):
            logger.critical("Uncaught exception: ", exc_info=(ex_type, value, tb))
            self.mp_pool_exit_preprocess()

    def __init__(self, op_name, num_parallel_workers, operations, max_row_size=16):
        super(_PythonMultiprocessing, self).__init__()
        self.op_name = op_name
        self.num_parallel_workers = num_parallel_workers
        self.operations = operations
        self.max_row_size = max_row_size

        self.process_pool = None
        self.op_id = -1

        self.arg_q_list = []
        self.res_q_list = []
        self.queues_map = {}
        self.next_queue = 0

        self.eot = None
        self.watch_dog = None
        self.workers = []
        self.ppid = os.getpid()
        self.hook = None

    def launch(self, op_id=-1):
        self.op_id = op_id
        logger.info("Launching new Python Multiprocessing pool for Op:" + str(self.op_id))
        self.create_pool()

    def create_pool(self):
        """

        Returns:

        """
        if get_enable_shared_mem():
            self.create_shared_memory()

        if self.process_pool is not None:
            raise Exception("Pool was already created, close it first.")

        # Let gc collect unrefrenced memory to avoid child processes in the pool to do it
        gc.collect()
        # Construct python multiprocessing pool.
        # The _pyfunc_worker_init is used to pass lambda function to subprocesses.
        self.process_pool = multiprocessing.Pool(processes=self.num_parallel_workers,
                                                 initializer=_pyfunc_worker_init,
                                                 initargs=(self.operations,
                                                           self.arg_q_list, self.res_q_list))

        self.gather_workers_info()

        self.hook = _PythonMultiprocessing._ExceptHookHandler()

        # The op (Map, Batch, etc) multiprocessing will launch a watch dog thread for monitoring sub processes
        self._launch_watch_dog()

        atexit.register(self.hook.mp_pool_exit_preprocess)
        # If Python version greater than 3.8, we need to close ThreadPool in atexit for unclean pool teardown.
        if sys.version_info >= (3, 8):
            atexit.register(self.process_pool.close)

    def terminate(self):
        logger.info("Terminating Python Multiprocessing pool for Op:" + str(self.op_id))
        self.close_pool()
        self.abort_watchdog()
        self.delete_shared_memory()
        self.process_pool = None

    def get_pids(self):
        # obtain process IDs from multiprocessing.pool
        return [w.pid for w in self.workers]

    def add_new_workers(self, num_new_workers):
        logger.info(
            "Increasing num_parallel_workers of Python Multiprocessing pool for Op:" + str(self.op_id) +
            ", old num_workers=" + str(self.num_parallel_workers) + " new num_workers" + str(self.num_parallel_workers +
                                                                                             num_new_workers) + ".")
        self.terminate()
        self.num_parallel_workers += num_new_workers
        self.launch(self.op_id)

    def remove_workers(self, num_removed_workers):
        logger.info(
            "Decreasing num_parallel_workers of Python Multiprocessing pool for Op:" + str(self.op_id) +
            ", old num_workers=" + str(self.num_parallel_workers) + " new num_workers" + str(self.num_parallel_workers -
                                                                                             num_removed_workers) + ".")
        self.terminate()
        self.num_parallel_workers -= num_removed_workers
        self.launch(self.op_id)

    def is_mp_enabled(self):
        return self.process_pool is not None

    def create_shared_memory(self):
        _check_shm_usage(self.num_parallel_workers, 1, self.max_row_size, 2)
        self.arg_q_list = []
        self.res_q_list = []
        self.queues_map = {}
        self.next_queue = 0

        for _ in range(self.num_parallel_workers):
            self.arg_q_list.append(_SharedQueue(1, max_rowsize=self.max_row_size))
            self.res_q_list.append(_SharedQueue(1, max_rowsize=self.max_row_size))

    def delete_shared_memory(self):
        """
        Call this method to delete any shared memory created for this pool.
        """
        if hasattr(self, 'arg_q_list') and self.arg_q_list is not None:
            arg_q_list_len = len(self.arg_q_list)
            for idx in range(arg_q_list_len):
                del self.arg_q_list[arg_q_list_len - idx - 1]
            del self.arg_q_list

        if hasattr(self, 'res_q_list') and self.res_q_list is not None:
            res_q_list_len = len(self.res_q_list)
            for idx in range(res_q_list_len):
                del self.res_q_list[res_q_list_len - idx - 1]
            del self.res_q_list

        #  recreate the lists for next pool creation
        self.arg_q_list = []
        self.res_q_list = []

    def gather_workers_info(self):
        """
        Collect the PIDs of the children processes.
        """
        self.workers = [w for w in self.process_pool._pool]  # pylint: disable=W0212
        pids = self.get_pids()
        logger.info("Op: " + str(self.op_id) + " Python multiprocessing pool workers' PIDs: " + str(pids))

    def execute(self, py_callable, idx, *args):
        """
        Execute
        """
        if self.is_running() and check_iterator_cleanup() is False:
            result, qid, ret = self._send(py_callable, idx, *args)
            if ret:
                return result

            # todo this check might be wrong
            while check_iterator_cleanup() is False:
                try:
                    return self._receive(result, qid)
                except multiprocessing.TimeoutError:
                    continue
                except KeyboardInterrupt:
                    _set_iterator_cleanup()
                    self.close_pool()
                    raise Exception("Multiprocess Op worker receives KeyboardInterrupt.")
            return (None,)
        return None

    def _send(self, py_callable, idx, *args):
        """
        The map/batch operator will use multiprocessing-pool apply_async interface to execute python function
        in a sub process, apply_async will release GIL temporarily. For better performance, we use shared memory
        feature and pass shared queue instead of multiprocess args.
        """
        ret = False
        qid = None
        if self.arg_q_list:
            tid = threading.get_ident()
            # Need to register each thread to use a different queue to send data to pool
            if tid not in self.queues_map:
                qid = self.next_queue
                self.next_queue += 1
                self.queues_map[tid] = qid
            else:
                qid = self.queues_map[tid]
            self.arg_q_list[qid].put(args)

            # This call will send the tensors along with Python callable index to the process pool.
            # Block, yield GIL. Current thread will reacquire GIL once result is returned.
            if self.is_running() and check_iterator_cleanup() is False:
                result = self.process_pool.apply_async(_pyfunc_worker_exec, [idx, qid, []])
            else:
                ret = True
                result = py_callable(*args)
        else:
            result = self.process_pool.apply_async(_pyfunc_worker_exec, [idx, -1, *args])
        return result, qid, ret

    def _receive(self, result, qid):
        """
        The map/batch operator will use multiprocessing-pool get interface to sync output data from a sub process,
        get interface will reacquire GIL. For better performance, we use shared memory feature and get data from
        shared queue directly.
        """
        if self.arg_q_list:
            r = result.get(30)
            if isinstance(r, ExceptionHandler):
                r.reraise()
            if r[0] != qid:
                raise Exception("In PyCallable, got results from wrong thread")
            r = self.res_q_list[qid].get()
            return r
        r = result.get(30)
        if isinstance(r, ExceptionHandler):
            r.reraise()
        return r

    # This wait function is for cleaning zombie subprocesses
    @staticmethod
    def wait_pid():
        """
        This function is used by the main process to release subprocess resources.
        """
        try:
            while True:
                child_pid, _ = os.waitpid(-1, os.WNOHANG)
                if child_pid == 0:
                    break
        except OSError:
            # waitpid may be failed for some reasons so we ignore this error
            pass

    # Dataset need watch_dog thread to monitoring fork multi-processing,
    # and thread can't be a member function otherwise python won't collect and release resources.
    @staticmethod
    def _watch_dog(eot, workers, pool=None):
        """
        This thread is for monitoring subprocesses forked by GeneratorDataset/map/batch
        """
        if not isinstance(workers, list):
            raise TypeError("[Internal Error] The 2nd parameter of watch dog thread should be list of process, " \
                            "but got {}.".format(type(workers)))
        if pool is not None and not isinstance(pool, multiprocessing.pool.Pool):
            raise TypeError("[Internal Error] The 3rd parameter of watch dog thread should be multiprocessing.Pool, " \
                            "but got {}".format(type(pool)))
        while not eot.is_set():
            clear_subprocess_timeout = 0
            # Monitoring and count how many subprocesses already exit
            clear_subprocess_timeout = _PythonMultiprocessing._monitor_subprocess_exit(workers)
            # If find subprocess exit, we will wait for 30s and do some waitpid operations
            if clear_subprocess_timeout > 0:
                if pool is not None:
                    # Python multiprocessing.pool has a bug, if sub process of pool is killed, pool will launch
                    # a new sub process, so we have to set worker_handler._state to TERMINATE to stop relaunching.
                    if pool._state == RUN:  # pylint: disable=W0212
                        pool._state = TERMINATE  # pylint: disable=W0212
                        pool._worker_handler._state = TERMINATE  # pylint: disable=W0212
                        pool._worker_handler.join()  # pylint: disable=W0212
                start = time.time()
                while time.time() - start < clear_subprocess_timeout:
                    # We need to distinguishing get_dataset_size or train finished normally and hang scenario.
                    # If get_dataset_size or train finished normally, _stop_subprocess can be execute and
                    # self.need_abort can be set to True. If main process is hang in get(), self.need_abort
                    # will never set to True, then we wait for 30s and kill main process
                    if eot.is_set():
                        return
                    # Sometimes subprocess may be zombie, so in 30s we can wait and do some useful tasks(waitpid).
                    _PythonMultiprocessing.wait_pid()
                # multiprocessing.queue may hang in .get() forever when put() process was killed.
                # We have to exit main process otherwise main process will hang.
                if pool is not None:
                    _PythonMultiprocessing._terminate_process(pool._pool)  # pylint: disable=W0212
                else:
                    _PythonMultiprocessing._terminate_process(workers)
                logger.critical("The subprocess of dataset may exit unexpected or be killed, "
                                "main process will exit. If this is not an artificial operation, you can use "
                                "ds.config.set_enable_watchdog(False) to block this error.")
                os.kill(os.getpid(), signal.SIGTERM)

    @staticmethod
    # Terminate subprocess launched by multiprocessing.pool
    def _terminate_process(workers):
        for w in workers:
            if w.exitcode is None:
                w.terminate()
        for w in workers:
            if w._closed is False:  # pylint: disable=W0212
                # We don't use w.join because join can only used in main process or join will raise an error.
                w._popen.wait()  # pylint: disable=W0212

    # Monitor the exit number of subprocesses
    @staticmethod
    def _monitor_subprocess_exit(workers):
        """
        To monitor whether process is exit.

        Args:
            workers (list of multiprocessing.Process): multiprocessing.Process.

        Returns:
            int, the timeout(in seconds) when process exit.
        """
        for w in workers:
            exit_code = w.exitcode
            if exit_code is not None:
                # For kill -9, we can exit quickly
                if exit_code == -9:
                    return 1
                # For kill -15, we still exit after 30s
                if exit_code == -15:
                    return 30
        return 0

    # Monitor the exit status of main process
    @staticmethod
    def process_still_alive(ppid):
        """
        We always hit dead lock when we use psutil or w.exitcode to check whether a process is still alive. So we use
        os.kill(ppid, 0) as the best solution when we want to check whether process is still alive.
        """
        try:
            os.kill(ppid, 0)
        except OSError:
            return False
        return True

    # When main process exit, subprocesses will be terminate
    @staticmethod
    def _clean_process(ppid, workers, pool=None):
        """
        This is the execute function of clean process, if we found main process is exit, we will clean subprocesses.

        :param ppid: The process id of main process.
        :param workers: The list of subprocesses.
        :param pool: multiprocessing.Pool object, we can get list of subprocesses from _pool.
        """
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        while _PythonMultiprocessing.process_still_alive(ppid):
            time.sleep(0.1)
        if pool is not None:
            # Python multiprocessing.pool has a bug, if sub process of pool is killed, pool will launch
            # a new sub process, so we have to set worker_handler._state to TERMINATE to stop relaunching.
            # But this pool is not the same object as it in main process, so we don't support kill main process then
            # kill subprocess.
            if pool._state == RUN:  # pylint: disable=W0212
                pool._state = TERMINATE  # pylint: disable=W0212
                pool._worker_handler._state = TERMINATE  # pylint: disable=W0212
                pool._worker_handler.join()  # pylint: disable=W0212
        if pool is not None:
            _PythonMultiprocessing._terminate_process(pool._pool)  # pylint: disable=W0212
        else:
            _PythonMultiprocessing._terminate_process(workers)
        os.kill(os.getpid(), signal.SIGTERM)

    def _launch_watch_dog(self):
        """
        We will launch a watchdog thread and a clean process to cleaning subprocess when there is process was killed.
        The watchdog thread will cleanup subprocesses and main process when one of the subprocesses was killed.
        The cleaning subprocess will cleanup subprocesses when main process was killed.
        """
        if platform.system().lower() != 'windows':
            self.cleaning_process = multiprocessing.Process(target=self._clean_process,
                                                            args=(self.ppid, self.workers, self.process_pool))
            self.cleaning_process.daemon = True
            self.cleaning_process.start()

            if get_enable_watchdog():
                self.eot = threading.Event()
                self.watch_dog = threading.Thread(target=self._watch_dog,
                                                  args=(self.eot, self.workers + [self.cleaning_process],
                                                        self.process_pool))
                self.watch_dog.daemon = True
                self.watch_dog.start()

    def _abort_watchdog(self):
        if not self.eot.is_set():
            self.eot.set()

    def abort_watchdog(self):
        if hasattr(self, 'watch_dog') and self.watch_dog is not None and hasattr(self, 'eot') and self.eot is not None:
            self._abort_watchdog()
        if hasattr(self, 'cleaning_process') and self.cleaning_process is not None:
            _PythonMultiprocessing._terminate_process([self.cleaning_process])

    def is_running(self):
        # note here: the RUN state of python3.7 and python3.8 is different:
        # python3.7: RUN = 0
        # python3.8: RUN = "RUN"
        # so we use self.pool._state == RUN instead and we can't use _state == 0 any more.
        if self.process_pool is not None and self.process_pool._state == RUN:  # pylint: disable=W0212
            return True
        return False

    def close_pool(self):
        if hasattr(self, 'process_pool') and self.process_pool is not None:
            self.process_pool.close()
            self.process_pool.join()

    def __del__(self):
        # Cleanup when the iter had been deleted from ITERATORS_LIST
        self.terminate()


class MapDataset(UnionBaseDataset):
    """
    The result of applying the Map operator to the input Dataset.

    Args:
        input_dataset (Dataset): Input Dataset to be mapped.
        operations (Union[list[TensorOperation], list[functions]]): A function mapping a nested structure of tensors
            to another nested structure of tensor (default=None).
        input_columns (Union[str, list[str]]): List of names of the input columns
            (default=None, the operations will be applied on the first columns in the dataset).
            The size of the list should match the number of inputs of the first operator.
        output_columns (Union[str, list[str]], optional): List of names of the output columns.
            The size of the list should match the number of outputs of the last operator
            (default=None, output columns will be the input columns, i.e., the columns will
            be replaced).
        column_order (list[str], optional): Specifies the list of all the columns you need in the whole
            dataset. The parameter is required when len(input_column) != len(output_column). Caution: the list here
            is not just the columns specified in parameter input_columns and output_columns.
        num_parallel_workers (int, optional): Number of workers to process the dataset
            in parallel (default=None).
        python_multiprocessing (bool, optional): Parallelize Python operations with multiple worker process. This
            option could be beneficial if the Python operation is computational heavy (default=False).
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing.
            (default=None, which means no cache is used).
        callbacks (DSCallback, list[DSCallback], optional): List of Dataset callbacks to be called (Default=None)
        max_rowsize(int, optional): Maximum size of row in MB that is used for shared memory allocation to copy
            data between processes.  This is only used if python_multiprocessing is set to True (default=16).
        offload (bool, optional): Flag to indicate whether offload is used (Default=None).

    Raises:
        ValueError: If len(input_columns) != len(output_columns) and column_order is not specified.
    """

    def __init__(self, input_dataset, operations=None, input_columns=None, output_columns=None, column_order=None,
                 num_parallel_workers=None, python_multiprocessing=False, cache=None, callbacks=None, max_rowsize=16,
                 offload=None):
        super().__init__(children=input_dataset, num_parallel_workers=num_parallel_workers, cache=cache)
        self.operations = to_list(operations)
        for op in self.operations:
            # user define c_vision.HWC2CHW without parentheses is error
            if type(op) == type:  # pylint: disable=unidiomatic-typecheck
                raise ValueError("Parameter operations's element of method map should be a dataset processing "
                                 "operation instance, but got: {}. It may be missing parentheses for "
                                 "instantiation.".format(op))
            if not isinstance(op, (c_transforms.TensorOperation, py_transforms.PyTensorOperation)) \
                    and not callable(op):
                raise ValueError("Parameter operations's element of method map should be a python function or "
                                 "class method which should be callable, but got: {}. It doesn't need parentheses "
                                 "for python function or class method.".format(op))
        self.operations = py_transforms.Compose.reduce(self.operations)
        self.input_columns = to_list(input_columns)
        self.output_columns = to_list(output_columns)
        self.column_order = replace_none(column_order, [])

        #  If output_columns were not provided then use input_columns
        self.output_columns = self.input_columns if not self.output_columns else self.output_columns

        if self.input_columns and self.output_columns \
                and len(self.input_columns) != len(self.output_columns) \
                and not self.column_order:
            raise ValueError("When length of input_columns and output_columns are not equal,"
                             " column_order must be specified.")

        self.python_multiprocessing = python_multiprocessing
        self.process_pool = None

        self.callbacks = to_list(callbacks)
        self.max_rowsize = max_rowsize
        self.offload = offload

    def parse(self, children=None):
        operations = []
        for op in self.operations:
            if op and getattr(op, 'parse', None):
                operations.append(op.parse())
            else:
                operations.append(op)

        callbacks = [cb.create_runtime_obj() for cb in self.callbacks]
        return cde.MapNode(children[0], operations, self.input_columns, self.output_columns, self.column_order,
                           callbacks, self.max_rowsize, OffloadToManualOffloadMode.get(self.offload), self.process_pool)

    def __deepcopy__(self, memodict):
        return self.__safe_deepcopy__(memodict, exclude=("operations", "callbacks", "__transfer_dataset__"))

    # Iterator bootstrap will be called on iterator construction.
    # A deep copy of Dataset object is created prior of iterator_bootstrap.
    # This method will create per iterator process pool and bind pyfunc execution to the pool.
    def iterator_bootstrap(self):
        """
        Per iterator bootstrap callback.
        """
        if self.python_multiprocessing:
            iter_specific_operations = []
            callable_list = []

            # If user didn't specify num_parallel_workers, set it to default
            if self.num_parallel_workers is None:
                self.num_parallel_workers = get_num_parallel_workers()

            # Pass #1, look for Python callables and build list
            for op in self.operations:
                # our c transforms is now callable and should not be run in Python multithreading
                if MapDataset.__operation_valid_for_multiprocessing(op):
                    callable_list.append(op)

            if callable_list:
                self.process_pool = _PythonMultiprocessing(str(self), self.num_parallel_workers, callable_list,
                                                           self.max_rowsize)
                # Pass #2
                idx = 0
                for op in self.operations:
                    # our c transforms is now callable and should not be run in Python multithreading
                    if MapDataset.__operation_valid_for_multiprocessing(op):
                        # Wrap Python callable into _PythonCallable
                        iter_specific_operations.append(_PythonCallable(op, idx, self.process_pool))
                        idx += 1
                    else:
                        # CPP ops remain the same
                        iter_specific_operations.append(op)
                self.operations = iter_specific_operations

    @staticmethod
    def __operation_valid_for_multiprocessing(op):
        if callable(op) and str(op).find("c_transform") < 0:
            return True
        return False


class FilterDataset(UnionBaseDataset):
    """
    The result of applying filter predicate to the input Dataset.

    Args:
        input_dataset (Dataset): Input Dataset to be mapped.
        predicate (callable): Python callable which returns a boolean value. If False then filter the element.
        input_columns (Union[str, list[str]], optional): List of names of the input columns
        (default=None, the predicate will be applied to all columns in the dataset).
        num_parallel_workers (int, optional): Number of workers to process the dataset
            in parallel (default=None).
    """

    def __init__(self, input_dataset, predicate, input_columns=None, num_parallel_workers=None):
        super().__init__(children=input_dataset, num_parallel_workers=num_parallel_workers)
        self.predicate = lambda *args: bool(predicate(*args))
        self.input_columns = to_list(input_columns)

    def parse(self, children=None):
        return cde.FilterNode(children[0], self.predicate, self.input_columns)


class RepeatDataset(UnionBaseDataset):
    """
    The result of applying Repeat operator to the input Dataset.

    Args:
        input_dataset (Dataset): Input Dataset to be repeated.
        count (int): Number of times the dataset will be repeated (default=-1, repeat indefinitely).
    """

    def __init__(self, input_dataset, count):
        super().__init__(children=input_dataset)
        self.count = replace_none(count, -1)

    def parse(self, children=None):
        return cde.RepeatNode(children[0], self.count)


class SkipDataset(UnionBaseDataset):
    """
    The result of applying Skip operator to the input Dataset.

    Args:
        input_dataset (Dataset): Input dataset to have elements skipped.
        count (int): Number of elements to be skipped in the dataset.
    """

    def __init__(self, input_dataset, count):
        super().__init__(input_dataset)
        self.count = count

    def parse(self, children=None):
        return cde.SkipNode(children[0], self.count)


class TakeDataset(UnionBaseDataset):
    """
    The result of applying Take operator to the input Dataset.

    Args:
        input_dataset (Dataset): Input Dataset to have elements taken from.
        count (int): Number of elements to be taken from the dataset.
    """

    def __init__(self, input_dataset, count):
        super().__init__(children=input_dataset)
        self.count = count

    def parse(self, children=None):
        return cde.TakeNode(children[0], self.count)


class ZipDataset(UnionBaseDataset):
    """
    The result of applying Zip operator to the input Dataset.

    Args:
        datasets (tuple): A tuple of datasets to be zipped together.

    Raises:
        TypeError: If dataset is not an instance of Dataset.
    """

    def __init__(self, datasets):
        super().__init__(children=datasets)

    def parse(self, children=None):
        return cde.ZipNode(children)

    def is_sync(self):
        return any([c.is_sync() for c in self.children])


class ConcatDataset(UnionBaseDataset):
    """
    The result of applying concat dataset operator to the input Dataset.

    Args:
        datasets (list): A list of datasets to be concatenated together.

    Raises:
        TypeError: If dataset is not an instance of Dataset.
        ValueError: If there is no samples in the one of the datasets.
    """

    def __init__(self, datasets):
        super().__init__(children=datasets)
        for dataset in datasets:
            if not isinstance(dataset, Dataset):
                raise TypeError("Invalid dataset, expected Dataset object, but got %s!" % type(dataset))
        self.datasets = datasets
        self._sampler = samplers.SequentialSampler(num_samples=None)

        self.children_sizes_ = [c.get_dataset_size() for c in self.children]
        child_index = 0
        for item in self.children_sizes_:
            if item == 0:
                raise ValueError("There are no samples in the dataset number %d. Please make sure there are "
                                 "valid samples in the dataset." % child_index)
            child_index += 1

        # _children_flag_and_nums: A list of pair<int ,int>.The first element of pair is flag that characterizes
        # whether the dataset is mappable. The second element of pair is length of the dataset
        self._children_flag_and_nums = []

        # _children_start_end_index_: A list of pair<int ,int>.The elements of pair are used to characterize
        # the valid position of the dataset corresponding to the subscript when sampling
        self._children_start_end_index_ = []
        for index, child in enumerate(self.children):
            tem_list = [-1, -1]
            self._children_start_end_index_.append(tem_list)
            dataset_len = self.children_sizes_[index]

            from mindspore.dataset.engine.datasets_user_defined import GeneratorDataset
            if isinstance(child, GeneratorDataset) and not hasattr(child.source, "__getitem__"):
                dataset_len = 0
                self.children_sizes_[index] = 0

            if isinstance(child, MappableDataset):
                self._children_flag_and_nums.append((0, dataset_len))
            else:
                self._children_flag_and_nums.append((1, dataset_len))

    def parse(self, children=None):
        return cde.ConcatNode(children, self._sampler, self._children_flag_and_nums, self._children_start_end_index_)

    def use_sampler(self, sampler):
        """
        Set the distributedSampler to concat dataset

        Args:
            sampler (Sampler): The sampler to use for the current dataset.
                Currently supported: DistributedSampler.

        Raises:
            TypeError: If the sampler is not an instance of DistributedSampler
            ValueError: If the parameter shuffle of sampler is True
            ValueError: If the parameter NumSamples of sampler is not None.
            ValueError: If num_shards <=0.
        """
        if not isinstance(sampler, samplers.DistributedSampler):
            raise TypeError("The parameter %s of concat must be DistributedSampler!" % sampler)

        if sampler.is_shuffled():
            raise ValueError("The parameter shuffle of DistributedSampler must be False!")

        if sampler.num_shards <= 0:
            raise ValueError("The parameter num_shards of DistributedSampler must be positive int!")

        if sampler.get_num_samples() is not None:
            raise ValueError("The parameter num_samples of DistributedSampler is not support to be set!")

        self.dataset_size = None

        self._sampler = sampler
        cumulative_samples_nums = 0
        for index, child in enumerate(self.children):
            if hasattr(child, 'sampler') and child.sampler.get_num_samples() is not None:
                raise ValueError("The parameter NumSamples of %s is not support to be set!" % child)

            if isinstance(child, BatchDataset):
                raise TypeError("The parameter %s of concat must not be BatchDataset!" % child)

            # if child is mappable and the length is greater than 0
            if not self._children_flag_and_nums[index][0] and self._children_flag_and_nums[index][1]:

                tem_value = cumulative_samples_nums + self._children_flag_and_nums[index][1]

                if not self._children_flag_and_nums[index][1] >= sampler.num_shards:
                    if tem_value < sampler.num_shards:
                        self._children_start_end_index_[index][0] = cumulative_samples_nums
                        self._children_start_end_index_[index][1] = tem_value
                    else:
                        self._children_start_end_index_[index][0] = cumulative_samples_nums
                        self._children_start_end_index_[index][1] = tem_value % sampler.num_shards

                tem_sampler = copy.deepcopy(sampler)
                tem_sampler.set_offset(cumulative_samples_nums)
                child.use_sampler(tem_sampler)

            cumulative_samples_nums += self.children_sizes_[index]
            cumulative_samples_nums %= sampler.num_shards


class RenameDataset(UnionBaseDataset):
    """
    The result of applying Rename operator to the input Dataset.

    Args:
        input_dataset (Dataset): Input Dataset to be Renamed.
        input_columns (Union[str, list[str]]): List of names of the input columns.
        output_columns (Union[str, list[str]]): List of names of the output columns.
    """

    def __init__(self, input_dataset, input_columns, output_columns):
        super().__init__(children=input_dataset)
        self.input_column_names = to_list(input_columns)
        self.output_column_names = to_list(output_columns)

    def parse(self, children=None):
        return cde.RenameNode(children[0], self.input_column_names, self.output_column_names)


def to_list(items):
    if items is None:
        return []
    if isinstance(items, tuple):
        return list(items)
    if not isinstance(items, list):
        return [items]
    return items


class ProjectDataset(UnionBaseDataset):
    """
    The result of applying Project operator to the input Dataset.

    Args:
        input_dataset (Dataset): Input Dataset to be Projected.
        columns (Union[str, list[str]]): List of names of the columns to project.
    """

    def __init__(self, input_dataset, columns):
        super().__init__(children=input_dataset)
        self.columns = to_list(columns)

    def parse(self, children=None):
        return cde.ProjectNode(children[0], self.columns)


class _ToDevice:
    """
    Internal class to handle sending data to device.
    """

    def __init__(self, dataset, num_epochs):
        ir_tree, self.api_tree = dataset.create_ir_tree()

        self._runtime_context = cde.PythonRuntimeContext()
        self._runtime_context.Init()
        self._to_device = cde.ToDevice(num_epochs)
        self._to_device.Init(ir_tree)
        self._runtime_context.AssignConsumer(self._to_device)

        ITERATORS_LIST.append(weakref.ref(self))
        _unset_iterator_cleanup()

    def send(self):
        self._to_device.Send()

    def _reset(self, step):
        self._to_device.Reset(step)

    def stop_send(self):
        """
        send stop send signal to pipeline, it is used when end of sequence is sent at the epoch end.
        """
        self._to_device.StopSend()

    def continue_send(self):
        """
        send continue send signal to pipeline, it is used when end of sequence is sent at the epoch end.
        """
        self._to_device.ContinueSend()

    def get_data_info(self):
        """
        Get type and shape of current batch.
        """
        return self._to_device.GetDataInfo()

    def release(self):
        """
        Manually terminate Device Queue instead of relying on out of scope destruction.
        """
        if hasattr(self, '_runtime_context') and self._runtime_context:
            if hasattr(self, '_to_device') and self._to_device:
                self._runtime_context.Terminate()
                del self._to_device
            del self._runtime_context

    def __deepcopy__(self, memodict):
        return self

    def get_offload_model(self, col_names):
        """
        Get offload model containing removed offload ops from pipeline.
        """
        offload_model = GetOffloadModel(self._to_device, col_names)
        return offload_model


class TransferDataset(Dataset):
    """
    The result of applying TDT operator to the input Dataset.

    Args:
        input_dataset (Dataset): Input Dataset to be transferred.
        send_epoch_end (bool, optional): Whether to send end of sequence to device or not (default=True).
        create_data_info_queue (bool, optional): Whether to create queue which stores
            types and shapes of data or not (default=False).

    Raises:
        TypeError: If device_type is empty.
        ValueError: If device_type is not 'Ascend', 'GPU' or 'CPU'.
        RuntimeError: If dataset is unknown.
    """

    def __init__(self, input_dataset, send_epoch_end=True, create_data_info_queue=False):
        super().__init__(children=input_dataset)
        self.queue_name = str(uuid.uuid1())
        self.device_type = context.get_context("device_target") if context else "CPU"
        self.device_id = context.get_context("device_id") if context else 0

        self._send_epoch_end = replace_none(send_epoch_end, True)
        self._create_data_info_queue = create_data_info_queue
        self._to_device = None
        self.column_name = input_dataset.get_col_names()

    def parse(self, children=None):
        total_batch = 0
        if hasattr(self.children[0], "__total_batch__"):
            total_batch = self.children[0].__total_batch__
        return cde.TransferNode(children[0], self.queue_name, self.device_type, self.device_id, self._send_epoch_end,
                                total_batch, self._create_data_info_queue)

    def create_dict_iterator(self, num_epochs=-1, output_numpy=False):
        raise RuntimeError("TransferDataset is not iterable.")

    def create_tuple_iterator(self, columns=None, num_epochs=-1, output_numpy=False, do_copy=True):
        raise RuntimeError("TransferDataset is not iterable.")

    def __iter__(self):
        raise RuntimeError("TransferDataset is not iterable.")

    def output_shapes(self):
        raise RuntimeError("TransferDataset does not support obtaining output_shapes.")

    def output_types(self):
        raise RuntimeError("TransferDataset does not support obtaining output_types.")

    @check_to_device_send
    def send(self, num_epochs=-1):
        """
        Send to device
        """
        if Dataset._noop_mode():
            return
        if self._to_device is not None:
            del self._to_device
        self._to_device = _ToDevice(self, num_epochs)
        self._to_device.send()

    def stop_send(self):
        if self._to_device is not None:
            self._to_device.stop_send()

    def continue_send(self):
        if self._to_device is not None:
            self._to_device.continue_send()

    def _reset(self, step):
        if self._to_device is not None:
            logger.info("Reset the dataset pipeline to step " + str(step))
            self._to_device._reset(step)  # pylint: disable=W0212

    def get_data_info(self):
        """
        Get type and shape of current batch
        """
        if self._to_device is not None:
            return self._to_device.get_data_info()
        raise RuntimeError("Calling get_data_info with bad state.")

    def get_offload_model(self):
        if self._to_device is not None:
            return self._to_device.get_offload_model(self.column_name)

        raise RuntimeError("get_offload_model, _to_device is None")

    def release(self):
        """
        Manually terminate Device Queue instead of relying on out of scope destruction.
        """
        if self._to_device is not None:
            self._to_device.release()


class Schema:
    """
    Class to represent a schema of a dataset.

    Args:
        schema_file(str): Path of the schema file (default=None).

    Returns:
        Schema object, schema info about dataset.

    Raises:
        RuntimeError: If schema file failed to load.

    Examples:
        >>> from mindspore import dtype as mstype
        >>>
        >>> # Create schema; specify column name, mindspore.dtype and shape of the column
        >>> schema = ds.Schema()
        >>> schema.add_column(name='col1', de_type=mstype.int64, shape=[2])
    """

    @check_schema
    def __init__(self, schema_file=None):
        self.schema_file = replace_none(schema_file, "")
        self.cpp_schema = cde.SchemaObj(self.schema_file)

    @check_add_column
    def add_column(self, name, de_type, shape=None):
        """
        Add new column to the schema.

        Args:
            name (str): The new name of the column.
            de_type (str): Data type of the column.
            shape (list[int], optional): Shape of the column
                (default=None, [-1] which is an unknown shape of rank 1).

        Raises:
            ValueError: If column type is unknown.
        """
        if isinstance(de_type, typing.Type):
            de_type = mstype_to_detype(de_type)
            col_type = str(de_type)
        else:
            col_type = str(cde.DataType(de_type))
        if shape is None:
            self.cpp_schema.add_column(name, col_type)
        else:
            self.cpp_schema.add_column(name, col_type, shape)

    def parse_columns(self, columns):
        """
        Parse the columns and add it to self.

        Args:
            columns (Union[dict, list[dict], tuple[dict]]): Dataset attribute information, decoded from schema file.

                - list[dict], 'name' and 'type' must be in keys, 'shape' optional.

                - dict, columns.keys() as name, columns.values() is dict, and 'type' inside, 'shape' optional.

        Raises:
            RuntimeError: If failed to parse columns.
            RuntimeError: If column's name field is missing.
            RuntimeError: If column's type field is missing.

        Examples:
            >>> from mindspore.dataset import Schema
            >>> schema = Schema()
            >>> columns1 = [{'name': 'image', 'type': 'int8', 'shape': [3, 3]},
            ...             {'name': 'label', 'type': 'int8', 'shape': [1]}]
            >>> schema.parse_columns(columns1)
            >>> columns2 = {'image': {'shape': [3, 3], 'type': 'int8'}, 'label': {'shape': [1], 'type': 'int8'}}
            >>> schema.parse_columns(columns2)
        """
        self.cpp_schema.parse_columns(json.dumps(columns, indent=2))

    def to_json(self):
        """
        Get a JSON string of the schema.

        Returns:
            str, JSON string of the schema.
        """
        return self.cpp_schema.to_json()

    def from_json(self, json_obj):
        """
        Get schema file from JSON object.

        Args:
            json_obj(dictionary): Object of JSON parsed.

        Raises:
            RuntimeError: if there is unknown item in the object.
            RuntimeError: if dataset type is missing in the object.
            RuntimeError: if columns are missing in the object.
        """
        self.cpp_schema.from_string(json.dumps(json_obj, indent=2))

    def __str__(self):
        return self.to_json()

    @staticmethod
    def get_num_rows(schema):
        schema_obj = schema
        if not isinstance(schema_obj, Schema):
            schema_obj = Schema(schema_obj)
        return schema_obj.cpp_schema.get_num_rows()


class DeserializedDataset(Dataset):
    def __init__(self, input_obj):
        super().__init__()
        self.input_obj = input_obj

    def parse(self, children=None):
        if isinstance(self.input_obj, dict):
            json_str = json.dumps(self.input_obj)
            return cde.Dataset.from_json_string(json_str)
        return cde.Dataset.from_json_file(self.input_obj)
