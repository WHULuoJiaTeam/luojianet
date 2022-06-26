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
"""Built-in iterators.
"""
from abc import abstractmethod
import os
import signal
import weakref
import numpy as np

import mindspore._c_dataengine as cde
from mindspore.common.tensor import Tensor
import mindspore.dataset.engine.offload as offload

from mindspore import log as logger

_ITERATOR_CLEANUP = False


def _set_iterator_cleanup():
    global _ITERATOR_CLEANUP
    _ITERATOR_CLEANUP = True


def _unset_iterator_cleanup():
    global _ITERATOR_CLEANUP
    _ITERATOR_CLEANUP = False


def check_iterator_cleanup():
    global _ITERATOR_CLEANUP
    return _ITERATOR_CLEANUP


ITERATORS_LIST = list()


def _cleanup():
    """Release all the Iterator."""
    _set_iterator_cleanup()
    for itr_ref in reversed(ITERATORS_LIST):
        itr = itr_ref()
        if itr is not None:
            itr.release()


class Iterator:
    """
    General Iterator over a dataset.

    Attributes:
        dataset: Dataset to be iterated over
    """

    def __init__(self, dataset, num_epochs=-1, output_numpy=False, do_copy=True):
        self._col_names = None

        # create a copy of tree and work on it.
        self.__ori_dataset = dataset

        self.ir_tree, self.dataset = dataset.create_ir_tree()

        self._runtime_context = cde.PythonRuntimeContext()
        self._runtime_context.Init()
        consumer = cde.PythonIteratorConsumer(num_epochs)
        consumer.Init(self.ir_tree)
        self._runtime_context.AssignConsumer(consumer)
        self._iterator = self._runtime_context.GetConsumer()

        self._transform_tensor = lambda t: t.as_array()
        if not output_numpy:
            def _transform(t, do_copy):
                array = t.as_array()
                if array.dtype.type is np.bytes_:
                    array = np.char.decode(array)
                if do_copy:
                    return Tensor(array)
                return Tensor.from_numpy(array)
            self._transform_tensor = lambda t: _transform(t, do_copy)
        self.__index = 0

        self.offload_model = None
        offload_model = offload.GetOffloadModel(consumer, self.__ori_dataset.get_col_names())

        # See if GetOffloadModel identified any operations set to be offloaded.
        if offload_model.transform_list != []:
            offload.check_concat_zip_dataset(self.__ori_dataset)
            self.offload_model = offload_model

        ITERATORS_LIST.append(weakref.ref(self))
        _unset_iterator_cleanup()

    def __iter__(self):
        return self

    def stop(self):
        """
        Manually terminate Python iterator instead of relying on out of scope destruction.
        """
        if hasattr(self, '_runtime_context') and self._runtime_context:
            if hasattr(self, '_iterator') and self._iterator:
                self._runtime_context.Terminate()
                del self._iterator
            del self._runtime_context
            del self.dataset

            # get weakref which is dead
            dead_iterator = []
            for index, item in enumerate(ITERATORS_LIST):
                # item() == None indicate the object is dead
                # id(item()) == id(self) indicate del self
                if item() is None or id(item()) == id(self):
                    dead_iterator.append(index)

            # del dead weakref
            for index in reversed(dead_iterator):
                ITERATORS_LIST.pop(index)

    def release(self):
        self.stop()

    def __del__(self):
        self.release()

    @abstractmethod
    def _get_next(self):
        raise RuntimeError("Calling base class Iterator's get_next is invalid.")

    def __next__(self):
        if not self._runtime_context:
            logger.warning("Iterator does not have a running C++ pipeline." +
                           "It might because Iterator stop() had been called, or C++ pipeline crashed silently.")
            raise RuntimeError("Iterator does not have a running C++ pipeline.")

        data = self._get_next()
        if not data:
            if self.__index == 0:
                logger.warning("No records available.")
            if self.__ori_dataset.dataset_size is None:
                self.__ori_dataset.dataset_size = self.__index
            raise StopIteration
        self.__index += 1

        if self.offload_model is not None:
            data = offload.apply_offload_iterators(data, self.offload_model)

        return data

    def __deepcopy__(self, memo):
        return self

    def _getters(self):
        """
        Get pipeline information.
        """
        getter = cde.TreeGetters()
        getter.Init(self.ir_tree)
        self._runtime_context.AssignConsumer(getter)
        self._col_names = getter.GetColumnNames()

    def get_col_names(self):
        """
        Get names of the columns in the dataset
        """
        if self._col_names is None:
            self._getters()
        return self._col_names

    def _reset(self, step):
        """
        Reset the iterator to the given step number.

        Args:
            step (int): Global step number.
        """
        self._iterator.Reset(step)


class DictIterator(Iterator):
    """
    The derived class of Iterator with dict type.
    """

    def _get_next(self):
        """
        Returns the next record in the dataset as dictionary

        Returns:
            Dict, the next record in the dataset.
        """
        try:
            return {k: self._transform_tensor(t) for k, t in self._iterator.GetNextAsMap().items()}
        except RuntimeError as err:
            # maybe "Out of memory" / "MemoryError" error
            err_info = str(err)
            if err_info.find("Out of memory") >= 0 or err_info.find("MemoryError") >= 0:
                logger.critical("Memory error occurred, process will exit.")
                os.kill(os.getpid(), signal.SIGKILL)
            raise err


class TupleIterator(Iterator):
    """
    The derived class of Iterator with list type.
    """

    def __init__(self, dataset, columns=None, num_epochs=-1, output_numpy=False, do_copy=True):
        if columns is not None:
            if not isinstance(columns, list):
                columns = [columns]
            dataset = dataset.project(columns)
        super().__init__(dataset, num_epochs, output_numpy, do_copy)

    def _get_next(self):
        """
        Returns the next record in the dataset as a list

        Returns:
            List, the next record in the dataset.
        """

        return [self._transform_tensor(t) for t in self._iterator.GetNextAsList()]


class DummyIterator:
    """
    A DummyIterator only work when env MS_ROLE="MS_PSERVER" or MS_ROLE="MS_SCHED"
    """

    def __init__(self, dataset, mode):
        self.mode = mode
        self.shapes = dataset.output_shapes()
        self.types = dataset.output_types()
        self.fetched_first = False

    def __get_tensor(self):
        tensor_row = []
        for np_shape, np_type in zip(self.shapes, self.types):
            input_np = np.zeros(np_shape, np_type)
            tensor = Tensor(input_np)
            tensor_row.append(tensor)
        return tensor_row

    def __iter__(self):
        return self

    def __next__(self):
        if self.mode == "tuple":
            if not self.fetched_first:
                self.fetched_first = True
                return self.__get_tensor()
        raise StopIteration()
