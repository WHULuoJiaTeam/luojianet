# Copyright 2020 Huawei Technologies Co., Ltd
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
# ============================================================================
"""Dataset help for minddata dataset"""
import math

from mindspore._checkparam import Validator
from mindspore.common.dtype import pytype_to_dtype
from mindspore.common.api import _cell_graph_executor
from mindspore.dataset.engine import offload
from .. import context, nn
from ._utils import _exec_datagraph, _get_types_and_shapes, _construct_tensor_list
from ..parallel._utils import _get_device_num, _get_global_rank, _need_to_full, _to_full_shapes, _get_pipeline_stages
from ..parallel._ps_context import _is_role_worker, _is_role_pserver, _is_role_sched, _is_ps_mode
from ..ops import operations as P


def _send_data(dataset, epoch_num):
    """Engine dataset to write data to tdt queue."""
    if not hasattr(dataset, '__has_sent__'):
        exec_dataset = dataset.__transfer_dataset__
        exec_dataset.send(epoch_num)
        dataset.__has_sent__ = True


def _send_data_no_flag(dataset, epoch_num):
    """Engine dataset to write data to tdt queue directly."""
    exec_dataset = dataset.__transfer_dataset__
    exec_dataset.send(epoch_num)


def _dynamic_sink_data(dataset, dataset_iter):
    """Special scenario for dataset with sink_size=1."""
    if hasattr(dataset_iter, "sink_size") and \
       dataset_iter.sink_size == 1 and \
       dataset.get_dataset_size() != 1 and \
       hasattr(dataset_iter, "sink_count") and \
       dataset_iter.sink_count == 1 and \
       context.get_context("device_target") == "Ascend":
        return True
    return False


def _dynamic_sink_exception_scenario(dataset_iter):
    """The exception scenario for dynamic data is not applicable."""
    _, dataset_shapes = dataset_iter.types_shapes()

    if _has_dynamic_shape(dataset_shapes) or (_is_role_worker() and _is_ps_mode()) or \
       context.get_context("mode") != context.GRAPH_MODE:
        return True
    return False


def _dynamic_sink_scenario(dataset, dataset_iter):
    """Special scenario with dynamic shape and sink_size=1."""
    flag = False
    if _dynamic_sink_data(dataset, dataset_iter) and not _dynamic_sink_exception_scenario(dataset_iter):
        flag = True

    return flag


class _DataWrapper(nn.Cell):
    """
    Wraps the input network with a dataset which automatically fetches data with 'GetNext' function from the
    dataset channel 'queue_name' and performs the forward computation.
    """

    def __init__(self, network, dataset_types, dataset_shapes, queue_name, min_shapes=None, max_shapes=None):
        super(_DataWrapper, self).__init__(auto_prefix=False, flags=network.get_flags())
        # Also copy the flag in `network` construct
        flags = getattr(network.__class__.construct, "_mindspore_flags", {})
        self.info = (dataset_types, dataset_shapes)
        self.add_flags(**flags)
        self.get_next = P.GetNext(dataset_types, dataset_shapes, len(dataset_types), queue_name)
        if min_shapes is not None and max_shapes is not None:
            Validator.check_value_type("min_shapes", min_shapes, [list, tuple])
            Validator.check_value_type("max_shapes", max_shapes, [list, tuple])
            self.get_next.add_prim_attr("min_shapes", min_shapes)
            self.get_next.add_prim_attr("max_shapes", max_shapes)
        self.network = network

    def construct(self):
        outputs = self.get_next()
        return self.network(*outputs)


def _generate_dataset_sink_mode_net(network, dataset_shapes, dataset_types, queue_name,
                                    min_shapes=None, max_shapes=None):
    if not isinstance(network, _DataWrapper):
        network = _DataWrapper(network, dataset_types, dataset_shapes, queue_name, min_shapes, max_shapes)
    return network


def _has_dynamic_shape(dataset_shapes):
    for shape in dataset_shapes:
        if -1 in shape:
            return True
    return False


def _generate_network_with_dataset(network, dataset_helper, queue_name):
    dataset_types, dataset_shapes = dataset_helper.types_shapes()
    (min_shapes, max_shapes) = (None, None) if not _has_dynamic_shape(dataset_shapes) \
        else dataset_helper.dynamic_min_max_shapes()
    network = _generate_dataset_sink_mode_net(network, dataset_shapes, dataset_types,
                                              queue_name, min_shapes, max_shapes)
    return network


class _DatasetAux:
    def __deepcopy__(self, memodict):
        return None


def _get_dataset_aux(dataset):
    if not hasattr(dataset, '__network_aux__'):
        dataset.__network_aux__ = _DatasetAux()
    return dataset.__network_aux__


def connect_network_with_dataset(network, dataset_helper):
    """
    Connect the `network` with dataset in `dataset_helper`.

    This function wraps the input network with 'GetNext' so that the data can be fetched automatically from the
    data channel corresponding to the 'queue_name' and passed to the input network during forward computation.

    Note:
        In the case of running the network on Ascend/GPU in graph mode, this function will wrap the input network with
        'GetNext', in other cases, the input network will be returned with no change.
        The 'GetNext' is required to get data only in sink mode, so this function is not applicable to no-sink mode.
        when dataset_helper's dataset_sink_mode is True, it can only be connected to one network.
    Args:
        network (Cell): The training network for dataset.
        dataset_helper (DatasetHelper): A class to process the MindData dataset, it provides the type, shape and queue
            name of the dataset to wrap the `GetNext`.

    Returns:
        Cell, a new network wrapped with 'GetNext' in the case of running the task on Ascend in graph mode, otherwise
        it is the input network.

    Raises:
        RuntimeError: If the API was not called in dataset sink mode.
    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import DatasetHelper
        >>> from mindspore import DatasetHelper, nn, connect_network_with_dataset
        >>> from mindspore import dataset as ds
        >>>
        >>> data = {"x": np.float32(np.random.rand(64, 10)), "y": np.random.randint(0, 5, (64,))}
        >>> train_dataset = ds.NumpySlicesDataset(data=data).batch(32)
        >>> dataset_helper = DatasetHelper(train_dataset, dataset_sink_mode=True)
        >>> net = nn.Dense(10, 5)
        >>> net_with_get_next = connect_network_with_dataset(net, dataset_helper)
    """
    dataset_iter = dataset_helper.iter
    dataset = dataset_iter.dataset
    aux = _get_dataset_aux(dataset)

    if isinstance(dataset_iter, _DatasetIterNormal):
        raise RuntimeError("The API 'connect_network_with_dataset' should be called in dataset sink mode.")

    if _is_role_sched() or _is_role_pserver():
        return network

    if not hasattr(aux, '__network__'):
        aux.__network__ = network

    if aux.__network__ is not network:
        raise ValueError("The dataset has been connected to other network, please check the code.")

    queue_name = dataset.__transfer_dataset__.queue_name
    if _dynamic_sink_scenario(dataset, dataset_iter):
        dataset_types, dataset_shapes = dataset_helper.get_data_info()
        dataset_types = [pytype_to_dtype(x) for x in dataset_types]

        key = str(dataset_types) + str(dataset_shapes)
        if hasattr(aux, '__network_manage__') and key in aux.__network_manage__:
            network = aux.__network_manage__[key]
        else:
            if _need_to_full():
                device_num = _get_device_num() // _get_pipeline_stages()
                dataset_shapes = _to_full_shapes(dataset_shapes, device_num)

            network = _generate_dataset_sink_mode_net(network, dataset_shapes, dataset_types, queue_name)
            aux.__network_manage__ = aux.__network_manage__ if hasattr(aux, '__network_manage__') else dict()
            aux.__network_manage__[key] = network
        return network

    if hasattr(aux, '__sink_network__'):
        network = aux.__sink_network__
    else:
        if not context.get_context("enable_ge") and context.get_context("device_target") in ("Ascend", "GPU"):
            network = offload.check_add_offload_sink_mode(dataset, dataset_helper, network)
            network = _generate_network_with_dataset(network, dataset_helper, queue_name)
            aux.__sink_network__ = network

    if _dynamic_sink_data(dataset, dataset_iter) and _dynamic_sink_exception_scenario(dataset_iter):
        dataset_helper.get_data_info()

    return network


class DatasetHelper:
    """
    DatasetHelper is a class to process the MindData dataset and provides the information of dataset.

    According to different contexts, change the iterations of dataset and use the same iteration for loop in different
    contexts.

    Note:
        The iteration of DatasetHelper will provide one epoch data.

    Args:
        dataset (Dataset): The dataset iterator. The dataset can be generated by dataset generator API in
                           :class:`mindspore.dataset`, such as :class:`mindspore.dataset.ImageFolderDataset`.
        dataset_sink_mode (bool): If the value is True, GetNext is employed to fetch the data at device through the
                                  dataset pipeline, otherwise fetch the data at host by iterating through the dataset.
                                  Default: True.
        sink_size (int): Control the amount of data in each sink.
                          If sink_size=-1, sink the complete dataset for each epoch.
                          If sink_size>0, sink sink_size data for each epoch.
                          Default: -1.
        epoch_num (int): The number of passes of the entire dataset to be sent. Default: 1.

    Examples:
        >>> import numpy as np
        >>> from mindspore import DatasetHelper, nn
        >>> from mindspore import dataset as ds
        >>>
        >>> data = {"x": np.float32(np.random.rand(64, 10)), "y": np.random.randint(0, 5, (64,))}
        >>> train_dataset = ds.NumpySlicesDataset(data=data).batch(32)
        >>> set_helper = DatasetHelper(train_dataset, dataset_sink_mode=False)
        >>>
        >>> net = nn.Dense(10, 5)
        >>> # Object of DatasetHelper is iterable
        >>> for next_element in set_helper:
        ...     # `next_element` includes data and label, using data to run the net
        ...     data = next_element[0]
        ...     net(data)
    """

    def __init__(self, dataset, dataset_sink_mode=True, sink_size=-1, epoch_num=1):
        dataset_sink_mode = Validator.check_bool(dataset_sink_mode)
        Validator.check_is_int(sink_size)
        if sink_size < -1 or sink_size == 0:
            raise ValueError("The 'sink_size' must be -1 or positive, but got sink_size {}.".format(sink_size))
        if sink_size == -1:
            sink_size = dataset.get_dataset_size()

        if dataset_sink_mode:
            if context.get_context("enable_ge"):
                iterclass = _DatasetIterGE
            else:
                if context.get_context("mode") == context.GRAPH_MODE:
                    if _is_role_sched() or _is_role_pserver():
                        iterclass = _DatasetIterPSServer
                    elif _is_role_worker() and _is_ps_mode():
                        iterclass = _DatasetIterPSWork
                    elif (context.get_context("device_target") == "Ascend") or \
                         (context.get_context("device_target") == "GPU"):
                        iterclass = _DatasetIterMSLoopSink
                    elif context.get_context("device_target") == "CPU":
                        raise RuntimeError("Currently dataset sink mode is not supported when the device "
                                           "target is CPU, please set dataset sink mode to False.")
                else:
                    iterclass = _DatasetIterPyNative
            self.iter = iterclass(dataset, sink_size, epoch_num)
        else:
            iterclass = _DatasetIterNormal
            self.iter = iterclass(dataset, epoch_num=epoch_num)

    def __iter__(self):
        return self.iter.__iter__()

    # A temp solution for loop sink. Delete later
    def types_shapes(self):
        """
        Get the types and shapes from dataset on the current configuration.

        Examples:
            >>> from mindspore import DatasetHelper
            >>>
            >>> train_dataset = create_custom_dataset()
            >>> dataset_helper = DatasetHelper(train_dataset, dataset_sink_mode=True)
            >>>
            >>> types, shapes = dataset_helper.types_shapes()
        """
        return self.iter.types_shapes()

    def sink_size(self):
        """
        Get sink_size for each iteration.

        Examples:
            >>> from mindspore import DatasetHelper
            >>>
            >>> train_dataset = create_custom_dataset()
            >>> dataset_helper = DatasetHelper(train_dataset, dataset_sink_mode=True, sink_size=-1)
            >>>
            >>> # if sink_size==-1, then will return the full size of source dataset.
            >>> sink_size = dataset_helper.sink_size()
        """
        return self.iter.get_sink_size()

    def stop_send(self):
        """Stop send data about data sink."""
        self.iter.stop_send()

    def release(self):
        """Free up resources about data sink."""
        self.iter.release()

    def continue_send(self):
        """Continue to send data to device at the beginning of epoch."""
        self.iter.continue_send()

    def _reset(self, step):
        """Reset the dataset to the provided step."""
        self.iter._reset(step) # pylint: disable=W0212

    def get_data_info(self):
        """
        In sink mode, it returns the types and shapes of the current data.
        Generally, it works in dynamic shape scenarios.
        """
        return self.iter.get_data_info()

    def dynamic_min_max_shapes(self):
        """
        Return the minimum and maximum data length of dynamic source dataset.

        Examples:
            >>> from mindspore import DatasetHelper
            >>>
            >>> train_dataset = create_custom_dataset()
            >>> # config dynamic shape
            >>> dataset.set_dynamic_columns(columns={"data1": [16, None, 83], "data2": [None]})
            >>> dataset_helper = DatasetHelper(train_dataset, dataset_sink_mode=True)
            >>>
            >>> min_shapes, max_shapes = dataset_helper.dynamic_min_max_shapes()
        """
        return self.iter.dynamic_min_max_shapes()


class _DatasetIter:
    """Base iter for dataset helper"""

    def __init__(self, dataset, sink_size, epoch_num):
        self.dataset = dataset
        self.sink_size = sink_size
        self.sink_count = self.get_sink_count(dataset)

        if not hasattr(dataset, '__transfer_dataset__'):
            if hasattr(dataset, '__loop_size__'):
                # PS mode does not support loop sink and need get the real sink size.
                if not (_is_role_worker() and _is_ps_mode()):
                    self.sink_size = dataset.__loop_size__
            create_data_info_queue = (sink_size == 1 and self.sink_count == 1 and dataset.get_dataset_size() != 1
                                      and context.get_context("device_target") == "Ascend")
            dataset.__transfer_dataset__ = _exec_datagraph(dataset, self.sink_size,
                                                           create_data_info_queue=create_data_info_queue)

            if not hasattr(dataset, '__no_send__'):
                _send_data(dataset, epoch_num)
        else:
            # if using an existed __transfer_dataset__, set the queue_name directly
            if not dataset.__transfer_dataset__.queue_name:
                _cell_graph_executor.set_queue_name(dataset.__transfer_dataset__.queue_name)
            _send_data_no_flag(dataset, epoch_num)

        self.stop_send = dataset.__transfer_dataset__.stop_send
        self.release = dataset.__transfer_dataset__.release
        self.continue_send = dataset.__transfer_dataset__.continue_send
        self.get_data_info = dataset.__transfer_dataset__.get_data_info
        self.dynamic_min_max_shapes = dataset.dynamic_min_max_shapes
        self.dataset_types, self.dataset_shapes = _get_types_and_shapes(dataset)
        if hasattr(dataset.__transfer_dataset__, "_reset"):
            self._reset = dataset.__transfer_dataset__._reset  # pylint: disable=W0212

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= self.sink_count:
            raise StopIteration()
        self.index += 1
        return self.op()

    def types_shapes(self):
        """
        Return the types and shapes of the dataset. The type and shape of each data in the dataset
        should be consistent.
        """
        return self.dataset_types, self.dataset_shapes

    def get_sink_count(self, dataset):
        sink_count = 1
        if hasattr(dataset, '__loop_size__'):
            loop_size = dataset.__loop_size__
            if loop_size <= dataset.get_dataset_size() and dataset.get_dataset_size() % loop_size != 0:
                raise ValueError(f"Dataset size {dataset.get_dataset_size()} and 'sink_size' {loop_size} "
                                 f"are not matched, dataset size should be divisible by 'sink_size'.")
            sink_count = math.ceil(dataset.get_dataset_size() / loop_size)
        return sink_count

    def get_sink_size(self):
        """get sink_size to device"""
        sink_size = 1
        if hasattr(self.dataset, '__loop_size__'):
            sink_size = self.dataset.__loop_size__
        elif _is_role_worker() and _is_ps_mode():
            # PS mode does not support loop sink.
            sink_size = 1
        else:
            if context.get_context("enable_ge") or context.get_context("device_target") == "Ascend" \
                    or context.get_context("device_target") == "GPU":
                if self.sink_size > 0:
                    sink_size = self.sink_size
                else:
                    sink_size = self.dataset.get_dataset_size()
        return sink_size


class _DatasetIterGE(_DatasetIter):
    """Iter for GE."""

    def __init__(self, dataset, sink_size, epoch_num):
        super().__init__(dataset, sink_size, epoch_num)
        self.sink_count = self.get_sink_count(dataset)
        batch_expand_num = 1
        if _need_to_full():
            batch_expand_num = _get_device_num() // _get_pipeline_stages()
        tensor_list_run = _construct_tensor_list(self.dataset_types, self.dataset_shapes, batch_expand_num)

        def op():
            return tensor_list_run

        self.op = op


class _DatasetIterPyNative(_DatasetIter):
    """Iter for context (mode=PYNATIVE_MODE)."""

    def __init__(self, dataset, sink_size, epoch_num):
        super().__init__(dataset, sink_size, epoch_num)
        if sink_size > 0:
            self.sink_count = sink_size
        else:
            self.sink_count = dataset.get_dataset_size()

        def op():
            return tuple()

        self.op = op


class _DatasetIterMSLoopSink(_DatasetIter):
    """Iter for context (device_target=Ascend)"""

    def __init__(self, dataset, sink_size, epoch_num):
        super().__init__(dataset, sink_size, epoch_num)
        self.sink_count = self.get_sink_count(dataset)
        # for self._parallel_mode equal to semi_auto_parallel or auto_parallel, and not using full_batch,
        # use a complete tensor to compile, and slice tensor to run. The batch dimension of tensors for
        # compile is device_number times the batch dimension of tensors for run. Now only support LoopSink.
        if _need_to_full():
            device_num = _get_device_num() // _get_pipeline_stages()
            self.dataset_shapes = _to_full_shapes(self.dataset_shapes, device_num)

        def op():
            return tuple()

        self.op = op


class _DatasetIterPSServer(_DatasetIter):
    """Iter for context on MS_PSERVER or MS_SCHED"""

    def __init__(self, dataset, sink_size, epoch_num):
        super().__init__(dataset, sink_size, epoch_num)
        self.sink_count = 1
        self.sink_size = 1
        self.op = None

        def op():
            return _construct_tensor_list(self.dataset_types, self.dataset_shapes, batch_expand_num=1)

        self.op = op


class _DatasetIterPSWork(_DatasetIter):
    """Iter for context on MS_WORKER"""

    def __init__(self, dataset, sink_size, epoch_num):
        super().__init__(dataset, sink_size, epoch_num)
        if sink_size > 0:
            self.sink_count = sink_size
        else:
            self.sink_count = dataset.get_dataset_size()

        def op():
            return tuple()

        self.op = op


class _DatasetIterNormal:
    """Iter for normal(non sink) mode, feed the data from host."""

    def __init__(self, dataset, epoch_num=-1):
        self.dataset = dataset
        self.device_num = _get_device_num()
        self.global_rank = _get_global_rank()
        self.iter = self.dataset.create_tuple_iterator(num_epochs=epoch_num, do_copy=True)

    def __iter__(self):
        return self

    def __next__(self):
        data = self.iter.__next__()
        return data


__all__ = ["DatasetHelper", "connect_network_with_dataset"]
