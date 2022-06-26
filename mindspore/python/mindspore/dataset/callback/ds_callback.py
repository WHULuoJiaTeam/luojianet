# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
Python callback class
"""
import threading
from mindspore._c_dataengine import PyDSCallback
from mindspore.train.callback import Callback
import mindspore.dataset as ds
from .validators import check_callback


class DSCallback:
    """
    Abstract base class used to build dataset callback classes.

    Users can obtain the dataset pipeline context through `ds_run_context`, including
    `cur_epoch_num`, `cur_step_num_in_epoch` and `cur_step_num`.

    Args:
        step_size (int, optional): The number of steps between adjacent `ds_step_begin`/`ds_step_end`
            calls. Default: 1, will be called at each step.

    Examples:
        >>> from mindspore.dataset import DSCallback
        >>> from mindspore.dataset.transforms import c_transforms
        >>>
        >>> class PrintInfo(DSCallback):
        ...     def ds_epoch_end(self, ds_run_context):
        ...         print(ds_run_context.cur_epoch_num)
        ...         print(ds_run_context.cur_step_num)
        >>>
        >>> dataset = ds.MnistDataset(mnist_dataset_dir, num_samples=100)
        >>> op = c_transforms.OneHot(10)
        >>> dataset = dataset.map(operations=op, callbacks=PrintInfo())
    """

    @check_callback
    def __init__(self, step_size=1):
        self.step_size = step_size

    def ds_begin(self, ds_run_context):
        """
        Called before the data pipeline is started.

        Args:
            ds_run_context (RunContext): Include some information of the data pipeline.
        """

    def ds_epoch_begin(self, ds_run_context):
        """
        Called before a new epoch is started.

        Args:
            ds_run_context (RunContext): Include some information of the data pipeline.
        """

    def ds_epoch_end(self, ds_run_context):
        """
        Called after an epoch is finished.

        Args:
            ds_run_context (RunContext): Include some information of the data pipeline.
        """

    def ds_step_begin(self, ds_run_context):
        """
        Called before a step start.

        Args:
            ds_run_context (RunContext): Include some information of the data pipeline.
        """

    def ds_step_end(self, ds_run_context):
        """
        Called after a step finished.

        Args:
            ds_run_context (RunContext): Include some information of the data pipeline.
        """

    def create_runtime_obj(self):
        """
        Internal method, creates a runtime (C++) object from the callback methods defined by the user.

        Returns:
            _c_dataengine.PyDSCallback.
        """
        c_cb = PyDSCallback(self.step_size)
        at_least_one = False

        if self.__class__.ds_begin != DSCallback.ds_begin:
            c_cb.set_begin(self.ds_begin)
            at_least_one = True

        if self.__class__.ds_epoch_begin != DSCallback.ds_epoch_begin:
            c_cb.set_epoch_begin(self.ds_epoch_begin)
            at_least_one = True
        if self.__class__.ds_epoch_end != DSCallback.ds_epoch_end:
            c_cb.set_epoch_end(self.ds_epoch_end)
            at_least_one = True

        if self.__class__.ds_step_begin != DSCallback.ds_step_begin:
            c_cb.set_step_begin(self.ds_step_begin)
            at_least_one = True
        if self.__class__.ds_step_end != DSCallback.ds_step_end:
            c_cb.set_step_end(self.ds_step_end)
            at_least_one = True

        if not at_least_one:
            raise AttributeError(
                "Inheriting Callback class without overriding any methods, check the usage of user defined Callback.")

        return c_cb


class WaitedDSCallback(Callback, DSCallback):
    r"""
    Abstract base class used to build dataset callback classes that are synchronized with the training callback class
    `mindspore.train.callback \
    <https://www.mindspore.cn/docs/en/r1.7/api_python/mindspore.train.html#mindspore.train.callback.Callback>`_.

    It can be used to execute a custom callback method before a step or an epoch, such as
    updating the parameters of operators according to the loss of the previous training epoch in auto augmentation.

    Users can obtain the network training context through `train_run_context`, such as
    `network`, `train_network`, `epoch_num`, `batch_num`, `loss_fn`, `optimizer`, `parallel_mode`,
    `device_number`, `list_callback`, `cur_epoch_num`, `cur_step_num`, `dataset_sink_mode`,
    `net_outputs`, etc., see
    `mindspore.train.callback \
    <https://www.mindspore.cn/docs/en/r1.7/api_python/mindspore.train.html#mindspore.train.callback.Callback>`_.

    Users can obtain the dataset pipeline context through `ds_run_context`, including
    `cur_epoch_num`, `cur_step_num_in_epoch` and `cur_step_num`.

    Note:
        Note that the call is triggered only at the beginning of the second step or epoch.

    Args:
       step_size (int, optional): The number of rows in each step, usually set equal to the batch size. Default: 1.

    Examples:
        >>> import mindspore.nn as nn
        >>> from mindspore.dataset import WaitedDSCallback
        >>> from mindspore import context
        >>> from mindspore.train import Model
        >>> from mindspore.train.callback import Callback
        >>>
        >>> context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
        >>>
        >>> # custom callback class for data synchronization in data pipeline
        >>> class MyWaitedCallback(WaitedDSCallback):
        ...     def __init__(self, events, step_size=1):
        ...         super().__init__(step_size)
        ...         self.events = events
        ...
        ...     # callback method to be executed by data pipeline before the epoch starts
        ...     def sync_epoch_begin(self, train_run_context, ds_run_context):
        ...         event = f"ds_epoch_begin_{ds_run_context.cur_epoch_num}_{ds_run_context.cur_step_num}"
        ...         self.events.append(event)
        ...
        ...     # callback method to be executed by data pipeline before the step starts
        ...     def sync_step_begin(self, train_run_context, ds_run_context):
        ...         event = f"ds_step_begin_{ds_run_context.cur_epoch_num}_{ds_run_context.cur_step_num}"
        ...         self.events.append(event)
        >>>
        >>> # custom callback class for data synchronization in network training
        >>> class MyMSCallback(Callback):
        ...     def __init__(self, events):
        ...         self.events = events
        ...
        ...     # callback method to be executed by network training after the epoch ends
        ...     def epoch_end(self, run_context):
        ...         cb_params = run_context.original_args()
        ...         event = f"ms_epoch_end_{cb_params.cur_epoch_num}_{cb_params.cur_step_num}"
        ...         self.events.append(event)
        ...
        ...     # callback method to be executed by network training after the step ends
        ...     def step_end(self, run_context):
        ...         cb_params = run_context.original_args()
        ...         event = f"ms_step_end_{cb_params.cur_epoch_num}_{cb_params.cur_step_num}"
        ...         self.events.append(event)
        >>>
        >>> # custom network
        >>> class Net(nn.Cell):
        ...     def construct(self, x, y):
        ...         return x
        >>>
        >>> # define a parameter that needs to be synchronized between data pipeline and network training
        >>> events = []
        >>>
        >>> # define callback classes of data pipeline and netwok training
        >>> my_cb1 = MyWaitedCallback(events, 1)
        >>> my_cb2 = MyMSCallback(events)
        >>> arr = [1, 2, 3, 4]
        >>>
        >>> # construct data pipeline
        >>> data = ds.NumpySlicesDataset((arr, arr), column_names=["c1", "c2"], shuffle=False)
        >>> # map the data callback object into the pipeline
        >>> data = data.map(operations=(lambda x: x), callbacks=my_cb1)
        >>>
        >>> net = Net()
        >>> model = Model(net)
        >>>
        >>> # add the data and network callback objects to the model training callback list
        >>> model.train(2, data, dataset_sink_mode=False, callbacks=[my_cb2, my_cb1])
    """

    def __init__(self, step_size=1):
        super().__init__()
        self.step_size = step_size
        self.step_event = threading.Event()
        self.step_run_context = None

        self.epoch_event = threading.Event()
        self.epoch_run_context = None

        self.training_ended = False

    def sync_epoch_begin(self, train_run_context, ds_run_context):
        """
        Called before a new dataset epoch is started and after the previous training epoch is ended.

        Args:
            train_run_context: Include some information of the model with feedback from the previous epoch.
            ds_run_context: Include some information of the data pipeline.
        """

    def sync_step_begin(self, train_run_context, ds_run_context):
        """
        Called before a new dataset step is started and after the previous training step is ended.

        Args:
            train_run_context: Include some information of the model with feedback from the previous step.
            ds_run_context: Include some information of the data pipeline.
        """

    def epoch_end(self, run_context):
        """
        Internal method, do not call/override. Defines epoch_end of Callback to release the wait in ds_epoch_begin.

        Args:
          run_context: Include some information of the model.
        """
        self.epoch_run_context = run_context
        self.epoch_event.set()

    def ds_epoch_begin(self, ds_run_context):
        """
        Internal method, do not call/override. Define mindspore.dataset.DSCallback.ds_epoch_begin
        to wait for mindspore.train.callback.Callback.epoch_end.

        Args:
          ds_run_context: Include some information of the data pipeline.
        """
        if ds_run_context.cur_epoch_num > 1:
            if not self.training_ended:
                success = self.epoch_event.wait(timeout=ds.config.get_callback_timeout())
                self.epoch_event.clear()
                if not success:
                    raise RuntimeError(f"ds_epoch_begin timed out after {ds.config.get_callback_timeout()} second(s).")
            # by the time this thread wakes up, self.epoch_run_context is already available
            self.sync_epoch_begin(self.epoch_run_context, ds_run_context)

    def step_end(self, run_context):
        """
        Internal method, do not call/override. Defines step_end of Callback to release the wait in ds_step_begin.

        Args:
          run_context: Include some information of the model.
        """
        self.step_run_context = run_context
        self.step_event.set()

    def ds_step_begin(self, ds_run_context):
        """
        Internal method, do not call/override. Define mindspore.dataset.DSCallback.ds_step_begin
        to wait for mindspore.train.callback.Callback.step_end.

        Args:
            ds_run_context: Include some information of the data pipeline.
        """
        if ds_run_context.cur_step_num > self.step_size:
            if not self.training_ended:
                success = self.step_event.wait(timeout=ds.config.get_callback_timeout())
                self.step_event.clear()
                if not success:
                    raise RuntimeError(f"ds_step_begin timed out after {ds.config.get_callback_timeout()} second(s).")
                # by the time this thread wakes up, self.epoch_run_context is already available
            self.sync_step_begin(self.step_run_context, ds_run_context)

    def create_runtime_obj(self):
        """
        Internal method, creates a runtime (C++) object from the callback methods defined by the user.

        Returns:
            _c_dataengine.PyDSCallback.
        """
        c_cb = PyDSCallback(self.step_size)
        at_least_one = False

        if self.__class__.sync_step_begin != WaitedDSCallback.sync_step_begin:
            c_cb.set_step_begin(self.ds_step_begin)
            at_least_one = True

        if self.__class__.sync_epoch_begin != WaitedDSCallback.sync_epoch_begin:
            c_cb.set_epoch_begin(self.ds_epoch_begin)
            at_least_one = True

        if not at_least_one:
            raise AttributeError(
                "Inheriting Callback class without overriding any methods, check the usage of user defined Callback.")

        return c_cb

    def end(self, run_context):
        """
        Internal method, release wait when the network training ends.

        Args:
          run_context: Include some information of the model.
        """
        self.epoch_end(run_context)
        self.step_end(run_context)
        self.training_ended = True
