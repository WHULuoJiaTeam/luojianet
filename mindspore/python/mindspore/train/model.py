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
# ============================================================================
"""Model."""
from collections.abc import Iterable
from functools import wraps

import os
import math
import numpy as np

from mindspore import log as logger
from .serialization import save_checkpoint, load_checkpoint
from .callback._checkpoint import ModelCheckpoint
from .callback._checkpoint import _chg_ckpt_file_name_if_same_exist
from ..common.tensor import Tensor
from ..nn.metrics import get_metrics
from .._checkparam import check_input_data, check_output_data, Validator
from .callback import _InternalCallbackParam, RunContext, _CallbackManager, Callback
from .. import context
from ..parallel._utils import _get_parallel_mode, _get_device_num, _get_global_rank, \
    _get_parameter_broadcast, _device_number_check, _parameter_broadcast_check, _parallel_predict_check
from ..parallel._ps_context import _is_role_worker, _is_role_pserver, _is_role_sched, _is_ps_mode, _cache_enable
from ..nn.metrics import Loss
from .. import nn
from ..boost import AutoBoost
from ..context import ParallelMode
from ..parallel._cost_model_context import _set_multi_subgraphs
from ..parallel._recovery_context import _set_recovery_context, _get_recovery_context
from .dataset_helper import DatasetHelper, connect_network_with_dataset
from . import amp
from ..common.api import _pynative_executor, _cell_graph_executor
from ..dataset.engine.datasets import _set_training_dataset, _reset_training_dataset

def _transfer_tensor_to_tuple(inputs):
    """
    If the input is a tensor, convert it to a tuple. If not, the output is unchanged.
    """
    if isinstance(inputs, Tensor):
        return (inputs,)

    return inputs


class _StepSync(Callback):
    @staticmethod
    def step_end(run_context):
        _pynative_executor.sync()


def _save_final_ckpt(func):
    """
    Decorator function, which saves the current checkpoint when an exception occurs during training.
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        obj = None
        if kwargs['callbacks'] and isinstance(kwargs['callbacks'], ModelCheckpoint):
            obj = kwargs['callbacks']
        if kwargs['callbacks'] and isinstance(kwargs['callbacks'], list):
            for item in kwargs['callbacks']:
                if isinstance(item, ModelCheckpoint):
                    obj = item
        if obj and obj._config and obj._config.exception_save:
            try:
                func(self, *args, **kwargs)
            except BaseException as e:
                prefix = _chg_ckpt_file_name_if_same_exist(obj._directory, obj._exception_prefix, True)
                cur_ckpoint_file = prefix + "-" + str(self._current_epoch_num) + "_" \
                    + str(self._current_step_num) + "_breakpoint.ckpt"
                cur_file = os.path.join(obj._directory, cur_ckpoint_file)
                save_checkpoint(self._train_network, cur_file, obj._config.integrated_save, obj._config.async_save,
                                obj._append_dict, obj._config.enc_key, obj._config.enc_mode)
                raise e
        else:
            func(self, *args, **kwargs)
    return wrapper


class Model:
    """
    High-Level API for training or inference.

    `Model` groups layers into an object with training and inference features based on the arguments.

    Note:
        If use mixed precision functions, need to set parameter `optimizer` at the same time,
        otherwise mixed precision functions do not take effect.
        When uses mixed precision functions, `global_step` in optimizer may be different from `cur_step_num` in Model.

    Args:
        network (Cell): A training or testing network.
        loss_fn (Cell): Objective function. If `loss_fn` is None, the `network` should contain the calculation of loss
                        and parallel if needed. Default: None.
        optimizer (Cell): Optimizer for updating the weights. If `optimizer` is None, the `network` needs to
                          do backpropagation and update weights. Default value: None.
        metrics (Union[dict, set]): A Dictionary or a set of metrics for model evaluation.
                                    eg: {'accuracy', 'recall'}. Default: None.
        eval_network (Cell): Network for evaluation. If not defined, `network` and `loss_fn` would be wrapped as
                             `eval_network` . Default: None.
        eval_indexes (list): It is used when eval_network is defined. If `eval_indexes` is None by default, all outputs
                             of the `eval_network` would be passed to metrics. If `eval_indexes` is set, it must contain
                             three elements: the positions of loss value, predicted value and label in outputs of the
                             `eval_network`. In this case, the loss value will be passed to the `Loss` metric, the
                             predicted value and label will be passed to other metrics.
                             :func: `mindindex.nn.metric.set_indexes` is recommended instead of `eval_indexes`.
                             Default: None.
        amp_level (str): Option for argument `level` in :func:`mindspore.build_train_network`, level for mixed
            precision training. Supports ["O0", "O2", "O3", "auto"]. Default: "O0".

            - "O0": Do not change.
            - "O2": Cast network to float16, keep BatchNorm run in float32, using dynamic loss scale.
            - "O3": Cast network to float16, the BatchNorm is also cast to float16, loss scale will not be used.
            - auto: Set level to recommended level in different devices. Set level to "O2" on GPU, set
              level to "O3" on Ascend. The recommended level is chosen by the expert experience, not applicable to all
              scenarios. User should specify the level for special network.

            "O2" is recommended on GPU, "O3" is recommended on Ascend.
            The BatchNorm strategy can be changed by `keep_batchnorm_fp32` settings in `kwargs`. `keep_batchnorm_fp32`
            must be a bool. The loss scale strategy can be changed by `loss_scale_manager` setting in `kwargs`.
            `loss_scale_manager` should be a subclass of :class:`mindspore.LossScaleManager`.
            The more detailed explanation of `amp_level` setting can be found at `mindspore.build_train_network`.

        boost_level (str): Option for argument `level` in `mindspore.boost`, level for boost mode
            training. Supports ["O0", "O1", "O2"]. Default: "O0".

            - "O0": Do not change.
            - "O1": Enable the boost mode, the performance is improved by about 20%, and
              the accuracy is the same as the original accuracy.
            - "O2": Enable the boost mode, the performance is improved by about 30%, and
              the accuracy is reduced by less than 3%.

            If you want to config boost mode by yourself, you can set boost_config_dict as `boost.py`.

    Examples:
        >>> from mindspore import Model, nn
        >>>
        >>> class Net(nn.Cell):
        ...     def __init__(self, num_class=10, num_channel=1):
        ...         super(Net, self).__init__()
        ...         self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        ...         self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        ...         self.fc1 = nn.Dense(16*5*5, 120, weight_init='ones')
        ...         self.fc2 = nn.Dense(120, 84, weight_init='ones')
        ...         self.fc3 = nn.Dense(84, num_class, weight_init='ones')
        ...         self.relu = nn.ReLU()
        ...         self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        ...         self.flatten = nn.Flatten()
        ...
        ...     def construct(self, x):
        ...         x = self.max_pool2d(self.relu(self.conv1(x)))
        ...         x = self.max_pool2d(self.relu(self.conv2(x)))
        ...         x = self.flatten(x)
        ...         x = self.relu(self.fc1(x))
        ...         x = self.relu(self.fc2(x))
        ...         x = self.fc3(x)
        ...         return x
        >>>
        >>> net = Net()
        >>> loss = nn.SoftmaxCrossEntropyWithLogits()
        >>> optim = nn.Momentum(params=net.trainable_params(), learning_rate=0.1, momentum=0.9)
        >>> model = Model(net, loss_fn=loss, optimizer=optim, metrics=None)
        >>> # For details about how to build the dataset, please refer to the function `create_dataset` in tutorial
        >>> # document on the official website:
        >>> # https://www.mindspore.cn/tutorials/zh-CN/r1.7/beginner/quick_start.html
        >>> dataset = create_custom_dataset()
        >>> model.train(2, dataset)
    """

    def __init__(self, network, loss_fn=None, optimizer=None, metrics=None, eval_network=None, eval_indexes=None,
                 amp_level="O0", boost_level="O0", **kwargs):
        self._network = network
        self._loss_fn = loss_fn
        self._optimizer = optimizer
        self._loss_scale_manager = None
        self._loss_scale_manager_set = False
        self._keep_bn_fp32 = None
        self._check_kwargs(kwargs)
        self._amp_level = amp_level
        self._boost_level = boost_level
        self._eval_network = eval_network
        self._process_amp_args(kwargs)
        self._parallel_mode = _get_parallel_mode()
        self._device_number = _get_device_num()
        self._global_rank = _get_global_rank()
        self._parameter_broadcast = _get_parameter_broadcast()
        self._metrics = metrics

        self._check_amp_level_arg(optimizer, amp_level)
        self._check_for_graph_cell(kwargs)
        self._build_boost_network(kwargs)
        self._train_network = self._build_train_network()
        self._build_eval_network(metrics, self._eval_network, eval_indexes)
        self._build_predict_network()
        self._current_epoch_num = 0
        self._current_step_num = 0

    def _check_for_graph_cell(self, kwargs):
        """Check for graph cell"""
        if not isinstance(self._network, nn.GraphCell):
            return
        if self._amp_level != "O0":
            logger.warning("amp_level will not work when network is a GraphCell.")

        if self._loss_fn is not None or self._optimizer is not None:
            raise ValueError("For 'Model', 'loss_fn' and 'optimizer' should be None when network is a GraphCell, "
                             "but got 'loss_fn': {}, 'optimizer': {}.".format(self._loss_fn, self._optimizer))
        if kwargs:
            raise ValueError("For 'Model', the '**kwargs' argument should be empty when network is a GraphCell.")

    def _process_amp_args(self, kwargs):
        if 'keep_batchnorm_fp32' in kwargs:
            self._keep_bn_fp32 = kwargs['keep_batchnorm_fp32']
        if 'loss_scale_manager' in kwargs:
            self._loss_scale_manager = kwargs['loss_scale_manager']
            self._loss_scale_manager_set = True

    def _check_amp_level_arg(self, optimizer, amp_level):
        if optimizer is None and amp_level != "O0":
            raise ValueError(
                "Auto mixed precision will not work because 'optimizer' is None.Please set amp_level='O0' "
                "to disable auto mixed precision or set 'optimizer' not be None to use auto mixed precision.")

    def _check_kwargs(self, kwargs):
        for arg in kwargs:
            if arg not in ['loss_scale_manager', 'keep_batchnorm_fp32', 'boost_config_dict', 'acc_level']:
                raise ValueError(f"The argument in 'kwargs' should be 'loss_scale_manager' or "
                                 f"'keep_batchnorm_fp32' or 'boost_config_dict' or 'acc_level', but got '{arg}'.")

    def _check_reuse_dataset(self, dataset):
        if not hasattr(dataset, '__model_hash__'):
            dataset.__model_hash__ = hash(self)
        if hasattr(dataset, '__model_hash__') and dataset.__model_hash__ != hash(self):
            raise RuntimeError("The dataset object had been used in other model by model.train(...), "
                               "please create a new dataset.")

    def _build_boost_network(self, kwargs):
        """Build the boost network."""
        boost_config_dict = ""
        if 'boost_config_dict' in kwargs:
            boost_config_dict = kwargs['boost_config_dict']
        if 'acc_level' in kwargs:
            self._boost_level = kwargs['acc_level']
            logger.warning("Next version acc_level will be removed, please replace with boost_level")
        processor = AutoBoost(self._boost_level, boost_config_dict)
        if processor.level not in ["O1", "O2"]:
            return
        if self._optimizer is None:
            logger.warning("In boost mode, the optimizer must be defined.")
            return
        if self._eval_network is None and self._metrics is None:
            logger.warning("In boost mode, the eval_network and metrics cannot be undefined at the same time.")
            return

        self._network, self._optimizer = processor.network_auto_process_train(self._network, self._optimizer)
        if self._eval_network is not None:
            self._eval_network = processor.network_auto_process_eval(self._eval_network)

    def _build_train_network(self):
        """Build train network"""
        network = self._network
        if self._loss_scale_manager is not None and self._optimizer is None:
            raise ValueError("The argument 'optimizer' can not be None when set 'loss_scale_manager'.")

        if self._optimizer:
            amp_config = {}
            if self._loss_scale_manager_set:
                amp_config['loss_scale_manager'] = self._loss_scale_manager
            if self._keep_bn_fp32 is not None:
                amp_config['keep_batchnorm_fp32'] = self._keep_bn_fp32
            network = amp.build_train_network(network,
                                              self._optimizer,
                                              self._loss_fn,
                                              level=self._amp_level,
                                              boost_level=self._boost_level,
                                              **amp_config)
        elif self._loss_fn:
            network = nn.WithLossCell(network, self._loss_fn)
        # If need to check if loss_fn is not None, but optimizer is None

        if self._parallel_mode in (ParallelMode.SEMI_AUTO_PARALLEL, ParallelMode.AUTO_PARALLEL):
            network.set_auto_parallel()
            if self._optimizer is None:
                # In this case, multiple optimizer(s) is supposed to be included in 'self._network'
                _set_multi_subgraphs()
        if self._network.get_inputs() is not None:
            network.set_inputs(*self._network.get_inputs())
        return network

    def _build_eval_network(self, metrics, eval_network, eval_indexes):
        """Build the network for evaluation."""
        self._metric_fns = get_metrics(metrics)
        if not self._metric_fns:
            return

        if eval_network is not None:
            if eval_indexes is not None and not (isinstance(eval_indexes, list) and len(eval_indexes) == 3):
                raise ValueError("The argument 'eval_indexes' must be a list or None. If 'eval_indexes' is a list, "
                                 "length of it must be three. But got 'eval_indexes' {}".format(eval_indexes))

            self._eval_network = eval_network
            self._eval_indexes = eval_indexes
        else:
            if self._loss_fn is None:
                raise ValueError(f"If `metrics` is set, `eval_network` must not be None. Do not set `metrics` if you"
                                 f" don't want an evaluation.\n"
                                 f"If evaluation is required, you need to specify `eval_network`, which will be used in"
                                 f" the framework to evaluate the model.\n"
                                 f"For the simple scenarios with one data, one label and one logits, `eval_network` is"
                                 f" optional, and then you can set `eval_network` or `loss_fn`. For the latter case,"
                                 f" framework will automatically build an evaluation network with `network` and"
                                 f" `loss_fn`.")

            self._eval_network = nn.WithEvalCell(self._network, self._loss_fn, self._amp_level in ["O2", "O3", "auto"])
            self._eval_indexes = [0, 1, 2]

        if self._parallel_mode in (ParallelMode.SEMI_AUTO_PARALLEL, ParallelMode.AUTO_PARALLEL):
            if self._optimizer is None:
                # In this case, multiple optimizer(s) is supposed to be included in 'self._network'
                _set_multi_subgraphs()
            self._eval_network.set_auto_parallel()

    def _build_predict_network(self):
        """Build the network for prediction."""
        self._predict_network = self._network
        if self._parallel_mode in (ParallelMode.SEMI_AUTO_PARALLEL, ParallelMode.AUTO_PARALLEL):
            # Unlike the cases in build_train_network() and build_eval_network(), 'multi_subgraphs' is not set
            self._predict_network.set_auto_parallel()

    def _clear_metrics(self):
        """Clear metrics local values."""
        for metric in self._metric_fns.values():
            metric.clear()

    def _update_metrics(self, outputs):
        """Update metrics local values."""
        if isinstance(outputs, Tensor):
            outputs = (outputs,)
        if not isinstance(outputs, tuple):
            raise ValueError(f"The argument 'outputs' should be tuple, but got {type(outputs)}.")

        if self._eval_indexes is not None and len(outputs) < 3:
            raise ValueError("The length of 'outputs' must be >= 3, but got {}".format(len(outputs)))

        for metric in self._metric_fns.values():
            if self._eval_indexes is None:
                metric.update(*outputs)
            else:
                if isinstance(metric, Loss):
                    metric.update(outputs[self._eval_indexes[0]])
                else:
                    metric.update(outputs[self._eval_indexes[1]], outputs[self._eval_indexes[2]])

    def _get_metrics(self):
        """Get metrics local values."""
        metrics = dict()
        for key, value in self._metric_fns.items():
            metrics[key] = value.eval()
        return metrics

    def _get_scaling_sens(self):
        """get the scaling sens"""
        scaling_sens = 1
        if self._loss_scale_manager is not None:
            scaling_sens = self._loss_scale_manager.get_loss_scale()
        if self._parallel_mode == ParallelMode.DATA_PARALLEL:
            scaling_sens /= self._device_number
        return scaling_sens

    def _exec_preprocess(self, is_train, dataset, dataset_sink_mode, sink_size=-1, epoch_num=1, dataset_helper=None):
        """Initializes dataset."""
        if is_train:
            network = self._train_network
            phase = 'train'
        else:
            network = self._eval_network
            phase = 'eval'

        if dataset_sink_mode and not is_train:
            dataset.__loop_size__ = 1

        if dataset_helper is None:
            dataset_helper = DatasetHelper(dataset, dataset_sink_mode, sink_size, epoch_num)

        if dataset_sink_mode:
            network = connect_network_with_dataset(network, dataset_helper)

        if is_train:
            _set_training_dataset(dataset_helper)  # pylint: disable=W0212


        network.set_train(is_train)
        network.phase = phase

        if self._parallel_mode in (ParallelMode.SEMI_AUTO_PARALLEL, ParallelMode.AUTO_PARALLEL):
            network.set_auto_parallel()

        return dataset_helper, network

    def _warmup_dataset(self, epoch, train_dataset, sink_size=-1):
        """
        Trigger dataset pipeline running before graph compiling.

        Args:
            epoch (int): Total number of iterations on the data.
            train_dataset (Dataset): A training dataset iterator. If `train_dataset` is defined, training graphs will be
                                     initialized. Default: None.
            sink_size (int): Control the amount of data in each sink. Default: -1.
        """
        if sink_size == -1:
            epoch_num = epoch
        else:
            epoch_num = math.ceil(epoch * sink_size / train_dataset.get_dataset_size())
            train_dataset.__total_batch__ = epoch * sink_size
        dataset_helper = None
        dataset_helper, _ = self._exec_preprocess(is_train=True,
                                                  dataset=train_dataset,
                                                  dataset_sink_mode=True,
                                                  sink_size=sink_size,
                                                  epoch_num=epoch_num,
                                                  dataset_helper=dataset_helper)
        train_dataset._dataset_helper = dataset_helper
        train_dataset._warmup_epoch = epoch

    def _init(self, train_dataset=None, valid_dataset=None, sink_size=-1, epoch=1):
        """
        Initialize compute graphs and data graphs with the sink mode.

        Note:
            Pre-init process only supports `GRAPH_MODE` and `Ascend` target currently.

        Args:
            train_dataset (Dataset): A training dataset iterator. If `train_dataset` is defined, training graphs will be
                                     initialized. Default: None.
            valid_dataset (Dataset): A evaluating dataset iterator. If `valid_dataset` is defined, evaluation graphs
                                     will be initialized, and `metrics` in `Model` can not be None. Default: None.
            sink_size (int): Control the amount of data in each sink. Default: -1.
            epoch (int): Total number of iterations on the data. Default: 1.
        """
        if context.get_context("mode") != context.GRAPH_MODE or context.get_context("device_target") != "Ascend":
            raise RuntimeError('Pre-init process only supports GRAPH MODE and Ascend target currently.')

        if not train_dataset and not valid_dataset:
            raise ValueError("The argument 'train_dataset' and 'valid_dataset' can not both be None or empty.")

        _device_number_check(self._parallel_mode, self._device_number)

        if train_dataset:
            _parameter_broadcast_check(self._parallel_mode, self._parameter_broadcast)
            if self._parameter_broadcast:
                self._train_network.set_broadcast_flag()

            train_dataset.__no_send__ = True
            train_dataset_helper, train_network = self._exec_preprocess(is_train=True,
                                                                        dataset=train_dataset,
                                                                        dataset_sink_mode=True,
                                                                        sink_size=sink_size)
            self._warmup_dataset(epoch, train_dataset, sink_size)
            if context.get_auto_parallel_context("pipeline_stages") > 1 and valid_dataset:
                train_network.add_flags_recursive(is_first_iteration=True)
            for inputs in train_dataset_helper:
                train_network.compile(*inputs)
                break

        if valid_dataset:
            if not self._metric_fns:
                raise RuntimeError("If define `valid_dataset`, metric fn can not be None or empty, "
                                   "you should set the argument 'metrics' for model.")

            valid_dataset.__no_send__ = True
            valid_dataset_helper, eval_network = self._exec_preprocess(is_train=False,
                                                                       dataset=valid_dataset,
                                                                       dataset_sink_mode=True)
            if context.get_auto_parallel_context("pipeline_stages") > 1:
                eval_network.add_flags_recursive(is_first_iteration=False)
            for inputs in valid_dataset_helper:
                eval_network.compile(*inputs)
                break

    @_save_final_ckpt
    def _train(self, epoch, train_dataset, callbacks=None, dataset_sink_mode=True, sink_size=-1):
        """
        Training.

        Args:
            epoch (int): Total number of iterations on the data.
            train_dataset (Dataset): A training dataset iterator. If there is no
                                     loss_fn, a tuple with multiple data (data1, data2, data3, ...) will be
                                     returned and passed to the network. Otherwise, a tuple (data, label) will
                                     be returned. The data and label would be passed to the network and loss
                                     function respectively.
            callbacks (list): List of callback objects which should be executed while training. Default: None.
            dataset_sink_mode (bool): Determine whether the data should be passed through the dataset channel.
                                      Default: True.
                                      Configure pynative mode or CPU, the training process will be performed with
                                      dataset not sink.
            sink_size (int): Control the amount of data in each sink. Default: -1.
        """
        epoch = Validator.check_positive_int(epoch)
        if self._parameter_broadcast:
            self._train_network.set_broadcast_flag()

        cb_params = _InternalCallbackParam()
        cb_params.train_network = self._train_network
        cb_params.epoch_num = epoch
        if dataset_sink_mode and sink_size > 0:
            cb_params.batch_num = sink_size
        else:
            cb_params.batch_num = train_dataset.get_dataset_size()
        cb_params.mode = "train"
        cb_params.loss_fn = self._loss_fn
        cb_params.optimizer = self._optimizer
        cb_params.parallel_mode = self._parallel_mode
        cb_params.device_number = self._device_number
        cb_params.train_dataset = train_dataset
        cb_params.list_callback = self._transform_callbacks(callbacks)
        if context.get_context("mode") == context.PYNATIVE_MODE:
            cb_params.list_callback.insert(0, _StepSync())
            callbacks = cb_params.list_callback
        cb_params.train_dataset_element = None
        cb_params.network = self._network
        if _is_role_pserver() or _is_role_sched():
            epoch = 1
        cb_params.last_save_ckpt_step = None
        cb_params.latest_ckpt_file = None

        # build callback list
        with _CallbackManager(callbacks) as list_callback:
            self._check_reuse_dataset(train_dataset)
            if not dataset_sink_mode:
                self._train_process(epoch, train_dataset, list_callback, cb_params)
            elif context.get_context("device_target") == "CPU":
                logger.info("The CPU cannot support dataset sink mode currently."
                            "So the training process will be performed with dataset not sink.")
                self._train_process(epoch, train_dataset, list_callback, cb_params)
            else:
                self._train_dataset_sink_process(epoch, train_dataset, list_callback, cb_params, sink_size)

    @staticmethod
    def _transform_callbacks(callbacks):
        """Transform callback to a list."""
        if callbacks is None:
            return []

        if isinstance(callbacks, Iterable):
            return list(callbacks)

        return [callbacks]

    def _train_dataset_sink_process(self, epoch, train_dataset, list_callback=None, cb_params=None, sink_size=-1):
        """
        Training process. The data would be passed to network through dataset channel.

        Args:
            epoch (int): Total number of iterations on the data.
            train_dataset (Dataset): A training dataset iterator. If there is no
                                     loss_fn, a tuple with multiple data (data1, data2, data3, ...) should be
                                     returned and passed to the network. Otherwise, a tuple (data, label) should
                                     be returned. The data and label would be passed to the network and loss
                                     function respectively.
            list_callback (Callback): Executor of callback list. Default: None.
            cb_params (_InternalCallbackParam): Callback parameters. Default: None.
            sink_size (int): Control the amount of data in each sink. Default: -1.
        """
        is_graph = (context.get_context("mode") == context.GRAPH_MODE)
        if sink_size == -1:
            epoch_num = epoch
        else:
            epoch_num = math.ceil(epoch * sink_size / train_dataset.get_dataset_size())
            train_dataset.__total_batch__ = epoch * sink_size

        cb_params.cur_step_num = 0
        cb_params.dataset_sink_mode = True

        run_context = RunContext(cb_params)
        list_callback.begin(run_context)
        # used to stop training for early stop, such as stopAtTIme or stopATStep
        should_stop = False
        dataset_helper = None
        if hasattr(train_dataset, '_dataset_helper'):
            dataset_helper = train_dataset._dataset_helper

        self.epoch_iter = 0

        self._check_enable_recovery()
        # Used to check whether need perform recovery for process which is restarted.
        self._check_need_load_ckpt(cb_params, train_dataset.get_dataset_size(), sink_size)

        while self.epoch_iter < epoch:
            cb_params.cur_epoch_num = self.epoch_iter + 1
            self._current_epoch_num = cb_params.cur_epoch_num
            self._current_step_num = 0
            list_callback.epoch_begin(run_context)
            dataset_helper, train_network = self._exec_preprocess(is_train=True,
                                                                  dataset=train_dataset,
                                                                  dataset_sink_mode=True,
                                                                  sink_size=sink_size,
                                                                  epoch_num=epoch_num,
                                                                  dataset_helper=dataset_helper)

            cb_params.train_network = train_network

            # Perform recovery for process which is restarted.
            self._reset_training_step_for_abnormal_process(cb_params)
            # Perform recovery for process which is not restarted.
            self._reset_training_step_for_normal_process(cb_params, dataset_helper)

            # For data sink dataset_helper only iter once, other wise iter epoch_size times.
            for inputs in dataset_helper:
                if is_graph:
                    cb_params.cur_step_num += dataset_helper.sink_size()
                else:
                    cb_params.cur_step_num += 1
                self._current_step_num = int((cb_params.cur_step_num - 1) % cb_params.batch_num + 1)
                cb_params.train_dataset_element = inputs
                list_callback.step_begin(run_context)
                outputs = train_network(*inputs)
                cb_params.net_outputs = outputs
                # In disaster recovery scenarios, need not to execute callbacks if this step executes failed.
                need_exec_callback_step_end = not (self.enable_recovery and _get_recovery_context("need_reset"))
                if need_exec_callback_step_end:
                    list_callback.step_end(run_context)

                if _is_role_pserver() or _is_role_sched():
                    os._exit(0)

            dataset_helper.continue_send()
            # In disaster recovery scenarios, need not to execute callbacks if this epoch executes failed.
            need_exec_callback_epoch_end = not (self.enable_recovery and _get_recovery_context("need_reset"))
            if need_exec_callback_epoch_end:
                list_callback.epoch_end(run_context)

            should_stop = should_stop or run_context.get_stop_requested()
            if should_stop:
                break

            need_reset_to_beginning = self.enable_recovery and _get_recovery_context("need_reset")\
                                      and not _get_recovery_context("latest_ckpt_file")
            if need_reset_to_beginning:
                self.epoch_iter = 0
                cb_params.cur_step_num = 0
                continue

            self.epoch_iter += 1

        dataset_helper.stop_send()
        dataset_helper.release()

        list_callback.end(run_context)

    def _check_enable_recovery(self):
        """
        Check whether enable recovery and execution mode consistency.
        """

        enable_recovery = _get_recovery_context("enable_recovery")
        if not enable_recovery:
            self.enable_recovery = False
        else:
            if context.get_context("mode") != context.GRAPH_MODE:
                raise RuntimeError("Recovery for training only support graph mode currently.")
            self.enable_recovery = enable_recovery and _is_role_worker()

    def _check_need_load_ckpt(self, cb_params, dataset_size, sink_size=-1):
        """
        Check whether need to load checkpoint after abnormal process restart.

        Args:
            cb_params (_InternalCallbackParam): Callback parameters.
            dataset_size (int): The number of batches in a dataset.
            sink_size (int): Control the amount of data in each sink. Default: -1.
        """

        if not self.enable_recovery:
            self.need_load_ckpt = False

        cb_params.latest_ckpt_file = _get_recovery_context("latest_ckpt_file")
        if cb_params.latest_ckpt_file:
            recovery_epoch_num = _get_recovery_context("latest_ckpt_epoch")
            recovery_step_num = _get_recovery_context("latest_ckpt_step")
            dataset_sink_size = sink_size if sink_size > 0 else dataset_size
            cb_params.cur_step_num = (recovery_epoch_num - 1) * dataset_sink_size + recovery_step_num
            cb_params.last_save_ckpt_step = cb_params.cur_step_num
            self.epoch_iter = recovery_epoch_num
            self.need_load_ckpt = True
        else:
            self.need_load_ckpt = False

    def _reset_training_step_for_abnormal_process(self, cb_params):
        """
        Execute recovery for abnormal exit process when restart.

        Args:
            cb_params (_InternalCallbackParam): Callback parameters.
        """

        if self.need_load_ckpt:
            try:
                load_checkpoint(cb_params.latest_ckpt_file, cb_params.train_network)
            except BaseException as e:
                os.remove(cb_params.latest_ckpt_file)
                raise RuntimeError(e.__str__() + ", load ckpt failed and remove the ckpt: "\
                                   + cb_params.latest_ckpt_file)
            _reset_training_dataset(cb_params.cur_step_num)
            self.need_load_ckpt = False

    def  _reset_training_step_for_normal_process(self, cb_params, dataset_helper):
        """
        Execute recovery for normal process when there is process exit abnormally.

        Args:
            cb_params (_InternalCallbackParam): Callback parameters.
            dataset_helper (DatasetHelper): A class to process the MindData dataset,
                it provides the type, shape and queue name of the dataset to wrap the `GetNext`.
        """

        if self.enable_recovery and _get_recovery_context("need_reset"):
            cb_params.latest_ckpt_file = _get_recovery_context("latest_ckpt_file")
            if cb_params.latest_ckpt_file:
                try:
                    load_checkpoint(cb_params.latest_ckpt_file, cb_params.train_network)
                except BaseException as e:
                    os.remove(cb_params.latest_ckpt_file)
                    raise RuntimeError(e.__str__() + ", load ckpt failed and remove the ckpt: "\
                         + cb_params.latest_ckpt_file)

                recovery_epoch_num = _get_recovery_context("latest_ckpt_epoch")
                recovery_step_num = _get_recovery_context("latest_ckpt_step")
                cb_params.cur_step_num = (recovery_epoch_num - 1) * dataset_helper.sink_size() + recovery_step_num
                self.epoch_iter = recovery_epoch_num
                cb_params.cur_epoch_num = self.epoch_iter + 1
                cb_params.last_save_ckpt_step = cb_params.cur_step_num
                _reset_training_dataset(cb_params.cur_step_num)
            else:
                _reset_training_dataset(0)

            _set_recovery_context(need_reset=False)


    def _train_process(self, epoch, train_dataset, list_callback=None, cb_params=None):
        """
        Training process. The data would be passed to network directly.

        Args:
            epoch (int): Total number of iterations on the data.
            train_dataset (Dataset): A training dataset iterator. If there is no
                                     loss_fn, a tuple with multiple data (data1, data2, data3, ...) should be
                                     returned and passed to the network. Otherwise, a tuple (data, label) should
                                     be returned. The data and label would be passed to the network and loss
                                     function respectively.
            list_callback (Callback): Executor of callback list. Default: None.
            cb_params (_InternalCallbackParam): Callback parameters. Default: None.
        """
        dataset_helper, _ = self._exec_preprocess(is_train=True,
                                                  dataset=train_dataset,
                                                  dataset_sink_mode=False,
                                                  epoch_num=epoch)
        cb_params.cur_step_num = 0
        cb_params.dataset_sink_mode = False
        run_context = RunContext(cb_params)
        list_callback.begin(run_context)
        # used to stop training for early stop, such as stopAtTIme or stopATStep
        should_stop = False
        for i in range(epoch):
            cb_params.cur_epoch_num = i + 1
            self._current_epoch_num = cb_params.cur_epoch_num
            self._current_step_num = 0

            list_callback.epoch_begin(run_context)

            for next_element in dataset_helper:
                len_element = len(next_element)
                next_element = _transfer_tensor_to_tuple(next_element)
                if self._loss_fn and len_element != 2:
                    raise ValueError("When 'loss_fn' is not None, 'train_dataset' should return "
                                     "two elements, but got {}, please check the number of elements "
                                     "returned by 'train_dataset'".format(len_element))
                cb_params.cur_step_num += 1
                self._current_step_num = int((cb_params.cur_step_num - 1) % cb_params.batch_num + 1)

                cb_params.train_dataset_element = next_element
                list_callback.step_begin(run_context)
                outputs = self._train_network(*next_element)
                cb_params.net_outputs = outputs
                if self._loss_scale_manager and self._loss_scale_manager.get_drop_overflow_update():
                    _, overflow, _ = outputs
                    overflow = np.all(overflow.asnumpy())
                    self._loss_scale_manager.update_loss_scale(overflow)

                list_callback.step_end(run_context)
                if _is_role_pserver() or _is_role_sched():
                    os._exit(0)
                should_stop = should_stop or run_context.get_stop_requested()
                if should_stop:
                    break

            train_dataset.reset()

            # if param is cache enable, flush data from cache to host before epoch end
            self._flush_from_cache(cb_params)

            list_callback.epoch_end(run_context)
            should_stop = should_stop or run_context.get_stop_requested()
            if should_stop:
                break

        list_callback.end(run_context)

    def train(self, epoch, train_dataset, callbacks=None, dataset_sink_mode=True, sink_size=-1):
        """
        Training API.

        When setting pynative mode or CPU, the training process will be performed with dataset not sink.

        Note:
            If dataset_sink_mode is True, data will be sent to device. If the device is Ascend, features
            of data will be transferred one by one. The limitation of data transmission per time is 256M.

            When dataset_sink_mode is True, the `step_end` method of the instance of Callback will be called at the end
            of epoch.

            If dataset_sink_mode is True, dataset will be bound to this model and cannot be used by other models.

            If sink_size > 0, each epoch of the dataset can be traversed unlimited times until you get sink_size
            elements of the dataset. The next epoch continues to traverse from the end position of the previous
            traversal.

            The interface builds the computational graphs and then executes the computational graphs. However, when
            the `Model.build` is executed first, it only performs the graphs execution.

        Args:
            epoch (int): Total training epochs. Generally, train network will be trained on complete dataset per epoch.
                         If `dataset_sink_mode` is set to True and `sink_size` is greater than 0, each epoch will
                         train `sink_size` steps instead of total steps of dataset.
            train_dataset (Dataset): A training dataset iterator. If `loss_fn` is defined, the data and label will be
                                     passed to the `network` and the `loss_fn` respectively, so a tuple (data, label)
                                     should be returned from dataset. If there is multiple data or labels, set `loss_fn`
                                     to None and implement calculation of loss in `network`,
                                     then a tuple (data1, data2, data3, ...) with all data returned from dataset will be
                                     passed to the `network`.
            callbacks (Optional[list[Callback], Callback]): List of callback objects or callback object,
                                                            which should be executed while training.
                                                            Default: None.
            dataset_sink_mode (bool): Determines whether to pass the data through dataset channel.
                                      Configure pynative mode or CPU, the training process will be performed with
                                      dataset not sink. Default: True.
            sink_size (int): Control the amount of data in each sink. `sink_size` is invalid if `dataset_sink_mode`
                             is False.
                             If sink_size = -1, sink the complete dataset for each epoch.
                             If sink_size > 0, sink sink_size data for each epoch.
                             Default: -1.

        Examples:
            >>> from mindspore import Model, nn, FixedLossScaleManager
            >>>
            >>> # For details about how to build the dataset, please refer to the tutorial
            >>> # document on the official website.
            >>> dataset = create_custom_dataset()
            >>> net = Net()
            >>> loss = nn.SoftmaxCrossEntropyWithLogits()
            >>> loss_scale_manager = FixedLossScaleManager()
            >>> optim = nn.Momentum(params=net.trainable_params(), learning_rate=0.1, momentum=0.9)
            >>> model = Model(net, loss_fn=loss, optimizer=optim, metrics=None, loss_scale_manager=loss_scale_manager)
            >>> model.train(2, dataset)
        """
        dataset_sink_mode = Validator.check_bool(dataset_sink_mode)
        if isinstance(self._train_network, nn.GraphCell) and dataset_sink_mode:
            raise ValueError("Dataset sink mode is currently not supported when training with a GraphCell.")

        if hasattr(train_dataset, '_warmup_epoch') and train_dataset._warmup_epoch != epoch:
            raise ValueError("When use Model.build to initialize model, the value of parameter 'epoch' in Model.build "
                             "should be equal to value in Model.train, but got the value of epoch in build {} and "
                             "the value of epoch in train {} separately."
                             .format(train_dataset._warmup_epoch, epoch))

        if dataset_sink_mode and _is_ps_mode() and not _cache_enable():
            raise ValueError("Parameter server mode does not support 'data_sink_mode=True'.")

        Validator.check_is_int(sink_size)
        dataset_size = train_dataset.get_dataset_size()
        if dataset_size == 0:
            raise ValueError("There is no valid data in dataset, please check dataset file firstly.")
        if sink_size == -1:
            sink_size = dataset_size
        if sink_size < -1 or sink_size == 0:
            raise ValueError("For 'Model.train', The argument 'sink_size' must be -1 or positive, "
                             "but got {}.".format(sink_size))

        _device_number_check(self._parallel_mode, self._device_number)

        self._train(epoch,
                    train_dataset,
                    callbacks=callbacks,
                    dataset_sink_mode=dataset_sink_mode,
                    sink_size=sink_size)

    def build(self, train_dataset=None, valid_dataset=None, sink_size=-1, epoch=1, jit_config=None):
        """
        Build computational graphs and data graphs with the sink mode.

        .. warning::
            This is an experimental prototype that is subject to change or deletion.

        Note:
            The interface builds the computational graphs, when the interface is executed first, 'Model.train' only
            performs the graphs execution. Pre-build process only supports `GRAPH_MODE` and `Ascend` target currently.
            It only supports dataset sink mode.

        Args:
            train_dataset (Dataset): A training dataset iterator. If `train_dataset` is defined, training graphs will be
                                     built. Default: None.
            valid_dataset (Dataset): An evaluating dataset iterator. If `valid_dataset` is defined, evaluation graphs
                                     will be built, and `metrics` in `Model` can not be None. Default: None.
            sink_size (int): Control the amount of data in each sink. Default: -1.
            epoch (int): Control the training epochs. Default: 1.
            jit_config (Union[str, str]): Control the jit config.
                                          By default, if set to None, the graph will compile as the default behavior.
                                          You can customize the compile config with a dictionary.
                                          For example, you can set {'jit_level': 'o0'} to control the jit level.
                                          The data that supports control is shown below. Default: None.

                                          - jit_level (string): Control the graph compile optimize level.
                                            Optional: o0/o1. Default: o1. If set to o0, the graph compiling will pass
                                            the combine like graph phase.

        Examples:
            >>> from mindspore import Model, nn, FixedLossScaleManager
            >>>
            >>> # For details about how to build the dataset, please refer to the tutorial
            >>> # document on the official website.
            >>> dataset = create_custom_dataset()
            >>> net = Net()
            >>> loss = nn.SoftmaxCrossEntropyWithLogits()
            >>> loss_scale_manager = FixedLossScaleManager()
            >>> optim = nn.Momentum(params=net.trainable_params(), learning_rate=0.1, momentum=0.9)
            >>> model = Model(net, loss_fn=loss, optimizer=optim, metrics=None, loss_scale_manager=loss_scale_manager)
            >>> model.build(dataset, epoch=2)
            >>> model.train(2, dataset)
        """
        if jit_config is not None:
            _cell_graph_executor.set_jit_config(jit_config)
        self._init(train_dataset, valid_dataset, sink_size, epoch)

    def _eval_dataset_sink_process(self, valid_dataset, list_callback=None, cb_params=None):
        """
        Evaluation. The data would be passed to network through dataset channel.

        Args:
            valid_dataset (Dataset): Dataset to evaluate the model.
            list_callback (Callback): Executor of callback list. Default: None.
            cb_params (_InternalCallbackParam): Callback parameters. Default: None.

        Returns:
            Dict, which returns the loss value and metrics values for the model in the test mode.
        """
        run_context = RunContext(cb_params)

        dataset_helper, eval_network = self._exec_preprocess(is_train=False,
                                                             dataset=valid_dataset,
                                                             dataset_sink_mode=True)
        cb_params.eval_network = eval_network
        cb_params.dataset_sink_mode = True
        list_callback.begin(run_context)
        list_callback.epoch_begin(run_context)
        for inputs in dataset_helper:
            cb_params.cur_step_num += 1
            list_callback.step_begin(run_context)
            outputs = eval_network(*inputs)
            cb_params.net_outputs = outputs
            list_callback.step_end(run_context)
            self._update_metrics(outputs)

        list_callback.epoch_end(run_context)
        metrics = self._get_metrics()
        cb_params.metrics = metrics
        list_callback.end(run_context)

        return metrics

    def _eval_process(self, valid_dataset, list_callback=None, cb_params=None):
        """
        Evaluation. The data would be passed to network directly.

        Args:
            valid_dataset (Dataset): Dataset to evaluate the model.
            list_callback (Callback): Executor of callback list. Default: None.
            cb_params (_InternalCallbackParam): Callback parameters. Default: None.

        Returns:
            Dict, which returns the loss value and metrics values for the model in the test mode.
        """
        run_context = RunContext(cb_params)
        cb_params.dataset_sink_mode = False
        list_callback.begin(run_context)
        dataset_helper, _ = self._exec_preprocess(is_train=False,
                                                  dataset=valid_dataset,
                                                  dataset_sink_mode=False)
        list_callback.epoch_begin(run_context)
        for next_element in dataset_helper:
            cb_params.cur_step_num += 1
            list_callback.step_begin(run_context)
            next_element = _transfer_tensor_to_tuple(next_element)
            outputs = self._eval_network(*next_element)
            cb_params.net_outputs = outputs
            list_callback.step_end(run_context)
            self._update_metrics(outputs)

        list_callback.epoch_end(run_context)
        valid_dataset.reset()
        metrics = self._get_metrics()
        cb_params.metrics = metrics
        list_callback.end(run_context)
        return metrics

    def eval(self, valid_dataset, callbacks=None, dataset_sink_mode=True):
        """
        Evaluation API.

        Configure to pynative mode or CPU, the evaluating process will be performed with dataset non-sink mode.

        Note:
            If dataset_sink_mode is True, data will be sent to device. At this point, the dataset will be bound to this
            model, so the dataset cannot be used by other models. If the device is Ascend, features
            of data will be transferred one by one. The limitation of data transmission per time is 256M.

            The interface builds the computational graphs and then executes the computational graphs. However, when
            the `Model.build` is executed first, it only performs the graphs execution.

        Args:
            valid_dataset (Dataset): Dataset to evaluate the model.
            callbacks (Optional[list(Callback), Callback]): List of callback objects or callback object,
                                                            which should be executed while evaluation.
                                                            Default: None.
            dataset_sink_mode (bool): Determines whether to pass the data through dataset channel.
                Default: True.

        Returns:
            Dict, the key is the metric name defined by users and the value is the metrics value for
            the model in the test mode.

        Examples:
            >>> from mindspore import Model, nn
            >>>
            >>> # For details about how to build the dataset, please refer to the tutorial
            >>> # document on the official website.
            >>> dataset = create_custom_dataset()
            >>> net = Net()
            >>> loss = nn.SoftmaxCrossEntropyWithLogits()
            >>> model = Model(net, loss_fn=loss, optimizer=None, metrics={'acc'})
            >>> acc = model.eval(dataset, dataset_sink_mode=False)
        """
        dataset_sink_mode = Validator.check_bool(dataset_sink_mode)

        _device_number_check(self._parallel_mode, self._device_number)
        if not self._metric_fns:
            raise ValueError("For Model.eval, the model argument 'metrics' can not be None or empty, "
                             "you should set the argument 'metrics' for model.")
        if isinstance(self._eval_network, nn.GraphCell) and dataset_sink_mode:
            raise ValueError("Sink mode is currently not supported when evaluating with a GraphCell.")

        cb_params = _InternalCallbackParam()
        cb_params.eval_network = self._eval_network
        cb_params.valid_dataset = valid_dataset
        cb_params.batch_num = valid_dataset.get_dataset_size()
        cb_params.mode = "eval"
        cb_params.cur_step_num = 0
        cb_params.list_callback = self._transform_callbacks(callbacks)
        cb_params.network = self._network

        self._clear_metrics()

        if context.get_context("device_target") == "CPU" and dataset_sink_mode:
            dataset_sink_mode = False
            logger.info("CPU cannot support dataset sink mode currently."
                        "So the evaluating process will be performed with dataset non-sink mode.")

        with _CallbackManager(callbacks) as list_callback:
            if dataset_sink_mode:
                return self._eval_dataset_sink_process(valid_dataset, list_callback, cb_params)
            return self._eval_process(valid_dataset, list_callback, cb_params)

    def predict(self, *predict_data):
        """
        Generate output predictions for the input samples.

        Args:
            predict_data (Optional[Tensor, list[Tensor], tuple[Tensor]]): The predict data, can be a single tensor,
                a list of tensor, or a tuple of tensor.

        Returns:
            Tensor, array(s) of predictions.

        Examples:
            >>> import numpy as np
            >>> import mindspore as ms
            >>> from mindspore import Model, Tensor
            >>>
            >>> input_data = Tensor(np.random.randint(0, 255, [1, 1, 32, 32]), ms.float32)
            >>> model = Model(Net())
            >>> result = model.predict(input_data)
        """
        self._predict_network.set_train(False)
        check_input_data(*predict_data, data_class=(int, float, str, None, Tensor))
        _parallel_predict_check()
        result = self._predict_network(*predict_data)

        check_output_data(result)
        return result

    def _infer_train_check(self, train_dataset, dataset_sink_mode, sink_size):
        """
        Check arguments of training.

        Args:
            train_dataset (Dataset): A training dataset iterator.
            dataset_sink_mode (bool): Determines whether to pass the data through dataset channel.
            sink_size (int): Control the amount of data in each sink.
        """
        if context.get_context("mode") != context.GRAPH_MODE:
            raise RuntimeError("Pre-compile process that generate parameter layout for the train network "
                               "only supports GRAPH MODE and Ascend target currently.")
        if _get_parallel_mode() not in (ParallelMode.SEMI_AUTO_PARALLEL, ParallelMode.AUTO_PARALLEL):
            raise RuntimeError("'infer_train_layout' only supports 'semi_auto_parallel' and 'auto_parallel' "
                               "mode, but got {}.".format(_get_parallel_mode()))
        dataset_sink_mode = Validator.check_bool(dataset_sink_mode)
        if not dataset_sink_mode:
            raise ValueError("Only dataset sink mode is supported for now.")
        if isinstance(self._train_network, nn.GraphCell) and dataset_sink_mode:
            raise ValueError("Dataset sink mode is currently not supported when training with a GraphCell.")
        Validator.check_is_int(sink_size)
        dataset_size = train_dataset.get_dataset_size()
        if dataset_size == 0:
            raise ValueError("There is no valid data in dataset, please check dataset file firstly.")
        if sink_size == -1:
            sink_size = dataset_size
        if sink_size < -1 or sink_size == 0:
            raise ValueError("For 'infer_train_layout', the argument 'sink_size' must be -1 or positive, "
                             "but got sink_size {}.".format(sink_size))

    def infer_train_layout(self, train_dataset, dataset_sink_mode=True, sink_size=-1):
        """
        Generate parameter layout for the train network in 'AUTO_PARALLEL' or 'SEMI_AUTO_PARALLEL' mode.
        Only dataset sink mode is supported for now.

        .. warning::
            This is an experimental prototype that is subject to change and/or deletion.

        Note:
            This is a pre-compile function. The arguments should be the same as model.train() function.

        Args:
            train_dataset (Dataset): A training dataset iterator. If there is no
                         loss_fn, a tuple with multiple data (data1, data2, data3, ...) should be
                         returned and passed to the network. Otherwise, a tuple (data, label) should
                         be returned. The data and label would be passed to the network and loss
                         function respectively.
            dataset_sink_mode (bool): Determines whether to pass the data through dataset channel.
                                      Configure pynative mode or CPU, the training process will be performed with
                                      dataset not sink. Default: True.
            sink_size (int): Control the amount of data in each sink.
                             If sink_size = -1, sink the complete dataset for each epoch.
                             If sink_size > 0, sink sink_size data for each epoch.
                             If dataset_sink_mode is False, set sink_size as invalid.
                             Default: -1.

        Returns:
            Dict, Parameter layout dictionary used for load distributed checkpoint

        Examples:
            >>> # This example should be run with multiple devices. Refer to the tutorial > Distributed Training on
            >>> # mindspore.cn.
            >>> import numpy as np
            >>> import mindspore as ms
            >>> from mindspore import Model, context, Tensor, nn, FixedLossScaleManager
            >>> from mindspore.context import ParallelMode
            >>> from mindspore.communication import init
            >>>
            >>> context.set_context(mode=context.GRAPH_MODE)
            >>> init()
            >>> context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL)
            >>>
            >>> # For details about how to build the dataset, please refer to the tutorial
            >>> # document on the official website.
            >>> dataset = create_custom_dataset()
            >>> net = Net()
            >>> loss = nn.SoftmaxCrossEntropyWithLogits()
            >>> loss_scale_manager = FixedLossScaleManager()
            >>> optim = nn.Momentum(params=net.trainable_params(), learning_rate=0.1, momentum=0.9)
            >>> model = Model(net, loss_fn=loss, optimizer=optim, metrics=None, loss_scale_manager=loss_scale_manager)
            >>> layout_dict = model.infer_train_layout(dataset)
        """
        self._infer_train_check(train_dataset, dataset_sink_mode, sink_size)

        train_dataset.__no_send__ = True
        train_dataset_helper, train_network = self._exec_preprocess(is_train=True,
                                                                    dataset=train_dataset,
                                                                    dataset_sink_mode=dataset_sink_mode,
                                                                    sink_size=sink_size)
        for inputs in train_dataset_helper:
            train_network.compile(*inputs)
            break
        train_dataset.__model_hash__ = hash(self)
        return train_network.parameter_layout_dict

    def infer_predict_layout(self, *predict_data):
        """
        Generate parameter layout for the predict network in 'AUTO_PARALLEL' or 'SEMI_AUTO_PARALLEL' mode.

        Data could be a single tensor or multiple tensors.

        Note:
            Batch data should be put together in one tensor.

        Args:
            predict_data (Optional[Tensor, list[Tensor], tuple[Tensor]]): The predict data, can be a single tensor,
                a list of tensor, or a tuple of tensor.

        Returns:
            Dict, Parameter layout dictionary used for load distributed checkpoint.
            Using as one of input parameters of load_distributed_checkpoint, always.

        Raises:
            RuntimeError: If not in GRAPH_MODE.

        Examples:
            >>> # This example should be run with multiple devices. Refer to the tutorial > Distributed Training on
            >>> # mindspore.cn.
            >>> import numpy as np
            >>> import mindspore as ms
            >>> from mindspore import Model, context, Tensor
            >>> from mindspore.context import ParallelMode
            >>> from mindspore.communication import init
            >>>
            >>> context.set_context(mode=context.GRAPH_MODE)
            >>> init()
            >>> context.set_auto_parallel_context(full_batch=True, parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL)
            >>> input_data = Tensor(np.random.randint(0, 255, [1, 1, 32, 32]), ms.float32)
            >>> model = Model(Net())
            >>> predict_map = model.infer_predict_layout(input_data)
        """
        if context.get_context("mode") != context.GRAPH_MODE:
            raise RuntimeError("Pre-compile process that generate parameter layout for the predict network "
                               "only supports GRAPH MODE and Ascend target currently.")
        if _get_parallel_mode() not in (ParallelMode.SEMI_AUTO_PARALLEL, ParallelMode.AUTO_PARALLEL):
            raise RuntimeError('Infer predict layout only supports semi auto parallel and auto parallel mode.')
        _parallel_predict_check()
        check_input_data(*predict_data, data_class=Tensor)

        predict_net = self._predict_network
        # Unlike the cases in build_train_network() and build_eval_network(), 'multi_subgraphs' is not set
        predict_net.set_auto_parallel()
        predict_net.set_train(False)
        predict_net.compile(*predict_data)
        return predict_net.parameter_layout_dict

    def _flush_from_cache(self, cb_params):
        """Flush cache data to host if tensor is cache enable."""
        params = cb_params.train_network.get_parameters()
        for param in params:
            if param.cache_enable:
                Tensor(param).flush_from_cache()

    @property
    def train_network(self):
        """
        Get the model's train network.

        Returns:
            Object, the instance of train network.
        """
        return self._train_network

    @property
    def predict_network(self):
        """
        Get the model's predict network.

        Returns:
            Object, the instance of predict network.
        """
        return self._predict_network

    @property
    def eval_network(self):
        """
        Get the model's eval network.

        Returns:
            Object, the instance of evaluate network.
        """
        return self._eval_network


__all__ = ["Model"]
