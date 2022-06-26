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
"""
The configuration module provides various functions to set and get the supported
configuration parameters, and read a configuration file.

Common imported modules in corresponding API examples are as follows:

.. code-block::

    import mindspore.dataset as ds
"""
import os
import platform
import random
from pathlib import Path
import numpy
import mindspore._c_dataengine as cde
from mindspore import log as logger
from .validator_helpers import replace_none

__all__ = ['set_sending_batches', 'load', '_init_device_info',
           'set_seed', 'get_seed',
           'set_prefetch_size', 'get_prefetch_size',
           'set_num_parallel_workers', 'get_num_parallel_workers',
           'set_numa_enable', 'get_numa_enable',
           'set_monitor_sampling_interval', 'get_monitor_sampling_interval',
           'set_callback_timeout', 'get_callback_timeout',
           'set_auto_num_workers', 'get_auto_num_workers',
           'set_enable_shared_mem', 'get_enable_shared_mem',
           'set_enable_autotune', 'get_enable_autotune',
           'set_autotune_interval', 'get_autotune_interval',
           'set_auto_offload', 'get_auto_offload',
           'set_enable_watchdog', 'get_enable_watchdog',
           'set_multiprocessing_timeout_interval', 'get_multiprocessing_timeout_interval']

INT32_MAX = 2147483647
UINT32_MAX = 4294967295

_config = cde.GlobalContext.config_manager()


def _init_device_info():
    """
    INTERNAL USE ONLY!
    As rank_id need to pass into deep layer for numa and device_queue.
    One process work with only one rank_id, In standalone scenario,
    rank_id may come from env 'CUDA_VISIBLE_DEVICES', For distribute
    scenario, rank_id come from _get_global_rank().
    """
    from mindspore import context
    from mindspore.parallel._auto_parallel_context import auto_parallel_context
    from mindspore.parallel._utils import _get_global_rank
    numa_enable = False
    numa_enable_env = os.getenv("DATASET_ENABLE_NUMA", None)
    if numa_enable_env and numa_enable_env.strip() == 'True':
        numa_enable = True
    if context.get_context("device_target") == "GPU":
        rank_id = _get_global_rank()
        parallel_mode = auto_parallel_context().get_parallel_mode()
        if parallel_mode == "stand_alone":
            rank_id = context.get_context("device_id")
        if numa_enable:
            _config.set_numa_enable(True)
        _config.set_rank_id(rank_id)
    elif context.get_context("device_target") == "Ascend":
        # Ascend is a special scenario, we'd better get rank info from env
        env_rank_size = os.getenv("RANK_SIZE", None)
        env_rank_id = os.getenv("RANK_ID", None)
        rank_size = 0
        rank_id = 0
        if env_rank_size and env_rank_id:
            try:
                rank_size = int(env_rank_size.strip())
                rank_id = int(env_rank_id.strip())
            except ValueError:
                raise ValueError("rank_size or rank_id is not int.")
        if rank_size > 1:
            if numa_enable:
                _config.set_numa_enable(True)
            _config.set_rank_id(rank_id)


def set_seed(seed):
    """
    Set the seed so the random generated number will be fixed for deterministic results.

    Note:
        This set_seed function sets the seed in the Python random library and numpy.random library
        for deterministic Python augmentations using randomness. This set_seed function should
        be called when iterator is created to reset the random seed.

    Args:
        seed(int): Random number seed. It is used to generate deterministic random numbers.

    Raises:
        TypeError: If `seed` isn't of type int.
        ValueError: If `seed` < 0 or `seed` > UINT32_MAX(4294967295).

    Examples:
        >>> # Set a new global configuration value for the seed value.
        >>> # Operations with randomness will use the seed value to generate random values.
        >>> ds.config.set_seed(1000)
    """
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed isn't of type int.")
    if seed < 0 or seed > UINT32_MAX:
        raise ValueError("seed given is not within the required range [0, UINT32_MAX(4294967295)].")
    _config.set_seed(seed)
    random.seed(seed)
    # numpy.random isn't thread safe
    numpy.random.seed(seed)


def get_seed():
    """
    Get random number seed. If the seed has been set, then will
    return the set value, otherwise it will return the default seed value
    which equals to std::mt19937::default_seed.

    Returns:
        int, random number seed.

    Examples:
        >>> # Get the global configuration of seed.
        >>> # If set_seed() is never called before, the default value(std::mt19937::default_seed) will be returned.
        >>> seed = ds.config.get_seed()
    """
    return _config.get_seed()


def set_prefetch_size(size):
    """
    Set the queue capacity of the thread in pipeline.

    Args:
        size (int): The length of the cache queue.

    Raises:
        TypeError: If `size` is not of type int.
        ValueError: If `size` <= 0 or `size` > INT32_MAX(2147483647).

    Note:
        Since total memory used for prefetch can grow very large with high number of workers,
        when the number of workers is greater than 4, the per worker prefetch size will be reduced.
        The actual prefetch size at runtime per-worker will be prefetchsize * (4 / num_parallel_workers).

    Examples:
        >>> # Set a new global configuration value for the prefetch size.
        >>> ds.config.set_prefetch_size(1000)
    """
    if not isinstance(size, int) or isinstance(size, bool):
        raise TypeError("size isn't of type int.")
    if size <= 0 or size > INT32_MAX:
        raise ValueError("size is not within the required range (0, INT32_MAX(2147483647)].")
    _config.set_op_connector_size(size)


def get_prefetch_size():
    """
    Get the prefetch size as for number of rows.
    If `set_prefetch_size` is never called before, the default value 16 will be returned.

    Returns:
        int, total number of rows to be prefetched.

    Examples:
        >>> # Get the global configuration of prefetch size.
        >>> # If set_prefetch_size() is never called before, the default value(16) will be returned.
        >>> prefetch_size = ds.config.get_prefetch_size()
    """
    return _config.get_op_connector_size()


def set_num_parallel_workers(num):
    """
    Set a new global configuration default value for the number of parallel workers.
    This setting will affect the parallelism of all dataset operation.

    Args:
        num (int): Number of parallel workers to be used as a default for each operation.

    Raises:
        TypeError: If `num` is not of type int.
        ValueError: If `num` <= 0 or `num` > INT32_MAX(2147483647).

    Examples:
        >>> # Set a new global configuration value for the number of parallel workers.
        >>> # Now parallel dataset operators will run with 8 workers.
        >>> ds.config.set_num_parallel_workers(8)
    """
    if not isinstance(num, int) or isinstance(num, bool):
        raise TypeError("num isn't of type int.")
    if num <= 0 or num > INT32_MAX:
        raise ValueError("Number of parallel workers given is not within the required range"
                         " (0, INT32_MAX(2147483647)].")
    _config.set_num_parallel_workers(num)


def get_num_parallel_workers():
    """
    Get the global configuration of number of parallel workers.
    This is the DEFAULT num_parallel_workers value used for each operation.

    Returns:
        int, number of parallel workers to be used as a default for each operation.

    Examples:
        >>> # Get the global configuration of parallel workers.
        >>> # If set_num_parallel_workers() is never called before, the default value(8) will be returned.
        >>> num_parallel_workers = ds.config.get_num_parallel_workers()
    """
    return _config.get_num_parallel_workers()


def set_numa_enable(numa_enable):
    """
    Set the default state of numa enabled. If numa_enable is True, need to ensure numa library is installed.

    Args:
        numa_enable (bool): Whether to use numa bind feature.

    Raises:
        TypeError: If `numa_enable` is not a boolean data type.

    Examples:
        >>> # Set a new global configuration value for the state of numa enabled.
        >>> # Now parallel dataset operators will run with numa bind function
        >>> ds.config.set_numa_enable(True)
    """
    if not isinstance(numa_enable, bool):
        raise TypeError("numa_enable must be a boolean dtype.")
    _config.set_numa_enable(numa_enable)


def get_numa_enable():
    """
    Get the state of numa to indicate enabled/disabled.
    This is the DEFAULT numa enabled value used for the all process.

    Returns:
        bool, the default state of numa enabled.

    Examples:
        >>> # Get the global configuration of numa.
        >>> numa_state = ds.config.get_numa_enable()
    """
    return _config.get_numa_enable()


def set_monitor_sampling_interval(interval):
    """
    Set the default interval (in milliseconds) for monitor sampling.

    Args:
        interval (int): Interval (in milliseconds) to be used for performance monitor sampling.

    Raises:
        TypeError: If `interval` is not type int.
        ValueError: If `interval` <= 0 or `interval` > INT32_MAX(2147483647).

    Examples:
        >>> # Set a new global configuration value for the monitor sampling interval.
        >>> ds.config.set_monitor_sampling_interval(100)
    """
    if not isinstance(interval, int) or isinstance(interval, bool):
        raise TypeError("interval isn't of type int.")
    if interval <= 0 or interval > INT32_MAX:
        raise ValueError("Interval given is not within the required range (0, INT32_MAX(2147483647)].")
    _config.set_monitor_sampling_interval(interval)


def get_monitor_sampling_interval():
    """
    Get the global configuration of sampling interval of performance monitor.
    If `set_monitor_sampling_interval` is never called before, the default value(1000) will be returned.

    Returns:
        int, interval (in milliseconds) for performance monitor sampling.

    Examples:
        >>> # Get the global configuration of monitor sampling interval.
        >>> # If set_monitor_sampling_interval() is never called before, the default value(1000) will be returned.
        >>> sampling_interval = ds.config.get_monitor_sampling_interval()
    """
    return _config.get_monitor_sampling_interval()


def set_auto_num_workers(enable):
    """
    Set num_parallel_workers for each op automatically(This feature is turned off by default).

    If turned on, the num_parallel_workers in each op will be adjusted automatically, possibly overwriting the
    num_parallel_workers passed in by user or the default value (if user doesn't pass anything) set by
    ds.config.set_num_parallel_workers().

    For now, this function is only optimized for YoloV3 dataset with per_batch_map (running map in batch).
    This feature aims to provide a baseline for optimized num_workers assignment for each operation.
    Operation whose num_parallel_workers is adjusted to a new value will be logged.

    Args:
        enable (bool): Whether to enable auto num_workers feature or not.

    Raises:
        TypeError: If `enable` is not of boolean type.

    Examples:
        >>> # Enable auto_num_worker feature, this might override the num_parallel_workers passed in by user
        >>> ds.config.set_auto_num_workers(True)
    """
    if not isinstance(enable, bool):
        raise TypeError("enable must be of type bool.")
    _config.set_auto_num_workers(enable)


def _set_auto_workers_config(option):
    """
    INTERNAL USE ONLY!
    Select the weight profile of auto_num_workers. currently these 7 options are supported.
    Option #0 leaf_num_workers:batch_num_workers:map_num_workers=1:1:1
    Option #1 leaf_num_workers:batch_num_workers:map_num_workers=2:1:1
    Option #2 leaf_num_workers:batch_num_workers:map_num_workers=1:2:1
    Option #3 leaf_num_workers:batch_num_workers:map_num_workers=1:1:2
    Option #4 leaf_num_workers:batch_num_workers:map_num_workers=2:2:1
    Option #5 leaf_num_workers:batch_num_workers:map_num_workers=2:1:2
    Option #6 leaf_num_workers:batch_num_workers:map_num_workers=1:2:2

    Args:
        option (int): The id of the profile to use.

    Raises:
        TypeError: If `option` is not of type int.
        ValueError: If `option` is not within the range of [0, 6].
    """
    if not isinstance(option, int) or isinstance(option, bool):
        raise TypeError("option isn't of type int.")
    if option < 0 or option > 6:
        raise ValueError("option isn't within the required range of [0, 6].")
    _config.set_auto_worker_config(option)


def get_auto_num_workers():
    """
    Get the setting (turned on or off) automatic number of workers.

    Returns:
        bool, whether auto number worker feature is turned on.

    Examples:
        >>> # Get the global configuration of auto number worker feature.
        >>> flag = ds.config.get_auto_num_workers()
    """
    return _config.get_auto_num_workers()


def set_callback_timeout(timeout):
    """
    Set the default timeout (in seconds) for DSWaitedCallback.

    Args:
        timeout (int): Timeout (in seconds) to be used to end the wait in DSWaitedCallback in case of a deadlock.

    Raises:
        TypeError: If `timeout` is not type int.
        ValueError: If `timeout` <= 0 or `timeout` > INT32_MAX(2147483647).

    Examples:
        >>> # Set a new global configuration value for the timeout value.
        >>> ds.config.set_callback_timeout(100)
    """
    if not isinstance(timeout, int) or isinstance(timeout, bool):
        raise TypeError("timeout isn't of type int.")
    if timeout <= 0 or timeout > INT32_MAX:
        raise ValueError("Timeout given is not within the required range.")
    _config.set_callback_timeout(timeout)


def get_callback_timeout():
    """
    Get the default timeout for WaitedDSCallback.

    Returns:
        int, Timeout (in seconds) to be used to end the wait in DSWaitedCallback in case of a deadlock.

    Examples:
        >>> # Get the global configuration of callback timeout.
        >>> # If set_callback_timeout() is never called before, the default value(60) will be returned.
        >>> callback_timeout = ds.config.get_callback_timeout()
    """
    return _config.get_callback_timeout()


def __str__():
    """
    String representation of the configurations.

    Returns:
        str, configurations.
    """
    return str(_config)


def load(file):
    """
    Load the project configuration from the file.

    Args:
        file (str): Path of the configuration file to be loaded.

    Raises:
        RuntimeError: If `file` is invalid and parsing fails.

    Examples:
        >>> # Set new default configuration according to values in the configuration file.
        >>> # example config file:
        >>> # {
        >>> #     "logFilePath": "/tmp",
        >>> #     "numParallelWorkers": 4,
        >>> #     "seed": 5489,
        >>> #     "monitorSamplingInterval": 30
        >>> # }
        >>> config_file = "/path/to/config/file"
        >>> ds.config.load(config_file)
    """
    _config.load(file)


def set_enable_autotune(enable, filepath_prefix=None):
    """
    Set whether to enable AutoTune. AutoTune is disabled by default.

    AutoTune is used to automatically adjust the global configuration of the data pipeline
    according to the workload of environmental resources during the training process to
    improve the speed of data processing.

    The optimized global configuration can be saved as a JSON file by setting `json_filepath`
    for subsequent reuse.

    Args:
        enable (bool): Whether to enable AutoTune.
        filepath_prefix (str, optional): The prefix filepath to save the optimized global configuration.
            The rank id and the json extension will be appended to the filepath_prefix string in multi-device training,
            rank id will be set to 0 in standalone training.
            For example, if filepath_prefix="/path/to/some/dir/prefixname" and rank_id is 1, then the path
            of the generated file will be "/path/to/some/dir/prefixname_1.json"
            If the file already exists, it will be automatically overwritten. Default: None,
            means not to save the configuration file, but the tuned result still can be checked through INFO log.

    Raises:
        TypeError: If `enable` is not of type boolean.
        TypeError: If `json_filepath` is not of type str.
        RuntimeError: If `json_filepath` is an empty string.
        RuntimeError: If `json_filepath` is a directory.
        RuntimeError: If `json_filepath` does not exist.
        RuntimeError: If `json_filepath` does not have write permission.

    Note:
        - When `enable` is False, `json_filepath` will be ignored.
        - The JSON file can be loaded by API `mindspore.dataset.deserialize` to build a tuned pipeline.
        - In distributed training scenario, set_enable_autotune() must be called after cluster communication has been
          initialized (mindspore.communication.management.init()), otherwise the AutoTune file will always suffix with
          rank id 0.

    An example of the generated JSON file is as follows. "remark" file will conclude that if the dataset has been
    tuned or not. "summary" filed will show the tuned configuration of dataset pipeline. Users can modify scripts
    based on the tuned result.

    .. code-block::

        {
            "remark": "The following file has been auto-generated by the Dataset AutoTune.",
            "summary": [
                "CifarOp(ID:5)       (num_parallel_workers: 2, prefetch_size:64)",
                "MapOp(ID:4)         (num_parallel_workers: 2, prefetch_size:64)",
                "MapOp(ID:3)         (num_parallel_workers: 2, prefetch_size:64)",
                "BatchOp(ID:2)       (num_parallel_workers: 8, prefetch_size:64)"
            ],
            "tree": {
                ...
            }
        }

    Examples:
        >>> # enable AutoTune and save optimized data pipeline configuration
        >>> ds.config.set_enable_autotune(True, "/path/to/autotune_out.json")
        >>>
        >>> # enable AutoTune
        >>> ds.config.set_enable_autotune(True)
    """
    if not isinstance(enable, bool):
        raise TypeError("enable must be of type bool.")

    save_autoconfig = bool(enable and filepath_prefix is not None)

    if filepath_prefix and not isinstance(filepath_prefix, str):
        raise TypeError("json_filepath must be a str value but was: {}.".format(filepath_prefix))

    if enable and filepath_prefix == "":
        raise RuntimeError("The value of json_filepath cannot be the empty string.")

    if not enable and filepath_prefix is not None:
        logger.warning("The value of json_filepath is ignored when enable is False.")

    if enable and filepath_prefix is None:
        logger.warning("Dataset AutoTune is enabled but no json path is specified, check INFO log for tuned result.")

    json_filepath = replace_none(filepath_prefix, "")

    rank_id = _get_rank_id()

    path = Path(json_filepath).resolve()

    if not path.is_dir():
        filename_prefix = path.name
        # append rank_id and json extension
        filename = filename_prefix + "_" + rank_id + ".json"
        json_filepath = str(path.with_name(filename))

    _config.set_enable_autotune(enable, save_autoconfig, json_filepath)


def _get_rank_id():
    """
    INTERNAL USE ONLY
    Get the rank id
    :return:rank_id
    """
    _init_device_info()
    rank_id = _config.get_rank_id()
    # default rank_id is -1 in ConfigManager
    if rank_id < 0:
        rank_id = os.getenv("RANK_ID")
        if not rank_id or not rank_id.isdigit():
            rank_id = "0"
    else:
        rank_id = str(rank_id)
    return rank_id


def get_enable_autotune():
    """
    Get whether AutoTune is currently enabled.

    Returns:
        bool, whether AutoTune is currently enabled.

    Examples:
        >>> # get the state of AutoTune
        >>> autotune_flag = ds.config.get_enable_autotune()
    """
    return _config.get_enable_autotune()


def set_autotune_interval(interval):
    """
    Set the configuration adjustment interval (in steps) for AutoTune.

    The default setting is 0, which will adjust the configuration after each epoch.
    Otherwise, the configuration will be adjusted every `interval` steps.

    Args:
        interval (int): Interval (in steps) to adjust the configuration of the data pipeline.

    Raises:
        TypeError: If `interval` is not of type int.
        ValueError: If `interval` is not non-negative.

    Examples:
        >>> # set a new interval for AutoTune
        >>> ds.config.set_autotune_interval(30)
    """
    if not isinstance(interval, int) or isinstance(interval, bool):
        raise TypeError("interval must be of type int.")
    if interval < 0 or interval > INT32_MAX:
        raise ValueError("Interval given is not within the required range [0, INT32_MAX(2147483647)].")
    _config.set_autotune_interval(interval)


def get_autotune_interval():
    """
    Get the current configuration adjustment interval (in steps) for AutoTune.

    Returns:
        int, the configuration adjustment interval (in steps) for AutoTune.

    Examples:
        >>> # get the global configuration of the autotuning interval
        >>> autotune_interval = ds.config.get_autotune_interval()
    """
    return _config.get_autotune_interval()


def get_enable_shared_mem():
    """
    Get the default state of shared mem enabled variable.

    Note:
        `get_enable_shared_mem` is not supported on Windows and MacOS platforms yet.

    Returns:
        bool, the state of shared mem enabled variable.

    Examples:
        >>> # Get the flag of shared memory feature.
        >>> shared_mem_flag = ds.config.get_enable_shared_mem()
    """
    # For Windows and MacOS we forbid shared mem function temporarily
    if platform.system().lower() in {"windows", "darwin"}:
        logger.warning("For Windows and MacOS we forbid shared mem function temporarily.")
        return False
    return _config.get_enable_shared_mem()


def set_enable_shared_mem(enable):
    """
    Set the default state of shared memory flag. If shared_mem_enable is True, will use shared memory queues
    to pass data to processes that are created for operators that set python_multiprocessing=True.

    Note:
        `set_enable_shared_mem` is not supported on Windows and MacOS platforms yet.

    Args:
        enable (bool): Whether to use shared memory in operators when python_multiprocessing=True.

    Raises:
        TypeError: If `enable` is not a boolean data type.

    Examples:
        >>> # Enable shared memory feature to improve the performance of Python multiprocessing.
        >>> ds.config.set_enable_shared_mem(True)
    """
    # For Windows and MacOS we forbid shared mem function temporarily
    if platform.system().lower() in {"windows", "darwin"}:
        logger.warning("For Windows and MacOS we forbid shared mem function temporarily.")
        return

    if not isinstance(enable, bool):
        raise TypeError("enable must be of type bool.")
    if enable:
        logger.warning("The shared memory is on, multiprocessing performance will be improved. "
                       "Note: the required shared memory can't exceeds 80% of the available shared memory.")
    _config.set_enable_shared_mem(enable)


def set_sending_batches(batch_num):
    """
    Set the default sending batches when training with sink_mode=True in Ascend device.

    Args:
        batch_num (int): the total sending batches, when batch_num is set, it will wait unless sending batches
         increase, default is 0 which means will send all batches in dataset.

    Raises:
        TypeError: If `batch_num` is not of type int.

    Examples:
        >>> # Set a new global configuration value for the sending batches
        >>> ds.config.set_sending_batches(10)
    """
    if not isinstance(batch_num, int) or isinstance(batch_num, bool):
        raise TypeError("batch_num must be an int dtype.")
    _config.set_sending_batches(batch_num)


def set_auto_offload(offload):
    """
    Set the automatic offload flag of the dataset. If set_auto_offload is True,
    automatically offload as many dataset operations from the CPU to the Device (GPU or Ascend).

    Args:
        offload (bool): Whether to use the automatic offload feature.

    Raises:
        TypeError: If offload is not a boolean data type.

    Examples:
        >>> # Enable automatic offload feature
        >>> ds.config.set_auto_offload(True)
    """
    if not isinstance(offload, bool):
        raise TypeError("offload must be a bool dtype")
    _config.set_auto_offload(offload)


def get_auto_offload():
    """
    Get the state of the automatic offload flag (True or False)

    Returns:
        bool, Whether the automatic offload feature is enabled.

    Example:
        >>> # Get the global configuration of the automatic offload feature.
        >>> auto_offload = ds.config.get_auto_offload()
    """
    return _config.get_auto_offload()


def set_enable_watchdog(enable):
    """
    Set the default state of watchdog Python thread as enabled, the default state of watchdog Python thread is enabled.
    Watchdog is a thread which cleans up hanging subprocesses.

    Args:
        enable (bool): Whether to launch a watchdog Python thread. System default: True.

    Raises:
        TypeError: If `enable` is not a boolean data type.

    Examples:
        >>> # Set a new global configuration value for the state of watchdog Python thread as enabled.
        >>> ds.config.set_enable_watchdog(True)
    """
    if not isinstance(enable, bool):
        raise TypeError("enable must be a boolean dtype.")
    _config.set_enable_watchdog(enable)


def get_enable_watchdog():
    """
    Get the state of watchdog Python thread to indicate enabled or disabled state.
    This is the DEFAULT watchdog Python thread state value used for the all processes.

    Returns:
        bool, the default state of watchdog Python thread enabled.

    Examples:
        >>> # Get the global configuration of watchdog Python thread.
        >>> watchdog_state = ds.config.get_enable_watchdog()
    """
    return _config.get_enable_watchdog()


def set_multiprocessing_timeout_interval(interval):
    """
    Set the default interval (in seconds) for multiprocessing/multithreading timeout when main process/thread gets
    data from subprocesses/child threads.

    Args:
        interval (int): Interval (in seconds) to be used for multiprocessing/multithreading timeout when main
          process/thread gets data from subprocess/child threads. System default: 300s.

    Raises:
        TypeError: If `interval` is not of type int.
        ValueError: If `interval` <= 0 or `interval` > INT32_MAX(2147483647).

    Examples:
        >>> # Set a new global configuration value for multiprocessing/multithreading timeout when getting data.
        >>> ds.config.set_multiprocessing_timeout_interval(300)
    """
    if not isinstance(interval, int) or isinstance(interval, bool):
        raise TypeError("interval isn't of type int.")
    if interval <= 0 or interval > INT32_MAX:
        raise ValueError("Interval given is not within the required range (0, INT32_MAX(2147483647)).")
    _config.set_multiprocessing_timeout_interval(interval)


def get_multiprocessing_timeout_interval():
    """
    Get the global configuration of multiprocessing/multithreading timeout when main process/thread gets data from
    subprocesses/child threads.

    Returns:
        int, interval (in seconds) for multiprocessing/multithreading timeout when main process/thread gets data from
        subprocesses/child threads (default is 300s).

    Examples:
        >>> # Get the global configuration of multiprocessing/multithreading timeout when main process/thread gets data
        >>> # from subprocesses/child threads. If set_multiprocessing_timeout_interval() is never called before, the
        >>> # default value(300) will be returned.
        >>> multiprocessing_timeout_interval = ds.config.get_multiprocessing_timeout_interval()
    """
    return _config.get_multiprocessing_timeout_interval()
