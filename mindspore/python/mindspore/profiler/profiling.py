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
"""Profiling api file."""
import os
import stat
import time
import json
from google.protobuf.json_format import MessageToJson

from mindspore import log as logger, context
from mindspore.communication.management import GlobalComm, get_rank, get_group_size
import mindspore._c_expression as c_expression
import mindspore._c_dataengine as cde
from mindspore.train.profiling_parallel_pb2 import ProfilingParallel
from mindspore.profiler.common.exceptions.exceptions import ProfilerFileNotFoundException, \
    ProfilerIOException, ProfilerException, ProfilerRawFileException
from mindspore.profiler.common.exceptions.exceptions import ProfilerPathErrorException
from mindspore.profiler.common.exceptions.exceptions import ProfilerDirNotFoundException
from mindspore.profiler.common.util import get_file_path, fwrite_format
from mindspore.profiler.common.validator.validate_path import \
    validate_and_normalize_path
from mindspore.profiler.parser.aicpu_data_parser import DataPreProcessParser
from mindspore.profiler.parser.framework_parser import FrameworkParser
from mindspore.profiler.parser.hwts_log_parser import HWTSLogParser
from mindspore.profiler.parser.integrator import Integrator, DeviceTarget
from mindspore.profiler.parser.integrator import GpuTimelineGenerator, CpuTimelineGenerator, AscendTimelineGenerator
from mindspore.profiler.parser.memory_usage_parser import MemoryUsageParser
from mindspore.profiler.parser.minddata_parser import MinddataParser
from mindspore.profiler.parser.minddata_analyzer import MinddataProfilingAnalyzer
from mindspore.profiler.parser.flops_parser import FlopsParser
from mindspore.profiler.parser.minddata_pipeline_parser import \
    MinddataPipelineParser
from mindspore.profiler.parser.optime_parser import OPComputeTimeParser
from mindspore.profiler.parser.step_trace_parser import GpuStepTraceParser, AscendStepTraceParser
from mindspore.profiler.parser.hccl_parser import HcclParser
from mindspore.profiler.parser.op_intermediate_parser import OPIntermediateParser

INIT_OP_NAME = 'Default/InitDataSetQueue'


def _environment_check():
    if c_expression.security.enable_security():
        raise RuntimeError("Profiler is not supported when MindSpore is compiled with \'-s on\'.")


class Profiler:
    """
    MindSpore users can use this class to collect the performance of neural networks.

    Args:
        output_path (str, optional): Output data path. Default: "./data".
        profile_communication (bool, optional): (Ascend only) Whether to collect communication performance data in
            a multi devices training,collect when True. Setting this parameter has no effect during single device
            training. Default: False.
        profile_memory (bool, optional): (Ascend only) Whether to collect tensor memory data, collect when True.
            Default: False.
        start_profile (bool, optional): The start_profile parameter controls whether to enable or disable performance
            data collection based on conditions. Default: True.

    Raises:
        RuntimeError: When the version of CANN does not match the version of MindSpore,
            MindSpore cannot parse the generated ascend_job_id directory structure.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import nn, context
        >>> from mindspore import Model
        >>> import mindspore.dataset as ds
        >>> from mindspore.profiler import Profiler
        >>>
        >>>
        >>> class Net(nn.Cell):
        ...     def __init__(self):
        ...         super(Net, self).__init__()
        ...         self.fc = nn.Dense(2,2)
        ...     def construct(self, x):
        ...         return self.fc(x)
        >>>
        >>> def generator():
        ...     for i in range(2):
        ...         yield (np.ones([2, 2]).astype(np.float32), np.ones([2]).astype(np.int32))
        >>>
        >>> def train(net):
        ...     optimizer = nn.Momentum(net.trainable_params(), 1, 0.9)
        ...     loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
        ...     data = ds.GeneratorDataset(generator, ["data", "label"])
        ...     model = Model(net, loss, optimizer)
        ...     model.train(1, data)
        >>>
        >>> if __name__ == '__main__':
        ...     # If the device_target is GPU, set the device_target to "GPU"
        ...     context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
        ...
        ...     # Init Profiler
        ...     # Note that the Profiler should be initialized after context.set_context and before model.train
        ...     # If you are running in parallel mode on Ascend, the Profiler should be initialized before HCCL
        ...     # initialized.
        ...     profiler = Profiler()
        ...
        ...     # Train Model
        ...     net = Net()
        ...     train(net)
        ...
        ...     # Profiler end
        ...     profiler.analyse()
    """

    _hwts_output_filename_target = "output_format_data_hwts_"
    _opcompute_output_filename_target = "output_op_compute_time_"
    _aicpu_op_output_filename_target = "output_data_preprocess_aicpu_"
    _has_analysed = False
    _has_initialized = False
    _ascend_profiling_options = ""
    _ascend_job_id = ""

    def __init__(self, **kwargs):
        if Profiler._has_initialized:
            msg = "Do not init twice in the profiler."
            raise RuntimeError(msg)
        self._filt_optype_names = []
        self._output_path = ""
        self._dev_id = ""
        Profiler._has_initialized = True
        self._dev_id = None
        self._cpu_profiler = None
        self._gpu_profiler = None
        self._init_time = None
        self._ascend_job_id = ''
        self._job_id_env = None
        self._filt_optype_names = ''
        self._output_path = ''
        self._rank_size = 0
        _environment_check()
        # get device_id and device_target
        self._get_devid_rankid_and_devtarget()
        self._get_output_path(kwargs)
        self._profile_communication = False
        self._has_started = False
        self._has_started_twice = False
        self.start_profile = True
        self._profile_memory = False

        # Setup and start MindData Profiling
        self._md_profiler = cde.GlobalContext.profiling_manager()
        self._md_profiler.init()
        self._decide_device_target(kwargs)
        if self.start_profile:
            self.start()

    def _decide_device_target(self, kwargs):
        """Complete Profiler initialization according to device_target."""
        if self._device_target:
            cpu_profiler = c_expression.CPUProfiler
            self._cpu_profiler = cpu_profiler.get_instance()
            self._cpu_profiler.init(self._output_path)

        if self._device_target and self._device_target == DeviceTarget.CPU.value:
            if context.get_context("mode") == context.PYNATIVE_MODE:
                raise RuntimeError(
                    "Pynative model is not supported on CPU currently.")

            self.start_profile = kwargs.pop("start_profile", True)
            if not isinstance(self.start_profile, bool):
                raise TypeError(f"For '{self.__class__.__name__}', the parameter start_profile must be bool, "
                                f"but got type {type(self.start_profile)}")

        if self._device_target and self._device_target == DeviceTarget.GPU.value:
            if context.get_context("mode") == context.PYNATIVE_MODE:
                raise RuntimeError(
                    "Pynative model is not supported on GPU currently.")
            self._parse_parameter_for_gpu(kwargs)

            gpu_profiler = c_expression.GPUProfiler
            self._gpu_profiler = gpu_profiler.get_instance()
            self._gpu_profiler.init(self._output_path)
            if GlobalComm.WORLD_COMM_GROUP == "nccl_world_group":
                self._dev_id = str(get_rank())
            os.environ['DEVICE_ID'] = self._dev_id

        elif self._device_target and self._device_target == DeviceTarget.ASCEND.value:
            self._init_time = int(time.time() * 10000000)
            logger.info("Profiling: profiling init time: %d", self._init_time)
            self._parse_parameter_for_ascend(kwargs)
            os.environ['DEVICE_ID'] = self._dev_id

            self._ascend_profiling_options = json.dumps(
                self._construct_profiling_options())
            # Characters longer than 2048 are ignored, resulting in profiling option resolution errors
            if len(self._ascend_profiling_options) > 2048:
                msg = f"For '{self.__class__.__name__}', the environment parameter length exceeds " \
                      f"the limit (2048), please input valid parameters."
                logger.critical(msg)
                raise ValueError(msg)
            # use context interface to open profiling, for the new mindspore version(after 2020.5.21)
            self._ascend_profiler = c_expression.AscendProfiler.get_instance()
            self._ascend_profiler.init(self._output_path, int(
                self._dev_id), self._ascend_profiling_options)
            base_profiling_container_path = os.path.join(
                self._output_path, "container")
            container_path = os.path.join(
                base_profiling_container_path, self._dev_id)
            data_path = os.path.join(container_path, "data")
            data_path = validate_and_normalize_path(data_path)
            if not os.path.exists(data_path):
                os.makedirs(data_path, exist_ok=True)

    def _construct_profiling_options(self):
        """
        Construct profiling options to determine which profiling data should be collected.
        """
        profile_memory = "off"
        if self._profile_memory:
            profile_memory = "on"
        profiler_communication = "off"
        if self._profile_communication:
            profiler_communication = "on"

        fp_point = os.environ.get("PROFILING_FP_START", "")
        bp_point = os.environ.get("PROFILING_BP_END", "")

        profiling_options = {
            "output": self._output_path,
            "fp_point": fp_point,
            "bp_point": bp_point,
            "training_trace": "on",
            "task_trace": "on",
            "aic_metrics": "ArithmeticUtilization",
            "aicpu": "on",
            "profile_memory": profile_memory,
            "hccl": profiler_communication
        }

        return profiling_options

    def _parse_parameter_for_gpu(self, kwargs):
        """Parse parameter in Proflier when the device target is GPU."""
        self.start_profile = kwargs.pop("start_profile", True)
        if not isinstance(self.start_profile, bool):
            raise TypeError(f"For '{self.__class__.__name__}', the parameter start_profile must be bool, "
                            f"but got type {type(self.start_profile)}")

        self._profile_communication = kwargs.pop("profile_communication", False)
        if not isinstance(self._profile_communication, bool):
            raise TypeError(f"For '{self.__class__.__name__}', the parameter profile_communication must be bool, "
                            f"but got type {type(self._profile_communication)}")
        if self._profile_communication:
            raise RuntimeError(f"The parameter profile_communication is not supported on GPU currently.")

        self._profile_memory = kwargs.pop("profile_memory", False)
        if not isinstance(self._profile_memory, bool):
            raise TypeError(f"For '{self.__class__.__name__}', the parameter _profile_memory must be bool, "
                            f"but got type {type(self._profile_memory)}")
        if self._profile_memory:
            raise RuntimeError(
                f"The parameter profile_memory is not supported on GPU currently.")

    def _parse_parameter_for_ascend(self, kwargs):
        """Parse parameter in Proflier when the device target is Ascend."""
        self.start_profile = kwargs.pop("start_profile", True)
        if not isinstance(self.start_profile, bool):
            raise TypeError(f"For '{self.__class__.__name__}', the parameter start_profile must be bool, "
                            f"but got type {type(self.start_profile)}")

        self._profile_communication = kwargs.pop("profile_communication", False)
        if not isinstance(self._profile_communication, bool):
            raise TypeError(f"For '{self.__class__.__name__}', the parameter profile_communication must be bool, "
                            f"but got type {type(self._profile_communication)}")
        if self._profile_communication:
            hccl_option = {"output": self._output_path, "task_trace": "on"}
            os.environ['PROFILING_OPTIONS'] = json.dumps(hccl_option)
            if not self.start_profile:
                raise RuntimeError(f"For '{self.__class__.__name__}', the parameter profile_communication can "
                                   f"not be True while starting profiler in the process of training.")

        self._profile_memory = kwargs.pop("profile_memory", False)
        if not isinstance(self._profile_memory, bool):
            raise TypeError(f"For '{self.__class__.__name__}', the parameter profile_memory must be bool, "
                            f"but got type '{type(self._profile_memory)}'")
        if kwargs:
            logger.warning("There are invalid params which don't work.")

        task_sink = os.getenv("GRAPH_OP_RUN")
        if task_sink and task_sink == "1":
            logger.warning(f"For '{self.__class__.__name__}', Profiling is not supported if set environment "
                           f"'GRAPH_OP_RUN' value to 1, which means model training task is not sink.")

    def _set_ascend_job_id(self, ascend_job_id):
        """Set output_path for offline parsing performance data."""
        self._ascend_job_id = validate_and_normalize_path(ascend_job_id)
        if not os.path.exists(self._ascend_job_id):
            msg = f"Invalid ascend_job_id: {self._ascend_job_id}, Please pass the absolute path of the JOB dir"
            logger.critical(msg)
            raise ValueError(msg)
        self._output_path, _ = os.path.split(self._ascend_job_id)

    def _is_offline_parser(self):
        """Return whether offline parser or online parser."""
        if self._device_target and self._device_target == DeviceTarget.ASCEND.value:
            return bool(self._ascend_job_id)
        return False

    def analyse(self):
        """
        Collect and analyze training performance data, support calls during and after training. The example shows above.
        """
        if Profiler._has_analysed:
            msg = "Do not analyze twice in the profiler."
            raise RuntimeError(msg)
        Profiler._has_analysed = True

        _environment_check()

        self._cpu_profiler.stop()

        if self._device_target and self._device_target == DeviceTarget.CPU.value:
            self._cpu_analyse()

        if self._device_target and self._device_target == DeviceTarget.GPU.value:
            self._gpu_analyse()

        elif self._device_target and self._device_target == DeviceTarget.ASCEND.value:
            self._ascend_analyse()
        logger.info("Profiling: all the data have been analyzed.")

    def _ascend_pynative_analyse(self):
        """Collect and analyse ascend pynative model performance data."""
        op_intermediate_parser = OPIntermediateParser(
            self._output_path, self._rank_id)
        op_intermediate_parser.parser_pynative_op_type()
        op_intermediate_parser.parser_pynative_op_intermediate_detail()

        job_id = self._get_profiling_job_id()
        logger.info("Profiling: job id is %s ", job_id)
        self._check_output_path(output_path=self._output_path)
        source_path = os.path.join(self._output_path, job_id)
        MinddataParser.execute(source_path, self._output_path, self._rank_id)

        pipeline_parser = MinddataPipelineParser(
            self._output_path, self._rank_id, self._output_path)
        logger.info(
            "Profiling: analyzing the minddata pipeline operator and queue.")
        pipeline_parser.parse()

        timeline_analyser = AscendTimelineGenerator(self._output_path, self._dev_id, self._rank_id,
                                                    self._rank_size, context.get_context("mode"))
        timeline_analyser.init_pynative_timeline()
        size_limit = 100 * 1024 * 1024  # 100MB
        timeline_analyser.write_timeline(size_limit)
        timeline_analyser.write_timeline_summary()

    def _ascend_analyse(self):
        """Collect and analyse ascend performance data."""
        self._rank_size = 1
        if self._profile_communication and not GlobalComm.INITED:
            self._profile_communication = False

        if GlobalComm.INITED:
            self._rank_size = get_group_size()

        if self._has_started:
            self.stop()
        else:
            logger.info(
                "No need to stop profiler because profiler has been stopped.")

        if context.get_context("mode") == context.PYNATIVE_MODE:
            self._ascend_pynative_analyse()
        else:
            self._ascend_graph_analyse()

    def _ascend_graph_op_analyse(self, source_path):
        """
        Ascend graph model hwts analyse.

        Returns:
            list[obj]: The list is: framework_parser, aicpu_data_parser, optime_parser, op_task_dict
        """
        # parse hwts.log.data.45.dev file, and get task profiling data
        hwts_output_filename = self._hwts_output_filename_target + self._rank_id + ".txt"
        hwts_output_filename = os.path.join(
            self._output_path, hwts_output_filename)
        source_path = validate_and_normalize_path(source_path)
        hwts_output_filename = validate_and_normalize_path(
            hwts_output_filename)
        hwtslog_parser = HWTSLogParser(source_path, hwts_output_filename)
        logger.info("Profiling: analyzing hwts data.")
        hwtslog_parser.execute()

        # parse Framework file, and get the relation of op and tasks
        framework_parser = FrameworkParser(
            source_path, self._rank_id, self._output_path)
        logger.info("Profiling: analyzing framework data.")
        framework_parser.parse()
        op_task_dict = framework_parser.to_task_id_full_op_name_dict()
        if not op_task_dict:
            raise RuntimeError('Profiling: fail to parse framework files.')

        # get op compute time from hwts data and framework data, write output_op_compute_time.txt
        opcompute_output_filename = self._opcompute_output_filename_target + \
            self._rank_id + ".txt"
        opcompute_output_filename = os.path.join(
            self._output_path, opcompute_output_filename)
        opcompute_output_filename = validate_and_normalize_path(
            opcompute_output_filename)
        optime_parser = OPComputeTimeParser(
            hwts_output_filename, opcompute_output_filename,
            op_task_dict, self._output_path, self._rank_id
        )
        logger.info("Profiling: analyzing the operation compute time.")
        optime_parser.execute()

        # parse DATA_PREPROCESS.dev.AICPU file, write output_data_preprocess_aicpu_x.txt
        output_data_preprocess_aicpu = self._aicpu_op_output_filename_target + \
            self._rank_id + ".txt"
        output_data_preprocess_aicpu = os.path.join(
            self._output_path, output_data_preprocess_aicpu)
        output_data_preprocess_aicpu = validate_and_normalize_path(
            output_data_preprocess_aicpu)
        aicpu_data_parser = DataPreProcessParser(
            source_path, output_data_preprocess_aicpu, op_task_dict)
        logger.info("Profiling: analyzing the data preprocess data.")
        aicpu_data_parser.execute()

        return [framework_parser, aicpu_data_parser, optime_parser, op_task_dict]

    def _ascend_graph_minddata_analyse(self, source_path):
        """Analyse mindadata for ascend graph model."""
        # Parsing minddata AICPU profiling
        logger.info("Profiling: analyzing the minddata AICPU data.")
        MinddataParser.execute(source_path, self._output_path, self._rank_id)

        # parse minddata pipeline operator and queue
        try:
            pipeline_parser = MinddataPipelineParser(
                self._output_path, self._rank_id, self._output_path)
            logger.info(
                "Profiling: analyzing the minddata pipeline operator and queue.")
            pipeline_parser.parse()
        except ProfilerException as err:
            logger.warning(err.message)
        finally:
            pass

        # Analyze minddata information
        try:
            md_analyzer = MinddataProfilingAnalyzer(
                self._output_path, self._rank_id, self._output_path)
            logger.info("Profiling: analyzing the minddata information.")
            md_analyzer.analyze()
        except ProfilerException as err:
            logger.warning(err.message)
        finally:
            pass

    def _ascend_graph_analyse(self):
        """Ascend graph mode analyse."""
        self._ascend_profiler.finalize()

        job_id = self._get_profiling_job_id()
        logger.info("Profiling: job id is %s ", job_id)

        self._check_output_path(output_path=self._output_path)
        source_path = os.path.join(self._output_path, job_id)
        op_parser_obj = self._ascend_graph_op_analyse(source_path)
        framework_parser = op_parser_obj[0]
        aicpu_data_parser = op_parser_obj[1]
        optime_parser = op_parser_obj[2]
        op_task_dict = op_parser_obj[3]

        self._ascend_graph_minddata_analyse(source_path)

        # analyse op compute time info
        try:
            logger.info("Profiling: analyzing the operation compute time.")
            self._analyser_op_info()
        except ProfilerException as err:
            logger.warning(err.message)
        finally:
            pass

        # analyse step trace info
        points = None
        is_training_mode_flag = False

        try:
            logger.info("Profiling: analyzing the step trace data.")
            points, is_training_mode_flag = self._analyse_step_trace(source_path, framework_parser)
        except ProfilerException as err:
            logger.warning(err.message)
        finally:
            pass

        # analyse timeline info
        try:
            logger.info("Profiling: analyzing the timeline data.")
            self._analyse_timeline(aicpu_data_parser, optime_parser, source_path)
        except (ProfilerIOException, ProfilerFileNotFoundException, RuntimeError) as err:
            logger.warning('Fail to write timeline data: %s', err)
        finally:
            pass

        self._analyse_memory(points)
        self._analyse_hccl()

        # get op FLOPs from aicore.data.x.slice.0 file, and compute FLOPS, write output_op_flops_x.txt
        flops_parser = FlopsParser(source_path, self._output_path, op_task_dict,
                                   self._dev_id, self._rank_id, is_training_mode_flag)
        logger.info("Profiling: analyzing the operation FLOPs.")
        flops_parser.execute()
        logger.info("Profiling: analyzing the parallel strategy.")
        self._analyse_parallel_strategy()

    @staticmethod
    def _check_output_path(output_path):
        """Checking path validity."""
        try:
            output_path = validate_and_normalize_path(output_path)
        except RuntimeError:
            raise ProfilerPathErrorException(
                f'profiling data output path {output_path} is invalid.')
        finally:
            pass
        if not os.path.isdir(output_path):
            raise ProfilerDirNotFoundException(output_path)
        return output_path

    def start(self):
        """
        Used for Ascend, GPU, start profiling. Profiling can be turned on based on step and epoch.

        Raises:
            RuntimeError: If the profiler has already started.
            RuntimeError: If MD profiling has stopped, repeated start action is not supported.
            RuntimeError: If the start_profile parameter is not set or is set to True.

        Examples:
             >>> class StopAtStep(Callback):
             >>>     def __init__(self, start_step, stop_step):
             ...         super(StopAtStep, self).__init__()
             ...         self.start_step = start_step
             ...         self.stop_step = stop_step
             ...         self.profiler = Profiler(start_profile=False)
             ...
             >>>     def step_begin(self, run_context):
             ...         cb_params = run_context.original_args()
             ...         step_num = cb_params.cur_step_num
             ...         if step_num == self.start_step:
             ...             self.profiler.start()
             ...
             >>>     def step_end(self, run_context):
             ...         cb_params = run_context.original_args()
             ...         step_num = cb_params.cur_step_num
             ...         if step_num == self.stop_step:
             ...             self.profiler.stop()
             ...
             >>>     def end(self, run_context):
             ...         self.profiler.analyse()
        """

        if not self.start_profile and context.get_context("mode") == context.PYNATIVE_MODE:
            raise RuntimeError("Pynative model does not support conditional collection of performance data.")

        self._start_time = int(time.time() * 10000000)
        logger.info("Profiling: start time: %d", self._start_time)

        if not self._has_started:
            if not self._has_started_twice:
                self._has_started = True
                self._has_started_twice = True
            else:
                raise RuntimeError("MindSpore Profiling has finished, repeated start and stop actions are not "
                                   "supported.")

        else:
            raise RuntimeError("The profiler has already started. Use profiler.start() only when start_profile value "
                               "is set to False.")

        # No need to start anything if parse profiling data offline
        if self._is_offline_parser():
            return

        self._md_profiler.start()
        self._cpu_profiler.step_profiling_enable(True)

        if self._device_target and self._device_target == DeviceTarget.GPU.value:
            self._gpu_profiler.step_profiling_enable(True)
        elif self._device_target and self._device_target == DeviceTarget.ASCEND.value:
            if context.get_context("mode") == context.PYNATIVE_MODE:
                self._ascend_pynative_start()
            else:
                self._ascend_graph_start()

    def _analyse_memory(self, points):
        """Analyse memory usage info."""
        if self._profile_memory:
            try:
                logger.info("Profiling: analyzing the memory usage info.")
                self._analyse_memory_usage(points)
            except (ProfilerIOException, ProfilerFileNotFoundException, ProfilerRawFileException) as err:
                logger.warning(err.message)
            finally:
                pass

    def _analyse_hccl(self):
        """Analyse hccl info."""
        if self._profile_communication:
            try:
                logger.info("Profiling: analyzing the hccl profiler info.")
                self._analyse_hccl_info()
            except (ProfilerIOException, ProfilerFileNotFoundException, ProfilerRawFileException) as err:
                logger.warning(err.message)
            finally:
                pass

    def _ascend_pynative_start(self):
        """Ascend pynative mode start profiling."""
        pynative_profiler = c_expression.PynativeProfiler
        self._pynative_profiler = pynative_profiler.get_instance()
        self._pynative_profiler.init(self._output_path)
        self._ascend_profiler.start()

    def _ascend_graph_start(self):
        """Ascend graph mode start profiling."""
        self._ascend_profiler.start()

    def stop(self):
        """
        Used for Ascend, GPU, stop profiling. Profiling can be turned off based on step and epoch.

        Raises:
            RuntimeError: If the profiler has not started, this function is disabled.

        Examples:
             >>> class StopAtEpoch(Callback):
             >>>     def __init__(self, start_epoch, stop_epoch):
             ...         super(StopAtEpoch, self).__init__()
             ...         self.start_epoch = start_epoch
             ...         self.stop_epoch = stop_epoch
             ...         self.profiler = Profiler(start_profile=False)
             ...
             >>>     def epoch_begin(self, run_context):
             ...         cb_params = run_context.original_args()
             ...         epoch_num = cb_params.cur_epoch_num
             ...         if epoch_num == self.start_epoch:
             ...             self.profiler.start()
             ...
             >>>     def epoch_end(self, run_context):
             ...         cb_params = run_context.original_args()
             ...         epoch_num = cb_params.cur_epoch_num
             ...         if epoch_num == self.stop_epoch:
             ...             self.profiler.stop()
             ...
             >>>     def end(self, run_context):
             ...         self.profiler.analyse()
        """
        if self._has_started:
            self._has_started = False
        else:
            raise RuntimeError("The profiler has not started, so can not stop. Please call the start() method "
                               "before calling the stop() method.")

        # No need to stop anything if parse profiling data offline
        if self._is_offline_parser():
            return

        self._md_profiler.stop()
        self._md_profiler.save(self._output_path)

        if self._device_target and self._device_target == DeviceTarget.GPU.value:
            self._gpu_profiler.stop()
        elif self._device_target and self._device_target == DeviceTarget.ASCEND.value:
            if context.get_context("mode") == context.PYNATIVE_MODE:
                self._pynative_profiler.stop()
            self._ascend_profiler.stop()

            self._stop_time = int(time.time() * 10000000)
            logger.info("Profiling: stop time: %d", self._stop_time)

    def _gpu_analyse(self):
        """Collect and analyse gpu performance data."""
        self._dev_id = context.get_context("device_id")
        self._rank_size = 1
        if GlobalComm.WORLD_COMM_GROUP == "nccl_world_group":
            self._dev_id = str(get_rank())

        if GlobalComm.INITED:
            self._rank_size = get_group_size()

        if self._has_started:
            self.stop()
        else:
            logger.info(
                "No need to stop profiler because profiler has been stopped.")

        reduce_op_type = self._get_step_reduce_op_type()
        timeline_generator = self._generate_timeline(reduce_op_type)

        # parse minddata pipeline operator and queue for GPU
        try:
            pipeline_parser = MinddataPipelineParser(
                self._output_path, self._dev_id, self._output_path)
            logger.info(
                "Profiling: analyzing the minddata pipeline operator and queue for GPU.")
            pipeline_parser.parse()
        except ProfilerException as err:
            logger.warning(err.message)

        # Analyze minddata information
        try:
            md_analyzer = MinddataProfilingAnalyzer(
                self._output_path, self._dev_id, self._output_path)
            logger.info("Profiling: analyzing the minddata information.")
            md_analyzer.analyze()
        except ProfilerException as err:
            logger.warning(err.message)

        # analyse step trace info
        try:
            logger.info("Profiling: analyzing the step trace info.")
            self._analyse_step_trace(
                is_training_mode_flag=timeline_generator.check_op_name(
                    'Gradients'),
                is_gpu_kernel_async_launch_flag=timeline_generator.is_gpu_kernel_async_launch()
            )
        except ProfilerException as err:
            logger.warning(err.message)
        finally:
            pass

        logger.warning(
            '\nThe GPU supports only the training mode or inference mode, '
            'it does not support train and infer at the same time.'
        )

    def _get_step_reduce_op_type(self):
        """Gets all communication operator names."""

        step_trace_original_filename = f'step_trace_profiling_{self._dev_id}.txt'
        step_trace_file_path = os.path.join(
            self._output_path, step_trace_original_filename)
        step_trace_file_path = validate_and_normalize_path(
            step_trace_file_path)
        reduce_op_type = []
        with open(step_trace_file_path, 'r') as f_obj:
            one_step_info = f_obj.readline().strip().split()
            # The communication operator starts at index 4.
            for reduce_item in one_step_info[4:]:
                reduce_op_type.append(reduce_item.split(',')[0].split('/')[-1])
        return reduce_op_type

    def _cpu_analyse(self):
        """Collect and analyse cpu performance data."""

        try:
            size_limit = 100 * 1024 * 1024  # 100MB
            timeline_generator = CpuTimelineGenerator(
                self._output_path, context.get_context("mode"))
            timeline_generator.init_timeline()
            timeline_generator.write_timeline(size_limit)
            timeline_generator.write_timeline_summary()
            return timeline_generator
        except (ProfilerIOException, ProfilerFileNotFoundException, RuntimeError) as err:
            logger.warning('Fail to write timeline data: %s', err)
            raise RuntimeError('Fail to write timeline data.')

    def _analyse_step_trace(self, source_path=None, framework_parser=None, is_training_mode_flag=True,
                            is_gpu_kernel_async_launch_flag=False):
        """
        Analyse step trace data and save the result.

        Args:
            source_path (str): The directory that contains the step trace original data.
            framework_parser (FrameworkParser): The framework parse instance.
            is_training_mode_flag (bool): Whether in training mode or not.
        """
        logger.info("Begin to parse step trace.")
        # construct output path
        dev_id = self._rank_id if self._device_target == DeviceTarget.ASCEND.value else self._dev_id
        step_trace_intermediate_file_path = os.path.join(
            self._output_path,
            f'step_trace_raw_{dev_id}_detail_time.csv'
        )
        point_info_file_path = os.path.join(
            self._output_path,
            f'step_trace_point_info_{dev_id}.json'
        )
        step_trace_intermediate_file_path = validate_and_normalize_path(
            step_trace_intermediate_file_path)
        point_info_file_path = validate_and_normalize_path(
            point_info_file_path)

        if self._device_target and self._device_target == DeviceTarget.GPU.value:
            input_file_path = os.path.join(
                self._output_path, f'step_trace_profiling_{self._dev_id}.txt')
            input_file_path = validate_and_normalize_path(input_file_path)
            parser = GpuStepTraceParser(input_dir=input_file_path,
                                        output_file_path=step_trace_intermediate_file_path,
                                        is_training_mode=is_training_mode_flag,
                                        is_gpu_kernel_async_launch=is_gpu_kernel_async_launch_flag)
            parser.parse_and_save()
            point_info = parser.record_point_info(point_info_file_path)
        else:
            # whether keep the first step
            skip_first_step_flag = framework_parser.check_op_name(INIT_OP_NAME)
            point_info = framework_parser.point_info
            # recognize inference or training mode
            is_training_mode_flag = framework_parser.check_op_name("Gradients")
            # parser the step trace files and save the result to disk
            source_path = validate_and_normalize_path(source_path)
            parser = AscendStepTraceParser(input_dir=source_path,
                                           output_file_path=step_trace_intermediate_file_path,
                                           skip_first_step=skip_first_step_flag,
                                           is_training_mode=is_training_mode_flag)
            parser.set_task_id_op_name_dict(
                framework_parser.to_task_id_full_op_name_dict())
            parser.parse_and_save()
            point_info = parser.record_point_info(point_info_file_path)
        # print parser result
        parser.show()
        logger.info("Finish saving the intermediate result: %s",
                    step_trace_intermediate_file_path)
        logger.info("The point info is: %s", point_info)

        return point_info, is_training_mode_flag

    def _analyse_timeline(self, aicpu_parser, optime_parser, source_path):
        """
        Analyse and parse timeline info.

        Args:
            aicpu_parser (DataPreProcessParser): The parser instance for AI CPU operator
                execution time calculation.
            optime_parser (OPComputeTimeParserParser): The parser instance for AI Core
                operator execution time calculation.
        """
        timeline_analyser = AscendTimelineGenerator(self._output_path, self._dev_id, self._rank_id,
                                                    self._rank_size, context.get_context("mode"))
        # Get framework info
        integrator = Integrator(self._output_path, self._rank_id)
        aicore_detail_data = integrator.get_aicore_detail_data()
        aicore_detail_data_size = len(aicore_detail_data)
        col_names = ['op_name', 'op_type', 'avg_execution_time', 'subgraph',
                     'full_op_name', 'op_info']
        framework_info = {
            'col_name': col_names,
            'object': aicore_detail_data,
            'size': aicore_detail_data_size
        }

        all_reduce_info = integrator.query_for_all_reduce()

        # Get timeline info
        logger.info('Start writing timeline info...')
        logger.info('Warm Prompt: It could take a few minutes if you are training '
                    'with a complex network or more than 10 steps.')
        # Add info into timeline, such as AI CPU, AllReduce, framework info.
        aicpu_info = aicpu_parser.query_aicpu_data()
        min_cycle_counter = min(
            aicpu_parser.min_cycle_counter, optime_parser.min_cycle_counter)
        timeline_analyser.init_timeline(all_reduce_info, framework_info, aicpu_info,
                                        min_cycle_counter, source_path)
        size_limit = 100 * 1024 * 1024  # 100MB
        timeline_analyser.write_timeline(size_limit)
        timeline_analyser.write_timeline_summary()

    def _generate_timeline(self, reduce_op_type):
        """Used for gpu, generate timeline info, write to json format file."""
        try:
            size_limit = 100 * 1024 * 1024  # 100MB
            timeline_generator = GpuTimelineGenerator(self._output_path, self._dev_id, self._rank_size,
                                                      context.get_context("mode"))
            timeline_generator.init_timeline(reduce_op_type)
            timeline_generator.write_timeline(size_limit)
            timeline_generator.write_timeline_summary()
            return timeline_generator
        except (ProfilerIOException, ProfilerFileNotFoundException, RuntimeError) as err:
            logger.warning('Fail to write timeline data: %s', err)
            raise RuntimeError('Fail to write timeline data.')

    def _analyse_memory_usage(self, points):
        """Analyse memory usage data."""
        integrator = Integrator(self._output_path, self._rank_id)
        aicore_detail_data = integrator.get_aicore_detail_data()
        memory_parser = MemoryUsageParser(self._output_path, self._rank_id)
        memory_parser.init_memory_usage_info(aicore_detail_data, points)
        memory_parser.write_memory_files()

    def _get_profiling_job_id(self):
        """Get profiling job id, which was generated by ada service.

        Returns:
            str, profiling job id.
        """

        if self._is_offline_parser():
            # The self._ascend_job_id directory like "/../PROF***" or "/../JOB***".
            job_id = self._ascend_job_id.rstrip('/').split('/')[-1]
            if job_id.startswith('PROF'):
                device_dir = [dir for dir in os.listdir(
                    self._ascend_job_id) if dir.startswith('device')]
                return os.path.join(job_id, device_dir[0])
            return job_id

        job_id = ""
        job_dirs = filter(lambda item: item.startswith('JOB') or item.startswith('PROF') and
                          os.path.isdir(os.path.join(self._output_path, item)),
                          os.listdir(self._output_path))
        sorted_job_dirs = sorted(job_dirs, key=lambda x: os.path.getmtime(os.path.join(self._output_path, x)),
                                 reverse=True)

        for dir_name in sorted_job_dirs:
            if dir_name.startswith('PROF'):
                prof_dir = os.path.join(self._output_path, dir_name)
                device_dir = [dir for dir in os.listdir(prof_dir)
                              if dir.startswith('device') and os.path.isdir(os.path.join(prof_dir, dir))]
                job_dir = os.path.join(
                    self._output_path, dir_name, device_dir[0])
            else:
                job_dir = os.path.join(self._output_path, dir_name)

            host_start_file_path = get_file_path(job_dir, "host_start.log")
            if host_start_file_path is None:
                logger.warning("Find profiling job path %s, but host_start.log not exist, "
                               "profiler will ignore this job dir.", job_dir)
                continue

            training_device_id = host_start_file_path.split('.')[-1]
            if self._dev_id != training_device_id:
                logger.warning("Find profiling find job path %s, but not current training device id. "
                               "Current training device id %s, but job path device id: %s, "
                               "profiler will ignore this job dir.", job_dir, self._dev_id, training_device_id)
                continue

            if not os.listdir(os.path.join(job_dir, 'data')):
                continue

            job_start_time = self._parse_host_start_log(host_start_file_path)
            if not job_start_time:
                logger.warning("Find profiling job path %s, but fail to get job start info, "
                               "profiler will ignore this job dir.", job_start_time)
                continue

            if int(job_start_time) < self._start_time:
                logger.warning("Find profiling job path %s, but start_time(%d) is earlier than this training "
                               "start_time(%d), profiler will ignore this job dir.",
                               job_dir, int(job_start_time), self._start_time)
                continue

            if dir_name.startswith('PROF'):
                job_id = os.path.join(dir_name, device_dir[0])
            else:
                job_id = dir_name
            break

        if not job_id:
            msg = "Fail to get profiling job, output path is {}, " \
                  "please check whether job dir or prof dir(name startswith JOB or PROF) in output path " \
                  "was generated, or may be the device id from job dir dismatch the " \
                  "device_id in current process.".format(self._output_path)
            raise RuntimeError(msg)

        return job_id

    @staticmethod
    def _parse_host_start_log(input_file):
        """
        Parse host start log file, get the start time of the job.

        Args:
             input_file (str): The file path of the host start log file.

        Returns:
            str, job start time.
        """

        job_start_time = ""
        with open(input_file) as f:
            for line in f.readlines():
                if "clock_realtime" in line:
                    # 16 means the first digit of the timestamp, len(line)-3 means the last.
                    job_start_time = line[16:len(line) - 3]

        return job_start_time

    def _analyser_op_info(self):
        """Analyse the operator information."""
        integrator = Integrator(self._output_path, self._rank_id)
        integrator.integrate()

        aicore_type_result = self._query_op_type_info()
        detail_file_path = os.path.join(
            self._output_path,
            'output_op_compute_time_detail_{}.txt'.format(self._rank_id)
        )
        fwrite_format(detail_file_path, data_source='title:op compute time')
        display_names = [
            'optype_name', 'compute_time(ms, per-step)',
            'called_times(per-step)', 'percent'
        ]
        fwrite_format(detail_file_path, data_source=" ".join(
            display_names), is_print=True)
        fwrite_format(detail_file_path,
                      data_source=aicore_type_result, is_print=True)

        op_type_order = [item[0] for item in aicore_type_result]
        aicore_detail_result = self._query_op_detail_info(op_type_order)

        fwrite_format(detail_file_path, data_source='', is_print=True)
        fwrite_format(detail_file_path, data_source='Detail:', is_print=True)
        fwrite_format(detail_file_path, data_source=" ".join(aicore_detail_result.get('col_name_detail')),
                      is_print=True)
        fwrite_format(detail_file_path, data_source=aicore_detail_result.get(
            'object'), is_print=True)

    def _query_op_type_info(self):
        """
        Query AICORE operator type information.

        Returns:
            list[list], the AICORE operator type and execution time information.
        """
        integrator = Integrator(self._output_path, self._rank_id)
        return integrator.get_aicore_data()

    def _query_op_detail_info(self, op_type_order):
        """
        Query AICORE operator detail information.

        Args:
            op_type_order(list): The name of the op type in order.

        Returns:
            dict, the AICORE operator detail information.
        """

        op_type_condition = {}
        if self._filt_optype_names:
            op_type_condition['not_in'] = self._filt_optype_names

        filter_condition = {
            'op_type': op_type_condition,
            'is_display_detail': False,
        }
        integrator = Integrator(self._output_path, self._rank_id)
        return integrator.query_and_sort_by_op_type(filter_condition, op_type_order)

    def _get_devid_rankid_and_devtarget(self):
        """Get device id and rank id and target of this training."""

        device_target = ""
        dev_id = ""
        rank_id = ""
        try:
            dev_id = str(context.get_context("device_id"))
            device_target = context.get_context("device_target").lower()
        except ValueError as err:
            logger.error("Profiling: fail to get context, %s", err)

        if not dev_id or not dev_id.isdigit():
            dev_id = os.getenv('DEVICE_ID')
        if not dev_id or not dev_id.isdigit():
            dev_id = "0"
            logger.warning("Fail to get DEVICE_ID, use 0 instead.")

        if device_target and device_target not in [DeviceTarget.ASCEND.value, DeviceTarget.GPU.value,
                                                   DeviceTarget.CPU.value]:
            msg = "Profiling: unsupported backend: %s" % device_target
            raise RuntimeError(msg)

        rank_id = os.getenv("RANK_ID")
        if not rank_id or not rank_id.isdigit():
            rank_id = "0"
            logger.warning(f"For '{self.__class__.__name__}', fail to get RANK_ID from environment, "
                           f"use 0 instead.")

        self._dev_id = dev_id
        self._device_target = device_target.lower()
        self._rank_id = rank_id

    def _get_output_path(self, kwargs):
        """Get output path of profiling data."""
        if os.getenv("MS_DIAGNOSTIC_DATA_PATH") and kwargs.get("output_path") is not None:
            logger.warning("Both parameter output_path and environment variable MS_DIAGNOSTIC_DATA_PATH"
                           " have values set, and the profiling data saving path is the value set "
                           "in parameter output_path")
        if kwargs.get("output_path") is None:
            if "output_path" in kwargs:
                kwargs.pop("output_path")
            # Environment variables are mainly set for the convenience of cloud profiler.
            output_path = os.getenv("MS_DIAGNOSTIC_DATA_PATH")
            if output_path:
                self._output_path = validate_and_normalize_path(output_path)
            else:
                output_path = "data"
                self._output_path = validate_and_normalize_path(output_path)
        else:
            output_path = kwargs.pop("output_path")
            self._output_path = validate_and_normalize_path(output_path)
        self._output_path = os.path.join(self._output_path, "profiler")
        if not os.path.exists(self._output_path):
            os.makedirs(self._output_path, exist_ok=True)
            os.chmod(self._output_path, stat.S_IRUSR |
                     stat.S_IWUSR | stat.S_IXUSR)
        else:
            logger.warning("The target dir already exists. "
                           "There may be some old profiling data, and they will be rewritten in the end.")

    def _analyse_hccl_info(self):
        """Analyse hccl info."""
        hccl_path = os.path.join(
            self._output_path, "hccl_info_{}".format(self._rank_id))
        if not os.path.exists(hccl_path):
            os.makedirs(hccl_path, exist_ok=True)
            os.chmod(hccl_path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
        logger.info("Start call the interface HCCLParseOP parsing hccl info...")
        logger.info('Warm Prompt: It could take a few minutes if you are training '
                    'with a complex network or more than 10 steps.')
        # Call the interface HCCLParseOP parsing hccl info.
        try:
            from hccl_parser.entry import hccl_parse_op
            hccl_parse_op(self._dev_id, self._output_path,
                          hccl_path, op_type='all')
        except ImportError as err:
            logger.critical("%s,please check if the hccl_parser-{version}-py3-none-any.whl is installed."
                            "The hccl_parser-{version}-py3-none-any.whl package is usually located "
                            "in the /usr/local/Ascend/tools Directory", err)
            raise ImportError(err)
        logger.info("Parse hccl info successfully.")
        logger.info("Start analyse hccl info.")
        hccl_parse = HcclParser(hccl_path, self._dev_id,
                                self._rank_id, self._output_path)
        hccl_parse.parse()
        logger.info("Analyse hccl info successfully.")

    def _analyse_parallel_strategy(self):
        """Analyse parallel strategy from proto binary to json."""
        binary_file = os.path.join(self._output_path, 'parallel_strategy_pb_{}.bin'.format(self._rank_id))
        binary_file = validate_and_normalize_path(binary_file)
        if not os.path.isfile(binary_file):
            return
        with open(binary_file, 'rb') as f:
            data = f.read()
        parallel = ProfilingParallel()
        parallel.ParseFromString(data)
        parallel_json = MessageToJson(parallel)

        json_file = os.path.join(self._output_path, 'parallel_strategy_{}.json'.format(self._rank_id))
        with os.fdopen(os.open(json_file, os.O_WRONLY | os.O_CREAT, 0o660), 'w') as f:
            f.write(parallel_json)
        os.remove(binary_file)
        