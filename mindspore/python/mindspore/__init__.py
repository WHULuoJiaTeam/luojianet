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
""".. MindSpore package."""

from .run_check import run_check
from . import common, dataset, mindrecord, train, log
from . import profiler, communication, numpy, parallel
from .common import *
from .mindrecord import *
from .ops import _op_impl
from .train import *
from .log import *
from .context import GRAPH_MODE, PYNATIVE_MODE, set_context, get_context, set_auto_parallel_context, \
                     get_auto_parallel_context, reset_auto_parallel_context, ParallelMode, set_ps_context, \
                     get_ps_context, reset_ps_context, set_fl_context, get_fl_context
from .version import __version__


__all__ = ["run_check"]
__all__.extend(__version__)
__all__.extend(common.__all__)
__all__.extend(train.__all__)
__all__.extend(log.__all__)
__all__.extend(context.__all__)
