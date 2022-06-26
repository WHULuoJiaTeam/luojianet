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

"""Implementation for internal polymorphism `uadd` operations."""
from mindspore.ops.composite import base

# uadd is a metagraph object which will return operation result regarding input
# using ".register" decorator
uadd = base.MultitypeFuncGraph("uadd", True)


@uadd.register("Tensor")
@uadd.register("Number")
def _uadd_scala(x):
    return x
