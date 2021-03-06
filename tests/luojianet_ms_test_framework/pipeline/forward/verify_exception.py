# Copyright 2021, 2022 LuoJiaNET Research and Development Group, Wuhan University
# Copyright 2021, 2022 Huawei Technologies Co., Ltd
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

"""Pipelines for exception checking."""

from ...components.facade.me_facade import MeFacadeFC
from ...components.function.get_function_from_config import IdentityBC
from ...components.function_inputs_policy.cartesian_product_on_id_for_function_inputs import IdCartesianProductFIPC
from ...components.inputs.get_inputs_from_config import IdentityDC

# pylint: disable=W0105
"""
Test if function raises expected Exception type. The pipeline is suitable for config in a ME style.

Example:
    verification_set = [
        ('func_raise_exception', {
            'block': (func_raise_exception, {'exception': ValueError}),
            'desc_inputs': [[1, 1], [2, 2]],
        })
    ]
"""
pipeline_for_verify_exception_for_case_by_case_config = [MeFacadeFC, IdentityDC, IdentityBC,
                                                         IdCartesianProductFIPC]
