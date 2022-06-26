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
""" test_dtype """
from luojianet_ms._c_expression import typing

from luojianet_ms.common.api import ms_function

number = typing.Number()
int64 = typing.Int(64)
t1 = typing.Tuple((int64, int64))


@ms_function
def try_type():
    return (number, int64, t1)


def test_dtype_convert():
    try_type()
