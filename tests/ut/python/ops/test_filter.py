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
""" test_filter """

from luojianet_ms.nn import Module


def is_odd(x):
    """ Judge whether the parameter is odd """

    if x % 2:
        return True
    return False


class NetWork(Module):
    """ NetWork definition """

    def __init__(self):
        super(NetWork, self).__init__()
        self.func = is_odd

    def call(self, list_):
        set_func = filter
        ret = set_func(self.func, list_)
        return ret


list1 = [1, 2, 3]
net1 = NetWork()
result = net1(list1)
assert result == (1, 3)
