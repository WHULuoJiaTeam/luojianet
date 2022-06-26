# Copyright 2022 Huawei Technologies Co., Ltd
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
import pytest

from mindspore.ops import operations as P
import mindspore.nn as nn
from mindspore.train import Model
from mindspore.common import set_seed
import mindspore.dataset as ds
from mindspore.train.callback import Callback
from mindspore import log as logger

set_seed(1)


def create_np_dataset(size):
    """
    Create dataset for train or test
    """
    data = ds.NumpySlicesDataset(list(range(1, size + 1)), shuffle=False)
    return data


def create_model():
    """
    Define and return a simple model
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.print = P.Print()

        def construct(self, x):
            self.print(x)
            return x

    net = Net()
    model_ = Model(net)

    return model_


class MyCallback(Callback):
    def __init__(self, dataset_size, reset_point):
        self.dataset_size = dataset_size
        self.reset_point = reset_point

    def epoch_end(self, run_context):
        cb_params = run_context.original_args()
        logger.info(f"Epoch #{cb_params.cur_epoch_num - 1} has ended")
        if cb_params.cur_epoch_num == self.reset_point:
            dataset = ds.engine.datasets._get_training_dataset()  # pylint: disable=W0212
            dataset._reset(self.reset_point * self.dataset_size)  # pylint: disable=W0212


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_dataset_reset_sink():
    """
    Feature: Dataset recovery
    Description: Test Dataset recovery when GPU (and sink mode) is used.
    Expectation: Training completes successfully
    """
    data = create_np_dataset(10)
    model = create_model()
    num_epochs = 3
    reset_point = 2  # 2nd epoch
    cb = MyCallback(dataset_size=data.get_dataset_size(), reset_point=reset_point)
    model.train(num_epochs, data, callbacks=[cb])


if __name__ == '__main__':
    test_dataset_reset_sink()
