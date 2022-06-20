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
"""test SummaryCollector."""
import os
import re
import shutil
import tempfile
import json
from collections import Counter
import numpy as np

import pytest
from luojianet_ms.common import set_seed
from luojianet_ms import nn, Tensor, context
from luojianet_ms.common.initializer import Normal
from luojianet_ms.nn.metrics import Loss
from luojianet_ms.nn.optim import Momentum
from luojianet_ms.ops import operations as P
from luojianet_ms.train import Model
from luojianet_ms.train.callback import SummaryCollector, SummaryLandscape
from tests.st.summary.dataset import create_mnist_dataset
from tests.summary_utils import SummaryReader
from tests.security_utils import security_off_wrap


def callback_fn():
    """A python function job"""
    network = LeNet5()
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    metrics = {"Loss": Loss()}
    model = Model(network, loss, metrics=metrics)
    ds_train = create_mnist_dataset("train", num_samples=6)
    return model, network, ds_train, metrics


class LeNet5(nn.Module):
    """
    Lenet network

    Args:
        num_class (int): Number of classes. Default: 10.
        num_channel (int): Number of channels. Default: 1.

    Returns:
        Tensor, output tensor
    Examples:
        >>> LeNet(num_class=10)

    """

    def __init__(self, num_class=10, num_channel=1, include_top=True):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.include_top = include_top
        if self.include_top:
            self.flatten = nn.Flatten()
            self.fc1 = nn.Dense(16 * 5 * 5, 120, weight_init=Normal(0.02))
            self.fc2 = nn.Dense(120, 84, weight_init=Normal(0.02))
            self.fc3 = nn.Dense(84, num_class, weight_init=Normal(0.02))

        self.scalar_summary = P.ScalarSummary()
        self.image_summary = P.ImageSummary()
        self.histogram_summary = P.HistogramSummary()
        self.tensor_summary = P.TensorSummary()
        self.channel = Tensor(num_channel)

    def call(self, x):
        """call."""
        self.image_summary('image', x)
        x = self.conv1(x)
        self.histogram_summary('histogram', x)
        x = self.relu(x)
        self.tensor_summary('tensor', x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        self.scalar_summary('scalar', self.channel)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        if not self.include_top:
            return x
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class TestSummary:
    """Test summary collector the basic function."""
    base_summary_dir = ''

    @classmethod
    def setup_class(cls):
        """Run before test this class."""
        device_id = int(os.getenv('DEVICE_ID')) if os.getenv('DEVICE_ID') else 0
        context.set_context(mode=context.GRAPH_MODE, device_id=device_id)
        cls.base_summary_dir = tempfile.mkdtemp(suffix='summary')

    @classmethod
    def teardown_class(cls):
        """Run after test this class."""
        if os.path.exists(cls.base_summary_dir):
            shutil.rmtree(cls.base_summary_dir)

    def _run_network(self, dataset_sink_mode=False, num_samples=2, **kwargs):
        """run network."""
        lenet = LeNet5()
        loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
        optim = Momentum(lenet.trainable_params(), learning_rate=0.1, momentum=0.9)
        model = Model(lenet, loss_fn=loss, optimizer=optim, metrics={'loss': Loss()})
        summary_dir = tempfile.mkdtemp(dir=self.base_summary_dir)
        summary_collector = SummaryCollector(summary_dir=summary_dir, collect_freq=2, **kwargs)

        ds_train = create_mnist_dataset("train", num_samples=num_samples)
        model.train(1, ds_train, callbacks=[summary_collector], dataset_sink_mode=dataset_sink_mode)

        ds_eval = create_mnist_dataset("test")
        model.eval(ds_eval, dataset_sink_mode=dataset_sink_mode, callbacks=[summary_collector])
        return summary_dir

    @pytest.mark.level0
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_gpu_training
    @pytest.mark.env_onecard
    @security_off_wrap
    def test_summary_with_sink_mode_false(self):
        """Test summary with sink mode false, and num samples is 64."""
        summary_dir = self._run_network(num_samples=10)

        tag_list = self._list_summary_tags(summary_dir)

        expected_tag_set = {'conv1.weight/auto', 'conv2.weight/auto', 'fc1.weight/auto', 'fc1.bias/auto',
                            'fc2.weight/auto', 'input_data/auto', 'loss/auto',
                            'histogram', 'image', 'scalar', 'tensor'}
        assert set(expected_tag_set) == set(tag_list)

        # num samples is 10, batch size is 2, so step is 5, collect freq is 2,
        # SummaryCollector will collect the first step and 2th, 4th step
        tag_count = 3
        for value in Counter(tag_list).values():
            assert value == tag_count

    @pytest.mark.level0
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_gpu_training
    @pytest.mark.env_onecard
    @security_off_wrap
    def test_summary_with_sink_mode_true(self):
        """Test summary with sink mode true, and num samples is 64."""
        summary_dir = self._run_network(dataset_sink_mode=True, num_samples=10)

        tag_list = self._list_summary_tags(summary_dir)

        # There will not record input data when dataset sink mode is True
        expected_tags = {'conv1.weight/auto', 'conv2.weight/auto', 'fc1.weight/auto', 'fc1.bias/auto',
                         'fc2.weight/auto', 'loss/auto', 'histogram', 'image', 'scalar', 'tensor'}
        assert set(expected_tags) == set(tag_list)

        tag_count = 1
        for value in Counter(tag_list).values():
            assert value == tag_count

    @pytest.mark.level0
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.env_onecard
    @security_off_wrap
    def test_summarycollector_user_defind(self):
        """Test SummaryCollector with user-defined."""
        summary_dir = self._run_network(dataset_sink_mode=True, num_samples=2,
                                        custom_lineage_data={'test': 'self test'},
                                        export_options={'tensor_format': 'npy'})

        tag_list = self._list_summary_tags(summary_dir)
        file_list = self._list_tensor_files(summary_dir)
        # There will not record input data when dataset sink mode is True
        expected_tags = {'conv1.weight/auto', 'conv2.weight/auto', 'fc1.weight/auto', 'fc1.bias/auto',
                         'fc2.weight/auto', 'loss/auto', 'histogram', 'image', 'scalar', 'tensor'}
        assert set(expected_tags) == set(tag_list)
        expected_files = {'tensor_1.npy'}
        assert set(expected_files) == set(file_list)

    @staticmethod
    def _list_summary_tags(summary_dir):
        """list summary tags."""
        summary_file_path = ''
        for file in os.listdir(summary_dir):
            if re.search("_MS", file):
                summary_file_path = os.path.join(summary_dir, file)
                break
        assert summary_file_path

        tags = list()
        with SummaryReader(summary_file_path) as summary_reader:

            while True:
                summary_event = summary_reader.read_event()
                if not summary_event:
                    break
                for value in summary_event.summary.value:
                    tags.append(value.tag)
        return tags

    @staticmethod
    def _list_tensor_files(summary_dir):
        """list tensor tags."""
        export_file_path = ''
        for file in os.listdir(summary_dir):
            if re.search("export_", file):
                export_file_path = os.path.join(summary_dir, file)
                break
        assert export_file_path
        tensor_file_path = os.path.join(export_file_path, "tensor")
        assert tensor_file_path

        tensors = list()
        for file in os.listdir(tensor_file_path):
            tensors.append(file)

        return tensors

    def _train_network(self, epoch=3, dataset_sink_mode=False, num_samples=2, **kwargs):
        """run network."""
        lenet = LeNet5()
        loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
        optim = Momentum(lenet.trainable_params(), learning_rate=0.01, momentum=0.9)
        model = Model(lenet, loss_fn=loss, optimizer=optim)
        summary_dir = tempfile.mkdtemp(dir=self.base_summary_dir)
        summary_collector = SummaryCollector(summary_dir=summary_dir, collect_freq=2, **kwargs)

        ds_train = create_mnist_dataset("train", num_samples=num_samples)
        model.train(epoch, ds_train, callbacks=[summary_collector], dataset_sink_mode=dataset_sink_mode)
        return summary_dir

    @staticmethod
    def _list_summary_collect_landscape_tags(summary_dir):
        """list summary landscape tags."""
        summary_dir_path = ''
        for file in os.listdir(summary_dir):
            if re.search("ckpt_dir", file):
                summary_dir_path = os.path.join(summary_dir, file)
                break
        assert summary_dir_path

        summary_file_path = ''
        for file in os.listdir(summary_dir_path):
            if re.search(".json", file):
                summary_file_path = os.path.join(summary_dir_path, file)
                break
        assert summary_file_path

        tags = list()
        with open(summary_file_path, 'r') as file:
            data = json.load(file)
        for key, value in data.items():
            tags.append(key)

            assert value
        return tags

    @staticmethod
    def _list_landscape_tags(summary_dir):
        """list landscape tags."""
        expected_tags = {'landscape_[1, 3]', 'landscape_[3]'}
        summary_list = []
        for file in os.listdir(summary_dir):
            if re.search("_MS", file):
                summary_file_path = os.path.join(summary_dir, file)
                summary_list = summary_list + [summary_file_path]
            else:
                continue

        assert summary_list

        tags = []
        for summary_path in summary_list:
            with SummaryReader(summary_path) as summary_reader:

                while True:
                    summary_event = summary_reader.read_event()
                    if not summary_event:
                        break
                    for value in summary_event.summary.value:
                        if value.tag in expected_tags:
                            tags.append(value.loss_landscape.landscape.z.float_data)
                            break
        return tags

    @pytest.mark.level0
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_gpu_training
    @pytest.mark.env_onecard
    @security_off_wrap
    def test_summary_collector_landscape(self):
        """Test summary collector with landscape."""
        set_seed(1)
        interval_1 = [1, 2, 3]
        num_samples = 6
        summary_dir = self._train_network(epoch=3, num_samples=num_samples,
                                          collect_specified_data={'collect_landscape':
                                                                      {'landscape_size': 4,
                                                                       'unit': 'epoch',
                                                                       'create_landscape': {'train': True,
                                                                                            'result': True},
                                                                       'num_samples': num_samples,
                                                                       'intervals': [interval_1]}})

        tag_list = self._list_summary_collect_landscape_tags(summary_dir)
        expected_tags = {'epoch_group', 'model_params_file_map', 'step_per_epoch', 'unit', 'num_samples',
                         'landscape_size', 'create_landscape'}
        assert set(expected_tags) == set(tag_list)
        device_id = int(os.getenv('DEVICE_ID')) if os.getenv('DEVICE_ID') else 0
        summary_landscape = SummaryLandscape(summary_dir)
        summary_landscape.gen_landscapes_with_multi_process(callback_fn, device_ids=[device_id])
        expected_pca_value = np.array([2.2795506, 2.2795567, 2.2795629, 2.2795689, 2.2795507, 2.2795567, 2.2795629,
                                       2.2795688, 2.2795505, 2.2795566, 2.2795628, 2.2795689, 2.2795505, 2.2795566,
                                       2.2795627, 2.2795687])
        expe_pca_value_asc = np.array([2.2795513, 2.2797892, 2.2800267, 2.2802642, 2.2795035, 2.2797413, 2.2799790,
                                       2.2802165, 2.2794557, 2.2796936, 2.2799315, 2.2801689, 2.2794084, 2.2796460,
                                       2.2798840, 2.2801217])
        expected_random_value = np.array([2.2732414, 2.2778292, 2.2829423, 2.2885174, 2.2725525, 2.2772029, 2.2822288,
                                          2.2875323, 2.2726187, 2.2771581, 2.2819989, 2.2875887, 2.2732263, 2.2774866,
                                          2.2823269, 2.2883627])
        expe_random_value_asc = np.array([2.2728422, 2.2777682, 2.2830053, 2.2886044, 2.2725447, 2.2772817, 2.2823694,
                                          2.2878207, 2.2726294, 2.2771712, 2.2820640, 2.2873242, 2.2731032, 2.2774446,
                                          2.2821348, 2.2871782])
        tag_list_landscape = self._list_landscape_tags(summary_dir)
        assert np.all(abs(expected_pca_value - tag_list_landscape[0]) < 1.e-6) or \
               np.all(abs(expe_pca_value_asc - tag_list_landscape[0]) < 1.e-6)
        assert np.all(abs(expected_random_value - tag_list_landscape[1]) < 1.e-6) or \
               np.all(abs(expe_random_value_asc - tag_list_landscape[1]) < 1.e-6)
