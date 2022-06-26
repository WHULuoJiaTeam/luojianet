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
import os

from mindspore import dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as CV
import mindspore.nn as nn
from mindspore import context, Tensor
import mindspore.ops as ops
from mindspore.nn.probability.dpn import ConditionalVAE
from mindspore.nn.probability.infer import ELBO, SVI

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
IMAGE_SHAPE = (-1, 1, 32, 32)
image_path = os.path.join('/home/workspace/mindspore_dataset/mnist', "train")


class Encoder(nn.Cell):
    def __init__(self, num_classes):
        super(Encoder, self).__init__()
        self.fc1 = nn.Dense(1024 + num_classes, 400)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.concat = ops.Concat(axis=1)
        self.one_hot = nn.OneHot(depth=num_classes)

    def construct(self, x, y):
        x = self.flatten(x)
        y = self.one_hot(y)
        input_x = self.concat((x, y))
        input_x = self.fc1(input_x)
        input_x = self.relu(input_x)
        return input_x


class Decoder(nn.Cell):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc2 = nn.Dense(400, 1024)
        self.sigmoid = nn.Sigmoid()
        self.reshape = ops.Reshape()

    def construct(self, z):
        z = self.fc2(z)
        z = self.reshape(z, IMAGE_SHAPE)
        z = self.sigmoid(z)
        return z


class CVAEWithLossCell(nn.WithLossCell):
    """
    Rewrite WithLossCell for CVAE
    """
    def construct(self, data, label):
        out = self._backbone(data, label)
        return self._loss_fn(out, label)


def create_dataset(data_path, batch_size=32, repeat_size=1,
                   num_parallel_workers=1):
    """
    create dataset for train or test
    """
    # define dataset
    mnist_ds = ds.MnistDataset(data_path)

    resize_height, resize_width = 32, 32
    rescale = 1.0 / 255.0
    shift = 0.0

    # define map operations
    resize_op = CV.Resize((resize_height, resize_width))  # Bilinear mode
    rescale_op = CV.Rescale(rescale, shift)
    hwc2chw_op = CV.HWC2CHW()

    # apply map operations on images
    mnist_ds = mnist_ds.map(operations=resize_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=rescale_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=hwc2chw_op, input_columns="image", num_parallel_workers=num_parallel_workers)

    # apply DatasetOps
    mnist_ds = mnist_ds.batch(batch_size)
    mnist_ds = mnist_ds.repeat(repeat_size)

    return mnist_ds


def test_svi_cvae():
    # define the encoder and decoder
    encoder = Encoder(num_classes=10)
    decoder = Decoder()
    # define the cvae model
    cvae = ConditionalVAE(encoder, decoder, hidden_size=400, latent_size=20, num_classes=10)
    # define the loss function
    net_loss = ELBO(latent_prior='Normal', output_prior='Normal')
    # define the optimizer
    optimizer = nn.Adam(params=cvae.trainable_params(), learning_rate=0.001)
    # define the training dataset
    ds_train = create_dataset(image_path, 128, 1)
    # define the WithLossCell modified
    net_with_loss = CVAEWithLossCell(cvae, net_loss)
    # define the variational inference
    vi = SVI(net_with_loss=net_with_loss, optimizer=optimizer)
    # run the vi to return the trained network.
    cvae = vi.run(train_dataset=ds_train, epochs=5)
    # get the trained loss
    trained_loss = vi.get_train_loss()
    # test function: generate_sample
    sample_label = Tensor([i for i in range(0, 8)] * 8, dtype=mstype.int32)
    generated_sample = cvae.generate_sample(sample_label, 64, IMAGE_SHAPE)
    # test function: reconstruct_sample
    for sample in ds_train.create_dict_iterator(output_numpy=True, num_epochs=1):
        sample_x = Tensor(sample['image'], dtype=mstype.float32)
        sample_y = Tensor(sample['label'], dtype=mstype.int32)
        reconstructed_sample = cvae.reconstruct_sample(sample_x, sample_y)
    print('The loss of the trained network is ', trained_loss)
    print('The shape of the generated sample is ', generated_sample.shape)
    print('The shape of the reconstructed sample is ', reconstructed_sample.shape)
