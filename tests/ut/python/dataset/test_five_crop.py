# Copyright 2020 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Testing FiveCrop in DE
"""
import pytest
import numpy as np

import mindspore.dataset as ds
import mindspore.dataset.transforms.py_transforms
import mindspore.dataset.vision.py_transforms as vision
from mindspore import log as logger
from util import visualize_list, save_and_check_md5

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"

GENERATE_GOLDEN = False

def test_five_crop_op(plot=False):
    """
    Test FiveCrop
    """
    logger.info("test_five_crop")

    # First dataset
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    transforms_1 = [
        vision.Decode(),
        vision.ToTensor(),
    ]
    transform_1 = mindspore.dataset.transforms.py_transforms.Compose(transforms_1)
    data1 = data1.map(operations=transform_1, input_columns=["image"])

    # Second dataset
    data2 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    transforms_2 = [
        vision.Decode(),
        vision.FiveCrop(200),
        lambda *images: np.stack([vision.ToTensor()(image) for image in images])  # 4D stack of 5 images
    ]
    transform_2 = mindspore.dataset.transforms.py_transforms.Compose(transforms_2)
    data2 = data2.map(operations=transform_2, input_columns=["image"])

    num_iter = 0
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        num_iter += 1
        image_1 = (item1["image"].transpose(1, 2, 0) * 255).astype(np.uint8)
        image_2 = item2["image"]

        logger.info("shape of image_1: {}".format(image_1.shape))
        logger.info("shape of image_2: {}".format(image_2.shape))

        logger.info("dtype of image_1: {}".format(image_1.dtype))
        logger.info("dtype of image_2: {}".format(image_2.dtype))
        if plot:
            visualize_list(np.array([image_1]*5), (image_2 * 255).astype(np.uint8).transpose(0, 2, 3, 1))

        # The output data should be of a 4D tensor shape, a stack of 5 images.
        assert len(image_2.shape) == 4
        assert image_2.shape[0] == 5


def test_five_crop_error_msg():
    """
    Test FiveCrop error message.
    """
    logger.info("test_five_crop_error_msg")

    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    transforms = [
        vision.Decode(),
        vision.FiveCrop(200),
        vision.ToTensor()
    ]
    transform = mindspore.dataset.transforms.py_transforms.Compose(transforms)
    data = data.map(operations=transform, input_columns=["image"])

    with pytest.raises(RuntimeError) as info:
        for _ in data:
            pass
    error_msg = "TypeError: __call__() takes 2 positional arguments but 6 were given"

    # error msg comes from ToTensor()
    assert error_msg in str(info.value)


def test_five_crop_md5():
    """
    Test FiveCrop with md5 check
    """
    logger.info("test_five_crop_md5")

    # First dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    transforms = [
        vision.Decode(),
        vision.FiveCrop(100),
        lambda *images: np.stack([vision.ToTensor()(image) for image in images])  # 4D stack of 5 images
    ]
    transform = mindspore.dataset.transforms.py_transforms.Compose(transforms)
    data = data.map(operations=transform, input_columns=["image"])
    # Compare with expected md5 from images
    filename = "five_crop_01_result.npz"
    save_and_check_md5(data, filename, generate_golden=GENERATE_GOLDEN)


if __name__ == "__main__":
    test_five_crop_op(plot=True)
    test_five_crop_error_msg()
    test_five_crop_md5()
