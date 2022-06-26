# Copyright 2019 Huawei Technologies Co., Ltd
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
# ==============================================================================
"""
Testing the rescale op in DE
"""
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as vision
from mindspore import log as logger
from util import visualize_image, diff_mse, save_and_check_md5

DATA_DIR = ["../data/dataset/test_tf_file_3_images/train-0000-of-0001.data"]
SCHEMA_DIR = "../data/dataset/test_tf_file_3_images/datasetSchema.json"

GENERATE_GOLDEN = False


def rescale_np(image):
    """
    Apply the rescale
    """
    image = image / 255.0
    image = image - 1.0
    return image


def get_rescaled(image_id):
    """
    Reads the image using DE ops and then rescales using Numpy
    """
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    decode_op = vision.Decode()
    data1 = data1.map(operations=decode_op, input_columns=["image"])
    num_iter = 0
    for item in data1.create_dict_iterator(num_epochs=1):
        image = item["image"].asnumpy()
        if num_iter == image_id:
            return rescale_np(image)
        num_iter += 1

    return None


def test_rescale_op(plot=False):
    """
    Test rescale
    """
    logger.info("Test rescale")
    data1 = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)

    # define map operations
    decode_op = vision.Decode()
    rescale_op = vision.Rescale(1.0 / 255.0, -1.0)

    # apply map operations on images
    data1 = data1.map(operations=decode_op, input_columns=["image"])

    data2 = data1.map(operations=rescale_op, input_columns=["image"])

    num_iter = 0
    for item1, item2 in zip(data1.create_dict_iterator(num_epochs=1, output_numpy=True),
                            data2.create_dict_iterator(num_epochs=1, output_numpy=True)):
        image_original = item1["image"]
        image_de_rescaled = item2["image"]
        image_np_rescaled = get_rescaled(num_iter)
        mse = diff_mse(image_de_rescaled, image_np_rescaled)
        assert mse < 0.001  # rounding error
        logger.info("image_{}, mse: {}".format(num_iter + 1, mse))
        num_iter += 1
        if plot:
            visualize_image(image_original, image_de_rescaled, mse, image_np_rescaled)


def test_rescale_md5():
    """
    Test Rescale with md5 comparison
    """
    logger.info("Test Rescale with md5 comparison")

    # generate dataset
    data = ds.TFRecordDataset(DATA_DIR, SCHEMA_DIR, columns_list=["image"], shuffle=False)
    decode_op = vision.Decode()
    rescale_op = vision.Rescale(1.0 / 255.0, -1.0)

    # apply map operations on images
    data = data.map(operations=decode_op, input_columns=["image"])
    data = data.map(operations=rescale_op, input_columns=["image"])

    # check results with md5 comparison
    filename = "rescale_01_result.npz"
    save_and_check_md5(data, filename, generate_golden=GENERATE_GOLDEN)


if __name__ == "__main__":
    test_rescale_op(plot=True)
    test_rescale_md5()
