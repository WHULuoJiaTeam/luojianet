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
"""
This module is to support vision augmentations. It includes two parts:
c_transforms and py_transforms. C_transforms is a high performance
image augmentation module which is developed with c++ opencv. Py_transforms
provide more kinds of image augmentations which are developed with Python PIL.

Common imported modules in corresponding API examples are as follows:

.. code-block::

    import mindspore.dataset.vision.c_transforms as c_vision
    import mindspore.dataset.vision.py_transforms as py_vision
    from mindspore.dataset.transforms import c_transforms

Descriptions of common data processing terms are as follows:

- TensorOperation, the base class of all data processing operations implemented in C++.
- PyTensorOperation, the base class of all data processing operations implemented in Python.
- ImageTensorOperation, the base class of all image processing operations. It is a derived class of TensorOperation.
"""
from . import c_transforms
from . import py_transforms
from .utils import Inter, Border, ConvertMode, ImageBatchFormat, SliceMode, AutoAugmentPolicy
