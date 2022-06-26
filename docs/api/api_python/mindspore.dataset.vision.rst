mindspore.dataset.vision
===================================

此模块用于图像数据增强，包括 `c_transforms` 和 `py_transforms` 两个子模块。
`c_transforms` 是使用 C++ OpenCv 开发的高性能图像增强模块。
`py_transforms` 是使用 Python Pillow 开发的图像增强模块。

API样例中常用的导入模块如下：

.. code-block::

    import mindspore.dataset.vision.c_transforms as c_vision
    import mindspore.dataset.vision.py_transforms as py_vision
    from mindspore.dataset.transforms import c_transforms

常用数据处理术语说明如下：

- TensorOperation，所有C++实现的数据处理操作的基类。
- PyTensorOperation，所有Python实现的数据处理操作的基类。
- ImageTensorOperation，所有图像数据处理操作的基类，派生自TensorOperation。

mindspore.dataset.vision.c_transforms
------------------------------------------------

.. mscnautosummary::
    :toctree: dataset_vision
    :nosignatures:
    :template: classtemplate.rst

    mindspore.dataset.vision.c_transforms.AutoContrast
    mindspore.dataset.vision.c_transforms.BoundingBoxAugment
    mindspore.dataset.vision.c_transforms.CenterCrop
    mindspore.dataset.vision.c_transforms.ConvertColor
    mindspore.dataset.vision.c_transforms.Crop
    mindspore.dataset.vision.c_transforms.CutMixBatch
    mindspore.dataset.vision.c_transforms.CutOut
    mindspore.dataset.vision.c_transforms.Decode
    mindspore.dataset.vision.c_transforms.Equalize
    mindspore.dataset.vision.c_transforms.GaussianBlur
    mindspore.dataset.vision.c_transforms.HorizontalFlip
    mindspore.dataset.vision.c_transforms.HWC2CHW
    mindspore.dataset.vision.c_transforms.Invert
    mindspore.dataset.vision.c_transforms.MixUpBatch
    mindspore.dataset.vision.c_transforms.Normalize
    mindspore.dataset.vision.c_transforms.NormalizePad
    mindspore.dataset.vision.c_transforms.Pad
    mindspore.dataset.vision.c_transforms.RandomAffine
    mindspore.dataset.vision.c_transforms.RandomColor
    mindspore.dataset.vision.c_transforms.RandomColorAdjust
    mindspore.dataset.vision.c_transforms.RandomCrop
    mindspore.dataset.vision.c_transforms.RandomCropDecodeResize
    mindspore.dataset.vision.c_transforms.RandomCropWithBBox
    mindspore.dataset.vision.c_transforms.RandomHorizontalFlip
    mindspore.dataset.vision.c_transforms.RandomHorizontalFlipWithBBox
    mindspore.dataset.vision.c_transforms.RandomPosterize
    mindspore.dataset.vision.c_transforms.RandomResize
    mindspore.dataset.vision.c_transforms.RandomResizedCrop
    mindspore.dataset.vision.c_transforms.RandomResizedCropWithBBox
    mindspore.dataset.vision.c_transforms.RandomResizeWithBBox
    mindspore.dataset.vision.c_transforms.RandomRotation
    mindspore.dataset.vision.c_transforms.RandomSelectSubpolicy
    mindspore.dataset.vision.c_transforms.RandomSharpness
    mindspore.dataset.vision.c_transforms.RandomSolarize
    mindspore.dataset.vision.c_transforms.RandomVerticalFlip
    mindspore.dataset.vision.c_transforms.RandomVerticalFlipWithBBox
    mindspore.dataset.vision.c_transforms.Rescale
    mindspore.dataset.vision.c_transforms.Resize
    mindspore.dataset.vision.c_transforms.ResizeWithBBox
    mindspore.dataset.vision.c_transforms.Rotate
    mindspore.dataset.vision.c_transforms.SlicePatches
    mindspore.dataset.vision.c_transforms.SoftDvppDecodeRandomCropResizeJpeg
    mindspore.dataset.vision.c_transforms.SoftDvppDecodeResizeJpeg
    mindspore.dataset.vision.c_transforms.UniformAugment
    mindspore.dataset.vision.c_transforms.VerticalFlip

mindspore.dataset.vision.py_transforms
-------------------------------------------------

.. mscnautosummary::
    :toctree: dataset_vision
    :nosignatures:
    :template: classtemplate.rst

    mindspore.dataset.vision.py_transforms.AutoContrast
    mindspore.dataset.vision.py_transforms.CenterCrop
    mindspore.dataset.vision.py_transforms.Cutout
    mindspore.dataset.vision.py_transforms.Decode
    mindspore.dataset.vision.py_transforms.Equalize
    mindspore.dataset.vision.py_transforms.FiveCrop
    mindspore.dataset.vision.py_transforms.Grayscale
    mindspore.dataset.vision.py_transforms.HsvToRgb
    mindspore.dataset.vision.py_transforms.HWC2CHW
    mindspore.dataset.vision.py_transforms.Invert
    mindspore.dataset.vision.py_transforms.LinearTransformation
    mindspore.dataset.vision.py_transforms.MixUp
    mindspore.dataset.vision.py_transforms.Normalize
    mindspore.dataset.vision.py_transforms.NormalizePad
    mindspore.dataset.vision.py_transforms.Pad
    mindspore.dataset.vision.py_transforms.RandomAffine
    mindspore.dataset.vision.py_transforms.RandomColor
    mindspore.dataset.vision.py_transforms.RandomColorAdjust
    mindspore.dataset.vision.py_transforms.RandomCrop
    mindspore.dataset.vision.py_transforms.RandomErasing
    mindspore.dataset.vision.py_transforms.RandomGrayscale
    mindspore.dataset.vision.py_transforms.RandomHorizontalFlip
    mindspore.dataset.vision.py_transforms.RandomPerspective
    mindspore.dataset.vision.py_transforms.RandomResizedCrop
    mindspore.dataset.vision.py_transforms.RandomRotation
    mindspore.dataset.vision.py_transforms.RandomSharpness
    mindspore.dataset.vision.py_transforms.RandomVerticalFlip
    mindspore.dataset.vision.py_transforms.Resize
    mindspore.dataset.vision.py_transforms.RgbToHsv
    mindspore.dataset.vision.py_transforms.TenCrop
    mindspore.dataset.vision.py_transforms.ToPIL
    mindspore.dataset.vision.py_transforms.ToTensor
    mindspore.dataset.vision.py_transforms.ToType
    mindspore.dataset.vision.py_transforms.UniformAugment

mindspore.dataset.vision.utils
-------------------------------

.. mscnautosummary::
    :toctree: dataset_vision
    :nosignatures:
    :template: classtemplate.rst

    mindspore.dataset.vision.Border
    mindspore.dataset.vision.ConvertMode
    mindspore.dataset.vision.ImageBatchFormat
    mindspore.dataset.vision.Inter
    mindspore.dataset.vision.SliceMode
