mindspore.dataset.vision.c_transforms.Rotate
============================================

.. py:class:: mindspore.dataset.vision.c_transforms.Rotate(degrees, resample=Inter.NEAREST, expand=False, center=None, fill_value=0)

    将输入图像旋转指定的度数。

    **参数：**

    - **degrees** (Union[int, float]) - 旋转角度。
    - **resample** (Inter, 可选) - 插值方式。 它可以是 [Inter.BILINEAR, Inter.NEAREST, Inter.BICUBIC] 中的任何一个，默认值：Inter.NEAREST。

      - Inter.BILINEAR，双线性插值。
      - Inter.NEAREST，最近邻插值。
      - Inter.BICUBIC，双三次插值。

    - **expand** (bool, 可选) - 若为True，将扩展图像尺寸大小使其足以容纳整个旋转图像；若为False，则保持图像尺寸大小不变。请注意，扩展时将假设图像为中心旋转且未进行平移。默认值：False。
    - **center** (tuple, 可选) - 可选的旋转中心，以图像左上角为原点，旋转中心的位置按照 (宽度, 高度) 格式指定。默认值：None，表示中心旋转。
    - **fill_value** (Union[int, tuple[int]], 可选) - 旋转图像之外区域的像素填充值。若输入3元素元组，将分别用于填充R、G、B通道；若输入整型，将以该值填充RGB通道。 `fill_value` 值必须在 [0, 255] 范围内，默认值：0。

    **异常：**

    - **TypeError** - 当 `degrees` 的类型不为int或float。
    - **TypeError** - 当 `resample` 的类型不为 :class:`mindspore.dataset.vision.Inter` 。
    - **TypeError** - 当 `expand` 的类型不为bool。
    - **TypeError** - 当 `center` 的类型不为tuple。
    - **TypeError** - 当 `fill_value` 的类型不为int或tuple[int]。
    - **ValueError** - 当 `fill_value` 取值不在[0, 255]范围内。
    - **RuntimeError** - 当输入图像的shape不为<H, W>或<H, W, C>。
