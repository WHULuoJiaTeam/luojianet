mindspore.dataset.vision.py_transforms.RandomAffine
===================================================

.. py:class:: mindspore.dataset.vision.py_transforms.RandomAffine(degrees, translate=None, scale=None, shear=None, resample=Inter.NEAREST, fill_value=0)

    对输入PIL图像进行随机仿射变换。

    **参数：**

    - **degrees** (Union[float, Sequence[float, float]]) - 旋转角度的随机选取范围，单位为度。若输入float，将从(- `degrees` , `degrees` )中随机生成旋转角度；若输入Sequence[float, float]，需按(min, max)顺序排列。
    - **translate** (Sequence[float, float]，可选) - 水平与垂直平移比例的随机选取范围，按照(tx, ty)顺序排列，水平与垂直平移的距离将分别从(-tx * width, tx * width)与(-ty * height, ty * height)中随机生成，默认值：None，表示不平移。
    - **scale** (Sequence[float, float]，可选) - 放缩比例的随机选取范围，默认值：None，表示不进行放缩。
    - **shear** (Union[float, Sequence[float, float], Sequence[float, float, float, float]]，可选) - 剪切角度的随机选取范围，单位为度。若输入float，将进行X轴剪切，剪切角度从(- `shear` , `shear` )中随机生成；若输入Sequence[float, float]，将进行X轴剪切，剪切角度从( `shear` [0], `shear` [1])中随机生成；若输入Sequence[int, int, int, int]，将分别进行X轴和Y轴剪切，剪切角度分别从( `shear` [0], `shear` [1])和( `shear` [2], `shear` [3])中随机生成。默认值：None，表示不进行剪切。
    - **resample** (Inter，可选) - 插值方式，取值可为 Inter.BILINEAR、Inter.NEAREST 或 Inter.BICUBIC。若输入的PIL图像模式为"1"或"P"，将直接使用 Inter.NEAREST 作为插值方式。默认值：Inter.NEAREST。

      - **Inter.BILINEAR**：双线性插值。
      - **Inter.NEAREST**：最近邻插值。
      - **Inter.BICUBIC**：双三次插值。

    - **fill_value** (Union[int, tuple[int, int, int]]，可选) - 变换图像之外区域的像素填充值。若输入int，将以该值填充RGB通道；若输入tuple[int, int, int]，将分别用于填充R、G、B通道。仅支持Pillow 5.0.0以上版本。默认值：0。

    **异常：**

    - **TypeError** - 当 `degrees` 的类型不为float或Sequence[float, float]。
    - **TypeError** - 当 `translate` 的类型不为Sequence[float, float]。
    - **TypeError** - 当 `scale` 的类型不为Sequence[float, float]。
    - **TypeError** - 当 `shear` 的类型不为float或Sequence[float, float]。
    - **TypeError** - 当 `resample` 的类型不为 :class:`mindspore.dataset.vision.Inter` 。
    - **TypeError** - 当 `fill_value` 的类型不为int或tuple[int, int, int]。
    - **ValueError** - 当 `degrees` 为负数。
    - **ValueError** - 当 `translate` 取值不在[-1.0, 1.0]范围内。
    - **ValueError** - 当 `scale` 为负数。
    - **ValueError** - 当 `shear` 不为正数。
    - **RuntimeError** - 当输入图像的shape不为<H, W>或<H, W, C>。
