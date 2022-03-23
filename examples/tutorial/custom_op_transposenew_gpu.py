import luojianet_ms.context as context
from luojianet_ms import Tensor
import luojianet_ms.ops as ops
import numpy as np
import pytest

context.set_context(device_target='GPU')


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_TensorAdd():
    x1 = Tensor(np.ones((3, 4), np.float32))
    x2 = Tensor(np.ones((3, 4), np.float32))
    y = ops.TensorAddV2()(x1, x2)
    print('result: ', y)