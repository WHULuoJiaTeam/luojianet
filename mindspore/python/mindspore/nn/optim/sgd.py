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
"""sgd"""
from mindspore.ops import functional as F, composite as C, operations as P
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
import mindspore.common.dtype as mstype
from mindspore._checkparam import Validator as validator
from .optimizer import Optimizer
from .optimizer import opt_init_args_register

_sgd_opt = C.MultitypeFuncGraph("sgd_opt")


@_sgd_opt.register("Function", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor")
def _tensor_run_opt_ext(opt, momentum, learning_rate, gradient, weight, accum, stat):
    """Apply sgd optimizer to the weight parameter using Tensor."""
    success = True
    success = F.depend(success, opt(weight, gradient, learning_rate, accum, momentum, stat))
    return success


class SGD(Optimizer):
    r"""
    Implements stochastic gradient descent. Momentum is optional.

    Introduction to SGD can be found at https://en.wikipedia.org/wiki/Stochastic_gradient_descent.
    Nesterov momentum is based on the formula from paper `On the importance of initialization and
    momentum in deep learning <http://proceedings.mlr.press/v28/sutskever13.html>`_.

    .. math::
            v_{t+1} = u \ast v_{t} + gradient \ast (1-dampening)

    If nesterov is True:

    .. math::
            p_{t+1} = p_{t} - lr \ast (gradient + u \ast v_{t+1})

    If nesterov is False:

    .. math::
            p_{t+1} = p_{t} - lr \ast v_{t+1}

    To be noticed, for the first step, :math:`v_{t+1} = gradient`

    Here : where p, v and u denote the parameters, accum, and momentum respectively.

    Args:
        params (Union[list[Parameter], list[dict]]): Must be list of `Parameter` or list of `dict`. When the
            `params` is a list of `dict`, the string "params", "lr", "grad_centralization" and
            "order_params" are the keys can be parsed.

            - params: Required. Parameters in current group. The value must be a list of `Parameter`.

            - lr: Optional. If "lr" in the keys, the value of corresponding learning rate will be used.
              If not, the `learning_rate` in optimizer will be used. Fixed and dynamic learning rate are supported.

            - weight_decay: Using different `weight_decay` by grouping parameters is currently not supported.

            - grad_centralization: Optional. Must be Boolean. If "grad_centralization" is in the keys, the set value
              will be used. If not, the `grad_centralization` is False by default. This configuration only works on the
              convolution layer.

            - order_params: Optional. When parameters is grouped, this usually is used to maintain the order of
              parameters that appeared in the network to improve performance. The value should be parameters whose
              order will be followed in optimizer.
              If `order_params` in the keys, other keys will be ignored and the element of 'order_params' must be in
              one group of `params`.

        learning_rate (Union[float, int, Tensor, Iterable, LearningRateSchedule]): Default: 0.1.

            - float: The fixed learning rate value. Must be equal to or greater than 0.

            - int: The fixed learning rate value. Must be equal to or greater than 0. It will be converted to float.

            - Tensor: Its value should be a scalar or a 1-D vector. For scalar, fixed learning rate will be applied.
              For vector, learning rate is dynamic, then the i-th step will take the i-th value as the learning rate.

            - Iterable: Learning rate is dynamic. The i-th step will take the i-th value as the learning rate.

            - LearningRateSchedule: Learning rate is dynamic. During training, the optimizer calls the instance of
              LearningRateSchedule with step as the input to get the learning rate of current step.

        momentum (float): A floating point value the momentum. must be at least 0.0. Default: 0.0.
        dampening (float): A floating point value of dampening for momentum. must be at least 0.0. Default: 0.0.
        weight_decay (float): Weight decay (L2 penalty). It must be equal to or greater than 0. Default: 0.0.
        nesterov (bool): Enables the Nesterov momentum. If use nesterov, momentum must be positive,
                         and dampening must be equal to 0.0. Default: False.
        loss_scale (float): A floating point value for the loss scale, which must be larger than 0.0. In general, use
            the default value. Only when `FixedLossScaleManager` is used for training and the `drop_overflow_update` in
            `FixedLossScaleManager` is set to False, then this value needs to be the same as the `loss_scale` in
            `FixedLossScaleManager`. Refer to class :class:`mindspore.FixedLossScaleManager` for more details.
            Default: 1.0.

    Inputs:
        - **gradients** (tuple[Tensor]) - The gradients of `params`, the shape is the same as `params`.

    Outputs:
        Tensor[bool], the value is True.

    Raises:
        ValueError: If the momentum, dampening or weight_decay value is less than 0.0.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import nn, Model
        >>>
        >>> net = Net()
        >>> #1) All parameters use the same learning rate and weight decay
        >>> optim = nn.SGD(params=net.trainable_params())
        >>>
        >>> #2) Use parameter groups and set different values
        >>> conv_params = list(filter(lambda x: 'conv' in x.name, net.trainable_params()))
        >>> no_conv_params = list(filter(lambda x: 'conv' not in x.name, net.trainable_params()))
        >>> group_params = [{'params': conv_params,'grad_centralization':True},
        ...                 {'params': no_conv_params, 'lr': 0.01},
        ...                 {'order_params': net.trainable_params()}]
        >>> optim = nn.SGD(group_params, learning_rate=0.1, weight_decay=0.0)
        >>> # The conv_params's parameters will use default learning rate of 0.1 and default weight decay of 0.0
        >>> # and grad centralization of True.
        >>> # The no_conv_params's parameters will use learning rate of 0.01 and default weight decay of 0.0 and grad
        >>> # centralization of False.
        >>> # The final parameters order in which the optimizer will be followed is the value of 'order_params'.
        >>>
        >>> loss = nn.SoftmaxCrossEntropyWithLogits()
        >>> model = Model(net, loss_fn=loss, optimizer=optim)
    """

    @opt_init_args_register
    def __init__(self, params, learning_rate=0.1, momentum=0.0, dampening=0.0, weight_decay=0.0, nesterov=False,
                 loss_scale=1.0):

        super(SGD, self).__init__(learning_rate, params, weight_decay, loss_scale)

        if isinstance(momentum, int):
            momentum = float(momentum)
        if not isinstance(momentum, float):
            raise TypeError("For 'SGD', the argument 'momentum' should be float type, "
                            "but got {}.".format(type(momentum)))

        if isinstance(momentum, float) and momentum < 0.0:
            raise ValueError("For 'SGD', the argument 'momentum' should be at least 0.0, "
                             "but got {}".format(momentum))

        if isinstance(dampening, int):
            dampening = float(dampening)
        if not isinstance(dampening, float):
            raise TypeError("For 'SGD', the argument 'dampening' should be float type, "
                            "but got {}.".format(type(dampening)))

        if dampening < 0.0:
            raise ValueError("For 'SGD', the argument 'dampening' should be at least 0.0, "
                             "but got 'dampening' {}".format(dampening))
        self.dampening = dampening

        if isinstance(weight_decay, int):
            weight_decay = float(weight_decay)

        validator.check_value_type("nesterov", nesterov, [bool], self.cls_name)

        if nesterov and (momentum <= 0.0 or dampening != 0.0):
            raise ValueError("For 'SGD', if 'nesterov' is true, 'momentum' must be > 0.0 and 'dampening' must "
                             "equal to 0.0, but got 'momentum' {}, 'dampening' {}".format(momentum, dampening))
        self.nesterov = nesterov

        self.opt = P.SGD(dampening, weight_decay, nesterov)

        self.momentum = Parameter(Tensor(momentum, mstype.float32), name="momentum")
        self.accum = self.parameters.clone(prefix="accum", init='zeros')
        self.stat = self.parameters.clone(prefix="stat", init='ones')

    def construct(self, gradients):
        params = self.parameters
        accum = self.accum
        stat = self.stat
        gradients = self.gradients_centralization(gradients)
        gradients = self.scale_grad(gradients)
        lr = self.get_lr()
        if self.is_group_lr:
            success = self.hyper_map_reverse(F.partial(_sgd_opt, self.opt, self.momentum),
                                             lr, gradients, params, accum, stat)
        else:
            success = self.hyper_map_reverse(F.partial(_sgd_opt, self.opt, self.momentum, lr),
                                             gradients, params, accum, stat)
        return success
