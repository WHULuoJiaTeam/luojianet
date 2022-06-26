# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
"""FTRL"""
from mindspore.ops import functional as F, composite as C, operations as P
from mindspore.common import Tensor
import mindspore.common.dtype as mstype
from mindspore._checkparam import Validator as validator
from mindspore._checkparam import Rel
from .optimizer import Optimizer, _apply_decay, _grad_scale
from .optimizer import opt_init_args_register

_ftrl_opt = C.MultitypeFuncGraph("ftrl_opt")


@_ftrl_opt.register("Function", "Function", "Function", "Function", "Number", "Number", "Number", "Tensor", "Tensor",
                    "RowTensor", "Tensor", "Tensor", "Bool", "Bool")
def _tensor_run_opt_with_sparse(opt, spars_opt, push, pull, l1, l2, lr_power, learning_rate, linear,
                                gradient, weight, moment, ps_parameter, cache_enable):
    """Apply sparse ftrl optimizer to the weight parameter when the gradient is sparse."""
    success = True
    indices = gradient.indices
    values = gradient.values
    if ps_parameter and not cache_enable:
        op_shape = P.Shape()
        shapes = (op_shape(weight), op_shape(moment), op_shape(linear), op_shape(values), op_shape(indices))
        success = F.depend(success, pull(push((values, indices), shapes), weight))
    else:
        success = F.depend(success, spars_opt(weight, moment, linear, values, indices))
    return success


@_ftrl_opt.register("Function", "Function", "Function", "Function", "Number", "Number", "Number", "Tensor", "Tensor",
                    "Tensor", "Tensor", "Tensor", "Bool", "Bool")
def _tensor_run_opt(opt, spars_opt, push, pull, l1, l2, lr_power, learning_rate, linear,
                    gradient, weight, moment, ps_parameter, cache_enable):
    """Apply ftrl optimizer to the weight parameter."""
    success = True
    if ps_parameter and not cache_enable:
        op_shape = P.Shape()
        success = F.depend(success, pull(push((gradient, learning_rate, l1, l2, lr_power),
                                              (op_shape(weight), op_shape(moment), op_shape(linear))), weight))
    else:
        success = F.depend(success, opt(weight, moment, linear, gradient, learning_rate, l1, l2, lr_power))
    return success


def _check_param(initial_accum, lr_power, l1, l2, use_locking, prim_name=None):
    """Check param."""
    validator.check_value_type("initial_accum", initial_accum, [float], prim_name)
    validator.check_number("initial_accum", initial_accum, 0.0, Rel.GE, prim_name)

    validator.check_value_type("lr_power", lr_power, [float], prim_name)
    validator.check_number("lr_power", lr_power, 0.0, Rel.LE, prim_name)

    validator.check_value_type("l1", l1, [float], prim_name)
    validator.check_number("l1", l1, 0.0, Rel.GE, prim_name)

    validator.check_value_type("l2", l2, [float], prim_name)
    validator.check_number("l2", l2, 0.0, Rel.GE, prim_name)

    validator.check_value_type("use_locking", use_locking, [bool], prim_name)


class FTRL(Optimizer):
    r"""
    Implements the FTRL algorithm.

    FTRL is an online convex optimization algorithm that adaptively chooses its regularization function
    based on the loss functions. Refer to paper `Adaptive Bound Optimization for Online Convex Optimization
    <https://arxiv.org/abs/1002.4908>`_. Refer to paper `Ad Click Prediction: a View from the Trenches
    <https://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf>`_ for engineering document.

    The updating formulas are as follows,

    .. math::

        \begin{array}{ll} \\
            m_{t+1} = m_{t} + g^2 \\
            u_{t+1} = u_{t} + g  - \frac{m_{t+1}^\text{-p} - m_{t}^\text{-p}}{\alpha } * \omega_{t} \\
            \omega_{t+1} =
            \begin{cases}
                \frac{(sign(u_{t+1}) * l1 - u_{t+1})}{\frac{m_{t+1}^\text{-p}}{\alpha } + 2 * l2 }
                    & \text{ if } |u_{t+1}| > l1 \\
                0.0
                    & \text{ otherwise }
            \end{cases}\\
        \end{array}

    :math:`m` represents accumulators, :math:`g` represents `grads`, :math:`t` represents the current step,
    :math:`u` represents the linear coefficient to be updated, :math:`p` represents `lr_power`, :math:`\alpha`
    represents `learning_rate`, :math:`\omega` represents `params`.

    Note:
        The sparse strategy is applied while the SparseGatherV2 operator is used for forward network. If the sparse
        strategy wants to be executed on the host, set the target to the CPU.
        The sparse feature is under continuous development.

        If parameters are not grouped, the `weight_decay` in optimizer will be applied on the network parameters without
        'beta' or 'gamma' in their names. Users can group parameters to change the strategy of decaying weight. When
        parameters are grouped, each group can set `weight_decay`, if not, the `weight_decay` in optimizer will be
        applied.

    Args:
        params (Union[list[Parameter], list[dict]]): Must be list of `Parameter` or list of `dict`. When the
            `params` is a list of `dict`, the string "params", "weight_decay", "grad_centralization" and "order_params"
            are the keys can be parsed.

            - params: Required. Parameters in current group. The value must be a list of `Parameter`.

            - lr: Using different learning rate by grouping parameters is currently not supported.

            - weight_decay: Optional. If "weight_decay" in the keys, the value of corresponding weight decay
              will be used. If not, the `weight_decay` in the optimizer will be used. It should be noted that weight
              decay can be a constant value or a Cell. It is a Cell only when dynamic weight decay is applied. Dynamic
              weight decay is similar to dynamic learning rate, users need to customize a weight decay schedule only
              with global step as input, and during training, the optimizer calls the instance of WeightDecaySchedule
              to get the weight decay value of current step.

            - grad_centralization: Optional. Must be Boolean. If "grad_centralization" is in the keys, the set value
              will be used. If not, the `grad_centralization` is False by default. This configuration only works on the
              convolution layer.

            - order_params: Optional. When parameters is grouped, this usually is used to maintain the order of
              parameters that appeared in the network to improve performance. The value should be parameters whose
              order will be followed in optimizer.
              If `order_params` in the keys, other keys will be ignored and the element of 'order_params' must be in
              one group of `params`.

        initial_accum (float): The starting value for accumulators `m`, must be zero or positive values. Default: 0.1.
        learning_rate (float): The learning rate value, must be zero or positive, dynamic learning rate is currently
            not supported. Default: 0.001.
        lr_power (float): Learning rate power controls how the learning rate decreases during training, must be less
            than or equal to zero. Use fixed learning rate if lr_power is zero. Default: -0.5.
        l1 (float): l1 regularization strength, must be greater than or equal to zero. Default: 0.0.
        l2 (float): l2 regularization strength, must be greater than or equal to zero. Default: 0.0.
        use_locking (bool): If true, use locks for updating operation. Default: False.
        loss_scale (float): Value for the loss scale. It must be greater than 0.0. In general, use the default value.
            Only when `FixedLossScaleManager` is used for training and the `drop_overflow_update` in
            `FixedLossScaleManager` is set to False, then this value needs to be the same as the `loss_scale` in
            `FixedLossScaleManager`. Refer to class :class:`mindspore.FixedLossScaleManager` for more details.
            Default: 1.0.
        weight_decay (Union[float, int, Cell]): Weight decay (L2 penalty). Default: 0.0.

            - float: The fixed weight decay value. Must be equal to or greater than 0.

            - int: The fixed weight decay value. Must be equal to or greater than 0. It will be converted to float.

            - Cell: Weight decay is dynamic. During training, the optimizer calls the instance of
              the Cell with step as the input to get the weight decay value of current step.

    Inputs:
        - **grads** (tuple[Tensor]) - The gradients of `params` in the optimizer, the shape is the same as the `params`
          in optimizer.

    Outputs:
        Tuple[Parameter], the updated parameters, the shape is the same as `params`.

    Raises:
        TypeError: If `initial_accum`, `learning_rate`, `lr_power`, `l1`, `l2` or `loss_scale` is not a float.
        TypeError: If element of `parameters` is neither Parameter nor dict.
        TypeError: If `weight_decay` is neither float nor int.
        TypeError: If `use_nesterov` is not a bool.
        ValueError: If `lr_power` is greater than 0.
        ValueError: If `loss_scale` is less than or equal to 0.
        ValueError: If `initial_accum`, `l1` or `l2` is less than 0.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> from mindspore import nn, Model
        >>>
        >>> net = Net()
        >>> #1) All parameters use the same learning rate and weight decay
        >>> optim = nn.FTRL(params=net.trainable_params())
        >>>
        >>> #2) Use parameter groups and set different values
        >>> conv_params = list(filter(lambda x: 'conv' in x.name, net.trainable_params()))
        >>> no_conv_params = list(filter(lambda x: 'conv' not in x.name, net.trainable_params()))
        >>> group_params = [{'params': conv_params, 'weight_decay': 0.01, 'grad_centralization':True},
        ...                 {'params': no_conv_params},
        ...                 {'order_params': net.trainable_params()}]
        >>> optim = nn.FTRL(group_params, learning_rate=0.1, weight_decay=0.0)
        >>> # The conv_params's parameters will use default learning rate of 0.1 and weight decay of 0.01 and grad
        >>> # centralization of True.
        >>> # The no_conv_params's parameters will use default learning rate of 0.1 will use default weight decay
        >>> # of 0.0 and grad centralization of False.
        >>> # The final parameters order in which the optimizer will be followed is the value of 'order_params'.
        >>>
        >>> loss = nn.SoftmaxCrossEntropyWithLogits()
        >>> model = Model(net, loss_fn=loss, optimizer=optim)
    """

    @opt_init_args_register
    def __init__(self, params, initial_accum=0.1, learning_rate=0.001, lr_power=-0.5, l1=0.0, l2=0.0,
                 use_locking=False, loss_scale=1.0, weight_decay=0.0):
        super(FTRL, self).__init__(learning_rate, params, weight_decay, loss_scale=loss_scale)
        if self.dynamic_lr or self.is_group_lr:
            raise ValueError(f"For 'FTRL', dynamic learning rate and group learning rate are currently not supported "
                             f"in FTRL, they should all be false, but got dynamic learning rate {self.dynamic_lr} and"
                             f" group learning rate {self.is_group_lr}.")
        _check_param(initial_accum, lr_power, l1, l2, use_locking, self.cls_name)
        self.moments = self.parameters.clone(prefix="moments", init=initial_accum)
        self.linear = self.parameters.clone(prefix="linear", init='zeros')
        self.l1 = l1
        self.l2 = l2
        self.lr = learning_rate
        self.lr_power = lr_power
        if not self.is_group:
            self.decay_flags = tuple((lambda: True)() for x in self.parameters)
        self.opt = P.ApplyFtrl(use_locking=use_locking)
        self.use_locking = use_locking
        self.sparse_opt = P.SparseApplyFtrl(learning_rate, l1, l2, lr_power, use_locking=use_locking)
        self._ps_pull = P.Pull()
        self._ps_push = P.Push("Ftrl", [0, 1, 2])
        self._ps_push.add_prim_attr("init_accum", initial_accum)
        self._ps_push.add_prim_attr("lr", learning_rate)
        self._ps_push.add_prim_attr("l1", l1)
        self._ps_push.add_prim_attr("l2", l2)
        self._ps_push.add_prim_attr("lr_power", lr_power)

    def construct(self, grads):
        params = self.parameters
        moments = self.moments
        linear = self.linear
        grads = self.decay_weight(grads)
        grads = self.gradients_centralization(grads)
        grads = self.scale_grad(grads)
        grads = self._grad_sparse_indices_deduplicate(grads)
        lr = self.get_lr()

        success = self.map_(F.partial(_ftrl_opt, self.opt, self.sparse_opt, self._ps_push, self._ps_pull,
                                      self.l1, self.l2, self.lr_power, lr),
                            linear, grads, params, moments, self.ps_parameters, self.cache_enable)
        return success

    @Optimizer.target.setter
    def target(self, value):
        """
        If the input value is set to "CPU", the parameters will be updated on the host using the Fused
        optimizer operation.
        """
        if not isinstance(value, str):
            raise TypeError("For 'FTRL', the property 'target' must be string type, "
                            "but got type {}.".format(type(value)))

        if value not in ('CPU', 'Ascend', 'GPU'):
            raise ValueError("For 'FTRL', the property 'target' must be 'CPU', 'Ascend' or 'GPU', "
                             "but got {}".format(value))

        if value == 'CPU':
            self.sparse_opt = P.FusedSparseFtrl(self.lr, self.l1, self.l2, self.lr_power, self.use_locking)
            self.sparse_opt.add_prim_attr("primitive_target", "CPU")
        else:
            self.sparse_opt = P.SparseApplyFtrl(self.lr, self.l1, self.l2, self.lr_power, self.use_locking)

        self._target = value
