# Copyright 2021 Huawei Technologies Co., Ltd
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
"""minimize"""
from typing import Optional
from typing import NamedTuple
from ...common import Tensor
from ._bfgs import minimize_bfgs


class OptimizeResults(NamedTuple):
    """Object holding optimization results.

    Args:
        x (Tensor): final solution.
        success (bool): ``True`` if optimization succeeded.
        status (int): solver specific return code. 0 means converged (nominal),
            1=max BFGS iters reached, 3=zoom failed, 4=saddle point reached,
            5=max line search iters reached, -1=undefined
        fun (float): final function value.
        jac (Tensor): final jacobian array.
        hess_inv (Tensor, optional): final inverse Hessian estimate.
        nfev (int): number of function calls used.
        njev (int): number of gradient evaluations.
        nit (int): number of iterations of the optimization algorithm.
    """
    x: Tensor
    success: bool
    status: int
    fun: float
    jac: Tensor
    hess_inv: Optional[Tensor]
    nfev: int
    njev: int
    nit: int


def minimize(func, x0, args=(), *, method, tol=None, options=None):
    """Minimization of scalar function of one or more variables.

    This API for this function matches SciPy with some minor deviations:

    - Gradients of ``func`` are calculated automatically using MindSpore's autodiff
      support when required.
    - The ``method`` argument is required. You must specify a solver.
    - Various optional arguments in the SciPy interface have not yet been
      implemented.
    - Optimization results may differ from SciPy due to differences in the line
      search implementation.

    Note:
        - `minimize` does not yet support differentiation or arguments in the form of
          multi-dimensional Tensor, but support for both is planned.

        - `minimize` is not supported on Windows platform yet.

    Args:
      func (Callable): the objective function to be minimized, :math:`fun(x, *args) -> float`,
        where `x` is a 1-D array with shape :math:`(n,)` and `args` is a tuple
        of the fixed parameters needed to completely specify the function.
        `fun` must support differentiation.
      x0 (Tensor): initial guess. Array of real elements of size :math:`(n,)`, where `n` is
        the number of independent variables.
      args (Tuple): extra arguments passed to the objective function. Default: ().
      method (str): solver type. Currently only `"BFGS"` is supported.
      tol (float, optional): tolerance for termination. For detailed control, use solver-specific
        options. Default: None.
      options (Mapping[str, Any], optional): a dictionary of solver options. All methods accept the following
        generic options, Default: None.

        - maxiter (int): Maximum number of iterations to perform. Depending on the
          method each iteration may use several function evaluations.

    Returns:
        OptimizeResults, object holding optimization results.

    Supported Platforms:
        ``CPU`` ``GPU``

    Examples:
        >>> import numpy as onp
        >>> from mindspore.scipy.optimize import minimize
        >>> from mindspore.common import Tensor
        >>> x0 = Tensor(onp.zeros(2).astype(onp.float32))
        >>> def func(p):
        >>>     x, y = p
        >>>     return (x ** 2 + y - 11.) ** 2 + (x + y ** 2 - 7.) ** 2
        >>> res = minimize(func, x0, method='BFGS', options=dict(maxiter=None, gtol=1e-6))
        >>> print(res.x)
        [3. 2.]
    """
    if options is None:
        options = {}

    if not isinstance(args, tuple):
        msg = "args argument to mindspore.scipy.optimize.minimize must be a tuple, got {}"
        raise TypeError(msg.format(args))

    def fun_with_args(args):
        def inner_func(x):
            return func(x, *args)

        return inner_func

    if method.lower() == 'bfgs':
        results = minimize_bfgs(fun_with_args(args), x0, **options)
        success = results.converged and not results.failed
        return OptimizeResults(x=results.x_k,
                               success=success,
                               status=results.status,
                               fun=results.f_k,
                               jac=results.g_k,
                               hess_inv=results.H_k,
                               nfev=results.nfev,
                               njev=results.ngev,
                               nit=results.k)

    raise ValueError("Method {} not recognized".format(method))
