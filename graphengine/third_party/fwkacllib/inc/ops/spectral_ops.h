/**
 * Copyright 2019 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file spectral_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_SPECTRAL_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_SPECTRAL_OPS_H_

#include "graph/operator.h"
#include "graph/operator_reg.h"

namespace ge {

/**
*@brief Computes the inverse 1-dimensional discrete Fourier transform over the
inner-most dimension of `x`. \n

*@par Inputs:
*x: A Tensor. Must be the following types: complex64, complex128. \n

*@par Outputs:
*y: A complex tensor of the same rank as `x`. \n

*@par Third-party framework compatibility
* Compatible with TensorFlow IFFT operator.
*/
REG_OP(IFFT)
    .INPUT(x, TensorType({DT_COMPLEX64,DT_COMPLEX128}))
    .OUTPUT(y, TensorType({DT_COMPLEX64,DT_COMPLEX128}))
    .OP_END_FACTORY_REG(IFFT)

/**
*@brief Real-valued fast Fourier transform . \n

*@par Inputs:
*@li input: A float32 tensor.
*@li fft_length: An int32 tensor of shape [1]. The FFT length . \n

*@par Outputs:
*y: A complex64 tensor of the same rank as `input`. The inner-most
dimension of `input` is replaced with the `fft_length / 2 + 1` unique
frequency components of its 1D Fourier transform . \n

*@par Third-party framework compatibility
* Compatible with TensorFlow RFFT operator.
*/
REG_OP(RFFT)
    .INPUT(input, TensorType({DT_FLOAT}))
    .INPUT(fft_length, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_COMPLEX64}))
    .OP_END_FACTORY_REG(RFFT)

/**
*@brief Inverse real-valued fast Fourier transform. \n

*@par Inputs:
*@li x: A complex64 tensor.
*@li fft_length: An int32 tensor of shape [1]. The FFT length. \n

*@par Outputs:
* y: A float32 tensor of the same rank as `input`. The inner-most
  dimension of `input` is replaced with the `fft_length` samples of its inverse
  1D Fourier transform. \n

*@par Third-party framework compatibility
* Compatible with TensorFlow IRFFT operator.
*/
REG_OP(IRFFT)
    .INPUT(x, TensorType({DT_COMPLEX64}))
    .INPUT(fft_length, TensorType({DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(IRFFT)


/**
*@brief 2D fast Fourier transform. \n

*@par Inputs:
*x: A complex64 tensor.

*@par Outputs:
*y: A complex64 tensor of the same shape as `input`. The inner-most 2
  dimensions of `input` are replaced with their 2D Fourier transform. \n

*@par Third-party framework compatibility
* Compatible with TensorFlow FFT2D operator.
*/
REG_OP(FFT2D)
    .INPUT(x, TensorType({DT_COMPLEX64, DT_COMPLEX128}))
    .OUTPUT(y, TensorType({DT_COMPLEX64, DT_COMPLEX128}))
    .OP_END_FACTORY_REG(FFT2D)

/**
*@brief Calculate the one-dimensional discrete Fourier transform on the
innermost dimension of the input. \n

*@par Inputs:
*x: A Tensor. Must be the following types: complex64, complex128. \n

*@par Outputs:
*y: A complex tensor with the same shape as input. The innermost dimension
of the input is replaced by its 1-dimensional Fourier transform. \n

*@par Third-party framework compatibility
* Compatible with TensorFlow FFT operator.
*/
REG_OP(FFT)
    .INPUT(x, TensorType({DT_COMPLEX64,DT_COMPLEX128}))
    .OUTPUT(y, TensorType({DT_COMPLEX64,DT_COMPLEX128}))
    .OP_END_FACTORY_REG(FFT)

/**
*@brief Calculate the inverse 1-dimensional discrete Fourier transform on the
innermost dimension of the input. \n

*@par Inputs:
*x: A Tensor. Must be the following types: complex64, complex128. \n

*@par Outputs:
*y: A complex tensor with the same shape as input. The innermost dimension
of the input is replaced by its inverse two-dimensional Fourier transform. \n

*@par Third-party framework compatibility
* Compatible with TensorFlow IFFT2D operator.
*/
REG_OP(IFFT2D)
    .INPUT(x, TensorType({DT_COMPLEX64,DT_COMPLEX128}))
    .OUTPUT(y, TensorType({DT_COMPLEX64,DT_COMPLEX128}))
    .OP_END_FACTORY_REG(IFFT2D)

}  // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_SPECTRAL_OPS_H_