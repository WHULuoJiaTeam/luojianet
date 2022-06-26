/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#include "unary_op_impl.cuh"
#include "include/cuda_fp16.h"
template <typename T>
__global__ void ExponentialKernel(const T *input, T *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = expf(input[i]);
  }
  return;
}
template <>
__global__ void ExponentialKernel(const double *input, double *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = exp(input[i]);
  }
  return;
}
template <>
__global__ void ExponentialKernel(const half *input, half *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = hexp(input[i]);
  }
  return;
}
template <typename T>
__global__ void Expm1Kernel(const T *input, T *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = expm1f(input[i]);
  }
  return;
}
template <>
__global__ void Expm1Kernel(const double *input, double *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = expm1(input[i]);
  }
  return;
}
template <typename T>
__global__ void LogarithmKernel(const T *input, T *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = logf(input[i]);
  }
  return;
}
template <>
__global__ void LogarithmKernel(const double *input, double *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = log(input[i]);
  }
  return;
}
template <>
__global__ void LogarithmKernel(const half *input, half *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = hlog(input[i]);
  }
  return;
}
template <typename T>
__global__ void Log1pKernel(const T *input, T *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = log1pf(input[i]);
  }
  return;
}
template <>
__global__ void Log1pKernel(const double *input, double *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = log1p(input[i]);
  }
  return;
}
template <typename T>
__global__ void ErfKernel(const T *input, T *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = erff(input[i]);
  }
  return;
}
template <>
__global__ void ErfKernel(const double *input, double *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = erf(input[i]);
  }
  return;
}
template <typename T>
__global__ void ErfcKernel(const T *input, T *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = erfcf(input[i]);
  }
  return;
}
template <>
__global__ void ErfcKernel(const double *input, double *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = erfc(input[i]);
  }
  return;
}
template <typename T>
__global__ void NegativeKernel(const T *input, T *output, const size_t count) {
  T neg_one = -1;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = neg_one * input[i];
  }
  return;
}
template <typename T>
__global__ void ReciprocalKernel(const T *input, T *output, const size_t count) {
  T one = 1.0;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = one / input[i];
  }
  return;
}
template <typename T>
__global__ void SquareKernel(const T *input, T *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = input[i] * input[i];
  }
  return;
}
template <typename T>
__global__ void SqrtKernel(const T *input, T *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = sqrtf(input[i]);
  }
  return;
}
template <>
__global__ void SqrtKernel(const double *input, double *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = sqrt(input[i]);
  }
  return;
}
template <>
__global__ void SqrtKernel(const half *input, half *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = hsqrt(input[i]);
  }
  return;
}
template <typename T>
__global__ void RsqrtKernel(const T *input, T *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = rsqrtf(input[i]);
  }
  return;
}
template <>
__global__ void RsqrtKernel(const double *input, double *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = rsqrt(input[i]);
  }
  return;
}
template <>
__global__ void RsqrtKernel(const half *input, half *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = hrsqrt(input[i]);
  }
  return;
}
template <typename T>
__global__ void SinKernel(const T *input, T *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = sinf(input[i]);
  }
  return;
}
template <>
__global__ void SinKernel(const double *input, double *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = sin(input[i]);
  }
  return;
}
template <>
__global__ void SinKernel(const half *input, half *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = hsin(input[i]);
  }
  return;
}
template <typename T>
__global__ void AsinKernel(const T *input, T *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = asinf(input[i]);
  }
  return;
}
template <>
__global__ void AsinKernel(const double *input, double *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = asin(input[i]);
  }
  return;
}
template <typename T>
__global__ void AsinhKernel(const T *input, T *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = asinhf(input[i]);
  }
  return;
}
template <>
__global__ void AsinhKernel(const double *input, double *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = asinh(input[i]);
  }
  return;
}
template <typename T>
__global__ void CosKernel(const T *input, T *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = cosf(input[i]);
  }
  return;
}
template <>
__global__ void CosKernel(const double *input, double *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = cos(input[i]);
  }
  return;
}
template <>
__global__ void CosKernel(const half *input, half *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = hcos(input[i]);
  }
  return;
}
template <typename T>
__global__ void ACosKernel(const T *input, T *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = acosf(input[i]);
  }
  return;
}
template <>
__global__ void ACosKernel(const double *input, double *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = acos(input[i]);
  }
  return;
}
template <typename T>
__global__ void AcoshKernel(const T *input, T *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = acoshf(input[i]);
  }
  return;
}
template <>
__global__ void AcoshKernel(const double *input, double *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = acosh(input[i]);
  }
  return;
}
template <typename T>
__global__ void AtanKernel(const T *input, T *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = atanf(input[i]);
  }
  return;
}
template <>
__global__ void AtanKernel(const double *input, double *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = atan(input[i]);
  }
  return;
}
template <typename T>
__global__ void AbsKernel(const T *input, T *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = abs(input[i]);
  }
  return;
}
template <>
__global__ void AbsKernel(const half *input, half *output, const size_t count) {
  half zero = 0.0;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = input[i] < zero ? -input[i] : input[i];
  }
  return;
}
template <typename T>
__global__ void AbsKernel(const Complex<T> *input, T *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = abs(input[i]);
  }
  return;
}
template <typename T>
__global__ void RealKernel(const Complex<T> *input, T *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = input[i].real();
  }
  return;
}
template <typename T>
__global__ void RealKernel(const T *input, T *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = input[i];
  }
  return;
}
template <typename T>
__global__ void ImagKernel(const Complex<T> *input, T *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = input[i].imag();
  }
  return;
}
template <typename T>
__global__ void ImagKernel(const T *input, T *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    T zero = 0;
    output[i] = zero;
  }
  return;
}
template <typename T>
__global__ void ConjKernel(const Complex<T> *input, Complex<T> *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = Complex<T>(input[i].real(), -input[i].imag());
  }
  return;
}
template <typename T>
__global__ void ConjKernel(const T *input, T *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = input[i];
  }
  return;
}
template <typename T>
__global__ void FloorKernel(const T *input, T *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = floorf(input[i]);
  }
  return;
}
template <>
__global__ void FloorKernel(const double *input, double *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = floor(input[i]);
  }
  return;
}
template <>
__global__ void FloorKernel(const half *input, half *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = hfloor(input[i]);
  }
  return;
}
template <typename T>
__global__ void RintKernel(const T *input, T *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = rintf(input[i]);
  }
  return;
}
template <>
__global__ void RintKernel(const double *input, double *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = rint(input[i]);
  }
  return;
}
template <>
__global__ void RintKernel(const half *input, half *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = hrint(input[i]);
  }
  return;
}
template <typename T>
__global__ void RoundKernel(const T *input, T *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = nearbyintf(input[i]);
  }
  return;
}
template <>
__global__ void RoundKernel(const double *input, double *output, const size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = nearbyint(input[i]);
  }
  return;
}
template <typename T>
__global__ void SignKernel(const T *input, T *output, const size_t count) {
  T zero = 0.0;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    T res;
    if (input[i] < zero) {
      res = -1;
    } else if (input[i] > zero) {
      res = 1;
    } else {
      res = 0;
    }
    output[i] = static_cast<T>(res);
  }
  return;
}
template <typename T>
void Exponential(const T *input, T *output, const size_t count, cudaStream_t cuda_stream) {
  ExponentialKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}
template <typename T>
void Expm1(const T *input, T *output, const size_t count, cudaStream_t cuda_stream) {
  Expm1Kernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}
template <typename T>
void Logarithm(const T *input, T *output, const size_t count, cudaStream_t cuda_stream) {
  LogarithmKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}
template <typename T>
void Log1p(const T *input, T *output, const size_t count, cudaStream_t cuda_stream) {
  Log1pKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}
template <typename T>
void Erf(const T *input, T *output, const size_t count, cudaStream_t cuda_stream) {
  ErfKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}
template <typename T>
void Erfc(const T *input, T *output, const size_t count, cudaStream_t cuda_stream) {
  ErfcKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}
template <typename T>
void Negative(const T *input, T *output, const size_t count, cudaStream_t cuda_stream) {
  NegativeKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}
template <typename T>
void Reciprocal(const T *input, T *output, const size_t count, cudaStream_t cuda_stream) {
  ReciprocalKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}
template <typename T>
void Square(const T *input, T *output, const size_t count, cudaStream_t cuda_stream) {
  SquareKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}
template <typename T>
void Pow(const T *input, T *output, const size_t count, cudaStream_t cuda_stream) {
  PowKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}
template <typename T>
void Sqrt(const T *input, T *output, const size_t count, cudaStream_t cuda_stream) {
  SqrtKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}
template <typename T>
void Sin(const T *input, T *output, const size_t count, cudaStream_t cuda_stream) {
  SinKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}
template <typename T>
void Cos(const T *input, T *output, const size_t count, cudaStream_t cuda_stream) {
  CosKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}
template <typename T>
void Asin(const T *input, T *output, const size_t count, cudaStream_t cuda_stream) {
  AsinKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}
template <typename T>
void ACos(const T *input, T *output, const size_t count, cudaStream_t cuda_stream) {
  ACosKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}
template <typename T>
void Atan(const T *input, T *output, const size_t count, cudaStream_t cuda_stream) {
  AtanKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}
template <typename T>
void Asinh(const T *input, T *output, const size_t count, cudaStream_t cuda_stream) {
  AsinhKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}
template <typename T>
void Acosh(const T *input, T *output, const size_t count, cudaStream_t cuda_stream) {
  AcoshKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}
template <typename T>
void Rsqrt(const T *input, T *output, const size_t count, cudaStream_t cuda_stream) {
  RsqrtKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}
template <typename T>
void Abs(const T *input, T *output, const size_t count, cudaStream_t cuda_stream) {
  AbsKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}
template <typename T>
void Abs(const Complex<T> *input, T *output, const size_t count, cudaStream_t cuda_stream) {
  AbsKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}
template <typename T>
void Real(const Complex<T> *input, T *output, const size_t count, cudaStream_t cuda_stream) {
  RealKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}
template <typename T>
void Real(const T *input, T *output, const size_t count, cudaStream_t cuda_stream) {
  RealKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}
template <typename T>
void Imag(const Complex<T> *input, T *output, const size_t count, cudaStream_t cuda_stream) {
  ImagKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}
template <typename T>
void Imag(const T *input, T *output, const size_t count, cudaStream_t cuda_stream) {
  ImagKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}
template <typename T>
void Conj(const Complex<T> *input, Complex<T> *output, const size_t count, cudaStream_t cuda_stream) {
  ConjKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}
template <typename T>
void Conj(const T *input, T *output, const size_t count, cudaStream_t cuda_stream) {
  ConjKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}
template <typename T>
void Floor(const T *input, T *output, const size_t count, cudaStream_t cuda_stream) {
  FloorKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}
template <typename T>
void Rint(const T *input, T *output, const size_t count, cudaStream_t cuda_stream) {
  RintKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}
template <typename T>
void Round(const T *input, T *output, const size_t count, cudaStream_t cuda_stream) {
  RoundKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}
template <typename T>
void Sign(const T *input, T *output, const size_t count, cudaStream_t cuda_stream) {
  SignKernel<<<GET_BLOCKS(count), GET_THREADS, 0, cuda_stream>>>(input, output, count);
  return;
}

// double
template CUDA_LIB_EXPORT void Exponential<double>(const double *input, double *output, const size_t count,
                                                  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Expm1<double>(const double *input, double *output, const size_t count,
                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Logarithm<double>(const double *input, double *output, const size_t count,
                                                cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Log1p<double>(const double *input, double *output, const size_t count,
                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Erf<double>(const double *input, double *output, const size_t count,
                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Erfc<double>(const double *input, double *output, const size_t count,
                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Negative<double>(const double *input, double *output, const size_t count,
                                               cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Reciprocal<double>(const double *input, double *output, const size_t count,
                                                 cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Square<double>(const double *input, double *output, const size_t count,
                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Sqrt<double>(const double *input, double *output, const size_t count,
                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Sin<double>(const double *input, double *output, const size_t count,
                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Cos<double>(const double *input, double *output, const size_t count,
                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Asin<double>(const double *input, double *output, const size_t count,
                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void ACos<double>(const double *input, double *output, const size_t count,
                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Atan<double>(const double *input, double *output, const size_t count,
                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Asinh<double>(const double *input, double *output, const size_t count,
                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Acosh<double>(const double *input, double *output, const size_t count,
                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Rsqrt<double>(const double *input, double *output, const size_t count,
                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Abs<double>(const double *input, double *output, const size_t count,
                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Floor<double>(const double *input, double *output, const size_t count,
                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Rint<double>(const double *input, double *output, const size_t count,
                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Round<double>(const double *input, double *output, const size_t count,
                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Sign<double>(const double *input, double *output, const size_t count,
                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Real<double>(const double *input, double *output, const size_t count,
                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Imag<double>(const double *input, double *output, const size_t count,
                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Conj<double>(const double *input, double *output, const size_t count,
                                           cudaStream_t cuda_stream);


// float
template CUDA_LIB_EXPORT void Exponential<float>(const float *input, float *output, const size_t count,
                                                 cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Expm1<float>(const float *input, float *output, const size_t count,
                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Logarithm<float>(const float *input, float *output, const size_t count,
                                               cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Log1p<float>(const float *input, float *output, const size_t count,
                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Erf<float>(const float *input, float *output, const size_t count,
                                         cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Erfc<float>(const float *input, float *output, const size_t count,
                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Negative<float>(const float *input, float *output, const size_t count,
                                              cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Reciprocal<float>(const float *input, float *output, const size_t count,
                                                cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Square<float>(const float *input, float *output, const size_t count,
                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Sqrt<float>(const float *input, float *output, const size_t count,
                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Sin<float>(const float *input, float *output, const size_t count,
                                         cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Cos<float>(const float *input, float *output, const size_t count,
                                         cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Asin<float>(const float *input, float *output, const size_t count,
                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void ACos<float>(const float *input, float *output, const size_t count,
                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Atan<float>(const float *input, float *output, const size_t count,
                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Asinh<float>(const float *input, float *output, const size_t count,
                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Acosh<float>(const float *input, float *output, const size_t count,
                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Rsqrt<float>(const float *input, float *output, const size_t count,
                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Abs<float>(const float *input, float *output, const size_t count,
                                         cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Floor<float>(const float *input, float *output, const size_t count,
                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Rint<float>(const float *input, float *output, const size_t count,
                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Round<float>(const float *input, float *output, const size_t count,
                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Sign<float>(const float *input, float *output, const size_t count,
                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Real<float>(const float *input, float *output, const size_t count,
                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Imag<float>(const float *input, float *output, const size_t count,
                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Conj<float>(const float *input, float *output, const size_t count,
                                          cudaStream_t cuda_stream);

// half
template CUDA_LIB_EXPORT void Exponential<half>(const half *input, half *output, const size_t count,
                                                cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Expm1<half>(const half *input, half *output, const size_t count,
                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Logarithm<half>(const half *input, half *output, const size_t count,
                                              cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Log1p<half>(const half *input, half *output, const size_t count,
                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Erf<half>(const half *input, half *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Erfc<half>(const half *input, half *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Negative<half>(const half *input, half *output, const size_t count,
                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Reciprocal<half>(const half *input, half *output, const size_t count,
                                               cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Square<half>(const half *input, half *output, const size_t count,
                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Sqrt<half>(const half *input, half *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Sin<half>(const half *input, half *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Cos<half>(const half *input, half *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Asin<half>(const half *input, half *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void ACos<half>(const half *input, half *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Atan<half>(const half *input, half *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Asinh<half>(const half *input, half *output, const size_t count,
                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Acosh<half>(const half *input, half *output, const size_t count,
                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Rsqrt<half>(const half *input, half *output, const size_t count,
                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Abs<half>(const half *input, half *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Floor<half>(const half *input, half *output, const size_t count,
                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Rint<half>(const half *input, half *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Round<half>(const half *input, half *output, const size_t count,
                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Sign<half>(const half *input, half *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Real<half>(const half *input, half *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Imag<half>(const half *input, half *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Conj<half>(const half *input, half *output, const size_t count, cudaStream_t cuda_stream);

// int8
template CUDA_LIB_EXPORT void Exponential<char>(const char *input, char *output, const size_t count,
                                                cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Expm1<char>(const char *input, char *output, const size_t count,
                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Logarithm<char>(const char *input, char *output, const size_t count,
                                              cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Log1p<char>(const char *input, char *output, const size_t count,
                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Erf<char>(const char *input, char *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Erfc<char>(const char *input, char *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Negative<char>(const char *input, char *output, const size_t count,
                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Reciprocal<char>(const char *input, char *output, const size_t count,
                                               cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Square<char>(const char *input, char *output, const size_t count,
                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Sqrt<char>(const char *input, char *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Sin<char>(const char *input, char *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Cos<char>(const char *input, char *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Asin<char>(const char *input, char *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void ACos<char>(const char *input, char *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Atan<char>(const char *input, char *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Asinh<char>(const char *input, char *output, const size_t count,
                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Acosh<char>(const char *input, char *output, const size_t count,
                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Rsqrt<char>(const char *input, char *output, const size_t count,
                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Abs<char>(const char *input, char *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Floor<char>(const char *input, char *output, const size_t count,
                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Rint<char>(const char *input, char *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Round<char>(const char *input, char *output, const size_t count,
                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Sign<char>(const char *input, char *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Real<char>(const char *input, char *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Imag<char>(const char *input, char *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Conj<char>(const char *input, char *output, const size_t count, cudaStream_t cuda_stream);

// uint8
template CUDA_LIB_EXPORT void Exponential<unsigned char>(const unsigned char *input, unsigned char *output,
                                                         const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Expm1<unsigned char>(const unsigned char *input, unsigned char *output,
                                                   const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Logarithm<unsigned char>(const unsigned char *input, unsigned char *output,
                                                       const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Log1p<unsigned char>(const unsigned char *input, unsigned char *output,
                                                   const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Erf<unsigned char>(const unsigned char *input, unsigned char *output, const size_t count,
                                                 cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Erfc<unsigned char>(const unsigned char *input, unsigned char *output, const size_t count,
                                                  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Negative<unsigned char>(const unsigned char *input, unsigned char *output,
                                                      const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Reciprocal<unsigned char>(const unsigned char *input, unsigned char *output,
                                                        const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Square<unsigned char>(const unsigned char *input, unsigned char *output,
                                                    const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Sqrt<unsigned char>(const unsigned char *input, unsigned char *output, const size_t count,
                                                  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Sin<unsigned char>(const unsigned char *input, unsigned char *output, const size_t count,
                                                 cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Cos<unsigned char>(const unsigned char *input, unsigned char *output, const size_t count,
                                                 cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Asin<unsigned char>(const unsigned char *input, unsigned char *output, const size_t count,
                                                  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void ACos<unsigned char>(const unsigned char *input, unsigned char *output, const size_t count,
                                                  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Atan<unsigned char>(const unsigned char *input, unsigned char *output, const size_t count,
                                                  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Asinh<unsigned char>(const unsigned char *input, unsigned char *output,
                                                   const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Acosh<unsigned char>(const unsigned char *input, unsigned char *output,
                                                   const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Rsqrt<unsigned char>(const unsigned char *input, unsigned char *output,
                                                   const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Abs<unsigned char>(const unsigned char *input, unsigned char *output, const size_t count,
                                                 cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Floor<unsigned char>(const unsigned char *input, unsigned char *output,
                                                   const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Rint<unsigned char>(const unsigned char *input, unsigned char *output, const size_t count,
                                                  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Round<unsigned char>(const unsigned char *input, unsigned char *output,
                                                   const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Sign<unsigned char>(const unsigned char *input, unsigned char *output, const size_t count,
                                                  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Real<unsigned char>(const unsigned char *input, unsigned char *output, const size_t count,
                                                  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Imag<unsigned char>(const unsigned char *input, unsigned char *output, const size_t count,
                                                  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Conj<unsigned char>(const unsigned char *input, unsigned char *output, const size_t count,
                                                  cudaStream_t cuda_stream);

// int32
template CUDA_LIB_EXPORT void Exponential<int>(const int *input, int *output, const size_t count,
                                               cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Expm1<int>(const int *input, int *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Logarithm<int>(const int *input, int *output, const size_t count,
                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Log1p<int>(const int *input, int *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Erf<int>(const int *input, int *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Erfc<int>(const int *input, int *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Negative<int>(const int *input, int *output, const size_t count,
                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Reciprocal<int>(const int *input, int *output, const size_t count,
                                              cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Square<int>(const int *input, int *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Sqrt<int>(const int *input, int *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Sin<int>(const int *input, int *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Cos<int>(const int *input, int *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Asin<int>(const int *input, int *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void ACos<int>(const int *input, int *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Atan<int>(const int *input, int *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Asinh<int>(const int *input, int *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Acosh<int>(const int *input, int *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Rsqrt<int>(const int *input, int *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Abs<int>(const int *input, int *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Floor<int>(const int *input, int *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Rint<int>(const int *input, int *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Round<int>(const int *input, int *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Sign<int>(const int *input, int *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Real<int>(const int *input, int *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Imag<int>(const int *input, int *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Conj<int>(const int *input, int *output, const size_t count, cudaStream_t cuda_stream);

// complex64
template CUDA_LIB_EXPORT void Real<float>(const Complex<float> *input, float *output, const size_t count,
                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Imag<float>(const Complex<float> *input, float *output, const size_t count,
                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Conj<float>(const Complex<float> *input, Complex<float> *output, const size_t count,
                                          cudaStream_t cuda_stream);

// complex128
template CUDA_LIB_EXPORT void Real<double>(const Complex<double> *input, double *output, const size_t count,
                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Imag<double>(const Complex<double> *input, double *output, const size_t count,
                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Conj<double>(const Complex<double> *input, Complex<double> *output, const size_t count,
                                           cudaStream_t cuda_stream);

// bool
template CUDA_LIB_EXPORT void Real<bool>(const bool *input, bool *output, const size_t count, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Imag<bool>(const bool *input, bool *output, const size_t count, cudaStream_t cuda_stream);

// int16
template CUDA_LIB_EXPORT void Real<int16_t>(const int16_t *input, int16_t *output, const size_t count,
                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Imag<int16_t>(const int16_t *input, int16_t *output, const size_t count,
                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Conj<int16_t>(const int16_t *input, int16_t *output, const size_t count,
                                            cudaStream_t cuda_stream);

// uint16
template CUDA_LIB_EXPORT void Real<uint16_t>(const uint16_t *input, uint16_t *output, const size_t count,
                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Imag<uint16_t>(const uint16_t *input, uint16_t *output, const size_t count,
                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Conj<uint16_t>(const uint16_t *input, uint16_t *output, const size_t count,
                                             cudaStream_t cuda_stream);

// uint32
template CUDA_LIB_EXPORT void Real<uint32_t>(const uint32_t *input, uint32_t *output, const size_t count,
                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Imag<uint32_t>(const uint32_t *input, uint32_t *output, const size_t count,
                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Conj<uint32_t>(const uint32_t *input, uint32_t *output, const size_t count,
                                             cudaStream_t cuda_stream);

// int64
template CUDA_LIB_EXPORT void Real<int64_t>(const int64_t *input, int64_t *output, const size_t count,
                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Imag<int64_t>(const int64_t *input, int64_t *output, const size_t count,
                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Conj<int64_t>(const int64_t *input, int64_t *output, const size_t count,
                                            cudaStream_t cuda_stream);

// uint64
template CUDA_LIB_EXPORT void Real<uint64_t>(const uint64_t *input, uint64_t *output, const size_t count,
                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Imag<uint64_t>(const uint64_t *input, uint64_t *output, const size_t count,
                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void Conj<uint64_t>(const uint64_t *input, uint64_t *output, const size_t count,
                                             cudaStream_t cuda_stream);
