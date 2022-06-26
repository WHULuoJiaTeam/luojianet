/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include <vector>
#include <iostream>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/broadcast_impl.cuh"
#include "include/cuda_fp16.h"

// Basic function
template <typename T>
struct GreaterFunc {
  __device__ __host__ __forceinline__ bool operator()(const T &lhs, const T &rhs) { return lhs > rhs; }
};

template <typename T>
struct LessFunc {
  __device__ __host__ __forceinline__ bool operator()(const T &lhs, const T &rhs) { return lhs < rhs; }
};

template <typename T>
struct EqualFunc {
  __device__ __host__ __forceinline__ bool operator()(const T &lhs, const T &rhs) { return lhs == rhs; }
};

template <>
struct EqualFunc<half> {
  __device__ __host__ __forceinline__ bool operator()(const half &lhs, const half &rhs) {
    return std::abs(__half2float(lhs) - __half2float(rhs)) < 1e-9;
  }
};

template <>
struct EqualFunc<float> {
  __device__ __host__ __forceinline__ bool operator()(const float &lhs, const float &rhs) {
    return std::abs(lhs - rhs) < 1e-9;
  }
};

template <typename T>
struct GreaterEqualFunc {
  __device__ __host__ __forceinline__ bool operator()(const T &lhs, const T &rhs) { return lhs >= rhs; }
};

template <>
struct GreaterEqualFunc<half> {
  __device__ __host__ __forceinline__ bool operator()(const half &lhs, const half &rhs) {
    return std::abs(__half2float(lhs) - __half2float(rhs)) < 1e-9
             ? true
             : (__half2float(lhs) > __half2float(rhs));
  }
};

template <>
struct GreaterEqualFunc<float> {
  __device__ __host__ __forceinline__ bool operator()(const float &lhs, const float &rhs) {
    return std::abs(lhs - rhs) < 1e-9 ? true : (lhs > rhs);
  }
};

template <typename T>
struct LessEqualFunc {
  __device__ __host__ __forceinline__ bool operator()(const T &lhs, const T &rhs) { return lhs <= rhs; }
};

template <>
struct LessEqualFunc<half> {
  __device__ __host__ __forceinline__ bool operator()(const half &lhs, const half &rhs) {
    return std::abs(__half2float(lhs) - __half2float(rhs)) < 1e-9
             ? true
             : (__half2float(lhs) < __half2float(rhs));
  }
};

template <>
struct LessEqualFunc<float> {
  __device__ __host__ __forceinline__ bool operator()(const float &lhs, const float &rhs) {
    return lhs <= rhs;
  }
};

template <typename T>
struct NotEqualFunc {
  __device__ __host__ __forceinline__ bool operator()(const T &lhs, const T &rhs) { return lhs != rhs; }
};

template <>
struct NotEqualFunc<half> {
  __device__ __host__ __forceinline__ bool operator()(const half &lhs, const half &rhs) {
    return std::abs(__half2float(lhs) - __half2float(rhs)) >= 1e-9;
  }
};

template <>
struct NotEqualFunc<float> {
  __device__ __host__ __forceinline__ bool operator()(const float &lhs, const float &rhs) {
    return std::abs(lhs - rhs) >= 1e-9;
  }
};

template <typename T>
struct LogicalAndFunc {
  __device__ __host__ __forceinline__ bool operator()(const T &lhs, const T &rhs) { return lhs && rhs; }
};

template <typename T>
struct LogicalOrFunc {
  __device__ __host__ __forceinline__ bool operator()(const T &lhs, const T &rhs) { return lhs || rhs; }
};

template <typename T>
struct MinimumFunc {
  __device__ __host__ __forceinline__ T operator()(const T &lhs, const T &rhs) { return lhs < rhs ? lhs : rhs; }
};

template <typename T>
struct MaximumFunc {
  __device__ __host__ __forceinline__ T operator()(const T &lhs, const T &rhs) { return lhs > rhs ? lhs : rhs; }
};

template <typename T>
struct PowerFunc {
  __device__ __host__ __forceinline__ T operator()(const T &lhs, const T &rhs) { return pow(lhs, rhs); }
};

template <>
struct PowerFunc<half> {
  __device__ __host__ __forceinline__ half operator()(const half &lhs, const half &rhs) {
    return __float2half(pow(__half2float(lhs), __half2float(rhs)));
  }
};

template <>
struct PowerFunc<half2> {
  __device__ __host__ __forceinline__ half2 operator()(const half2 &lhs, const half2 &rhs) {
    float2 base = __half22float2(lhs);
    float2 index = __half22float2(rhs);
    base.x = pow(base.x, index.x);
    base.y = pow(base.y, index.y);
    return __float22half2_rn(base);
  }
};

template <typename T>
struct RealDivFunc {
  __device__ __host__ __forceinline__ T operator()(const T &lhs, const T &rhs) { return (lhs / rhs); }
  __device__ __host__ __forceinline__ Complex<T> operator()(const Complex<T> &lhs, const T &rhs) { return (lhs / rhs); }
  __device__ __host__ __forceinline__ Complex<T> operator()(const T &lhs, const Complex<T> &rhs) { return (lhs / rhs); }
  __device__ __host__ __forceinline__ Complex<T> operator()(const Complex<T> &lhs, const Complex<T> &rhs) {
    return (lhs / rhs);
  }
};

template <typename T>
struct ComplexFunc {
  __device__ __host__ __forceinline__ Complex<T> operator()(const T &lhs, const T &rhs) { return Complex<T>(lhs, rhs); }
};

template <typename T>
struct DivFunc {
  __device__ __host__ __forceinline__ T operator()(const T &lhs, const T &rhs) { return (lhs / rhs); }
  __device__ __host__ __forceinline__ Complex<T> operator()(const Complex<T> &lhs, const T &rhs) { return (lhs / rhs); }
  __device__ __host__ __forceinline__ Complex<T> operator()(const T &lhs, const Complex<T> &rhs) { return (lhs / rhs); }
  __device__ __host__ __forceinline__ Complex<T> operator()(const Complex<T> &lhs, const Complex<T> &rhs) {
    return (lhs / rhs);
  }
};

template <typename T>
struct MulFunc {
  __device__ __host__ __forceinline__ T operator()(const T &lhs, const T &rhs) { return (lhs * rhs); }
  __device__ __host__ __forceinline__ Complex<T> operator()(const Complex<T> &lhs, const T &rhs) { return (lhs * rhs); }
  __device__ __host__ __forceinline__ Complex<T> operator()(const T &lhs, const Complex<T> &rhs) { return (lhs * rhs); }
  __device__ __host__ __forceinline__ Complex<T> operator()(const Complex<T> &lhs, const Complex<T> &rhs) {
    return (lhs * rhs);
  }
};

template <typename T>
struct SubFunc {
  __device__ __host__ __forceinline__ T operator()(const T &lhs, const T &rhs) { return (lhs - rhs); }
  __device__ __host__ __forceinline__ Complex<T> operator()(const Complex<T> &lhs, const T &rhs) { return (lhs - rhs); }
  __device__ __host__ __forceinline__ Complex<T> operator()(const T &lhs, const Complex<T> &rhs) { return (lhs - rhs); }
  __device__ __host__ __forceinline__ Complex<T> operator()(const Complex<T> &lhs, const Complex<T> &rhs) {
    return (lhs - rhs);
  }
};

template <typename T>
struct AddFunc {
  __device__ __host__ __forceinline__ T operator()(const T &lhs, const T &rhs) { return (lhs + rhs); }
  __device__ __host__ __forceinline__ Complex<T> operator()(const Complex<T> &lhs, const T &rhs) { return (lhs + rhs); }
  __device__ __host__ __forceinline__ Complex<T> operator()(const T &lhs, const Complex<T> &rhs) { return (lhs + rhs); }
  __device__ __host__ __forceinline__ Complex<T> operator()(const Complex<T> &lhs, const Complex<T> &rhs) {
    return (lhs + rhs);
  }
};
// DivNoNan check if rhs is less than epsilon
template <typename T>
struct DivNoNanFunc {
  // default T is float
  __device__ __host__ __forceinline__ T operator()(const T &lhs, const T &rhs) {
    return rhs < kFloatEplison && rhs > -kFloatEplison ? 0.0 : (lhs / rhs);
  }
};

template <>
struct DivNoNanFunc<int> {
  __device__ __host__ __forceinline__ int operator()(const int &lhs, const int &rhs) {
    return rhs == 0 ? 0 : (lhs / rhs);
  }
};

template <>
struct DivNoNanFunc<half> {
  __device__ __host__ __forceinline__ half operator()(const half &lhs, const half &rhs) {
    if (__half2float(rhs) < (0.00007) && __half2float(rhs) > -0.00007) {
      return static_cast<half>(0.0);
    }
    return __float2half_rn(__half2float(lhs) / __half2float(rhs));
  }
};

template <>
struct DivNoNanFunc<half2> {
  __device__ __host__ __forceinline__ half2 operator()(const half2 &lhs, const half2 &rhs) {
    float2 l = __half22float2(lhs);
    float2 r = __half22float2(rhs);
    if ((r.x < kFloatEplison && r.x > -kFloatEplison) || (r.y < kFloatEplison && r.y > -kFloatEplison)) {
      l.x = 0.0;
      l.y = 0.0;
    } else {
      l.x = l.x / r.x;
      l.y = l.y / r.y;
    }
    return __float22half2_rn(l);
  }
};

// convert to float to fix accuracy issue
template <typename T>
struct FloorDivFunc {
  __device__ __host__ __forceinline__ T operator()(const T &lhs, const T &rhs) {
    return floorf(static_cast<float>(lhs) / static_cast<float>(rhs));
  }
};
template <>
struct FloorDivFunc<int64_t> {
  __device__ __host__ __forceinline__ int64_t operator()(const int64_t &lhs, const int64_t &rhs) {
    return floor(static_cast<double>(lhs) / static_cast<double>(rhs));
  }
};
template <>
struct FloorDivFunc<int32_t> {
  __device__ __host__ __forceinline__ int32_t operator()(const int32_t &lhs, const int32_t &rhs) {
    return floor(static_cast<double>(lhs) / static_cast<double>(rhs));
  }
};
template <>
struct FloorDivFunc<uint64_t> {
  __device__ __host__ __forceinline__ int64_t operator()(const uint64_t &lhs, const uint64_t &rhs) {
    return floor(static_cast<double>(lhs) / static_cast<double>(rhs));
  }
};
template <>
struct FloorDivFunc<uint32_t> {
  __device__ __host__ __forceinline__ uint32_t operator()(const uint32_t &lhs, const uint32_t &rhs) {
    return floor(static_cast<double>(lhs) / static_cast<double>(rhs));
  }
};

template <>
struct FloorDivFunc<half> {
  __device__ __host__ __forceinline__ half operator()(const half &lhs, const half &rhs) {
    return floorf(__half2float(lhs) / __half2float(rhs));
  }
};

template <>
struct FloorDivFunc<half2> {
  __device__ __host__ __forceinline__ half2 operator()(const half2 &lhs, const half2 &rhs) {
    float2 l = __half22float2(lhs);
    float2 r = __half22float2(rhs);
    l.x = floorf(l.x / r.x);
    l.y = floorf(l.y / r.y);
    return __float22half2_rn(l);
  }
};

template <typename T>
struct ModFunc {
  __device__ __host__ __forceinline__ T operator()(const T &lhs, const T &rhs) {
    T data_div = lhs / rhs;
    T data_div_min = data_div < 0.0 ? data_div : 0.0;
    T data_div_max = data_div > 0.0 ? data_div : 0.0;
    T data_div_max_floor = floorf(data_div_max);
    T data_div_min_ceil = ceilf(data_div_min);
    T data_div_res = data_div_max_floor + data_div_min_ceil;
    return lhs - data_div_res * rhs;
  }
};

template <>
struct ModFunc<half> {
  __device__ __host__ __forceinline__ half operator()(const half &lhs, const half &rhs) {
    float l = __half2float(lhs);
    float r = __half2float(rhs);
    float data_div = l / r;
    float data_div_min = data_div < 0.0 ? data_div : 0.0;
    float data_div_max = data_div > 0.0 ? data_div : 0.0;
    float data_div_max_floor = floorf(data_div_max);
    float data_div_min_ceil = ceilf(data_div_min);
    float data_div_res = data_div_max_floor + data_div_min_ceil;
    return __float2half_rn(l - data_div_res * r);
  }
};

template <>
struct ModFunc<half2> {
  __device__ __host__ __forceinline__ half2 operator()(const half2 &lhs, const half2 &rhs) {
    float2 l = __half22float2(lhs);
    float2 r = __half22float2(rhs);
    float2 data_div;
    data_div.x = l.x / r.x;
    data_div.y = l.y / r.y;
    data_div.x = data_div.x < 0.0 ? ceilf(data_div.x) : floorf(data_div.x);
    data_div.y = data_div.y < 0.0 ? ceilf(data_div.y) : floorf(data_div.y);
    data_div.x = l.x - data_div.x * r.x;
    data_div.y = l.y - data_div.y * r.y;
    return __float22half2_rn(data_div);
  }
};

template <typename T>
struct FloorModFunc {
  __device__ __host__ __forceinline__ T operator()(const T &lhs, const T &rhs) {
    T res = lhs - floorf(lhs / rhs) * rhs;
    res = (std::abs(res) > 1e-9) && ((res < 0.0) != (rhs < 0.0)) ? res + rhs : res;
    return res;
  }
};

template <>
struct FloorModFunc<half> {
  __device__ __host__ __forceinline__ half operator()(const half &lhs, const half &rhs) {
    float l = __half2float(lhs);
    float r = __half2float(rhs);
    float res = l - floorf(l / r) * r;
    res = (std::abs(res) > 1e-9) && ((res < 0.0) != (r < 0.0)) ? res + r : res;
    return __float2half_rn(res);
  }
};

template <>
struct FloorModFunc<half2> {
  __device__ __host__ __forceinline__ half2 operator()(const half2 &lhs, const half2 &rhs) {
    float2 l = __half22float2(lhs);
    float2 r = __half22float2(rhs);
    float2 res;
    res.x = l.x - floorf(l.x / r.x) * r.x;
    res.y = l.y - floorf(l.y / r.y) * r.y;
    res.x = (std::abs(res.x) > 1e-9) && ((res.x < 0.0) != (r.x < 0.0)) ? res.x + r.x : res.x;
    res.y = (std::abs(res.y) > 1e-9) && ((res.y < 0.0) != (r.y < 0.0)) ? res.y + r.y : res.y;
    return __float22half2_rn(res);
  }
};

// the FloorModFunc specializations for uint32_t and uint64_t are there
// because of a 'more than one instance of overloaded function "std::abs"'
// error. I realize the specializations are exactly the same, but I found
// no good alternative.
template <>
struct FloorModFunc<int32_t> {
  __device__ __host__ __forceinline__ int32_t operator()(const int32_t &lhs, const int32_t &rhs) {
    int32_t res = lhs - floor(static_cast<double>(lhs) / static_cast<double>(rhs)) * rhs;
    res = (res > 1e-9) && ((res < 0.0) != (rhs < 0.0)) ? res + rhs : res;
    return res;
  }
};

template <>
struct FloorModFunc<int64_t> {
  __device__ __host__ __forceinline__ int64_t operator()(const int64_t &lhs, const int64_t &rhs) {
    int64_t res = lhs - floor(static_cast<double>(lhs) / static_cast<double>(rhs)) * rhs;
    res = (res > 1e-9) && ((res < 0.0) != (rhs < 0.0)) ? res + rhs : res;
    return res;
  }
};

template <>
struct FloorModFunc<uint32_t> {
  __device__ __host__ __forceinline__ int32_t operator()(const uint32_t &lhs, const uint32_t &rhs) {
    int32_t res = lhs - floor(static_cast<double>(lhs) / static_cast<double>(rhs)) * rhs;
    res = (res > 1e-9) && ((res < 0.0) != (rhs < 0.0)) ? res + rhs : res;
    return res;
  }
};

template <>
struct FloorModFunc<uint64_t> {
  __device__ __host__ __forceinline__ int64_t operator()(const uint64_t &lhs, const uint64_t &rhs) {
    int64_t res = lhs - floor(static_cast<double>(lhs) / static_cast<double>(rhs)) * rhs;
    res = (res > 1e-9) && ((res < 0.0) != (rhs < 0.0)) ? res + rhs : res;
    return res;
  }
};

template <typename T>
struct AbsGradFunc {
  __device__ __forceinline__ T operator()(const T &lhs, const T &rhs) {
    T zero = 0.0;
    return lhs < zero ? -rhs : lhs > zero ? rhs : zero;
  }
};

template <>
struct AbsGradFunc<half2> {
  __device__ __forceinline__ half2 operator()(const half2 &lhs, const half2 &rhs) {
    half2 zero(0.0, 0.0);
    return lhs < zero ? -rhs : lhs > zero ? rhs : zero;
  }
};

template <typename T>
struct SquaredDifferenceFunc {
  __device__ __forceinline__ T operator()(const T &lhs, const T &rhs) {
    T diff = lhs - rhs;
    return diff * diff;
  }
};

template <typename T>
struct TruncateDivFunc {
  __device__ __forceinline__ T operator()(const T &lhs, const T &rhs) {
    T res = static_cast<T>(static_cast<double>(lhs) / static_cast<double>(rhs));
    return res;
  }
};

template <>
struct TruncateDivFunc<half> {
  __device__ __forceinline__ half operator()(const half &lhs, const half &rhs) {
    float res = __half2float(lhs) / __half2float(rhs);
    return __float2half_rn(res);
  }
};

template <>
struct TruncateDivFunc<half2> {
  __device__ __host__ __forceinline__ half2 operator()(const half2 &lhs, const half2 &rhs) {
    float2 l = __half22float2(lhs);
    float2 r = __half22float2(rhs);
    float2 res;
    res.x = l.x / r.x;
    res.y = l.y / r.y;
    return __float22half2_rn(res);
  }
};

template <typename T>
struct TruncateModFunc {
  __device__ __forceinline__ T operator()(const T &lhs, const T &rhs) {
    T res = static_cast<T>(lhs - static_cast<int>(lhs / rhs) * rhs);
    return res;
  }
};

template <>
struct TruncateModFunc<half> {
  __device__ __forceinline__ half operator()(const half &lhs, const half &rhs) {
    float l = __half2float(lhs);
    float r = __half2float(rhs);
    float res = l - static_cast<int>(l / r) * r;
    return __float2half_rn(res);
  }
};

template <>
struct TruncateModFunc<half2> {
  __device__ __host__ __forceinline__ half2 operator()(const half2 &lhs, const half2 &rhs) {
    float2 l = __half22float2(lhs);
    float2 r = __half22float2(rhs);
    float2 res;
    res.x = l.x - static_cast<int>(l.x / r.x) * r.x;
    res.y = l.y - static_cast<int>(l.y / r.y) * r.y;
    return __float22half2_rn(res);
  }
};

template <typename T>
struct Atan2Func {
  __device__ __host__ __forceinline__ T operator()(const T &lhs, const T &rhs) { return atan2f(lhs, rhs); }
};

template <>
struct Atan2Func<double> {
  __device__ __host__ __forceinline__ double operator()(const double &lhs, const double &rhs) {
    return atan2(lhs, rhs);
  }
};

template <>
struct Atan2Func<half> {
  __device__ __host__ __forceinline__ half operator()(const half &lhs, const half &rhs) {
    float l = __half2float(lhs);
    float r = __half2float(rhs);
    float res = atan2f(l, r);
    return __float2half_rn(res);
  }
};

template <>
struct Atan2Func<half2> {
  __device__ __host__ __forceinline__ half2 operator()(const half2 &lhs, const half2 &rhs) {
    float2 l = __half22float2(lhs);
    float2 r = __half22float2(rhs);
    float2 res;
    res.x = atan2f(l.x, r.x);
    res.y = atan2f(l.y, r.y);
    return __float22half2_rn(res);
  }
};

// Element-wise Comparison
template <typename T, typename Func>
__global__ void ElewiseCmpKernel(const int nums, const T *x0, const T *x1, bool *y) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < nums; pos += blockDim.x * gridDim.x) {
    y[pos] = Func()(x0[pos], x1[pos]);
  }
}

template <typename T>
void ElewiseCmp(const int &nums, enum BroadcastOpType op, const T *x0, const T *x1, bool *y, cudaStream_t stream) {
  switch (op) {
    case BROADCAST_TYPE_GREATER:
      return ElewiseCmpKernel<T, GreaterFunc<T>><<<(nums + 255) / 256, 256, 0, stream>>>(nums, x0, x1, y);
    case BROADCAST_TYPE_LESS:
      return ElewiseCmpKernel<T, LessFunc<T>><<<(nums + 255) / 256, 256, 0, stream>>>(nums, x0, x1, y);
    case BROADCAST_TYPE_EQUAL:
      return ElewiseCmpKernel<T, EqualFunc<T>><<<(nums + 255) / 256, 256, 0, stream>>>(nums, x0, x1, y);
    case BROADCAST_TYPE_GREATER_EQUAL:
      return ElewiseCmpKernel<T, GreaterEqualFunc<T>><<<(nums + 255) / 256, 256, 0, stream>>>(nums, x0, x1, y);
    case BROADCAST_TYPE_LESS_EQUAL:
      return ElewiseCmpKernel<T, LessEqualFunc<T>><<<(nums + 255) / 256, 256, 0, stream>>>(nums, x0, x1, y);
    case BROADCAST_TYPE_NOT_EQUAL:
      return ElewiseCmpKernel<T, NotEqualFunc<T>><<<(nums + 255) / 256, 256, 0, stream>>>(nums, x0, x1, y);
    case BROADCAST_TYPE_LOGICAL_AND:
      return ElewiseCmpKernel<T, LogicalAndFunc<T>><<<(nums + 255) / 256, 256, 0, stream>>>(nums, x0, x1, y);
    case BROADCAST_TYPE_LOGICAL_OR:
      return ElewiseCmpKernel<T, LogicalOrFunc<T>><<<(nums + 255) / 256, 256, 0, stream>>>(nums, x0, x1, y);
    default:
      break;
  }
}

template CUDA_LIB_EXPORT void ElewiseCmp(const int &nums, enum BroadcastOpType op,
                                         const double *x0, const double *x1, bool *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void ElewiseCmp(const int &nums, enum BroadcastOpType op,
                                         const float *x0, const float *x1, bool *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void ElewiseCmp(const int &nums, enum BroadcastOpType op,
                                         const half *x0, const half *x1, bool *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void ElewiseCmp(const int &nums, enum BroadcastOpType op,
                                         const int *x0, const int *x1, bool *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void ElewiseCmp(const int &nums, enum BroadcastOpType op,
                                         const int8_t *x0, const int8_t *x1, bool *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void ElewiseCmp(const int &nums, enum BroadcastOpType op,
                                         const uint8_t *x0, const uint8_t *x1, bool *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void ElewiseCmp(const int &nums, enum BroadcastOpType op,
                                         const int64_t *x0, const int64_t *x1, bool *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void ElewiseCmp(const int &nums, enum BroadcastOpType op,
                                         const int16_t *x0, const int16_t *x1, bool *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void ElewiseCmp(const int &nums, enum BroadcastOpType op,
                                         const uint16_t *x0, const uint16_t *x1, bool *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void ElewiseCmp(const int &nums, enum BroadcastOpType op,
                                         const uint32_t *x0, const uint32_t *x1, bool *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void ElewiseCmp(const int &nums, enum BroadcastOpType op,
                                         const uint64_t *x0, const uint64_t *x1, bool *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void ElewiseCmp(const int &nums, enum BroadcastOpType op,
                                         const bool *x0, const bool *x1, bool *y, cudaStream_t stream);
// Element-wise ArithMetic
template <typename T, typename Func>
__global__ void ElewiseArithKernel(const int nums, const T *x0, const T *x1, T *y) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < nums; pos += blockDim.x * gridDim.x) {
    y[pos] = Func()(x0[pos], x1[pos]);
  }
}

template <typename T1, typename T2, typename T3, typename Func>
__global__ void ElewiseArithComplexKernel(const int nums, const T1 *x0, const T2 *x1, Complex<T3> *y) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < nums; pos += blockDim.x * gridDim.x) {
    y[pos] = Func()(x0[pos], x1[pos]);
  }
}

template <typename T>
void ElewiseArithKernel(const int &nums, enum BroadcastOpType op, const T *x0, const T *x1, T *y, cudaStream_t stream) {
  switch (op) {
    case BROADCAST_TYPE_MINIMUM:
      return ElewiseArithKernel<T, MinimumFunc<T>><<<(nums + 255) / 256, 256, 0, stream>>>(nums, x0, x1, y);
    case BROADCAST_TYPE_MAXIMUM:
      return ElewiseArithKernel<T, MaximumFunc<T>><<<(nums + 255) / 256, 256, 0, stream>>>(nums, x0, x1, y);
    case BROADCAST_TYPE_POWER:
      return ElewiseArithKernel<T, PowerFunc<T>><<<(nums + 255) / 256, 256, 0, stream>>>(nums, x0, x1, y);
    case BROADCAST_TYPE_REALDIV:
      return ElewiseArithKernel<T, RealDivFunc<T>><<<(nums + 255) / 256, 256, 0, stream>>>(nums, x0, x1, y);
    case BROADCAST_TYPE_MUL:
      return ElewiseArithKernel<T, MulFunc<T>><<<(nums + 255) / 256, 256, 0, stream>>>(nums, x0, x1, y);
    case BROADCAST_TYPE_SUB:
      return ElewiseArithKernel<T, SubFunc<T>><<<(nums + 255) / 256, 256, 0, stream>>>(nums, x0, x1, y);
    case BROADCAST_TYPE_ADD:
      return ElewiseArithKernel<T, AddFunc<T>><<<(nums + 255) / 256, 256, 0, stream>>>(nums, x0, x1, y);
    case BROADCAST_TYPE_FLOORDIV:
      return ElewiseArithKernel<T, FloorDivFunc<T>><<<(nums + 255) / 256, 256, 0, stream>>>(nums, x0, x1, y);
    case BROADCAST_TYPE_ABSGRAD:
      return ElewiseArithKernel<T, AbsGradFunc<T>><<<(nums + 255) / 256, 256, 0, stream>>>(nums, x0, x1, y);
    case BROADCAST_TYPE_DIV:
      return ElewiseArithKernel<T, DivFunc<T>><<<(nums + 255) / 256, 256, 0, stream>>>(nums, x0, x1, y);
    case BROADCAST_TYPE_DIVNONAN:
      return ElewiseArithKernel<T, DivNoNanFunc<T>><<<(nums + 255) / 256, 256, 0, stream>>>(nums, x0, x1, y);
    case BROADCAST_TYPE_SQUARED_DIFFERENCE:
      return ElewiseArithKernel<T, SquaredDifferenceFunc<T>><<<(nums + 255) / 256, 256, 0, stream>>>(nums, x0, x1, y);
    case BROADCAST_TYPE_TRUNCATEDIV:
      return ElewiseArithKernel<T, TruncateDivFunc<T>><<<(nums + 255) / 256, 256, 0, stream>>>(nums, x0, x1, y);
    case BROADCAST_TYPE_TRUNCATEMOD:
      return ElewiseArithKernel<T, TruncateModFunc<T>><<<(nums + 255) / 256, 256, 0, stream>>>(nums, x0, x1, y);
    case BROADCAST_TYPE_MOD:
      return ElewiseArithKernel<T, ModFunc<T>><<<(nums + 255) / 256, 256, 0, stream>>>(nums, x0, x1, y);
    case BROADCAST_TYPE_FLOORMOD:
      return ElewiseArithKernel<T, FloorModFunc<T>><<<(nums + 255) / 256, 256, 0, stream>>>(nums, x0, x1, y);
    case BROADCAST_TYPE_ATAN2:
      return ElewiseArithKernel<T, Atan2Func<T>><<<(nums + 255) / 256, 256, 0, stream>>>(nums, x0, x1, y);
    default:
      break;
  }
}

template <typename T1, typename T2, typename T3>
void ElewiseArithComplexKernel(const int &nums, enum BroadcastOpType op, const T1 *x0, const T2 *x1, Complex<T3> *y,
                               cudaStream_t stream) {
  switch (op) {
    case BROADCAST_TYPE_ADD:
      return ElewiseArithComplexKernel<T1, T2, T3, AddFunc<T3>>
        <<<(nums + 255) / 256, 256, 0, stream>>>(nums, x0, x1, y);
    case BROADCAST_TYPE_SUB:
      return ElewiseArithComplexKernel<T1, T2, T3, SubFunc<T3>>
        <<<(nums + 255) / 256, 256, 0, stream>>>(nums, x0, x1, y);
    case BROADCAST_TYPE_MUL:
      return ElewiseArithComplexKernel<T1, T2, T3, MulFunc<T3>>
        <<<(nums + 255) / 256, 256, 0, stream>>>(nums, x0, x1, y);
    case BROADCAST_TYPE_DIV:
      return ElewiseArithComplexKernel<T1, T2, T3, DivFunc<T3>>
        <<<(nums + 255) / 256, 256, 0, stream>>>(nums, x0, x1, y);
    case BROADCAST_TYPE_REALDIV:
      return ElewiseArithComplexKernel<T1, T2, T3, RealDivFunc<T3>>
        <<<(nums + 255) / 256, 256, 0, stream>>>(nums, x0, x1, y);
    default:
      break;
  }
}

template <typename T>
void ElewiseArithComplexKernel(const int &nums, enum BroadcastOpType op, const T *x0, const T *x1, Complex<T> *y,
                               cudaStream_t stream) {
  if (op == BROADCAST_TYPE_COMPLEX) {
    return ElewiseArithComplexKernel<T, T, T, ComplexFunc<T>>
      <<<(nums + 255) / 256, 256, 0, stream>>>(nums, x0, x1, y);
  }
}

template <typename T>
void ElewiseArith(const int &nums, enum BroadcastOpType op, const T *x0, const T *x1, T *y, cudaStream_t stream) {
  return ElewiseArithKernel(nums, op, x0, x1, y, stream);
}

template <>
void ElewiseArith(const int &nums, enum BroadcastOpType op, const half *x0, const half *x1, half *y,
                  cudaStream_t stream) {
  // `>` return true iff both half result are true. fallback to half
  if (nums % 2 == 0 && op != BROADCAST_TYPE_MINIMUM && op != BROADCAST_TYPE_MAXIMUM && op != BROADCAST_TYPE_ABSGRAD) {
    ElewiseArithKernel<half2>(nums / 2, op, reinterpret_cast<const half2 *>(x0), reinterpret_cast<const half2 *>(x1),
                              reinterpret_cast<half2 *>(y), stream);
  } else {
    return ElewiseArithKernel(nums, op, x0, x1, y, stream);
  }
}

template <typename T1, typename T2, typename T3>
void ElewiseComplexArith(const int &nums, enum BroadcastOpType op, const T1 *x0, const T2 *x1, Complex<T3> *y,
                         cudaStream_t stream) {
  return ElewiseArithComplexKernel(nums, op, x0, x1, y, stream);
}

template CUDA_LIB_EXPORT void ElewiseArith(const int &nums, enum BroadcastOpType op,
                                           const double *x0, const double *x1, double *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void ElewiseArith(const int &nums, enum BroadcastOpType op,
                                           const float *x0, const float *x1, float *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void ElewiseArith(const int &nums, enum BroadcastOpType op,
                                           const half *x0, const half *x1, half *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void ElewiseArith(const int &nums, enum BroadcastOpType op,
                                           const int *x0, const int *x1, int *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void ElewiseArith(const int &nums, enum BroadcastOpType op,
                                           const int8_t *x0, const int8_t *x1, int8_t *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void ElewiseArith(const int &nums, enum BroadcastOpType op,
                                           const uint8_t *x0, const uint8_t *x1, uint8_t *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void ElewiseArith(const int &nums, enum BroadcastOpType op,
                                           const int64_t *x0, const int64_t *x1, int64_t *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void ElewiseArith(const int &nums, enum BroadcastOpType op,
                                           const int16_t *x0, const int16_t *x1, int16_t *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void ElewiseArith(const int &nums, enum BroadcastOpType op,
                                           const uint16_t *x0, const uint16_t *x1, uint16_t *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void ElewiseArith(const int &nums, enum BroadcastOpType op,
                                           const uint32_t *x0, const uint32_t *x1, uint32_t *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void ElewiseArith(const int &nums, enum BroadcastOpType op,
                                           const uint64_t *x0, const uint64_t *x1, uint64_t *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void ElewiseArith(const int &nums, enum BroadcastOpType op,
                                           const bool *x0, const bool *x1, bool *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void ElewiseComplexArith(const int &nums, enum BroadcastOpType op, const Complex<float> *x0,
                                                  const Complex<float> *x1, Complex<float> *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void ElewiseComplexArith(const int &nums, enum BroadcastOpType op, const Complex<float> *x0,
                                                  const float *x1, Complex<float> *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void ElewiseComplexArith(const int &nums, enum BroadcastOpType op, const float *x0,
                                                  const Complex<float> *x1, Complex<float> *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void ElewiseComplexArith(const int &nums, enum BroadcastOpType op, const Complex<double> *x0,
                                                  const Complex<double> *x1, Complex<double> *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void ElewiseComplexArith(const int &nums, enum BroadcastOpType op, const Complex<double> *x0,
                                                  const double *x1, Complex<double> *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void ElewiseComplexArith(const int &nums, enum BroadcastOpType op, const double *x0,
                                                  const Complex<double> *x1, Complex<double> *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void ElewiseComplexArith(const int &nums, enum BroadcastOpType op, const float *x0,
                                                  const float *x1, Complex<float> *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void ElewiseComplexArith(const int &nums, enum BroadcastOpType op, const double *x0,
                                                  const double *x1, Complex<double> *y, cudaStream_t stream);

// Broadcast comparison
__device__ __forceinline__ size_t Index(const size_t &index, const size_t &dim) { return dim == 1 ? 0 : index; }

template <typename T, typename Func>
__global__ void BroadcastCmpKernel(const size_t l0, const size_t l1, const size_t l2, const size_t l3, const size_t l4,
                                   const size_t l5, const size_t l6, const size_t r0, const size_t r1, const size_t r2,
                                   const size_t r3, const size_t r4, const size_t r5, const size_t r6, const size_t d0,
                                   const size_t d1, const size_t d2, const size_t d3, const size_t d4, const size_t d5,
                                   const size_t d6, const T *x0, const T *x1, bool *y) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < d0 * d1 * d2 * d3 * d4 * d5 * d6;
       pos += blockDim.x * gridDim.x) {
    size_t i = pos / (d1 * d2 * d3 * d4 * d5 * d6) % d0;
    size_t j = pos / (d2 * d3 * d4 * d5 * d6) % d1;
    size_t k = pos / (d3 * d4 * d5 * d6) % d2;
    size_t l = pos / (d4 * d5 * d6) % d3;
    size_t m = pos / (d5 * d6) % d4;
    size_t n = pos / d6 % d5;
    size_t o = pos % d6;

    size_t l_index = Index(i, l0) * l1 * l2 * l3 * l4 * l5 * l6;
    l_index += Index(j, l1) * l2 * l3 * l4 * l5 * l6;
    l_index += Index(k, l2) * l3 * l4 * l5 * l6;
    l_index += Index(l, l3) * l4 * l5 * l6;
    l_index += Index(m, l4) * l5 * l6;
    l_index += Index(n, l5) * l6;
    l_index += Index(o, l6);
    size_t r_index = Index(i, r0) * r1 * r2 * r3 * r4 * r5 * r6;
    r_index += Index(j, r1) * r2 * r3 * r4 * r5 * r6;
    r_index += Index(k, r2) * r3 * r4 * r5 * r6;
    r_index += Index(l, r3) * r4 * r5 * r6;
    r_index += Index(m, r4) * r5 * r6;
    r_index += Index(n, r5) * r6;
    r_index += Index(o, r6);
    y[pos] = Func()(x0[l_index], x1[r_index]);
  }
}

template <typename T>
void BroadcastCmp(const std::vector<size_t> &x0_dims, const std::vector<size_t> &x1_dims,
                  const std::vector<size_t> &y_dims, enum BroadcastOpType op, const T *x0, const T *x1, bool *y,
                  cudaStream_t stream) {
  size_t size = 1;
  for (auto d : y_dims) {
    size *= d;
  }

  switch (op) {
    case BROADCAST_TYPE_GREATER:
      return BroadcastCmpKernel<T, GreaterFunc<T>><<<(size + 255) / 256, 256, 0, stream>>>(
        x0_dims[0], x0_dims[1], x0_dims[2], x0_dims[3], x0_dims[4], x0_dims[5], x0_dims[6], x1_dims[0], x1_dims[1],
        x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5], x1_dims[6], y_dims[0], y_dims[1], y_dims[2], y_dims[3],
        y_dims[4], y_dims[5], y_dims[6], x0, x1, y);
    case BROADCAST_TYPE_LESS:
      return BroadcastCmpKernel<T, LessFunc<T>><<<(size + 255) / 256, 256, 0, stream>>>(
        x0_dims[0], x0_dims[1], x0_dims[2], x0_dims[3], x0_dims[4], x0_dims[5], x0_dims[6], x1_dims[0], x1_dims[1],
        x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5], x1_dims[6], y_dims[0], y_dims[1], y_dims[2], y_dims[3],
        y_dims[4], y_dims[5], y_dims[6], x0, x1, y);
    case BROADCAST_TYPE_EQUAL:
      return BroadcastCmpKernel<T, EqualFunc<T>><<<(size + 255) / 256, 256, 0, stream>>>(
        x0_dims[0], x0_dims[1], x0_dims[2], x0_dims[3], x0_dims[4], x0_dims[5], x0_dims[6], x1_dims[0], x1_dims[1],
        x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5], x1_dims[6], y_dims[0], y_dims[1], y_dims[2], y_dims[3],
        y_dims[4], y_dims[5], y_dims[6], x0, x1, y);
    case BROADCAST_TYPE_GREATER_EQUAL:
      return BroadcastCmpKernel<T, GreaterEqualFunc<T>><<<(size + 255) / 256, 256, 0, stream>>>(
        x0_dims[0], x0_dims[1], x0_dims[2], x0_dims[3], x0_dims[4], x0_dims[5], x0_dims[6], x1_dims[0], x1_dims[1],
        x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5], x1_dims[6], y_dims[0], y_dims[1], y_dims[2], y_dims[3],
        y_dims[4], y_dims[5], y_dims[6], x0, x1, y);
    case BROADCAST_TYPE_LESS_EQUAL:
      return BroadcastCmpKernel<T, LessEqualFunc<T>><<<(size + 255) / 256, 256, 0, stream>>>(
        x0_dims[0], x0_dims[1], x0_dims[2], x0_dims[3], x0_dims[4], x0_dims[5], x0_dims[6], x1_dims[0], x1_dims[1],
        x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5], x1_dims[6], y_dims[0], y_dims[1], y_dims[2], y_dims[3],
        y_dims[4], y_dims[5], y_dims[6], x0, x1, y);
    case BROADCAST_TYPE_NOT_EQUAL:
      return BroadcastCmpKernel<T, NotEqualFunc<T>><<<(size + 255) / 256, 256, 0, stream>>>(
        x0_dims[0], x0_dims[1], x0_dims[2], x0_dims[3], x0_dims[4], x0_dims[5], x0_dims[6], x1_dims[0], x1_dims[1],
        x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5], x1_dims[6], y_dims[0], y_dims[1], y_dims[2], y_dims[3],
        y_dims[4], y_dims[5], y_dims[6], x0, x1, y);
    case BROADCAST_TYPE_LOGICAL_AND:
      return BroadcastCmpKernel<T, LogicalAndFunc<T>><<<(size + 255) / 256, 256, 0, stream>>>(
        x0_dims[0], x0_dims[1], x0_dims[2], x0_dims[3], x0_dims[4], x0_dims[5], x0_dims[6], x1_dims[0], x1_dims[1],
        x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5], x1_dims[6], y_dims[0], y_dims[1], y_dims[2], y_dims[3],
        y_dims[4], y_dims[5], y_dims[6], x0, x1, y);
    case BROADCAST_TYPE_LOGICAL_OR:
      return BroadcastCmpKernel<T, LogicalOrFunc<T>><<<(size + 255) / 256, 256, 0, stream>>>(
        x0_dims[0], x0_dims[1], x0_dims[2], x0_dims[3], x0_dims[4], x0_dims[5], x0_dims[6], x1_dims[0], x1_dims[1],
        x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5], x1_dims[6], y_dims[0], y_dims[1], y_dims[2], y_dims[3],
        y_dims[4], y_dims[5], y_dims[6], x0, x1, y);
    default:
      break;
  }
}

template CUDA_LIB_EXPORT void BroadcastCmp(const std::vector<size_t> &x0_dims, const std::vector<size_t> &x1_dims,
                                           const std::vector<size_t> &y_dims, enum BroadcastOpType op, const double *x0,
                                           const double *x1, bool *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void BroadcastCmp(const std::vector<size_t> &x0_dims, const std::vector<size_t> &x1_dims,
                                           const std::vector<size_t> &y_dims, enum BroadcastOpType op, const float *x0,
                                           const float *x1, bool *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void BroadcastCmp(const std::vector<size_t> &x0_dims, const std::vector<size_t> &x1_dims,
                                           const std::vector<size_t> &y_dims, enum BroadcastOpType op, const half *x0,
                                           const half *x1, bool *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void BroadcastCmp(const std::vector<size_t> &x0_dims, const std::vector<size_t> &x1_dims,
                                           const std::vector<size_t> &y_dims, enum BroadcastOpType op, const int *x0,
                                           const int *x1, bool *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void BroadcastCmp(const std::vector<size_t> &x0_dims, const std::vector<size_t> &x1_dims,
                                           const std::vector<size_t> &y_dims, enum BroadcastOpType op, const int8_t *x0,
                                           const int8_t *x1, bool *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void BroadcastCmp(const std::vector<size_t> &x0_dims, const std::vector<size_t> &x1_dims,
                                           const std::vector<size_t> &y_dims, enum BroadcastOpType op,
                                           const uint8_t *x0, const uint8_t *x1, bool *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void BroadcastCmp(const std::vector<size_t> &x0_dims, const std::vector<size_t> &x1_dims,
                                           const std::vector<size_t> &y_dims, enum BroadcastOpType op,
                                           const int64_t *x0, const int64_t *x1, bool *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void BroadcastCmp(const std::vector<size_t> &x0_dims, const std::vector<size_t> &x1_dims,
                                           const std::vector<size_t> &y_dims, enum BroadcastOpType op,
                                           const int16_t *x0, const int16_t *x1, bool *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void BroadcastCmp(const std::vector<size_t> &x0_dims, const std::vector<size_t> &x1_dims,
                                           const std::vector<size_t> &y_dims, enum BroadcastOpType op,
                                           const uint16_t *x0, const uint16_t *x1, bool *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void BroadcastCmp(const std::vector<size_t> &x0_dims, const std::vector<size_t> &x1_dims,
                                           const std::vector<size_t> &y_dims, enum BroadcastOpType op,
                                           const uint32_t *x0, const uint32_t *x1, bool *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void BroadcastCmp(const std::vector<size_t> &x0_dims, const std::vector<size_t> &x1_dims,
                                           const std::vector<size_t> &y_dims, enum BroadcastOpType op,
                                           const uint64_t *x0, const uint64_t *x1, bool *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void BroadcastCmp(const std::vector<size_t> &x0_dims, const std::vector<size_t> &x1_dims,
                                           const std::vector<size_t> &y_dims, enum BroadcastOpType op, const bool *x0,
                                           const bool *x1, bool *y, cudaStream_t stream);
// Broadcast Arithmetic
template <typename T, typename Func>
__global__ void BroadcastArithKernel(const size_t l0, const size_t l1, const size_t l2, const size_t l3,
                                     const size_t l4, const size_t l5, const size_t l6, const size_t r0,
                                     const size_t r1, const size_t r2, const size_t r3, const size_t r4,
                                     const size_t r5, const size_t r6, const size_t d0, const size_t d1,
                                     const size_t d2, const size_t d3, const size_t d4, const size_t d5,
                                     const size_t d6, const T *x0, const T *x1, T *y) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < d0 * d1 * d2 * d3 * d4 * d5 * d6;
       pos += blockDim.x * gridDim.x) {
    size_t i = pos / (d1 * d2 * d3 * d4 * d5 * d6) % d0;
    size_t j = pos / (d2 * d3 * d4 * d5 * d6) % d1;
    size_t k = pos / (d3 * d4 * d5 * d6) % d2;
    size_t l = pos / (d4 * d5 * d6) % d3;
    size_t m = pos / (d5 * d6) % d4;
    size_t n = pos / d6 % d5;
    size_t o = pos % d6;

    size_t l_index = Index(i, l0) * l1 * l2 * l3 * l4 * l5 * l6;
    l_index += Index(j, l1) * l2 * l3 * l4 * l5 * l6;
    l_index += Index(k, l2) * l3 * l4 * l5 * l6;
    l_index += Index(l, l3) * l4 * l5 * l6;
    l_index += Index(m, l4) * l5 * l6;
    l_index += Index(n, l5) * l6;
    l_index += Index(o, l6);
    size_t r_index = Index(i, r0) * r1 * r2 * r3 * r4 * r5 * r6;
    r_index += Index(j, r1) * r2 * r3 * r4 * r5 * r6;
    r_index += Index(k, r2) * r3 * r4 * r5 * r6;
    r_index += Index(l, r3) * r4 * r5 * r6;
    r_index += Index(m, r4) * r5 * r6;
    r_index += Index(n, r5) * r6;
    r_index += Index(o, r6);
    y[pos] = Func()(x0[l_index], x1[r_index]);
  }
}

template <typename T1, typename T2, typename T3, typename Func>
__global__ void BroadcastComplexArithKernel(const size_t l0, const size_t l1, const size_t l2, const size_t l3,
                                            const size_t l4, const size_t l5, const size_t l6, const size_t r0,
                                            const size_t r1, const size_t r2, const size_t r3, const size_t r4,
                                            const size_t r5, const size_t r6, const size_t d0, const size_t d1,
                                            const size_t d2, const size_t d3, const size_t d4, const size_t d5,
                                            const size_t d6, const T1 *x0, const T2 *x1, Complex<T3> *y) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < d0 * d1 * d2 * d3 * d4 * d5 * d6;
       pos += blockDim.x * gridDim.x) {
    size_t i = pos / (d1 * d2 * d3 * d4 * d5 * d6) % d0;
    size_t j = pos / (d2 * d3 * d4 * d5 * d6) % d1;
    size_t k = pos / (d3 * d4 * d5 * d6) % d2;
    size_t l = pos / (d4 * d5 * d6) % d3;
    size_t m = pos / (d5 * d6) % d4;
    size_t n = pos / d6 % d5;
    size_t o = pos % d6;

    size_t l_index = Index(i, l0) * l1 * l2 * l3 * l4 * l5 * l6;
    l_index += Index(j, l1) * l2 * l3 * l4 * l5 * l6;
    l_index += Index(k, l2) * l3 * l4 * l5 * l6;
    l_index += Index(l, l3) * l4 * l5 * l6;
    l_index += Index(m, l4) * l5 * l6;
    l_index += Index(n, l5) * l6;
    l_index += Index(o, l6);
    size_t r_index = Index(i, r0) * r1 * r2 * r3 * r4 * r5 * r6;
    r_index += Index(j, r1) * r2 * r3 * r4 * r5 * r6;
    r_index += Index(k, r2) * r3 * r4 * r5 * r6;
    r_index += Index(l, r3) * r4 * r5 * r6;
    r_index += Index(m, r4) * r5 * r6;
    r_index += Index(n, r5) * r6;
    r_index += Index(o, r6);
    y[pos] = Func()(x0[l_index], x1[r_index]);
  }
}

template <typename T>
void BroadcastArith(const std::vector<size_t> &x0_dims, const std::vector<size_t> &x1_dims,
                    const std::vector<size_t> &y_dims, enum BroadcastOpType op, const T *x0, const T *x1, T *y,
                    cudaStream_t stream) {
  size_t size = 1;
  for (auto d : y_dims) {
    size *= d;
  }
  switch (op) {
    case BROADCAST_TYPE_MAXIMUM:
      return BroadcastArithKernel<T, MaximumFunc<T>><<<(size + 255) / 256, 256, 0, stream>>>(
        x0_dims[0], x0_dims[1], x0_dims[2], x0_dims[3], x0_dims[4], x0_dims[5], x0_dims[6], x1_dims[0], x1_dims[1],
        x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5], x1_dims[6], y_dims[0], y_dims[1], y_dims[2], y_dims[3],
        y_dims[4], y_dims[5], y_dims[6], x0, x1, y);
    case BROADCAST_TYPE_MINIMUM:
      return BroadcastArithKernel<T, MinimumFunc<T>><<<(size + 255) / 256, 256, 0, stream>>>(
        x0_dims[0], x0_dims[1], x0_dims[2], x0_dims[3], x0_dims[4], x0_dims[5], x0_dims[6], x1_dims[0], x1_dims[1],
        x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5], x1_dims[6], y_dims[0], y_dims[1], y_dims[2], y_dims[3],
        y_dims[4], y_dims[5], y_dims[6], x0, x1, y);
    case BROADCAST_TYPE_POWER:
      return BroadcastArithKernel<T, PowerFunc<T>><<<(size + 255) / 256, 256, 0, stream>>>(
        x0_dims[0], x0_dims[1], x0_dims[2], x0_dims[3], x0_dims[4], x0_dims[5], x0_dims[6], x1_dims[0], x1_dims[1],
        x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5], x1_dims[6], y_dims[0], y_dims[1], y_dims[2], y_dims[3],
        y_dims[4], y_dims[5], y_dims[6], x0, x1, y);
    case BROADCAST_TYPE_REALDIV:
      return BroadcastArithKernel<T, RealDivFunc<T>><<<(size + 255) / 256, 256, 0, stream>>>(
        x0_dims[0], x0_dims[1], x0_dims[2], x0_dims[3], x0_dims[4], x0_dims[5], x0_dims[6], x1_dims[0], x1_dims[1],
        x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5], x1_dims[6], y_dims[0], y_dims[1], y_dims[2], y_dims[3],
        y_dims[4], y_dims[5], y_dims[6], x0, x1, y);
    case BROADCAST_TYPE_MUL:
      return BroadcastArithKernel<T, MulFunc<T>><<<(size + 255) / 256, 256, 0, stream>>>(
        x0_dims[0], x0_dims[1], x0_dims[2], x0_dims[3], x0_dims[4], x0_dims[5], x0_dims[6], x1_dims[0], x1_dims[1],
        x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5], x1_dims[6], y_dims[0], y_dims[1], y_dims[2], y_dims[3],
        y_dims[4], y_dims[5], y_dims[6], x0, x1, y);
    case BROADCAST_TYPE_SUB:
      return BroadcastArithKernel<T, SubFunc<T>><<<(size + 255) / 256, 256, 0, stream>>>(
        x0_dims[0], x0_dims[1], x0_dims[2], x0_dims[3], x0_dims[4], x0_dims[5], x0_dims[6], x1_dims[0], x1_dims[1],
        x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5], x1_dims[6], y_dims[0], y_dims[1], y_dims[2], y_dims[3],
        y_dims[4], y_dims[5], y_dims[6], x0, x1, y);
    case BROADCAST_TYPE_ADD:
      return BroadcastArithKernel<T, AddFunc<T>><<<(size + 255) / 256, 256, 0, stream>>>(
        x0_dims[0], x0_dims[1], x0_dims[2], x0_dims[3], x0_dims[4], x0_dims[5], x0_dims[6], x1_dims[0], x1_dims[1],
        x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5], x1_dims[6], y_dims[0], y_dims[1], y_dims[2], y_dims[3],
        y_dims[4], y_dims[5], y_dims[6], x0, x1, y);
    case BROADCAST_TYPE_FLOORDIV:
      return BroadcastArithKernel<T, FloorDivFunc<T>><<<(size + 255) / 256, 256, 0, stream>>>(
        x0_dims[0], x0_dims[1], x0_dims[2], x0_dims[3], x0_dims[4], x0_dims[5], x0_dims[6], x1_dims[0], x1_dims[1],
        x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5], x1_dims[6], y_dims[0], y_dims[1], y_dims[2], y_dims[3],
        y_dims[4], y_dims[5], y_dims[6], x0, x1, y);
    case BROADCAST_TYPE_ABSGRAD:
      return BroadcastArithKernel<T, AbsGradFunc<T>><<<(size + 255) / 256, 256, 0, stream>>>(
        x0_dims[0], x0_dims[1], x0_dims[2], x0_dims[3], x0_dims[4], x0_dims[5], x0_dims[6], x1_dims[0], x1_dims[1],
        x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5], x1_dims[6], y_dims[0], y_dims[1], y_dims[2], y_dims[3],
        y_dims[4], y_dims[5], y_dims[6], x0, x1, y);
    case BROADCAST_TYPE_DIV:
      return BroadcastArithKernel<T, DivFunc<T>><<<(size + 255) / 256, 256, 0, stream>>>(
        x0_dims[0], x0_dims[1], x0_dims[2], x0_dims[3], x0_dims[4], x0_dims[5], x0_dims[6], x1_dims[0], x1_dims[1],
        x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5], x1_dims[6], y_dims[0], y_dims[1], y_dims[2], y_dims[3],
        y_dims[4], y_dims[5], y_dims[6], x0, x1, y);
    case BROADCAST_TYPE_DIVNONAN:
      return BroadcastArithKernel<T, DivNoNanFunc<T>><<<(size + 255) / 256, 256, 0, stream>>>(
        x0_dims[0], x0_dims[1], x0_dims[2], x0_dims[3], x0_dims[4], x0_dims[5], x0_dims[6], x1_dims[0], x1_dims[1],
        x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5], x1_dims[6], y_dims[0], y_dims[1], y_dims[2], y_dims[3],
        y_dims[4], y_dims[5], y_dims[6], x0, x1, y);
    case BROADCAST_TYPE_SQUARED_DIFFERENCE:
      return BroadcastArithKernel<T, SquaredDifferenceFunc<T>><<<(size + 255) / 256, 256, 0, stream>>>(
        x0_dims[0], x0_dims[1], x0_dims[2], x0_dims[3], x0_dims[4], x0_dims[5], x0_dims[6], x1_dims[0], x1_dims[1],
        x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5], x1_dims[6], y_dims[0], y_dims[1], y_dims[2], y_dims[3],
        y_dims[4], y_dims[5], y_dims[6], x0, x1, y);
    case BROADCAST_TYPE_TRUNCATEDIV:
      return BroadcastArithKernel<T, TruncateDivFunc<T>><<<(size + 255) / 256, 256, 0, stream>>>(
        x0_dims[0], x0_dims[1], x0_dims[2], x0_dims[3], x0_dims[4], x0_dims[5], x0_dims[6], x1_dims[0], x1_dims[1],
        x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5], x1_dims[6], y_dims[0], y_dims[1], y_dims[2], y_dims[3],
        y_dims[4], y_dims[5], y_dims[6], x0, x1, y);
    case BROADCAST_TYPE_TRUNCATEMOD:
      return BroadcastArithKernel<T, TruncateModFunc<T>><<<(size + 255) / 256, 256, 0, stream>>>(
        x0_dims[0], x0_dims[1], x0_dims[2], x0_dims[3], x0_dims[4], x0_dims[5], x0_dims[6], x1_dims[0], x1_dims[1],
        x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5], x1_dims[6], y_dims[0], y_dims[1], y_dims[2], y_dims[3],
        y_dims[4], y_dims[5], y_dims[6], x0, x1, y);
    case BROADCAST_TYPE_MOD:
      return BroadcastArithKernel<T, ModFunc<T>><<<(size + 255) / 256, 256, 0, stream>>>(
        x0_dims[0], x0_dims[1], x0_dims[2], x0_dims[3], x0_dims[4], x0_dims[5], x0_dims[6], x1_dims[0], x1_dims[1],
        x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5], x1_dims[6], y_dims[0], y_dims[1], y_dims[2], y_dims[3],
        y_dims[4], y_dims[5], y_dims[6], x0, x1, y);
    case BROADCAST_TYPE_FLOORMOD:
      return BroadcastArithKernel<T, FloorModFunc<T>><<<(size + 255) / 256, 256, 0, stream>>>(
        x0_dims[0], x0_dims[1], x0_dims[2], x0_dims[3], x0_dims[4], x0_dims[5], x0_dims[6], x1_dims[0], x1_dims[1],
        x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5], x1_dims[6], y_dims[0], y_dims[1], y_dims[2], y_dims[3],
        y_dims[4], y_dims[5], y_dims[6], x0, x1, y);
    case BROADCAST_TYPE_ATAN2:
      return BroadcastArithKernel<T, Atan2Func<T>><<<(size + 255) / 256, 256, 0, stream>>>(
        x0_dims[0], x0_dims[1], x0_dims[2], x0_dims[3], x0_dims[4], x0_dims[5], x0_dims[6], x1_dims[0], x1_dims[1],
        x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5], x1_dims[6], y_dims[0], y_dims[1], y_dims[2], y_dims[3],
        y_dims[4], y_dims[5], y_dims[6], x0, x1, y);
    default:
      break;
  }
}

template <typename T1, typename T2, typename T3>
void BroadcastComplexArith(const std::vector<size_t> &x0_dims, const std::vector<size_t> &x1_dims,
                           const std::vector<size_t> &y_dims, enum BroadcastOpType op, const T1 *x0, const T2 *x1,
                           Complex<T3> *y, cudaStream_t stream) {
  size_t size = 1;
  for (auto d : y_dims) {
    size *= d;
  }
  switch (op) {
    case BROADCAST_TYPE_ADD:
      return BroadcastComplexArithKernel<T1, T2, T3, AddFunc<T3>><<<(size + 255) / 256, 256, 0, stream>>>(
        x0_dims[0], x0_dims[1], x0_dims[2], x0_dims[3], x0_dims[4], x0_dims[5], x0_dims[6], x1_dims[0], x1_dims[1],
        x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5], x1_dims[6], y_dims[0], y_dims[1], y_dims[2], y_dims[3],
        y_dims[4], y_dims[5], y_dims[6], x0, x1, y);
    case BROADCAST_TYPE_SUB:
      return BroadcastComplexArithKernel<T1, T2, T3, SubFunc<T3>><<<(size + 255) / 256, 256, 0, stream>>>(
        x0_dims[0], x0_dims[1], x0_dims[2], x0_dims[3], x0_dims[4], x0_dims[5], x0_dims[6], x1_dims[0], x1_dims[1],
        x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5], x1_dims[6], y_dims[0], y_dims[1], y_dims[2], y_dims[3],
        y_dims[4], y_dims[5], y_dims[6], x0, x1, y);
    case BROADCAST_TYPE_MUL:
      return BroadcastComplexArithKernel<T1, T2, T3, MulFunc<T3>><<<(size + 255) / 256, 256, 0, stream>>>(
        x0_dims[0], x0_dims[1], x0_dims[2], x0_dims[3], x0_dims[4], x0_dims[5], x0_dims[6], x1_dims[0], x1_dims[1],
        x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5], x1_dims[6], y_dims[0], y_dims[1], y_dims[2], y_dims[3],
        y_dims[4], y_dims[5], y_dims[6], x0, x1, y);
    case BROADCAST_TYPE_DIV:
      return BroadcastComplexArithKernel<T1, T2, T3, DivFunc<T3>><<<(size + 255) / 256, 256, 0, stream>>>(
        x0_dims[0], x0_dims[1], x0_dims[2], x0_dims[3], x0_dims[4], x0_dims[5], x0_dims[6], x1_dims[0], x1_dims[1],
        x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5], x1_dims[6], y_dims[0], y_dims[1], y_dims[2], y_dims[3],
        y_dims[4], y_dims[5], y_dims[6], x0, x1, y);
    case BROADCAST_TYPE_REALDIV:
      return BroadcastComplexArithKernel<T1, T2, T3, RealDivFunc<T3>><<<(size + 255) / 256, 256, 0, stream>>>(
        x0_dims[0], x0_dims[1], x0_dims[2], x0_dims[3], x0_dims[4], x0_dims[5], x0_dims[6], x1_dims[0], x1_dims[1],
        x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5], x1_dims[6], y_dims[0], y_dims[1], y_dims[2], y_dims[3],
        y_dims[4], y_dims[5], y_dims[6], x0, x1, y);
    default:
      break;
  }
}

template <typename T>
void BroadcastComplexArith(const std::vector<size_t> &x0_dims, const std::vector<size_t> &x1_dims,
                         const std::vector<size_t> &y_dims, enum BroadcastOpType op, const T *x0, const T *x1,
                         Complex<T> *y, cudaStream_t stream) {
  size_t size = 1;
  for (auto d : y_dims) {
    size *= d;
  }
  if (op == BROADCAST_TYPE_COMPLEX) {
    return BroadcastComplexArithKernel<T, T, T, ComplexFunc<T>><<<(size + 255) / 256, 256, 0, stream>>>(
      x0_dims[0], x0_dims[1], x0_dims[2], x0_dims[3], x0_dims[4], x0_dims[5], x0_dims[6], x1_dims[0], x1_dims[1],
      x1_dims[2], x1_dims[3], x1_dims[4], x1_dims[5], x1_dims[6], y_dims[0], y_dims[1], y_dims[2], y_dims[3], y_dims[4],
      y_dims[5], y_dims[6], x0, x1, y);
  }
}

template CUDA_LIB_EXPORT void BroadcastArith(const std::vector<size_t> &x0_dims, const std::vector<size_t> &x1_dims,
                                             const std::vector<size_t> &y_dims, enum BroadcastOpType op,
                                             const double *x0, const double *x1, double *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void BroadcastArith(const std::vector<size_t> &x0_dims, const std::vector<size_t> &x1_dims,
                                             const std::vector<size_t> &y_dims, enum BroadcastOpType op,
                                             const float *x0, const float *x1, float *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void BroadcastArith(const std::vector<size_t> &x0_dims, const std::vector<size_t> &x1_dims,
                                             const std::vector<size_t> &y_dims, enum BroadcastOpType op, const half *x0,
                                             const half *x1, half *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void BroadcastArith(const std::vector<size_t> &x0_dims, const std::vector<size_t> &x1_dims,
                                             const std::vector<size_t> &y_dims, enum BroadcastOpType op, const int *x0,
                                             const int *x1, int *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void BroadcastArith(const std::vector<size_t> &x0_dims, const std::vector<size_t> &x1_dims,
                                             const std::vector<size_t> &y_dims, enum BroadcastOpType op,
                                             const int8_t *x0, const int8_t *x1, int8_t *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void BroadcastArith(const std::vector<size_t> &x0_dims, const std::vector<size_t> &x1_dims,
                                             const std::vector<size_t> &y_dims, enum BroadcastOpType op,
                                             const uint8_t *x0, const uint8_t *x1, uint8_t *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void BroadcastArith(const std::vector<size_t> &x0_dims, const std::vector<size_t> &x1_dims,
                                             const std::vector<size_t> &y_dims, enum BroadcastOpType op,
                                             const int64_t *x0, const int64_t *x1, int64_t *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void BroadcastArith(const std::vector<size_t> &x0_dims, const std::vector<size_t> &x1_dims,
                                             const std::vector<size_t> &y_dims, enum BroadcastOpType op,
                                             const int16_t *x0, const int16_t *x1, int16_t *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void BroadcastArith(const std::vector<size_t> &x0_dims, const std::vector<size_t> &x1_dims,
                                             const std::vector<size_t> &y_dims, enum BroadcastOpType op,
                                             const uint16_t *x0, const uint16_t *x1, uint16_t *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void BroadcastArith(const std::vector<size_t> &x0_dims, const std::vector<size_t> &x1_dims,
                                             const std::vector<size_t> &y_dims, enum BroadcastOpType op,
                                             const uint32_t *x0, const uint32_t *x1, uint32_t *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void BroadcastArith(const std::vector<size_t> &x0_dims, const std::vector<size_t> &x1_dims,
                                             const std::vector<size_t> &y_dims, enum BroadcastOpType op,
                                             const uint64_t *x0, const uint64_t *x1, uint64_t *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void BroadcastArith(const std::vector<size_t> &x0_dims, const std::vector<size_t> &x1_dims,
                                             const std::vector<size_t> &y_dims, enum BroadcastOpType op, const bool *x0,
                                             const bool *x1, bool *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void BroadcastComplexArith(const std::vector<size_t> &x0_dims,
                                                    const std::vector<size_t> &x1_dims,
                                                    const std::vector<size_t> &y_dims, enum BroadcastOpType op,
                                                    const Complex<float> *x0, const Complex<float> *x1,
                                                    Complex<float> *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void BroadcastComplexArith(const std::vector<size_t> &x0_dims,
                                                    const std::vector<size_t> &x1_dims,
                                                    const std::vector<size_t> &y_dims, enum BroadcastOpType op,
                                                    const Complex<float> *x0, const float *x1, Complex<float> *y,
                                                    cudaStream_t stream);
template CUDA_LIB_EXPORT void BroadcastComplexArith(const std::vector<size_t> &x0_dims,
                                                    const std::vector<size_t> &x1_dims,
                                                    const std::vector<size_t> &y_dims, enum BroadcastOpType op,
                                                    const float *x0, const Complex<float> *x1, Complex<float> *y,
                                                    cudaStream_t stream);
template CUDA_LIB_EXPORT void BroadcastComplexArith(const std::vector<size_t> &x0_dims,
                                                    const std::vector<size_t> &x1_dims,
                                                    const std::vector<size_t> &y_dims, enum BroadcastOpType op,
                                                    const Complex<double> *x0, const Complex<double> *x1,
                                                    Complex<double> *y, cudaStream_t stream);
template CUDA_LIB_EXPORT void BroadcastComplexArith(const std::vector<size_t> &x0_dims,
                                                    const std::vector<size_t> &x1_dims,
                                                    const std::vector<size_t> &y_dims, enum BroadcastOpType op,
                                                    const Complex<double> *x0, const double *x1, Complex<double> *y,
                                                    cudaStream_t stream);
template CUDA_LIB_EXPORT void BroadcastComplexArith(const std::vector<size_t> &x0_dims,
                                                    const std::vector<size_t> &x1_dims,
                                                    const std::vector<size_t> &y_dims, enum BroadcastOpType op,
                                                    const double *x0, const Complex<double> *x1, Complex<double> *y,
                                                    cudaStream_t stream);
template CUDA_LIB_EXPORT void BroadcastComplexArith(const std::vector<size_t> &x0_dims,
                                                    const std::vector<size_t> &x1_dims,
                                                    const std::vector<size_t> &y_dims, enum BroadcastOpType op,
                                                    const double *x0, const double *x1, Complex<double> *y,
                                                    cudaStream_t stream);
template CUDA_LIB_EXPORT void BroadcastComplexArith(const std::vector<size_t> &x0_dims,
                                                    const std::vector<size_t> &x1_dims,
                                                    const std::vector<size_t> &y_dims, enum BroadcastOpType op,
                                                    const float *x0, const float *x1, Complex<float> *y,
                                                    cudaStream_t stream);

// BroadcastTo
template <typename T>
__global__ void BroadcastToKernel(const size_t i0, const size_t i1, const size_t i2, const size_t i3, const size_t o0,
                                  const size_t o1, const size_t o2, const size_t o3, const T *input_addr,
                                  T *output_addr) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < o0 * o1 * o2 * o3; pos += blockDim.x * gridDim.x) {
    size_t i = pos / (o1 * o2 * o3) % o0;
    size_t j = pos / (o2 * o3) % o1;
    size_t k = pos / o3 % o2;
    size_t l = pos % o3;

    size_t input_idx = Index(i, i0) * i1 * i2 * i3 + Index(j, i1) * i2 * i3 + Index(k, i2) * i3 + Index(l, i3);
    output_addr[pos] = input_addr[input_idx];
  }
}

template <typename T>
void BroadcastTo(const size_t &i0, const size_t &i1, const size_t &i2, const size_t &i3, const size_t &o0,
                 const size_t &o1, const size_t &o2, const size_t &o3, const T *input_addr, T *output_addr,
                 cudaStream_t stream) {
  size_t nums = o0 * o1 * o2 * o3;
  BroadcastToKernel<<<GET_BLOCKS(nums), GET_THREADS, 0, stream>>>(i0, i1, i2, i3, o0, o1, o2, o3, input_addr,
                                                                  output_addr);
}

template CUDA_LIB_EXPORT void BroadcastTo(const size_t &i0, const size_t &i1, const size_t &i2, const size_t &i3,
                                          const size_t &o0, const size_t &o1, const size_t &o2, const size_t &o3,
                                          const double *input_addr, double *output_addr, cudaStream_t stream);
template CUDA_LIB_EXPORT void BroadcastTo(const size_t &i0, const size_t &i1, const size_t &i2, const size_t &i3,
                                          const size_t &o0, const size_t &o1, const size_t &o2, const size_t &o3,
                                          const float *input_addr, float *output_addr, cudaStream_t stream);
template CUDA_LIB_EXPORT void BroadcastTo(const size_t &i0, const size_t &i1, const size_t &i2, const size_t &i3,
                                          const size_t &o0, const size_t &o1, const size_t &o2, const size_t &o3,
                                          const half *input_addr, half *output_addr, cudaStream_t stream);
template CUDA_LIB_EXPORT void BroadcastTo(const size_t &i0, const size_t &i1, const size_t &i2, const size_t &i3,
                                          const size_t &o0, const size_t &o1, const size_t &o2, const size_t &o3,
                                          const int16_t *input_addr, int16_t *output_addr, cudaStream_t stream);
template CUDA_LIB_EXPORT void BroadcastTo(const size_t &i0, const size_t &i1, const size_t &i2, const size_t &i3,
                                          const size_t &o0, const size_t &o1, const size_t &o2, const size_t &o3,
                                          const int32_t *input_addr, int32_t *output_addr, cudaStream_t stream);
template CUDA_LIB_EXPORT void BroadcastTo(const size_t &i0, const size_t &i1, const size_t &i2, const size_t &i3,
                                          const size_t &o0, const size_t &o1, const size_t &o2, const size_t &o3,
                                          const int64_t *input_addr, int64_t *output_addr, cudaStream_t stream);
template CUDA_LIB_EXPORT void BroadcastTo(const size_t &i0, const size_t &i1, const size_t &i2, const size_t &i3,
                                          const size_t &o0, const size_t &o1, const size_t &o2, const size_t &o3,
                                          const bool *input_addr, bool *output_addr, cudaStream_t stream);
