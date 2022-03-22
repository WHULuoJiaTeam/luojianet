/**
 * Copyright 2021, 2022 LuoJiaNET Research and Development Group, Wuhan University
 * Copyright 2021, 2022 Huawei Technologies Co., Ltd
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

#include "common/math/fp16_math.h"
#include "external/register/register_types.h"

namespace ge {
fp16_t sqrt(fp16_t fp) {
  fp16_t ret;
  // Convert half precision float number to double
  double dVal = fp;
  // Calculate double number square root
  double dSqrt = std::sqrt(dVal);
  // calculate result
  ret = dSqrt;
  return ret;
}

fp16_t rsqrt(fp16_t fp) {
  fp16_t ret;
  // Convert half precision float number to double
  double dVal = fp;
  // Calculate double number square root and reciprocal
  double drSqrt = 1.0 / std::sqrt(dVal);
  // calculate result
  ret = drSqrt;
  return ret;
}

fp16_t rcp(fp16_t fp) {
  fp16_t ret;
  // Convert half precision float number to double
  double dVal = fp;
  // Calculate double number reciprocal
  double dRcp = 1.0 / dVal;
  // calculate result
  ret = dRcp;
  return ret;
}

fp16_t exp(fp16_t fp) {
  fp16_t ret;
  // Convert half precision float number to double
  double dVal = fp;
  // Calculate double number exponential
  double dExp = std::exp(dVal);
  // calculate result
  ret = dExp;

  return ret;
}

fp16_t pow2(fp16_t fp) {
  fp16_t ret;
  // Convert half precision float number to double
  double dVal = fp;
  // Calculate double number binary exponential
  double dExp2 = std::pow(kDim2, dVal);
  // calculate result
  ret = dExp2;

  return ret;
}

fp16_t pow10(fp16_t fp) {
  fp16_t ret;
  // Convert half precision float number to double
  double dVal = fp;
  // Calculate double number decimal exponential
  double dExp10 = std::pow(kDim10, dVal);
  // calculate result
  ret = dExp10;

  return ret;
}

fp16_t ln(fp16_t fp) {
  fp16_t ret;
  // Convert half precision float number to double
  double dVal = fp;
  // Calculate double number natural logarithm
  double dLn = std::log(dVal);
  // calculate result
  ret = dLn;

  return ret;
}

fp16_t log2(fp16_t fp) {
  fp16_t ret;
  // Convert half precision float number to double
  double dVal = fp;
  // Calculate double number binary logarithm
  double dLog2 = std::log2(dVal);
  // calculate result
  ret = dLog2;

  return ret;
}

fp16_t log10(fp16_t fp) {
  fp16_t ret;
  // Convert half precision float number to double
  double dVal = fp;
  // Calculate double number binary logarithm
  double dLog10 = std::log10(dVal);
  // calculate result
  ret = dLog10;

  return ret;
}

fp16_t cos(fp16_t fp) {
  fp16_t ret;
  // Convert half precision float number to double
  double dVal = fp;
  // Calculate double number cos
  double dCos = std::cos(dVal);
  // calculate result
  ret = dCos;

  return ret;
}

fp16_t sin(fp16_t fp) {
  fp16_t ret;
  // Convert half precision float number to double
  double dVal = fp;
  // Calculate double number sine
  double dSin = std::sin(dVal);
  // calculate result
  ret = dSin;

  return ret;
}

fp16_t abs(fp16_t fp) {
  fp16_t ret;
  ret.val = (fp.val & kFp16AbsMax);
  return ret;
}

fp16_t max(fp16_t fp1, fp16_t fp2) {
  if (fp1 >= fp2) {
    return fp1;
  } else {
    return fp2;
  }
}

fp16_t min(fp16_t fp1, fp16_t fp2) {
  if (fp1 <= fp2) {
    return fp1;
  } else {
    return fp2;
  }
}
}  // namespace ge
