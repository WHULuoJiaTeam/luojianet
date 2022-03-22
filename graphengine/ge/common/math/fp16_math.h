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

#ifndef GE_COMMON_MATH_FP16_MATH_H_
#define GE_COMMON_MATH_FP16_MATH_H_

#include "common/fp16_t.h"

namespace ge {
/// @ingroup fp16_t mathematics method
/// @param [in] fp fp16_t object to be calculate
/// @brief   Calculates fp16_t square root function of input fp
/// @return  Returns fp16_t square root of fp
fp16_t sqrt(fp16_t fp);
/// @ingroup fp16_t mathematics method
/// @param [in] fp fp16_t object to be calculate
/// @brief   Calculates fp16_t reciprocal square root function of input fp
/// @return  Returns fp16_t reciprocal square root of fp
fp16_t rsqrt(fp16_t fp);
/// @ingroup fp16_t mathematics method
/// @param [in] fp fp16_t object to be calculate
/// @brief   Calculates fp16_t reciprocal function of input fp
/// @return  Returns fp16_t reciprocal of fp
fp16_t rcp(fp16_t fp);
/// @ingroup fp16_t mathematics method
/// @param [in] fp fp16_t object to be calculate
/// @brief   Calculates fp16_t natural exponential function of input fp
/// @return  Returns fp16_t natural exponential function of fp
fp16_t exp(fp16_t fp);
/// @ingroup fp16_t mathematics method
/// @param [in] fp fp16_t object to be calculate
/// @brief   Calculates fp16_t binary exponential function of input fp
/// @return  Returns fp16_t binary exponential function of fp
fp16_t pow2(fp16_t fp);
/// @ingroup fp16_t mathematics method
/// @param [in] fp_v fp16_t object to be calculate
/// @brief   Calculates fp16_t decimal exponential function of input fp
/// @return  Returns fp16_t decimal exponential function of fp
fp16_t pow10(fp16_t fp);
/// @ingroup fp16_t mathematics method
/// @param [in] fp_v fp16_t object to be calculate
/// @brief   Calculate fp16_t natural logarithm of fp16_t
/// @return  Returns fp16_t natural logarithm of fp
fp16_t ln(fp16_t fp);
/// @ingroup fp16_t mathematics method
/// @param [in] fp fp16_t object to be calculate
/// @brief   Calculate fp16_t binary logarithm of fp16_t
/// @return  Returns fp16_t binary logarithm of fp
fp16_t log2(fp16_t fp);
/// @ingroup fp16_t mathematics method
/// @param [in] fp fp16_t object to be calculate
/// @brief   Calculate fp16_t decimal logarithm of fp16_t
/// @return  Returns fp16_t decimal logarithm of fp
fp16_t log10(fp16_t fp);
/// @ingroup fp16_t mathematics method
/// @param [in] fp fp16_t object to be calculate
/// @brief   Calculate fp16_t cosine of fp16_t
/// @return  Returns fp16_t cosine of fp
fp16_t cos(fp16_t fp);
/// @ingroup fp16_t mathematics method
/// @param [in] fp fp16_t object to be calculate
/// @brief   Calculate fp16_t sine of fp16_t
/// @return  Returns fp16_t sine of fp
fp16_t sin(fp16_t fp);
/// @ingroup fp16_t mathematics method
/// @param [in] fp fp16_t object to be calculate
/// @brief   Calculate the absolute value(the sign bit is 0) of the give value
/// @return  Returns fp16_t absolute value(the sign bit is 0) of of fp
fp16_t abs(fp16_t fp);
/// @ingroup fp16_t mathematics method
/// @param [in] fp1 fp16_t object to be compare
/// @param [in] fp2 fp16_t object to be compare
/// @brief   Calculate the maximum fp16_t of fp1 and fp2
/// @return  Returns maximum fp16_t of fp1 and fp2
fp16_t max(fp16_t fp1, fp16_t fp2);
/// @ingroup fp16_t mathematics method
/// @param [in] fp1 fp16_t object to be compare
/// @param [in] fp2 fp16_t object to be compare
/// @brief   Calculate the minimum fp16_t of fp1 and fp2
/// @return  Returns minimum fp16_t of fp1 and fp2
fp16_t min(fp16_t fp1, fp16_t fp2);
}  // namespace ge
#endif  // GE_COMMON_MATH_FP16_MATH_H_