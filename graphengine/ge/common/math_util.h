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

#ifndef GE_COMMON_MATH_UTIL_H_
#define GE_COMMON_MATH_UTIL_H_

#include <securec.h>
#include <algorithm>
#include <cmath>

#include "framework/common/debug/log.h"
#include "framework/common/types.h"
#include "framework/common/util.h"
#include "mmpa/mmpa_api.h"

namespace ge {
/**
* @ingroup domi_calibration
* @brief  Initializes an input array to a specified value
* @param [in]  n        array initialization length
* @param [in]  alpha    initialization value
* @param [out]  output  array to be initialized
* @return      Status
*/
template <typename Dtype>
Status NnSet(const int32_t n, const Dtype alpha, Dtype *output) {
  GE_CHECK_NOTNULL(output);

  if (alpha == 0) {
    if (sizeof(Dtype) * n < SECUREC_MEM_MAX_LEN) {
      errno_t err = memset_s(output, sizeof(Dtype) * n, 0, sizeof(Dtype) * n);
      GE_CHK_BOOL_RET_STATUS(err == EOK, PARAM_INVALID, "memset_s err");
    } else {
      uint64_t size = static_cast<uint64_t>(sizeof(Dtype) * n);
      uint64_t step = SECUREC_MEM_MAX_LEN - (SECUREC_MEM_MAX_LEN % sizeof(Dtype));
      uint64_t times = size / step;
      uint64_t remainder = size % step;
      uint64_t i = 0;
      while (i < times) {
        errno_t err = memset_s(output + i * (step / sizeof(Dtype)), step, 0, step);
        GE_CHK_BOOL_RET_STATUS(err == EOK, PARAM_INVALID, "memset_s err");
        i++;
      }
      if (remainder != 0) {
        errno_t err = memset_s(output + i * (step / sizeof(Dtype)), remainder, 0, remainder);
        GE_CHK_BOOL_RET_STATUS(err == EOK, PARAM_INVALID, "memset_s err");
      }
    }
  }

  for (int32_t i = 0; i < n; ++i) {
    output[i] = alpha;
  }
  return SUCCESS;
}
}  // end namespace ge

#endif  //  GE_COMMON_MATH_UTIL_H_
