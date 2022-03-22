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

#include "graph/load/model_manager/aipp_utils.h"

#include <string>

#include "framework/common/debug/log.h"
#include "framework/common/op/ge_op_utils.h"
#include "framework/common/util.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"

#include "framework/common/debug/ge_log.h"

namespace ge {
#define AIPP_CONVERT_TO_AIPP_INFO(KEY) aipp_info.KEY = aipp_params->KEY()

#define AIPP_CONVERT_TO_AIPP_INFO_WITH_INDEX(KEY, INDEX)                             \
  do {                                                                               \
    if (aipp_params->KEY##_size() > 0) {                                             \
      aipp_info.KEY = aipp_params->KEY(INDEX);                                       \
    }                                                                                \
  } while (0)

Status AippUtils::ConvertAippParams2AippInfo(domi::AippOpParams *aipp_params, AippConfigInfo &aipp_info) {
  GE_CHECK_NOTNULL(aipp_params);
  AIPP_CONVERT_TO_AIPP_INFO(aipp_mode);
  AIPP_CONVERT_TO_AIPP_INFO(input_format);
  AIPP_CONVERT_TO_AIPP_INFO(related_input_rank);
  AIPP_CONVERT_TO_AIPP_INFO(src_image_size_w);
  AIPP_CONVERT_TO_AIPP_INFO(src_image_size_h);
  AIPP_CONVERT_TO_AIPP_INFO(crop);
  AIPP_CONVERT_TO_AIPP_INFO(load_start_pos_w);
  AIPP_CONVERT_TO_AIPP_INFO(load_start_pos_h);
  AIPP_CONVERT_TO_AIPP_INFO(crop_size_w);
  AIPP_CONVERT_TO_AIPP_INFO(crop_size_h);
  AIPP_CONVERT_TO_AIPP_INFO(resize);
  AIPP_CONVERT_TO_AIPP_INFO(resize_output_w);
  AIPP_CONVERT_TO_AIPP_INFO(resize_output_h);
  AIPP_CONVERT_TO_AIPP_INFO(padding);
  AIPP_CONVERT_TO_AIPP_INFO(left_padding_size);
  AIPP_CONVERT_TO_AIPP_INFO(right_padding_size);
  AIPP_CONVERT_TO_AIPP_INFO(top_padding_size);
  AIPP_CONVERT_TO_AIPP_INFO(bottom_padding_size);
  AIPP_CONVERT_TO_AIPP_INFO(csc_switch);
  AIPP_CONVERT_TO_AIPP_INFO(rbuv_swap_switch);
  AIPP_CONVERT_TO_AIPP_INFO(ax_swap_switch);
  AIPP_CONVERT_TO_AIPP_INFO(single_line_mode);
  AIPP_CONVERT_TO_AIPP_INFO_WITH_INDEX(matrix_r0c0, 0);
  AIPP_CONVERT_TO_AIPP_INFO_WITH_INDEX(matrix_r0c1, 0);
  AIPP_CONVERT_TO_AIPP_INFO_WITH_INDEX(matrix_r0c2, 0);
  AIPP_CONVERT_TO_AIPP_INFO_WITH_INDEX(matrix_r1c0, 0);
  AIPP_CONVERT_TO_AIPP_INFO_WITH_INDEX(matrix_r1c1, 0);
  AIPP_CONVERT_TO_AIPP_INFO_WITH_INDEX(matrix_r1c2, 0);
  AIPP_CONVERT_TO_AIPP_INFO_WITH_INDEX(matrix_r2c0, 0);
  AIPP_CONVERT_TO_AIPP_INFO_WITH_INDEX(matrix_r2c1, 0);
  AIPP_CONVERT_TO_AIPP_INFO_WITH_INDEX(matrix_r2c2, 0);
  AIPP_CONVERT_TO_AIPP_INFO_WITH_INDEX(output_bias_0, 0);
  AIPP_CONVERT_TO_AIPP_INFO_WITH_INDEX(output_bias_1, 0);
  AIPP_CONVERT_TO_AIPP_INFO_WITH_INDEX(output_bias_2, 0);
  AIPP_CONVERT_TO_AIPP_INFO_WITH_INDEX(input_bias_0, 0);
  AIPP_CONVERT_TO_AIPP_INFO_WITH_INDEX(input_bias_1, 0);
  AIPP_CONVERT_TO_AIPP_INFO_WITH_INDEX(input_bias_2, 0);
  AIPP_CONVERT_TO_AIPP_INFO(mean_chn_0);
  AIPP_CONVERT_TO_AIPP_INFO(mean_chn_1);
  AIPP_CONVERT_TO_AIPP_INFO(mean_chn_2);
  AIPP_CONVERT_TO_AIPP_INFO(mean_chn_3);
  AIPP_CONVERT_TO_AIPP_INFO(min_chn_0);
  AIPP_CONVERT_TO_AIPP_INFO(min_chn_1);
  AIPP_CONVERT_TO_AIPP_INFO(min_chn_2);
  AIPP_CONVERT_TO_AIPP_INFO(min_chn_3);
  AIPP_CONVERT_TO_AIPP_INFO_WITH_INDEX(var_reci_chn_0, 0);
  AIPP_CONVERT_TO_AIPP_INFO_WITH_INDEX(var_reci_chn_1, 0);
  AIPP_CONVERT_TO_AIPP_INFO_WITH_INDEX(var_reci_chn_2, 0);
  AIPP_CONVERT_TO_AIPP_INFO_WITH_INDEX(var_reci_chn_3, 0);
  AIPP_CONVERT_TO_AIPP_INFO(support_rotation);
  AIPP_CONVERT_TO_AIPP_INFO(max_src_image_size);
  return SUCCESS;
}
}  // namespace ge
