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

#include <cce/dnn.h>

#include "hccl/hcom.h"

HcclResult hcom_all_gather(const char *tag, void *input_count_ptr, void *output_ptr, u64 input_count,
                           HcclDataType data_type, const char *group, rtStream_t stream) {
  return HCCL_SUCCESS;
}

HcclResult hcom_broadcast(const char *tag, void *ptr, u64 count, HcclDataType data_type, u32 root,
                          const char *group, rtStream_t stream) {
  return HCCL_SUCCESS;
}

HcclResult hcom_all_reduce(const char *tag, void *input_ptr, void *output_ptr, u64 count, HcclDataType data_type,
                           HcclReduceOp op, const char *group, rtStream_t stream) {
  return HCCL_SUCCESS;
}

HcclResult hcom_get_split_strategy(const char *group, const struct model_feature *feature, u32 max_segment_num,
                                   u32 *segment_num, u32 *segment_idx) {
  return HCCL_SUCCESS;
}

HcclResult hcom_reduce_scatter(const char *tag, void *input_ptr, void *output_ptr, u64 count,
                               HcclDataType data_type, HcclReduceOp op, const char *group, rtStream_t stream) {
  return HCCL_SUCCESS;
}

HcclResult HcomExecEnqueueAllToAllV(HcomAllToAllVParams params, std::function<void(HcclResult status)> callback) {
  return HCCL_SUCCESS;
}

HcclResult HcomExecEnqueueGatherAllToAllV(HcomGatherAllToAllVParams params,
std::function<void(HcclResult status)> callback) {
  return HCCL_SUCCESS;
}


