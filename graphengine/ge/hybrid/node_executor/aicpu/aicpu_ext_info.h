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

#ifndef GE_HYBRID_AICPU_EXT_INFO_H_
#define GE_HYBRID_AICPU_EXT_INFO_H_

#include "external/ge/ge_api_error_codes.h"
#include "cce/fwk_adpt_struct.h"
#include "cce/aicpu_engine_struct.h"
#include "graph/op_desc.h"
#include "graph/ge_tensor.h"

namespace ge {
namespace hybrid {
using AicpuShapeAndType = aicpu::FWKAdapter::ShapeAndType;
using AicpuExtInfo = aicpu::FWKAdapter::ExtInfo;
using AsyncWaitInfo = aicpu::FWKAdapter::AsyncWait;
using AicpuSessionInfo = SessionInfo;

class AicpuExtInfoHandler {
 public:
  AicpuExtInfoHandler(std::string node_name, uint32_t input_num, uint32_t output_num, UnknowShapeOpType unknown_type)
      : node_name_(std::move(node_name)),
        input_num_(input_num),
        output_num_(output_num),
        unknown_type_(unknown_type) {
  }

  ~AicpuExtInfoHandler() = default;

  uint8_t *GetExtInfo() const {
    return ext_info_.get();
  }
  size_t GetExtInfoLen() const {
    return ext_info_len_;
  }

  Status Parse(const std::string &ext_info);

  Status UpdateInputShapeAndType(uint32_t input_index, const GeTensorDesc &input_desc);

  Status UpdateOutputShapeAndType(uint32_t output_index, const GeTensorDesc &output_desc);

  Status UpdateSessionInfo(uint64_t session_id, uint64_t kernel_id, bool sess_flag);

  Status UpdateSessionInfoSessionId(uint64_t session_id);

  Status UpdateExecuteMode(bool flag);

  Status UpdateEventId(uint32_t event_id);

  Status GetOutputShapeAndType(uint32_t output_index, GeShape &shape, DataType &data_type);

  bool IsNeedRefreshIOAddr();
  int32_t GetTopicTypeFlag() const { return topic_type_flag_; }

 private:

  Status ParseExtShapeType(AicpuExtInfo *aicpu_ext_info);
  Status ParseExtInputShape(AicpuExtInfo *aicpu_ext_info);
  Status ParseExtOutputShape(AicpuExtInfo *aicpu_ext_info);
  Status ParseExtSessionInfo(AicpuExtInfo *aicpu_ext_info);
  Status ParseExtBitMap(AicpuExtInfo *aicpu_ext_info);
  Status ParseExtUpdateAddr(AicpuExtInfo *aicpu_ext_info);
  Status ParseExtTopicType(AicpuExtInfo *aicpu_ext_info);
  Status ParseExtAsyncWait(AicpuExtInfo *aicpu_ext_info);

  static Status UpdateShapeAndType(const GeShape &shape,
                                   DataType data_type,
                                   AicpuShapeAndType *shape_and_type);

  static void GetShapeAndType(const AicpuShapeAndType *shape_and_type,
                              GeShape &shape,
                              DataType &data_type);

 private:
  int32_t TopicTypeToRtsFlag(int32_t topic_type);

  const std::string node_name_;
  const uint32_t input_num_;
  const uint32_t output_num_;
  UnknowShapeOpType unknown_type_;
  AicpuSessionInfo *session_info_ = nullptr;
  AsyncWaitInfo *async_wait_ = nullptr;
  uint64_t *bit_map_ = nullptr;
  uint32_t *update_addr_ = nullptr;
  int32_t topic_type_flag_ = -1;

  std::unique_ptr<uint8_t[]> ext_info_;
  size_t ext_info_len_ = 0;
  std::vector<AicpuShapeAndType *> input_shape_and_type_;
  std::vector<AicpuShapeAndType *> output_shape_and_type_;
};
}  // namespace hybrid
}  // namespace ge
#endif // GE_HYBRID_AICPU_EXT_INFO_H_