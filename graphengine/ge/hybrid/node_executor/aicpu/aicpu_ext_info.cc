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

#include "hybrid/node_executor/aicpu/aicpu_ext_info.h"
#include "framework/common/util.h"
#include "framework/common/fmk_error_codes.h"
#include "framework/common/debug/log.h"

namespace ge {
namespace hybrid {
namespace {
// if dim count is not reach kMaxShapeDims(8), use INT64_MIN to mark dim end.
constexpr int64_t kDimEndFlag = INT64_MIN;
const std::map<int32_t, int32_t> kTopicTypeToRtsFlagMap {
    {static_cast<int32_t>(aicpu::FWKAdapter::FWK_ADPT_TOPIC_DEVICE_ONLY), 0},
    {static_cast<int32_t>(aicpu::FWKAdapter::FWK_ADPT_TOPIC_DEVICE_FIRST), RT_KERNEL_DEVICE_FIRST},
    {static_cast<int32_t>(aicpu::FWKAdapter::FWK_ADPT_TOPIC_HOST_ONLY), RT_KERNEL_HOST_ONLY},
    {static_cast<int32_t>(aicpu::FWKAdapter::FWK_ADPT_TOPIC_HOST_FIRST), RT_KERNEL_HOST_FIRST}
};
}

Status AicpuExtInfoHandler::Parse(const std::string &ext_info) {
  GELOGI("Node[%s] parse ext info start.", node_name_.c_str());
  if (ext_info.empty()) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID,
           "[Check][Param:ext_info]Node[%s] parse ext info failed as ext info is empty.", node_name_.c_str());
    REPORT_INNER_ERROR("E19999", "Node[%s] parse ext info failed as ext info is empty.", node_name_.c_str());
    return ACL_ERROR_GE_PARAM_INVALID;
  }

  ext_info_len_ = ext_info.size();
  ext_info_.reset(new(std::nothrow)uint8_t[ext_info_len_]);
  GE_CHECK_NOTNULL(ext_info_);

  if (memcpy_s(ext_info_.get(), ext_info_len_, ext_info.c_str(), ext_info.size()) != EOK) {
    GELOGE(ACL_ERROR_GE_MEMORY_OPERATE_FAILED, "[Update][ext_info_][%s] Failed to copy ext info", node_name_.c_str());
    REPORT_CALL_ERROR("E19999", "[%s] Failed to copy ext info.", node_name_.c_str());
    return ACL_ERROR_GE_MEMORY_OPERATE_FAILED;
  }

  input_shape_and_type_.clear();
  output_shape_and_type_.clear();

  auto ext_info_data = ext_info_.get();
  size_t offset = 0;
  while (offset + sizeof(AicpuExtInfo) <= ext_info_len_) {
    auto aicpu_ext_info = reinterpret_cast<AicpuExtInfo *>(ext_info_data + offset);
    GELOGD("Ext infoType=%d, infoLen=%u.", aicpu_ext_info->infoType, aicpu_ext_info->infoLen);
    switch (aicpu_ext_info->infoType) {
      case aicpu::FWKAdapter::FWK_ADPT_EXT_SHAPE_TYPE:
        GE_CHK_STATUS_RET(ParseExtShapeType(aicpu_ext_info), "[Parse][ExtShapeType] failed.");
        break;
      case aicpu::FWKAdapter::FWK_ADPT_EXT_INPUT_SHAPE:
        GE_CHK_STATUS_RET(ParseExtInputShape(aicpu_ext_info), "[Parse][ExtInputShape] failed.");
        break;
      case aicpu::FWKAdapter::FWK_ADPT_EXT_OUTPUT_SHAPE:
        GE_CHK_STATUS_RET(ParseExtOutputShape(aicpu_ext_info), "[Parse][ExtOutputShape] failed.");
        break;
      case aicpu::FWKAdapter::FWK_ADPT_EXT_SESSION_INFO:
        GE_CHK_STATUS_RET(ParseExtSessionInfo(aicpu_ext_info), "[Parse][ExtSessionInfo] failed.");
        break;
      case aicpu::FWKAdapter::FWK_ADPT_EXT_BITMAP:
        GE_CHK_STATUS_RET(ParseExtBitMap(aicpu_ext_info), "[Parse][ExtBitMap] failed.");
        break;
      case aicpu::FWKAdapter::FWK_ADPT_EXT_UPDATE_ADDR:
        GE_CHK_STATUS_RET(ParseExtUpdateAddr(aicpu_ext_info), "[Parse][ExtUpdateAddr] failed.");
        break;
      case aicpu::FWKAdapter::FWK_ADPT_EXT_TOPIC_TYPE:
        GE_CHK_STATUS_RET(ParseExtTopicType(aicpu_ext_info), "[Parse][ExtTopicType] failed.");
        break;
      case aicpu::FWKAdapter::FWK_ADPT_EXT_ASYNCWAIT:
        GE_CHK_STATUS_RET(ParseExtAsyncWait(aicpu_ext_info), "[Parse][ExtAsyncWait] failed.");
        break;
      default:
        GELOGD("Node[%s] ignore infoType=%d, infoLen=%u.",
               node_name_.c_str(), aicpu_ext_info->infoType, aicpu_ext_info->infoLen);
        break;
    }
    offset += sizeof(AicpuExtInfo);
    offset += aicpu_ext_info->infoLen;
  }

  GE_IF_BOOL_EXEC(offset != ext_info_len_,
                  REPORT_INNER_ERROR("E19999", "Node[%s] ext_info format error, parse not reach end,"
                                     "offset=%zu, ext_info_len=%zu.", node_name_.c_str(), offset, ext_info_len_);
                  GELOGE(ACL_ERROR_GE_PARAM_INVALID, "[Check][Size]Node[%s] ext_info format error,"
                         "parse not reach end, offset=%zu, ext_info_len=%zu.",
                         node_name_.c_str(), offset, ext_info_len_);
                  return ACL_ERROR_GE_PARAM_INVALID;);
  GELOGI("Node[%s] parse ext info end.", node_name_.c_str());
  return SUCCESS;
}

Status AicpuExtInfoHandler::ParseExtAsyncWait(AicpuExtInfo *aicpu_ext_info) {
  if (aicpu_ext_info->infoLen != sizeof(AsyncWaitInfo)) {
    REPORT_INNER_ERROR("E19999",
                       "Node[%s] parse ext async wait info failed as infoLen must be %zu but %u.",
                       node_name_.c_str(), sizeof(AsyncWaitInfo), aicpu_ext_info->infoLen);
    GELOGE(ACL_ERROR_GE_PARAM_INVALID,
           "[Check][DataLen]Node[%s] parse ext async wait info failed as infoLen must be %zu but %u.",
           node_name_.c_str(), sizeof(AsyncWaitInfo), aicpu_ext_info->infoLen);
    return ACL_ERROR_GE_PARAM_INVALID;
  }

  async_wait_ = reinterpret_cast<AsyncWaitInfo *>(aicpu_ext_info->infoMsg);
  GELOGI("Node[%s] parse async wait info success infoLen=%u.", node_name_.c_str(), aicpu_ext_info->infoLen);
  return SUCCESS;
}

Status AicpuExtInfoHandler::ParseExtShapeType(AicpuExtInfo *aicpu_ext_info) {
  GE_IF_BOOL_EXEC(aicpu_ext_info->infoLen != sizeof(int32_t),
                  REPORT_INNER_ERROR("E19999", "Node[%s] parse ext shape type failed as infoLen must be %zu but %u.",
                                     node_name_.c_str(), sizeof(int32_t), aicpu_ext_info->infoLen);
                  GELOGE(ACL_ERROR_GE_PARAM_INVALID,
                         "[Check][Size]Node[%s] parse ext shape type failed as infoLen must be %zu but %u.",
                         node_name_.c_str(), sizeof(int32_t), aicpu_ext_info->infoLen);
                  return ACL_ERROR_GE_PARAM_INVALID;);

  auto type = reinterpret_cast<const int32_t *>(aicpu_ext_info->infoMsg);

  GE_IF_BOOL_EXEC(*type != unknown_type_,
                  REPORT_INNER_ERROR("E19999", "Node[%s] parse ext shape type failed as need %d but %d.",
                                     node_name_.c_str(), unknown_type_, *type);
                  GELOGE(ACL_ERROR_GE_PARAM_INVALID,
                         "[Check][Type]Node[%s] parse ext shape type failed as need %d but %d.",
                         node_name_.c_str(), unknown_type_, *type);
                  return ACL_ERROR_GE_PARAM_INVALID;);
  GELOGI("Node[%s] parse ext shape type success infoLen=%u.", node_name_.c_str(), aicpu_ext_info->infoLen);
  return SUCCESS;
}

Status AicpuExtInfoHandler::ParseExtInputShape(AicpuExtInfo *aicpu_ext_info) {
  auto need_len = input_num_ * sizeof(AicpuShapeAndType);
  GE_IF_BOOL_EXEC(aicpu_ext_info->infoLen != need_len,
                  REPORT_INNER_ERROR("E19999", "Node[%s] parse ext input shape failed as infoLen must be "
                                     "input_num[%u]*sizeof(ShapeAndType)[%zu] but %u.",
                                     node_name_.c_str(), input_num_, sizeof(AicpuShapeAndType),
                                     aicpu_ext_info->infoLen);
                  GELOGE(ACL_ERROR_GE_PARAM_INVALID,
                         "[Check][DataLen]Node[%s] parse ext input shape failed as infoLen must be "
                         "input_num[%u]*sizeof(ShapeAndType)[%zu] but %u.",
                         node_name_.c_str(), input_num_, sizeof(AicpuShapeAndType), aicpu_ext_info->infoLen);
                  return ACL_ERROR_GE_PARAM_INVALID;);

  auto input = reinterpret_cast<AicpuShapeAndType *>(aicpu_ext_info->infoMsg);

  for (uint32_t index = 0; index < input_num_; ++index) {
    input_shape_and_type_.emplace_back(&input[index]);
  }
  GELOGI("Node[%s] parse ext input shape success infoLen=%u.", node_name_.c_str(), aicpu_ext_info->infoLen);
  return SUCCESS;
}

Status AicpuExtInfoHandler::ParseExtOutputShape(AicpuExtInfo *aicpu_ext_info) {
  if (unknown_type_ == DEPEND_COMPUTE) {
    GELOGD("Node[%s] is depend compute type no need ext output shape, ignore it, infoLen=%u.",
           node_name_.c_str(), aicpu_ext_info->infoLen);
    return SUCCESS;
  }
  auto need_len = output_num_ * sizeof(AicpuShapeAndType);
  GE_IF_BOOL_EXEC(aicpu_ext_info->infoLen != need_len,
                  REPORT_INNER_ERROR("E19999", "Node[%s] parse ext output shape failed as infoLen must be "
                                     "output_num[%u]*sizeof(ShapeAndType)[%zu] but %u.",
                                     node_name_.c_str(), output_num_, sizeof(AicpuShapeAndType),
                                     aicpu_ext_info->infoLen);
                  GELOGE(ACL_ERROR_GE_PARAM_INVALID,
                         "[Check][DataLen]Node[%s] parse ext output shape failed as infoLen must be "
                         "output_num[%u]*sizeof(ShapeAndType)[%zu] but %u.",
                         node_name_.c_str(), output_num_, sizeof(AicpuShapeAndType), aicpu_ext_info->infoLen);
                  return ACL_ERROR_GE_PARAM_INVALID;);

  auto output = reinterpret_cast<AicpuShapeAndType *>(aicpu_ext_info->infoMsg);
  for (uint32_t index = 0; index < output_num_; ++index) {
    output_shape_and_type_.emplace_back(&output[index]);
  }
  GELOGI("Node[%s] parse ext output shape success infoLen=%u.", node_name_.c_str(), aicpu_ext_info->infoLen);
  return SUCCESS;
}

Status AicpuExtInfoHandler::ParseExtSessionInfo(AicpuExtInfo *aicpu_ext_info) {
  GE_IF_BOOL_EXEC(aicpu_ext_info->infoLen != sizeof(AicpuSessionInfo),
                  REPORT_INNER_ERROR("E19999",
                                     "Node[%s] parse ext session info failed as infoLen must be %zu but %u.",
                                     node_name_.c_str(), sizeof(SessionInfo), aicpu_ext_info->infoLen);
                  GELOGE(ACL_ERROR_GE_PARAM_INVALID,
                         "[Check][DataLen]Node[%s] parse ext session info failed as infoLen must be %zu but %u.",
                         node_name_.c_str(), sizeof(SessionInfo), aicpu_ext_info->infoLen);
                  return ACL_ERROR_GE_PARAM_INVALID;);

  session_info_ = reinterpret_cast<AicpuSessionInfo *>(aicpu_ext_info->infoMsg);
  GELOGI("Node[%s] parse session info success infoLen=%u.", node_name_.c_str(), aicpu_ext_info->infoLen);
  return SUCCESS;
}

Status AicpuExtInfoHandler::ParseExtBitMap(AicpuExtInfo *aicpu_ext_info) {
  GE_IF_BOOL_EXEC(aicpu_ext_info->infoLen != sizeof(uint64_t),
                  REPORT_INNER_ERROR("E19999",
                                     "Node[%s] parse bit_map info failed as infoLen must be %zu but %u.",
                                     node_name_.c_str(), sizeof(uint64_t), aicpu_ext_info->infoLen);
                  GELOGE(PARAM_INVALID,
                         "[Check][DataLen]Node[%s] parse bit_map info failed as infoLen must be %zu but %u.",
                         node_name_.c_str(), sizeof(uint64_t), aicpu_ext_info->infoLen);
                  return PARAM_INVALID;);

  bit_map_ = reinterpret_cast<uint64_t *>(aicpu_ext_info->infoMsg);
  GELOGI("Node[%s] bit_map info success infoLen=%u.", node_name_.c_str(), aicpu_ext_info->infoLen);
  return SUCCESS;
}

Status AicpuExtInfoHandler::ParseExtUpdateAddr(AicpuExtInfo *aicpu_ext_info) {
  GE_IF_BOOL_EXEC(aicpu_ext_info->infoLen != sizeof(uint32_t),
                  REPORT_INNER_ERROR("E19999",
                                     "Node[%s] parse update_addr info failed as infoLen must be %zu but %u.",
                                     node_name_.c_str(), sizeof(uint32_t), aicpu_ext_info->infoLen);
                  GELOGE(PARAM_INVALID,
                         "[Check][DataLen]Node[%s] parse update_addr info failed as infoLen must be %zu but %u.",
                         node_name_.c_str(), sizeof(uint32_t), aicpu_ext_info->infoLen);
                  return PARAM_INVALID;);

  update_addr_ = reinterpret_cast<uint32_t *>(aicpu_ext_info->infoMsg);
  GELOGI("Node[%s] update_addr info success infoLen=%u.", node_name_.c_str(), aicpu_ext_info->infoLen);
  return SUCCESS;
}

Status AicpuExtInfoHandler::ParseExtTopicType(AicpuExtInfo *aicpu_ext_info) {
  if (aicpu_ext_info->infoLen != sizeof(int32_t)) {
    REPORT_INNER_ERROR("E19999",
                       "Node[%s] parse topic_type info failed as infoLen must be %zu but %u.",
                       node_name_.c_str(), sizeof(int32_t), aicpu_ext_info->infoLen);
    GELOGE(ACL_ERROR_GE_PARAM_INVALID,
           "[Check][DataLen]Node[%s] parse topic_type info failed as infoLen must be %zu but %u.",
           node_name_.c_str(), sizeof(int32_t), aicpu_ext_info->infoLen);
    return ACL_ERROR_GE_PARAM_INVALID;
  }
  GE_CHECK_NOTNULL(aicpu_ext_info->infoMsg);
  auto type_info = reinterpret_cast<int32_t *>(aicpu_ext_info->infoMsg);
  int32_t type = *type_info;

  topic_type_flag_ = TopicTypeToRtsFlag(type);
  if (topic_type_flag_ == -1) {
    REPORT_INNER_ERROR("E19999", "Node[%s] parse ext topic type failed as need %d %d %d %d but %d.",
                       node_name_.c_str(),
                       aicpu::FWKAdapter::FWK_ADPT_TOPIC_DEVICE_ONLY,
                       aicpu::FWKAdapter::FWK_ADPT_TOPIC_DEVICE_FIRST,
                       aicpu::FWKAdapter::FWK_ADPT_TOPIC_HOST_ONLY,
                       aicpu::FWKAdapter::FWK_ADPT_TOPIC_HOST_FIRST,
                       type);
    GELOGE(ACL_ERROR_GE_PARAM_INVALID,
           "[Check][Type]Node[%s] parse ext shape type failed as need %d %d %d %d but %d.",
           node_name_.c_str(),
           aicpu::FWKAdapter::FWK_ADPT_TOPIC_DEVICE_ONLY,
           aicpu::FWKAdapter::FWK_ADPT_TOPIC_DEVICE_FIRST,
           aicpu::FWKAdapter::FWK_ADPT_TOPIC_HOST_ONLY,
           aicpu::FWKAdapter::FWK_ADPT_TOPIC_HOST_FIRST,
           type);
    return ACL_ERROR_GE_PARAM_INVALID;
  }

  GELOGI("Node[%s] parse ext topic type info success infoLen=%u, topic_type=%d, rts_flag=%d.",
         node_name_.c_str(), aicpu_ext_info->infoLen, type, topic_type_flag_);
  return SUCCESS;
}

Status AicpuExtInfoHandler::UpdateExecuteMode(bool flag) {
  if (bit_map_ == nullptr) {
    GELOGD("There is no bit_map in ext_info, no need update.");
    return SUCCESS;
  }
  if (flag) {
    *(bit_map_) |= 1;
  } else {
    *(bit_map_) &= ~1;
  }
  return SUCCESS;
}

Status AicpuExtInfoHandler::UpdateSessionInfo(uint64_t session_id, uint64_t kernel_id, bool sess_flag) {
  if (session_info_ == nullptr) {
    GELOGD("There is no session info in ext_info, no need update.");
    return SUCCESS;
  }

  session_info_->sessionId = session_id;
  session_info_->kernelId = kernel_id;
  session_info_->sessFlag = sess_flag;
  return SUCCESS;
}

Status AicpuExtInfoHandler::UpdateEventId(uint32_t event_id) {
  if (async_wait_ == nullptr) {
    REPORT_INNER_ERROR("E19999", "async_wait_ is nullptr.");
    GELOGE(FAILED, "[Check][async_wait_] async_wait_ is nullptr.");
    return FAILED;
  }
  async_wait_->waitType = 1;
  async_wait_->waitId = event_id;
  return SUCCESS;
}

Status AicpuExtInfoHandler::UpdateSessionInfoSessionId(uint64_t session_id) {
  if (session_info_ == nullptr) {
    GELOGD("There is no session info in ext_info, no need update.");
    return SUCCESS;
  }

  session_info_->sessionId = session_id;
  session_info_->sessFlag = true;
  return SUCCESS;
}

Status AicpuExtInfoHandler::UpdateInputShapeAndType(uint32_t input_index, const GeTensorDesc &input_desc) {
  GE_CHECK_LE(input_index, input_num_);
  const auto &shape = input_desc.GetShape();

  GE_CHK_STATUS_RET(UpdateShapeAndType(shape, input_desc.GetDataType(), input_shape_and_type_[input_index]),
                    "[Update][ShapeAndType] failed, Node[%s] input[%u] .",
                    node_name_.c_str(), input_index);
  return SUCCESS;
}

Status AicpuExtInfoHandler::UpdateOutputShapeAndType(uint32_t output_index, const GeTensorDesc &output_desc) {
  GE_IF_BOOL_EXEC((unknown_type_ == DEPEND_COMPUTE),
                  REPORT_INNER_ERROR("E19999", "Node[%s] is depend compute is no need update output shape"
                                     "and type by ext.", node_name_.c_str());
                  GELOGE(ACL_ERROR_GE_INTERNAL_ERROR,
                         "[Check][Type]Node[%s] is depend compute is no need update output shape and type by ext.",
                         node_name_.c_str());
                  return ACL_ERROR_GE_INTERNAL_ERROR;);
  GE_CHECK_LE(output_index, output_num_);
  auto shape = output_desc.GetShape();

  // shape range need use range update shape
  if (unknown_type_ == DEPEND_SHAPE_RANGE) {
    std::vector<std::pair<int64_t, int64_t>> range;
    auto range_ret = output_desc.GetShapeRange(range);
    GE_IF_BOOL_EXEC(range_ret != GRAPH_SUCCESS,
                    REPORT_INNER_ERROR("E19999", "Node[%s] is shape range type but get GetShapeRange failed, ret=%u",
                                       node_name_.c_str(), range_ret);
                    GELOGE(ACL_ERROR_GE_INTERNAL_ERROR,
                           "[Invoke][GetShapeRange]Node[%s] is shape range type but get GetShapeRange failed, ret=%u",
                           node_name_.c_str(), range_ret);
                    return ACL_ERROR_GE_INTERNAL_ERROR;);
    for (size_t k = 0; k < range.size(); ++k) {
      if (shape.GetDim(k) < 0 && k < range.size()) {
        GELOGD("Node[%s] output[%u] update dim[%zu] from %ld to range max %ld.",
               node_name_.c_str(), output_index, k, shape.GetDim(k), range[k].second);
        shape.SetDim(k, range[k].second);
      }
    }
  }

  return UpdateShapeAndType(shape, output_desc.GetDataType(), output_shape_and_type_[output_index]);
}

Status AicpuExtInfoHandler::GetOutputShapeAndType(uint32_t output_index, GeShape &shape, DataType &data_type) {
  GE_IF_BOOL_EXEC((unknown_type_ == DEPEND_COMPUTE),
                  REPORT_INNER_ERROR("E19999",
                                     "Node[%s] is depend compute type can not get output shape and type by ext.",
                                     node_name_.c_str());
                  GELOGE(INTERNAL_ERROR,
                         "[Check][Type]Node[%s] is depend compute type can not get output shape and type by ext.",
                         node_name_.c_str());
                  return INTERNAL_ERROR;);
  GetShapeAndType(output_shape_and_type_[output_index], shape, data_type);
  return SUCCESS;
}

bool AicpuExtInfoHandler::IsNeedRefreshIOAddr() {
  return update_addr_ != nullptr && *update_addr_ != static_cast<uint32_t>(aicpu::FWKAdapter::FWK_ADPT_UPDATE_NULL);
}

Status AicpuExtInfoHandler::UpdateShapeAndType(const GeShape &shape, DataType data_type,
                                               AicpuShapeAndType *shape_and_type) {
  auto dim_num = shape.GetDimNum();
  if (dim_num > aicpu::FWKAdapter::kMaxShapeDims) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID,
           "[Check][DimNum]Update shape and type failed, as dim_num %zu is over max shape dims %u.",
           dim_num, aicpu::FWKAdapter::kMaxShapeDims);
    REPORT_INNER_ERROR("E19999", "Update shape and type failed, as dim_num %zu is over max shape dims %u.",
                       dim_num, aicpu::FWKAdapter::kMaxShapeDims);
    return ACL_ERROR_GE_PARAM_INVALID;
  }
  size_t index = 0;
  for (; index < dim_num; ++index) {
    shape_and_type->dims[index] = shape.GetDim(index);
  }
  if (index < aicpu::FWKAdapter::kMaxShapeDims) {
    shape_and_type->dims[index] = kDimEndFlag;
  }

  // now only support update shape, type is not support
  return SUCCESS;
}

void AicpuExtInfoHandler::GetShapeAndType(const AicpuShapeAndType *shape_and_type,
                                          GeShape &shape,
                                          DataType &data_type) {
  std::vector<int64_t> dims;
  for (uint32_t index = 0; index < aicpu::FWKAdapter::kMaxShapeDims; ++index) {
    auto tmpDim = shape_and_type->dims[index];
    if (tmpDim == kDimEndFlag) {
      break;
    }
    dims.emplace_back(tmpDim);
  }
  data_type = static_cast<DataType>(shape_and_type->type);
  shape = GeShape(dims);
}

int32_t AicpuExtInfoHandler::TopicTypeToRtsFlag(int32_t topic_type) {
  auto it = kTopicTypeToRtsFlagMap.find(topic_type);
  if (it != kTopicTypeToRtsFlagMap.end()) {
    return it->second;
  }

  return -1;
}
}  // namespace hybrid
}  // namespace ge
