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
#include "plugin/device/ascend/kernel/aicpu/aicpu_util.h"
#include "proto/types.pb.h"
#include "runtime/mem.h"
#include "include/common/utils/convert_utils.h"

namespace luojianet_ms {
namespace kernel {
static const std::map<int32_t, int32_t> kMsProtoDataTypeMap = {
  {luojianet_ms::TypeId::kTypeUnknown, luojianet_ms::DataType::MS_UNKNOWN},
  {luojianet_ms::TypeId::kNumberTypeBool, luojianet_ms::DataType::MS_BOOL},
  {luojianet_ms::TypeId::kNumberTypeInt, luojianet_ms::DataType::MS_INT32},
  {luojianet_ms::TypeId::kNumberTypeInt8, luojianet_ms::DataType::MS_INT8},
  {luojianet_ms::TypeId::kNumberTypeInt16, luojianet_ms::DataType::MS_INT16},
  {luojianet_ms::TypeId::kNumberTypeInt32, luojianet_ms::DataType::MS_INT32},
  {luojianet_ms::TypeId::kNumberTypeInt64, luojianet_ms::DataType::MS_INT64},
  {luojianet_ms::TypeId::kNumberTypeUInt, luojianet_ms::DataType::MS_UINT32},
  {luojianet_ms::TypeId::kNumberTypeUInt8, luojianet_ms::DataType::MS_UINT8},
  {luojianet_ms::TypeId::kNumberTypeUInt16, luojianet_ms::DataType::MS_UINT16},
  {luojianet_ms::TypeId::kNumberTypeUInt32, luojianet_ms::DataType::MS_UINT32},
  {luojianet_ms::TypeId::kNumberTypeUInt64, luojianet_ms::DataType::MS_UINT64},
  {luojianet_ms::TypeId::kNumberTypeFloat16, luojianet_ms::DataType::MS_FLOAT16},
  {luojianet_ms::TypeId::kNumberTypeFloat, luojianet_ms::DataType::MS_FLOAT32},
  {luojianet_ms::TypeId::kNumberTypeFloat32, luojianet_ms::DataType::MS_FLOAT32},
  {luojianet_ms::TypeId::kNumberTypeFloat64, luojianet_ms::DataType::MS_FLOAT64},
  {luojianet_ms::TypeId::kNumberTypeComplex64, luojianet_ms::DataType::MS_COMPLEX64},
  {luojianet_ms::TypeId::kNumberTypeComplex128, luojianet_ms::DataType::MS_COMPLEX128},
};

static const std::map<int32_t, int32_t> kProtoDataTypeToMsDataTypeMap = {
  {luojianet_ms::DataType::MS_UNKNOWN, luojianet_ms::TypeId::kTypeUnknown},
  {luojianet_ms::DataType::MS_BOOL, luojianet_ms::TypeId::kNumberTypeBool},
  {luojianet_ms::DataType::MS_INT32, luojianet_ms::TypeId::kNumberTypeInt32},
  {luojianet_ms::DataType::MS_INT8, luojianet_ms::TypeId::kNumberTypeInt8},
  {luojianet_ms::DataType::MS_INT16, luojianet_ms::TypeId::kNumberTypeInt16},
  {luojianet_ms::DataType::MS_INT64, luojianet_ms::TypeId::kNumberTypeInt64},
  {luojianet_ms::DataType::MS_UINT8, luojianet_ms::TypeId::kNumberTypeUInt8},
  {luojianet_ms::DataType::MS_UINT16, luojianet_ms::TypeId::kNumberTypeUInt16},
  {luojianet_ms::DataType::MS_UINT32, luojianet_ms::TypeId::kNumberTypeUInt32},
  {luojianet_ms::DataType::MS_UINT64, luojianet_ms::TypeId::kNumberTypeUInt64},
  {luojianet_ms::DataType::MS_FLOAT16, luojianet_ms::TypeId::kNumberTypeFloat16},
  {luojianet_ms::DataType::MS_FLOAT32, luojianet_ms::TypeId::kNumberTypeFloat32},
  {luojianet_ms::DataType::MS_FLOAT64, luojianet_ms::TypeId::kNumberTypeFloat64},
  {luojianet_ms::DataType::MS_COMPLEX64, luojianet_ms::TypeId::kNumberTypeComplex64},
  {luojianet_ms::DataType::MS_COMPLEX128, luojianet_ms::TypeId::kNumberTypeComplex128},
};

int AicpuOpUtil::MsTypeToProtoType(TypeId ms_type) {
  auto iter = kMsProtoDataTypeMap.find(ms_type);
  if (iter == kMsProtoDataTypeMap.end()) {
    MS_LOG(ERROR) << "UnSupported ms_type value" << static_cast<int>(ms_type);
    return -1;
  }
  return iter->second;
}

int AicpuOpUtil::ProtoTypeToMsType(int proto_type) {
  auto iter = kProtoDataTypeToMsDataTypeMap.find(proto_type);
  if (iter == kProtoDataTypeToMsDataTypeMap.end()) {
    MS_LOG(ERROR) << "UnSupported proto_type value:" << proto_type;
    return -1;
  }
  return iter->second;
}
}  // namespace kernel
}  // namespace luojianet_ms
