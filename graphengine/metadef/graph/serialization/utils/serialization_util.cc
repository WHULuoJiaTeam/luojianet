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


#include "serialization_util.h"
#include <map>

namespace ge {
const std::map<DataType, ::ge::proto::DataType> kDataTypeMap = {
    {DT_UNDEFINED, proto::DT_UNDEFINED},
    {DT_FLOAT, proto::DT_FLOAT},
    {DT_FLOAT16, proto::DT_FLOAT16},
    {DT_INT8, proto::DT_INT8},
    {DT_UINT8, proto::DT_UINT8},
    {DT_INT16, proto::DT_INT16},
    {DT_UINT16, proto::DT_UINT16},
    {DT_INT32, proto::DT_INT32},
    {DT_INT64, proto::DT_INT64},
    {DT_UINT32, proto::DT_UINT32},
    {DT_UINT64, proto::DT_UINT64},
    {DT_BOOL, proto::DT_BOOL},
    {DT_DOUBLE, proto::DT_DOUBLE},
    {DT_DUAL, proto::DT_DUAL},
    {DT_DUAL_SUB_INT8, proto::DT_DUAL_SUB_INT8},
    {DT_DUAL_SUB_UINT8, proto::DT_DUAL_SUB_UINT8},
    {DT_COMPLEX64, proto::DT_COMPLEX64},
    {DT_COMPLEX128, proto::DT_COMPLEX128},
    {DT_QINT8, proto::DT_QINT8},
    {DT_QINT16, proto::DT_QINT16},
    {DT_QINT32, proto::DT_QINT32},
    {DT_QUINT8, proto::DT_QUINT8},
    {DT_QUINT16, proto::DT_QUINT16},
    {DT_RESOURCE, proto::DT_RESOURCE},
    {DT_STRING_REF, proto::DT_STRING_REF},
    {DT_STRING, proto::DT_STRING},
    {DT_VARIANT, proto::DT_VARIANT},
    {DT_BF16, proto::DT_BF16},
    {DT_INT4, proto::DT_INT4},
    {DT_UINT1, proto::DT_UINT1},
    {DT_INT2, proto::DT_INT2},
    {DT_UINT2, proto::DT_UINT2}
};

void SerializationUtil::GeDataTypeToProto(const ge::DataType ge_type, proto::DataType &proto_type) {
  auto iter = kDataTypeMap.find(ge_type);
  if (iter != kDataTypeMap.end()) {
    proto_type = iter->second;
    return;
  }
  proto_type = proto::DT_UNDEFINED;
}

void SerializationUtil::ProtoDataTypeToGe(const proto::DataType proto_type, ge::DataType &ge_type) {
  for (auto iter : kDataTypeMap) {
    if (iter.second == proto_type) {
      ge_type = iter.first;
      return;
    }
  }
  ge_type = DT_UNDEFINED;
}

}  // namespace ge