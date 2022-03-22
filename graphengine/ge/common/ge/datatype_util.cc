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

#include "common/ge/datatype_util.h"
#include "proto/ge_ir.pb.h"

#include <map>

namespace {
const std::vector<ge::DataType> kEmptyDatatypeVector;
std::map<ge::DataType, std::vector<ge::DataType>> g_translatable_data_type = {
    // key:src datatype, value:dst datatype
    {ge::DT_FLOAT, {ge::DT_FLOAT16, ge::DT_FLOAT}},
    {ge::DT_BOOL, {ge::DT_INT32}},
    {ge::DT_FLOAT16, {ge::DT_FLOAT, ge::DT_FLOAT16}},
    {ge::DT_INT64, {ge::DT_INT32}}};

std::map<ge::DataType, std::vector<ge::DataType>> g_reverse_translatable_data_type = {
    // key:dst datatype,value:src datatype
    {ge::DT_FLOAT16, {ge::DT_FLOAT, ge::DT_FLOAT16}},
    {ge::DT_INT32, {ge::DT_BOOL, ge::DT_INT64}},
    {ge::DT_FLOAT, {ge::DT_FLOAT16, ge::DT_FLOAT}}};

std::map<ge::DataType, ge::proto::DataType> g_dump_data_type_map = {
    // key:ge datatype,value:proto datatype
    {ge::DT_UNDEFINED, ge::proto::DT_UNDEFINED},
    {ge::DT_FLOAT, ge::proto::DT_FLOAT},
    {ge::DT_FLOAT16, ge::proto::DT_FLOAT16},
    {ge::DT_INT8, ge::proto::DT_INT8},
    {ge::DT_UINT8, ge::proto::DT_UINT8},
    {ge::DT_INT16, ge::proto::DT_INT16},
    {ge::DT_UINT16, ge::proto::DT_UINT16},
    {ge::DT_INT32, ge::proto::DT_INT32},
    {ge::DT_INT64, ge::proto::DT_INT64},
    {ge::DT_UINT32, ge::proto::DT_UINT32},
    {ge::DT_UINT64, ge::proto::DT_UINT64},
    {ge::DT_BOOL, ge::proto::DT_BOOL},
    {ge::DT_DOUBLE, ge::proto::DT_DOUBLE},
    {ge::DT_DUAL, ge::proto::DT_DUAL},
    {ge::DT_DUAL_SUB_INT8, ge::proto::DT_DUAL_SUB_INT8},
    {ge::DT_DUAL_SUB_UINT8, ge::proto::DT_DUAL_SUB_UINT8},
    {ge::DT_COMPLEX64, ge::proto::DT_COMPLEX64},
    {ge::DT_COMPLEX128, ge::proto::DT_COMPLEX128},
    {ge::DT_QINT8, ge::proto::DT_QINT8},
    {ge::DT_QINT16, ge::proto::DT_QINT16},
    {ge::DT_QINT32, ge::proto::DT_QINT32},
    {ge::DT_QUINT8, ge::proto::DT_QUINT8},
    {ge::DT_QUINT16, ge::proto::DT_QUINT16},
    {ge::DT_RESOURCE, ge::proto::DT_RESOURCE},
    {ge::DT_STRING_REF, ge::proto::DT_STRING_REF},
    {ge::DT_STRING, ge::proto::DT_STRING},
    {ge::DT_VARIANT, ge::proto::DT_VARIANT},
};
}  // namespace

namespace ge {
bool DataTypeUtil::DataTypeTranslatable(const ge::DataType &src_out_data_type, const ge::DataType &dst_in_data_type) {
  auto search = g_translatable_data_type.find(src_out_data_type);
  if (search == g_translatable_data_type.end()) {
    return false;
  }

  for (auto data_type : search->second) {
    if (data_type == dst_in_data_type) {
      return true;
    }
  }

  return false;
}

const std::vector<ge::DataType> &DataTypeUtil::GetTranslatableDataTypesBySrc(const ge::DataType &src_out_data_type) {
  auto search = g_translatable_data_type.find(src_out_data_type);
  if (search == g_translatable_data_type.end()) {
    return kEmptyDatatypeVector;
  }

  return search->second;
}

const std::vector<ge::DataType> &DataTypeUtil::GetTranslatableDataTypesByDst(const ge::DataType &dst_in_data_type) {
  auto search = g_reverse_translatable_data_type.find(dst_in_data_type);
  if (search == g_reverse_translatable_data_type.end()) {
    return kEmptyDatatypeVector;
  }

  return search->second;
}

int32_t DataTypeUtil::GetIrDataType(ge::DataType data_type) {
  auto iter = g_dump_data_type_map.find(data_type);
  if (iter == g_dump_data_type_map.end()) {
    return static_cast<int32_t>(ge::proto::DT_UNDEFINED);
  }

  return static_cast<int32_t>(iter->second);
}
}  // namespace ge
