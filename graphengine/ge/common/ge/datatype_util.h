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

#ifndef GE_COMMON_GE_DATATYPE_UTIL_H_
#define GE_COMMON_GE_DATATYPE_UTIL_H_

#include <map>
#include <vector>

#include "external/graph/types.h"

namespace ge {
static const int32_t kGeSizeFloat = sizeof(float);
static const int32_t kGeSizeHalfFloat = sizeof(float) / 2;
static const int32_t kGeSizeInt8 = sizeof(int8_t);
static const int32_t kGeSizeInt16 = sizeof(int16_t);
static const int32_t kGeSizeInt32 = sizeof(int32_t);
static const int32_t kGeSizeInt64 = sizeof(int64_t);
static const int32_t kGeSizeUint8 = sizeof(uint8_t);
static const int32_t kGeSizeBool = sizeof(bool);
static const int32_t kGeSizeDouble = sizeof(double);
static const int32_t kGeSizeUint64 = sizeof(uint64_t);
static const int32_t kGeSizeUint16 = sizeof(uint16_t);
static const int32_t kGeSizeUint32 = sizeof(uint32_t);

static std::map<ge::DataType, int32_t> CONST_OPDATA_TYPE_SIZE_MAP = {
    {ge::DT_FLOAT, kGeSizeFloat},   {ge::DT_FLOAT16, kGeSizeHalfFloat}, {ge::DT_INT8, kGeSizeInt8},
    {ge::DT_INT16, kGeSizeInt16},   {ge::DT_INT32, kGeSizeInt32},       {ge::DT_INT64, kGeSizeInt64},
    {ge::DT_UINT8, kGeSizeUint8},   {ge::DT_UINT16, kGeSizeUint16},     {ge::DT_UINT32, kGeSizeUint32},
    {ge::DT_UINT64, kGeSizeUint64}, {ge::DT_DOUBLE, kGeSizeDouble},     {ge::DT_BOOL, kGeSizeBool}};

class DataTypeUtil {
 public:
  static bool DataTypeTranslatable(const ge::DataType &src_out_data_type, const ge::DataType &dst_in_data_type);
  static const std::vector<ge::DataType> &GetTranslatableDataTypesBySrc(const ge::DataType &src_out_data_type);
  static const std::vector<ge::DataType> &GetTranslatableDataTypesByDst(const ge::DataType &dst_in_data_type);
  static int32_t GetIrDataType(ge::DataType data_type);
};
}  // namespace ge
#endif  // GE_COMMON_GE_DATATYPE_UTIL_H_
