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
#ifndef LUOJIANET_MS_CCSRC_MINDDATA_DATASET_INCLUDE_TYPEID_H_
#define LUOJIANET_MS_CCSRC_MINDDATA_DATASET_INCLUDE_TYPEID_H_

#include "luojianet_ms/core/ir/dtype/type_id.h"
#include "minddata/dataset/core/data_type.h"

namespace luojianet_ms {
namespace dataset {
inline dataset::DataType MSTypeToDEType(TypeId data_type) {
  switch (data_type) {
    case kNumberTypeBool:
      return dataset::DataType(dataset::DataType::DE_BOOL);
    case kNumberTypeInt8:
      return dataset::DataType(dataset::DataType::DE_INT8);
    case kNumberTypeUInt8:
      return dataset::DataType(dataset::DataType::DE_UINT8);
    case kNumberTypeInt16:
      return dataset::DataType(dataset::DataType::DE_INT16);
    case kNumberTypeUInt16:
      return dataset::DataType(dataset::DataType::DE_UINT16);
    case kNumberTypeInt32:
      return dataset::DataType(dataset::DataType::DE_INT32);
    case kNumberTypeUInt32:
      return dataset::DataType(dataset::DataType::DE_UINT32);
    case kNumberTypeInt64:
      return dataset::DataType(dataset::DataType::DE_INT64);
    case kNumberTypeUInt64:
      return dataset::DataType(dataset::DataType::DE_UINT64);
    case kNumberTypeFloat16:
      return dataset::DataType(dataset::DataType::DE_FLOAT16);
    case kNumberTypeFloat32:
      return dataset::DataType(dataset::DataType::DE_FLOAT32);
    case kNumberTypeFloat64:
      return dataset::DataType(dataset::DataType::DE_FLOAT64);
    case kObjectTypeString:
      return dataset::DataType(dataset::DataType::DE_STRING);
    default:
      return dataset::DataType(dataset::DataType::DE_UNKNOWN);
  }
}

inline TypeId DETypeToMSType(dataset::DataType data_type) {
  switch (data_type.value()) {
    case dataset::DataType::DE_BOOL:
      return luojianet_ms::TypeId::kNumberTypeBool;
    case dataset::DataType::DE_INT8:
      return luojianet_ms::TypeId::kNumberTypeInt8;
    case dataset::DataType::DE_UINT8:
      return luojianet_ms::TypeId::kNumberTypeUInt8;
    case dataset::DataType::DE_INT16:
      return luojianet_ms::TypeId::kNumberTypeInt16;
    case dataset::DataType::DE_UINT16:
      return luojianet_ms::TypeId::kNumberTypeUInt16;
    case dataset::DataType::DE_INT32:
      return luojianet_ms::TypeId::kNumberTypeInt32;
    case dataset::DataType::DE_UINT32:
      return luojianet_ms::TypeId::kNumberTypeUInt32;
    case dataset::DataType::DE_INT64:
      return luojianet_ms::TypeId::kNumberTypeInt64;
    case dataset::DataType::DE_UINT64:
      return luojianet_ms::TypeId::kNumberTypeUInt64;
    case dataset::DataType::DE_FLOAT16:
      return luojianet_ms::TypeId::kNumberTypeFloat16;
    case dataset::DataType::DE_FLOAT32:
      return luojianet_ms::TypeId::kNumberTypeFloat32;
    case dataset::DataType::DE_FLOAT64:
      return luojianet_ms::TypeId::kNumberTypeFloat64;
    case dataset::DataType::DE_STRING:
      return luojianet_ms::TypeId::kObjectTypeString;
    default:
      return kTypeUnknown;
  }
}
}  // namespace dataset
}  // namespace luojianet_ms

#endif  // LUOJIANET_MS_CCSRC_MINDDATA_DATASET_INCLUDE_TYPEID_H_
