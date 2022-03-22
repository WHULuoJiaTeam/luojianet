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

#ifndef GE_COMMON_FORMATS_FORMAT_TRANSFERS_DATATYPE_TRANSFER_H_
#define GE_COMMON_FORMATS_FORMAT_TRANSFERS_DATATYPE_TRANSFER_H_

#include <functional>
#include <memory>
#include <vector>

#include "register/register_format_transfer.h"
#include "external/graph/types.h"
#include "framework/common/ge_inner_error_codes.h"

namespace ge {
namespace formats {
struct CastArgs {
  const uint8_t *data;
  size_t src_data_size;
  DataType src_data_type;
  DataType dst_data_type;
};

class DataTypeTransfer {
 public:
  Status TransDataType(const CastArgs &args, TransResult &result);
};

std::shared_ptr<DataTypeTransfer> BuildDataTypeTransfer(const CastArgs &args);

bool DataTypeTransferExists(const CastArgs &args);
}  // namespace formats
}  // namespace ge

#endif  // GE_COMMON_FORMATS_FORMAT_TRANSFERS_DATATYPE_TRANSFER_H_
