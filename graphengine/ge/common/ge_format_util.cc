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

#include "framework/common/ge_format_util.h"
#include "common/formats/formats.h"

namespace ge {
Status GeFormatUtil::TransShape(const TensorDesc &src_desc, Format dst_format, std::vector<int64_t> &dst_shape) {
  return formats::TransShape(src_desc.GetFormat(), src_desc.GetShape().GetDims(), src_desc.GetDataType(), dst_format,
                             dst_shape);
}
}  // namespace ge
