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

#include "transfer_range_according_to_format.h"
#include "graph/types.h"

namespace transformer {

bool RangeTransferAccordingToFormat::GetRangeAccordingToFormat(RangeAndFormat &range_and_format_info) {
  /* The default new range is old range */
  std::vector<int64_t> range_upper_old;
  std::vector<int64_t> range_low_old;
  for (auto &i : range_and_format_info.old_range) {
    range_low_old.emplace_back(i.first);
    range_upper_old.emplace_back(i.second);
  }

  ge::GeShape shape_low(range_low_old);
  ge::GeShape shape_upper(range_upper_old);
  transformer::ShapeAndFormat shape_and_format_info_low {shape_low, range_and_format_info.old_format,
      range_and_format_info.new_format, range_and_format_info.current_data_type};
  transformer::ShapeAndFormat shape_and_format_info_upper {shape_upper, range_and_format_info.old_format,
      range_and_format_info.new_format, range_and_format_info.current_data_type};
  ShapeTransferAccordingToFormat shape_transfer;
  bool res = (shape_transfer.GetShapeAccordingToFormat(shape_and_format_info_low) &&
      shape_transfer.GetShapeAccordingToFormat(shape_and_format_info_upper));
  if (!res || (shape_low.GetDimNum() != shape_upper.GetDimNum())) {
    return false;
  }
  range_and_format_info.new_range.clear();
  for (size_t i = 0; i < range_and_format_info.new_range.size(); ++i) {
    range_and_format_info.new_range.emplace_back(shape_low.GetDim(i), shape_upper.GetDim(i));
  }
  return res;
}
};  // namespace fe
