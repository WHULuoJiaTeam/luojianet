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

#ifndef FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_RANGE_FORMAT_TRANSFER_TRANSFER_RANGE_ACCORDING_TO_FORMAT_H_
#define FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_RANGE_FORMAT_TRANSFER_TRANSFER_RANGE_ACCORDING_TO_FORMAT_H_

#include <functional>
#include <vector>
#include <algorithm>
#include "graph/compute_graph.h"
#include "transfer_shape_according_to_format.h"
#include "graph/debug/ge_log.h"
#include "graph/debug/ge_util.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/type_utils.h"

namespace transformer {
struct RangeAndFormatInfo {
  ge::GeShape old_shape;
  std::vector<std::pair<int64_t, int64_t>> old_range;
  std::vector<std::pair<int64_t, int64_t>> &new_range;
  ge::Format old_format;
  ge::Format new_format;
  ge::DataType current_data_type;
  CalcShapeExtraAttr extra_attr;
  RangeAndFormatInfo(ge::GeShape old_shape, std::vector<std::pair<int64_t, int64_t>> old_range,
                     std::vector<std::pair<int64_t, int64_t>> &new_range, ge::Format old_format,
                     ge::Format new_format, ge::DataType current_data_type) :
          old_shape(old_shape), old_range(old_range), new_range(new_range), old_format(old_format),
          new_format(new_format), current_data_type(current_data_type), extra_attr(CalcShapeExtraAttr()) {}
  RangeAndFormatInfo(ge::GeShape old_shape, std::vector<std::pair<int64_t, int64_t>> old_range,
                     std::vector<std::pair<int64_t, int64_t>> &new_range, ge::Format old_format,
                     ge::Format new_format, ge::DataType current_data_type, CalcShapeExtraAttr extra_attr) :
          old_shape(old_shape), old_range(old_range), new_range(new_range), old_format(old_format),
          new_format(new_format), current_data_type(current_data_type), extra_attr(extra_attr) {}
};

using RangeAndFormat = struct RangeAndFormatInfo;

class RangeTransferAccordingToFormat {
 public:
  RangeTransferAccordingToFormat() = default;

  ~RangeTransferAccordingToFormat() = default;

  RangeTransferAccordingToFormat(const RangeTransferAccordingToFormat &) = delete;

  RangeTransferAccordingToFormat &operator=(const RangeTransferAccordingToFormat &) = delete;

  static bool GetRangeAccordingToFormat(RangeAndFormat &input_and_output_info);
};
}  // namespace fe

#endif  // FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_RANGE_FORMAT_TRANSFER_TRANSFER_RANGE_ACCORDING_TO_FORMAT_H_
