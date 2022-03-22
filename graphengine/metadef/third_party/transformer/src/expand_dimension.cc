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

#include "expand_dimension.h"
#include "framework/common/debug/ge_log.h"

namespace transformer {
bool GetDefaultReshapeType(const ge::Format &original_format, const size_t &old_dims_size, std::string &reshape_type) {
  int32_t default_key = GenerateReshapeTypeKey(original_format, old_dims_size);
  auto iter = DEFAULT_RESHAPE_TYPE.find(default_key);
  if (iter == DEFAULT_RESHAPE_TYPE.end()) {
    GELOGW("dim size %zu is invalid.", old_dims_size);
    return false;
  }

  reshape_type = iter->second;
  return true;
}

bool IsExpandNecessary(const size_t &old_dims_size, const ge::Format &original_format, const ge::Format &final_format,
                       const std::string &reshape_type, size_t &full_size) {
  /* 1. Check whether the old dim size is full. Full size is not necessary for expand. */
  auto iter_full_size = FULL_SIZE_OF_FORMAT.find(original_format);
  if (iter_full_size == FULL_SIZE_OF_FORMAT.end()) {
    GELOGW("Original Format %u is invalid.", original_format);
    return false;
  } else {
    if (old_dims_size >= iter_full_size->second) {
      return false;
    }
  }
  /* 2. Check whether the final format does not need expanding demension. */
  bool no_need_reshape_flag = reshape_type == RESHAPE_TYPE_FORBIDDEN || final_format == ge::FORMAT_FRACTAL_NZ ||
                              (original_format == ge::FORMAT_ND && final_format == ge::FORMAT_FRACTAL_Z);
  if (no_need_reshape_flag) {
    return false;
  }
  full_size = iter_full_size->second;
  return true;
}

bool IsReshapeTypeValid(const ge::Format &original_format, const size_t &old_dims_size,
                        const std::string &reshape_type) {
  if (reshape_type.empty()) {
    return old_dims_size == 0;
  }
  int32_t pos = -1;
  uint32_t format_key = GenerateFormatKey(original_format);
  uint32_t axis_key = 0;
  for (const char &dim : reshape_type) {
    axis_key = format_key | (static_cast<uint32_t>(dim) & 0xff);
    auto iter = AXIS_INDEX_OF_FORMAT.find(axis_key);
    if (iter == AXIS_INDEX_OF_FORMAT.end()) {
      return false;
    }
    if (iter->second > pos) {
      pos = iter->second;
    } else {
      return false;
    }
  }

  return true;
}

void ExpandByReshapeType(ge::GeShape &shape, const ge::Format &original_format,
                         const size_t &old_dims_size, const size_t &full_size, const std::string &reshape_type) {
  GELOGD("Expand tensor by reshape type %s.", reshape_type.c_str());
  if (reshape_type == "CN") {
    /* If the reshape type is CN, we will consider the original format is HWCN. */
    if (old_dims_size < DIMENSION_NUM_TWO) {
      GELOGW("old dims size %zu is less than 2. Reshape type is %s.", old_dims_size, reshape_type.c_str());
      return;
    }
    int64_t dim_0 = shape.GetDim(0);
    int64_t dim_1 = shape.GetDim(1);
    shape.SetDimNum(4);
    shape.SetDim(0, 1);
    shape.SetDim(1, 1);
    shape.SetDim(2, dim_0);
    shape.SetDim(3, dim_1);
    /* In this case the final format must be HWCN, we just return true */
    return;
  } else {
    /* Build a array with all 1 of full size. Then we will substitute some of the 1 with the original axis value. */
    for (size_t i = old_dims_size; i < full_size; i++) {
      shape.AppendDim(1);
    }
    if (reshape_type.empty() || old_dims_size == 0) {
      return;
    }

    uint32_t format_key = GenerateFormatKey(original_format);
    uint32_t axis_key = 0;
    for (int32_t i = static_cast<int32_t>(old_dims_size) - 1; i >= 0; i--) {
      axis_key = format_key | (static_cast<uint32_t>(reshape_type.at(i)) & 0xff);
      auto iter_axis_index = AXIS_INDEX_OF_FORMAT.find(axis_key);
      if (iter_axis_index == AXIS_INDEX_OF_FORMAT.end()) {
        continue;
      }
      if (iter_axis_index->second == i) {
        continue;
      }
      shape.SetDim(iter_axis_index->second, shape.GetDim(i));
      shape.SetDim(i, 1);
    }
  }
}

bool ExpandDimension(const std::string &op_type, const ge::Format &original_format, const ge::Format &final_format,
                     const uint32_t &tensor_index, const std::string &reshape_type, ge::GeShape &shape) {
  /* 1. Check expanding necessary. */
  size_t full_size = 0;
  size_t old_dims_size = shape.GetDimNum();
  if (!IsExpandNecessary(old_dims_size, original_format, final_format, reshape_type, full_size)) {
    return true;
  }

  /* 2. Check whether the reshape type is consistent with the original format.
   * If not consistent, just return and report a warning. */
  std::string valid_reshape_type = reshape_type;
  if (!IsReshapeTypeValid(original_format, old_dims_size, reshape_type)) {
    if (!GetDefaultReshapeType(original_format, old_dims_size, valid_reshape_type)) {
      return true;
    }
  }

  /* 3. Check whether the dimension of original shape is less than or equal to
   * the length of reshape type. If the dimension of original shape if larger,
   * we cannot find suitable posotion for all axis in original shape and we just return. */
  if (old_dims_size > valid_reshape_type.length()) {
    GELOGW("Dimension %zu of tensor %u of %s is larger than the length of reshape type which is %zu.",
           old_dims_size, tensor_index, op_type.c_str(), valid_reshape_type.length());
    return true;
  }

  /* 4. Expand dimension. */
  ExpandByReshapeType(shape, original_format, old_dims_size, full_size, valid_reshape_type);
  return true;
}

bool ExpandRangeDimension(const std::string &op_type, const ge::Format &original_format,
    const ge::Format &final_format, const uint32_t &tensor_index, const std::string &reshape_type,
    std::vector<std::pair<int64_t, int64_t>> &ranges) {
  std::vector<int64_t> range_upper;
  std::vector<int64_t> range_low;
  for (auto &i : ranges) {
    range_low.emplace_back(i.first);
    range_upper.emplace_back(i.second);
  }

  ge::GeShape shape_low(range_low);
  ge::GeShape shape_upper(range_upper);
  bool res = ExpandDimension(op_type, original_format, final_format, tensor_index, reshape_type, shape_low) &&
      ExpandDimension(op_type, original_format, final_format, tensor_index, reshape_type, shape_upper);
  if (!res || (shape_low.GetDimNum() != shape_upper.GetDimNum())) {
    return false;
  }
  ranges.clear();
  for (size_t idx = 0; idx < shape_low.GetDimNum(); ++idx) {
    ranges.emplace_back(std::pair<int64_t, int64_t>(shape_low.GetDim(idx), shape_upper.GetDim(idx)));
  }
  return res;
}
} // namespace transformer
