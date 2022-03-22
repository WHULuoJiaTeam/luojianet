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

#include "common/formats/utils/formats_trans_utils.h"

#include <cstdint>

#include "common/formats/utils/formats_definitions.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/debug/log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "graph/utils/type_utils.h"

namespace ge {
namespace formats {
int64_t GetCubeSizeByDataType(DataType data_type) {
  // Current cube does not support 4 bytes and longer data
  auto size = GetSizeByDataType(data_type);
  if (size <= 0) {
    std::string error = "Failed to get cube size, the data type " +
        FmtToStr(TypeUtils::DataTypeToSerialString(data_type)) + " is invalid";
    GE_WARNINGLOG_AND_ERRORMSG(error.c_str());
    return -1;
  } else if (size == 1) {
    return kCubeSize * 2;  // 32 bytes cube size
  } else {
    return kCubeSize;
  }
}

std::string ShapeToString(const GeShape &shape) {
  return ShapeToString(shape.GetDims());
}

std::string ShapeToString(const std::vector<int64_t> &shape) {
  return JoinToString(shape);
}

std::string RangeToString(const std::vector<std::pair<int64_t, int64_t>> &ranges) {
  bool first = true;
  std::stringstream ss;
  ss << "[";
  for (const auto &range : ranges) {
    if (first) {
      first = false;
    } else {
      ss << ",";
    }
    ss << "{";
    ss << range.first << "," << range.second;
    ss << "}";
  }
  ss << "]";
  return ss.str();
}

int64_t GetItemNumByShape(const std::vector<int64_t> &shape) {
  int64_t num = 1;
  for (auto dim : shape) {
    num *= dim;
  }
  return num;
}

bool CheckShapeValid(const std::vector<int64_t> &shape, const int64_t expect_dims) {
  if (expect_dims <= 0 || shape.size() != static_cast<size_t>(expect_dims)) {
    std::string error = "Invalid shape, dims num " + FmtToStr(shape.size()) +
        ", expect " + FmtToStr(expect_dims);
    GE_WARNINGLOG_AND_ERRORMSG(error.c_str());
    return false;
  }
  return IsShapeValid(shape);
}

bool IsShapeValid(const std::vector<int64_t> &shape) {
  if (shape.empty()) {
    return false;
  }
  int64_t num = 1;
  for (auto dim : shape) {
    if (dim < 0) {
      std::string error = "Invalid negative dims in the shape " +  FmtToStr(ShapeToString(shape));
      GE_WARNINGLOG_AND_ERRORMSG(error.c_str());
      return false;
    }
    if (dim != 0 && kShapeItemNumMAX / dim < num) {
      std::string error = "Shape overflow, the total count should be less than " + FmtToStr(kShapeItemNumMAX);
      GE_WARNINGLOG_AND_ERRORMSG(error.c_str());
      return false;
    }
    num *= dim;
  }
  return true;
}

bool IsShapeEqual(const GeShape &src, const GeShape &dst) {
  if (src.GetDims().size() != dst.GetDims().size()) {
    return false;
  }

  for (size_t i = 0; i < src.GetDims().size(); ++i) {
    if (src.GetDim(i) != dst.GetDim(i)) {
      return false;
    }
  }
  return true;
}

bool IsTransShapeSrcCorrect(const TransArgs &args, std::vector<int64_t> &expect_shape) {
  if (args.src_shape != expect_shape) {
    std::string error = "Failed to trans format from" +
        FmtToStr(TypeUtils::FormatToSerialString(args.src_format)) + " to " +
        FmtToStr(TypeUtils::FormatToSerialString(args.dst_format)) + ", invalid relationship between src shape " +
        FmtToStr(ShapeToString(args.src_shape)) + " and dst " +
        FmtToStr(ShapeToString(args.dst_shape));
    GE_WARNINGLOG_AND_ERRORMSG(error.c_str());
    return false;
  }
  return true;
}

bool IsTransShapeDstCorrect(const TransArgs &args, std::vector<int64_t> &expect_shape) {
  if (!args.dst_shape.empty() && args.dst_shape != expect_shape) {
    std::string error = "Failed to trans format from " +
        FmtToStr(TypeUtils::FormatToSerialString(args.src_format)) + " to " +
        FmtToStr(TypeUtils::FormatToSerialString(args.dst_format)) + ", the dst shape" +
        FmtToStr(ShapeToString(args.dst_shape)) + " is invalid, expect" +
        FmtToStr(ShapeToString(expect_shape));
    GE_WARNINGLOG_AND_ERRORMSG(error.c_str());
    return false;
  }
  return true;
}
}  // namespace formats
}  // namespace ge
