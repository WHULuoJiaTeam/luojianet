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

#include "common/formats/format_transfers/format_transfer_fractal_zz.h"

#include <securec.h>
#include <memory>

#include "common/formats/utils/formats_definitions.h"
#include "common/formats/utils/formats_trans_utils.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/debug/log.h"
#include "framework/common/types.h"
#include "graph/utils/type_utils.h"

namespace ge {
namespace formats {
namespace {
const int kDimSize4D = 4;

const size_t kSingleDim = 1;

const size_t kNdDimIndexN = 0;
const size_t kNdDimIndexH = 1;
const size_t kNdDimIndexW = 2;

const size_t kDimDValueBNdFZz = 2;  // dim d-value between Nd and FractalZz

const size_t kNdDimCountBackwardsW = 1;
const size_t kNdDimCountBackwardsWH = 2;

const size_t kFZzDimCountBackwardsW0 = 1;
const size_t kFZzDimCountBackwardsW0H0 = 2;
const size_t kFZzDimCountBackwardsW0H0W1 = 3;
const size_t kFZzDimCountBackwardsW0H0W1H1 = 4;
bool IsDataTypeSupport(DataType d_type) { return GetSizeByDataType(d_type) > 0; }

using ShapeVector = std::vector<int64_t>;
bool CheckShape(Format format, const ShapeVector &shape) {
  switch (format) {
    case FORMAT_ND:
      return IsShapeValid(shape);
    case FORMAT_NCHW:
    case FORMAT_NHWC:
      return CheckShapeValid(shape, kDimSize4D);
    default:
      std::string error = "Trans format between " + FmtToStr(TypeUtils::FormatToSerialString(format)) +
                          " and FORMAT_FRACTAL_ZZ is not supported.";
      GE_ERRORLOG_AND_ERRORMSG(ACL_ERROR_GE_FORMAT_INVALID, error.c_str());
      return false;
  }
}

/**
 * After the conversion to two-dimensional matrix, the memory arrangement is small z and large Z.
 * @src_shape: N*H*W
 * @dst_shape: N*H1*W1*H0*w0
 * @return
 */
Status TransShapeToFracZz(const ShapeVector &src_shape, DataType data_type, ShapeVector &dst_shape,
                          ShapeVector &hw_shape) {
  dst_shape.clear();
  hw_shape.clear();
  auto w0 = GetCubeSizeByDataType(data_type);
  auto h0 = GetCubeSizeByDataType(data_type);
  switch (src_shape.size()) {
    case kSingleDim:
      dst_shape.push_back(DIM_DEFAULT_VALUE);
      dst_shape.push_back(Ceil(src_shape[kNdDimIndexN], w0));
      dst_shape.push_back(h0);
      dst_shape.push_back(w0);
      hw_shape.push_back(DIM_DEFAULT_VALUE);
      hw_shape.push_back(DIM_DEFAULT_VALUE);
      hw_shape.push_back(src_shape[kNdDimIndexN]);
      if (!IsShapeValid(dst_shape)) {
        GELOGE(ACL_ERROR_GE_SHAPE_INVALID, "[Check][DSTShape]Failed, dst shape %s",
               ShapeToString(dst_shape).c_str());
        REPORT_CALL_ERROR("E19999", "Failed to check dst shape %s",
                          ShapeToString(dst_shape).c_str());
        return ACL_ERROR_GE_SHAPE_INVALID;
      }
      return SUCCESS;
    default:
      auto size = src_shape.size();
      int64_t times = 1;
      for (size_t i = 0; i != size - kDimDValueBNdFZz; i++) {
        dst_shape.push_back(src_shape[i]);
        times *= src_shape[i];
      }
      dst_shape.push_back(Ceil(src_shape[size - kNdDimCountBackwardsWH], h0));
      dst_shape.push_back(Ceil(src_shape[size - kNdDimCountBackwardsW], w0));
      dst_shape.push_back(h0);
      dst_shape.push_back(w0);
      hw_shape.push_back(times);
      hw_shape.push_back(src_shape[size - kNdDimCountBackwardsWH]);
      hw_shape.push_back(src_shape[size - kNdDimCountBackwardsW]);
      if (!IsShapeValid(dst_shape)) {
        GELOGE(ACL_ERROR_GE_SHAPE_INVALID, "[Check][DSTShape]Failed, dst shape %s",
               ShapeToString(dst_shape).c_str());
        REPORT_CALL_ERROR("E19999", "Failed to check dst shape %s",
                          ShapeToString(dst_shape).c_str());
        return ACL_ERROR_GE_SHAPE_INVALID;
      }
      return SUCCESS;
  }
}

Status CheckShapeRelation(const TransArgs &args, ShapeVector &hw_shape) {
  ShapeVector expect_src_shape;
  auto ret = TransShapeToFracZz(args.dst_shape, args.src_data_type, expect_src_shape, hw_shape);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Trans][ShapeToFracZz] Failed from %s to %s, shape %s to %s, data type %s",
           TypeUtils::FormatToSerialString(args.dst_format).c_str(),
           TypeUtils::FormatToSerialString(args.src_format).c_str(),
           ShapeToString(args.dst_shape).c_str(),
           ShapeToString(args.src_shape).c_str(),
           TypeUtils::DataTypeToSerialString(args.src_data_type).c_str());
    REPORT_CALL_ERROR("E19999", "Failed to trans shape from %s to %s, shape %s to %s, data type %s",
                      TypeUtils::FormatToSerialString(args.dst_format).c_str(),
                      TypeUtils::FormatToSerialString(args.src_format).c_str(),
                      ShapeToString(args.dst_shape).c_str(),
                      ShapeToString(args.src_shape).c_str(),
                      TypeUtils::DataTypeToSerialString(args.src_data_type).c_str());
    return ret;
  }
  if (!IsTransShapeSrcCorrect(args, expect_src_shape)) {
    return ACL_ERROR_GE_SHAPE_INVALID;
  }
  return SUCCESS;
}

Status TransFormatFromNdToFracZz(const TransArgs &args, TransResult &result, const ShapeVector &hw_shape) {
  int size = GetSizeByDataType(args.src_data_type);
  int64_t dst_size = GetItemNumByShape(args.dst_shape) * size;
  if (dst_size == 0) {
    result.length = static_cast<size_t>(dst_size);
    return SUCCESS;
  }

  std::shared_ptr<uint8_t> dst(new (std::nothrow) uint8_t[dst_size](), std::default_delete<uint8_t[]>());
  if (dst == nullptr) {
    GELOGE(ACL_ERROR_GE_MEMORY_ALLOCATION, "[Allocate][DSTMemory]Failed to allcoate memory "
           "for dst buf %ld when trans format from %s to %s",
           dst_size, TypeUtils::FormatToSerialString(args.src_format).c_str(),
           TypeUtils::FormatToSerialString(args.dst_format).c_str());
    REPORT_CALL_ERROR("E19999", "Failed to allcoate memory for dst buf %ld "
                      "when trans format from %s to %s",
                      dst_size, TypeUtils::FormatToSerialString(args.src_format).c_str(),
                      TypeUtils::FormatToSerialString(args.dst_format).c_str());
    return ACL_ERROR_GE_MEMORY_ALLOCATION;
  }
  // The src&dst_shape can be written as times*H*W & times*H1*W1*H0*W0, respectively. dst_shape_size >= kDimNum4D
  auto times = hw_shape.at(kNdDimIndexN);
  auto h = hw_shape.at(kNdDimIndexH);
  auto w = hw_shape.at(kNdDimIndexW);
  auto hw = h * w;

  auto shape_size = args.dst_shape.size();
  auto h1 = args.dst_shape[shape_size - kFZzDimCountBackwardsW0H0W1H1];
  auto w1 = args.dst_shape[shape_size - kFZzDimCountBackwardsW0H0W1];
  auto h0 = args.dst_shape[shape_size - kFZzDimCountBackwardsW0H0];
  auto w0 = args.dst_shape[shape_size - kFZzDimCountBackwardsW0];
  auto h0w0 = h0 * w0;
  auto w1h0w0 = w1 * h0w0;
  auto h1w1h0w0 = h1 * w1h0w0;
  auto num_w1 = w / w0;

  for (int64_t times_idx = 0; times_idx < times; times_idx++) {
    auto times_head = times_idx * h1w1h0w0;
    auto src_times_head = times_idx * hw;
    for (int64_t h1_idx = 0; h1_idx < h1; h1_idx++) {
      auto h1_head = times_head + h1_idx * w1h0w0;
      auto src_h1_head = h1_idx * h0;
      for (int64_t h0_idx = 0; h0_idx < h0 && h0_idx + src_h1_head < h; h0_idx++) {
        auto h0_head = h1_head + h0_idx * w0;
        auto src_h_head = src_times_head + (src_h1_head + h0_idx) * w;
        for (int64_t w1_idx = 0; w1_idx < num_w1; w1_idx++) {
          auto src_offset = (src_h_head + w1_idx * w0) * size;
          auto dst_offset = (h0_head + w1_idx * h0w0) * size;
          auto protected_size = dst_size - dst_offset < static_cast<int64_t>(SECUREC_MEM_MAX_LEN)
                                  ? dst_size - dst_offset
                                  : static_cast<int64_t>(SECUREC_MEM_MAX_LEN);
          GE_CHECK_GE(protected_size, 0);
          auto ret = memcpy_s(dst.get() + dst_offset, static_cast<size_t>(protected_size), args.data + src_offset,
                              static_cast<size_t>(size * w0));
          if (ret != EOK) {
            GELOGE(ACL_ERROR_GE_MEMORY_OPERATE_FAILED, "[Operate][DSTMemory]Failed at offset %ld, "
                   "error-code %d",
                   dst_offset, ret);
            REPORT_CALL_ERROR("E19999", "Failed to operate dst memory at offset %ld, error-code %d",
                              dst_offset, ret);
            return ACL_ERROR_GE_MEMORY_OPERATE_FAILED;
          }
        }
        auto w1_head = num_w1 * w0;
        auto w0_head = h0_head + num_w1 * h0w0;
        for (int64_t w0_idx = 0; w0_idx + w1_head < w; w0_idx++) {
          auto src_w_idx = w1_head + w0_idx;
          auto src_offset = (src_h_head + src_w_idx) * size;
          auto dst_offset = (w0_head + w0_idx) * size;
          auto protected_size = dst_size - dst_offset < static_cast<int64_t>(SECUREC_MEM_MAX_LEN)
                                  ? dst_size - dst_offset
                                  : static_cast<int64_t>(SECUREC_MEM_MAX_LEN);
          GE_CHECK_GE(protected_size, 0);
          auto ret = memcpy_s(dst.get() + dst_offset, static_cast<size_t>(protected_size), args.data + src_offset,
                              static_cast<size_t>(size));
          if (ret != EOK) {
            GELOGE(ACL_ERROR_GE_MEMORY_OPERATE_FAILED, "[Operate][DSTMemory]Failed at offset %ld, "
                   "error-code %d",
                   dst_offset, ret);
            REPORT_CALL_ERROR("E19999", "Failed to operate dst memory at offset %ld, error-code %d",
                              dst_offset, ret);
            return ACL_ERROR_GE_MEMORY_OPERATE_FAILED;
          }
        }
      }
    }
  }
  result.data = dst;
  result.length = static_cast<size_t>(dst_size);
  return SUCCESS;
}

Status TransFormatFromFracZzToNd(const TransArgs &args, TransResult &result, const ShapeVector &dst_hw_shape) {
  int size = GetSizeByDataType(args.src_data_type);
  int64_t dst_size = GetItemNumByShape(args.dst_shape) * size;
  if (dst_size == 0) {
    result.length = static_cast<size_t>(dst_size);
    return SUCCESS;
  }

  std::shared_ptr<uint8_t> dst(new (std::nothrow) uint8_t[dst_size](), std::default_delete<uint8_t[]>());
  if (dst == nullptr) {
    GELOGE(ACL_ERROR_GE_MEMORY_ALLOCATION, "[Allocate][DSTMemory]Failed to allcoate memory "
           "for dst buf %ld when trans format from %s to %s",
           dst_size, TypeUtils::FormatToSerialString(args.src_format).c_str(),
           TypeUtils::FormatToSerialString(args.dst_format).c_str());
    REPORT_CALL_ERROR("E19999", "Failed to allcoate memory for dst buf %ld "
                      "when trans format from %s to %s",
                      dst_size, TypeUtils::FormatToSerialString(args.src_format).c_str(),
                      TypeUtils::FormatToSerialString(args.dst_format).c_str());
    return ACL_ERROR_GE_MEMORY_ALLOCATION;
  }

  // The src&dst_shape can be written as times*H*W & times*H1*W1*H0*W0, respectively. dst_shape_size >= kDimNum4D
  auto times = dst_hw_shape.at(kNdDimIndexN);
  auto h = dst_hw_shape.at(kNdDimIndexH);
  auto w = dst_hw_shape.at(kNdDimIndexW);
  auto hw = h * w;

  auto shape_size = args.src_shape.size();
  auto h1 = args.src_shape[shape_size - kFZzDimCountBackwardsW0H0W1H1];
  auto w1 = args.src_shape[shape_size - kFZzDimCountBackwardsW0H0W1];
  auto h0 = args.src_shape[shape_size - kFZzDimCountBackwardsW0H0];
  auto w0 = args.src_shape[shape_size - kFZzDimCountBackwardsW0];
  auto h0w0 = h0 * w0;
  auto w1h0w0 = w1 * h0w0;
  auto h1w1h0w0 = h1 * w1h0w0;
  auto num_w1 = w / w0;

  for (int64_t times_idx = 0; times_idx < times; times_idx++) {
    auto times_head = times_idx * h1w1h0w0;
    auto dst_times_head = times_idx * hw;
    for (int64_t h1_idx = 0; h1_idx < h1; h1_idx++) {
      auto h1_head = times_head + h1_idx * w1h0w0;
      auto dst_h1_head = h1_idx * h0;
      for (int64_t h0_idx = 0; h0_idx < h0 && h0_idx + dst_h1_head < h; h0_idx++) {
        auto h0_head = h1_head + h0_idx * w0;
        auto dst_h_head = dst_times_head + (dst_h1_head + h0_idx) * w;
        for (int64_t w1_idx = 0; w1_idx < num_w1; w1_idx++) {
          auto src_offset = (h0_head + w1_idx * h0w0) * size;
          auto dst_offset = (dst_h_head + w1_idx * w0) * size;
          auto protected_size = dst_size - dst_offset < static_cast<int64_t>(SECUREC_MEM_MAX_LEN)
                                  ? dst_size - dst_offset
                                  : static_cast<int64_t>(SECUREC_MEM_MAX_LEN);
          GE_CHECK_GE(protected_size, 0);
          auto ret = memcpy_s(dst.get() + dst_offset, static_cast<size_t>(protected_size), args.data + src_offset,
                              static_cast<size_t>(size * w0));
          if (ret != EOK) {
            GELOGE(ACL_ERROR_GE_MEMORY_OPERATE_FAILED, "[Operate][DSTMemory]Failed at offset %ld, "
                   "error-code %d",
                   dst_offset, ret);
            REPORT_CALL_ERROR("E19999", "Failed to operate dst memory at offset %ld, error-code %d",
                              dst_offset, ret);
            return ACL_ERROR_GE_MEMORY_OPERATE_FAILED;
          }
        }
        auto w1_head = num_w1 * w0;
        auto w0_head = h0_head + num_w1 * h0w0;
        for (int64_t w0_idx = 0; w0_idx + w1_head < w; w0_idx++) {
          auto src_offset = (w0_head + w0_idx) * size;
          auto dst_w_idx = w1_head + w0_idx;
          auto dst_offset = (dst_h_head + dst_w_idx) * size;
          auto protected_size = dst_size - dst_offset < static_cast<int64_t>(SECUREC_MEM_MAX_LEN)
                                  ? dst_size - dst_offset
                                  : static_cast<int64_t>(SECUREC_MEM_MAX_LEN);
          GE_CHECK_GE(protected_size, 0);
          auto ret = memcpy_s(dst.get() + dst_offset, static_cast<size_t>(protected_size), args.data + src_offset,
                              static_cast<size_t>(size));
          if (ret != EOK) {
            GELOGE(ACL_ERROR_GE_MEMORY_OPERATE_FAILED, "[Operate][DSTMemory]Failed at offset %ld, "
                   "error-code %d",
                   dst_offset, ret);
            REPORT_CALL_ERROR("E19999", "Failed to operate dst memory at offset %ld, error-code %d",
                              dst_offset, ret);
            return ACL_ERROR_GE_MEMORY_OPERATE_FAILED;
          }
        }
      }
    }
  }
  result.data = dst;
  result.length = static_cast<size_t>(dst_size);
  return SUCCESS;
}
}  // namespace

Status FormatTransferFractalZz::TransFormat(const TransArgs &args, TransResult &result) {
  if (!IsDataTypeSupport(args.src_data_type)) {
    GELOGE(ACL_ERROR_GE_DATATYPE_INVALID,
           "[Check][Datatype]Failed, not support trans format from %s to %s, "
           "src shape %s, dst shape %s, data type %s",
           TypeUtils::FormatToSerialString(args.src_format).c_str(),
           TypeUtils::FormatToSerialString(args.dst_format).c_str(),
           ShapeToString(args.src_shape).c_str(),
           ShapeToString(args.dst_shape).c_str(),
           TypeUtils::DataTypeToSerialString(args.src_data_type).c_str());
    REPORT_INNER_ERROR("E19999", "Check datatype failed, not support trans format "
                       "from %s to %s, src shape %s, dst shape %s, data type %s",
                       TypeUtils::FormatToSerialString(args.src_format).c_str(),
                       TypeUtils::FormatToSerialString(args.dst_format).c_str(),
                       ShapeToString(args.src_shape).c_str(),
                       ShapeToString(args.dst_shape).c_str(),
                       TypeUtils::DataTypeToSerialString(args.src_data_type).c_str());
    return ACL_ERROR_GE_DATATYPE_INVALID;
  }
  if (!CheckShape(args.src_format, args.src_shape) || !IsShapeValid(args.dst_shape)) {
    GELOGE(ACL_ERROR_GE_SHAPE_INVALID,
           "[Check][Shape]Failed, not support trans format from %s to %s, "
           "src shape %s, dst shape %s, data type %s",
           TypeUtils::FormatToSerialString(args.src_format).c_str(),
           TypeUtils::FormatToSerialString(args.dst_format).c_str(),
           ShapeToString(args.src_shape).c_str(),
           ShapeToString(args.dst_shape).c_str(),
           TypeUtils::DataTypeToSerialString(args.src_data_type).c_str());
    REPORT_CALL_ERROR("E19999",  "Check shape failed, not support trans format from %s to %s, "
                      "src shape %s, dst shape %s, data type %s",
                      TypeUtils::FormatToSerialString(args.src_format).c_str(),
                      TypeUtils::FormatToSerialString(args.dst_format).c_str(),
                      ShapeToString(args.src_shape).c_str(),
                      ShapeToString(args.dst_shape).c_str(),
                      TypeUtils::DataTypeToSerialString(args.src_data_type).c_str());
    return ACL_ERROR_GE_SHAPE_INVALID;
  }
  GELOGD("Begin to trans format from %s to %s, src shape %s, dst shape %s, data type %s",
         TypeUtils::FormatToSerialString(args.src_format).c_str(),
         TypeUtils::FormatToSerialString(args.dst_format).c_str(), ShapeToString(args.src_shape).c_str(),
         ShapeToString(args.dst_shape).c_str(), TypeUtils::DataTypeToSerialString(args.src_data_type).c_str());
  ShapeVector expect_shape;
  ShapeVector hw_shape;
  auto ret = TransShapeToFracZz(args.src_shape, args.src_data_type, expect_shape, hw_shape);
  if (ret != SUCCESS) {
    return ret;
  }
  if (!IsTransShapeDstCorrect(args, expect_shape)) {
    return ACL_ERROR_GE_SHAPE_INVALID;
  }
  return TransFormatFromNdToFracZz(args, result, hw_shape);
}

Status FormatTransferFractalZz::TransShape(Format src_format, const ShapeVector &src_shape, DataType data_type,
                                           Format dst_format, ShapeVector &dst_shape) {
  if (!IsDataTypeSupport(data_type)) {
    GELOGE(ACL_ERROR_GE_DATATYPE_INVALID,
           "[Check][Datatype]Failed, not support trans format from %s to %s, "
           "src shape %s, data type %s",
           TypeUtils::FormatToSerialString(src_format).c_str(),
           TypeUtils::FormatToSerialString(dst_format).c_str(),
           ShapeToString(src_shape).c_str(),
           TypeUtils::DataTypeToSerialString(data_type).c_str());
    REPORT_INNER_ERROR("E19999", "Check datatype failed, not support trans format from %s to %s, "
                       "src shape %s, data type %s",
                       TypeUtils::FormatToSerialString(src_format).c_str(),
                       TypeUtils::FormatToSerialString(dst_format).c_str(),
                       ShapeToString(src_shape).c_str(),
                       TypeUtils::DataTypeToSerialString(data_type).c_str());
    return ACL_ERROR_GE_DATATYPE_INVALID;
  }
  if (!CheckShape(src_format, src_shape)) {
    GELOGE(ACL_ERROR_GE_SHAPE_INVALID,
           "[Check][Shape]Failed, not support trans format from %s to %s, "
           "src shape %s, data type %s",
           TypeUtils::FormatToSerialString(src_format).c_str(),
           TypeUtils::FormatToSerialString(dst_format).c_str(),
           ShapeToString(src_shape).c_str(),
           TypeUtils::DataTypeToSerialString(data_type).c_str());
    REPORT_CALL_ERROR("E19999", "Check shape failed, not support trans format from %s to %s, "
                      "src shape %s, data type %s",
                      TypeUtils::FormatToSerialString(src_format).c_str(),
                      TypeUtils::FormatToSerialString(dst_format).c_str(),
                      ShapeToString(src_shape).c_str(),
                      TypeUtils::DataTypeToSerialString(data_type).c_str());
    return ACL_ERROR_GE_SHAPE_INVALID;
  }
  ShapeVector hw_shape;
  return TransShapeToFracZz(src_shape, data_type, dst_shape, hw_shape);
}

Status FormatTransferFractalZzND::TransFormat(const TransArgs &args, TransResult &result) {
  if (!IsDataTypeSupport(args.src_data_type)) {
    GELOGE(ACL_ERROR_GE_DATATYPE_INVALID,
           "[Check][Datatype]Failed, not support trans format from %s to %s, "
           "src shape %s, dst shape %s, data type %s",
           TypeUtils::FormatToSerialString(args.src_format).c_str(),
           TypeUtils::FormatToSerialString(args.dst_format).c_str(),
           ShapeToString(args.src_shape).c_str(),
           ShapeToString(args.dst_shape).c_str(),
           TypeUtils::DataTypeToSerialString(args.src_data_type).c_str());
    REPORT_INNER_ERROR("E19999", "Check datatype Failed, not support trans format from %s to %s, "
                       "src shape %s, dst shape %s, data type %s",
                       TypeUtils::FormatToSerialString(args.src_format).c_str(),
                       TypeUtils::FormatToSerialString(args.dst_format).c_str(),
                       ShapeToString(args.src_shape).c_str(),
                       ShapeToString(args.dst_shape).c_str(),
                       TypeUtils::DataTypeToSerialString(args.src_data_type).c_str());
    return ACL_ERROR_GE_DATATYPE_INVALID;
  }

  if (!IsShapeValid(args.src_shape) || !CheckShape(args.dst_format, args.dst_shape)) {
    GELOGE(ACL_ERROR_GE_SHAPE_INVALID, "[Check][Shape]Failed, not support trans format "
           "from %s to %s, src shape %s, dst shape %s, data type %s",
           TypeUtils::FormatToSerialString(args.src_format).c_str(),
           TypeUtils::FormatToSerialString(args.dst_format).c_str(),
           ShapeToString(args.src_shape).c_str(),
           ShapeToString(args.dst_shape).c_str(),
           TypeUtils::DataTypeToSerialString(args.src_data_type).c_str());
    REPORT_CALL_ERROR("E19999", "Check shape failed, not support trans format from %s to %s, "
                      "src shape %s, dst shape %s, data type %s",
                      TypeUtils::FormatToSerialString(args.src_format).c_str(),
                      TypeUtils::FormatToSerialString(args.dst_format).c_str(),
                      ShapeToString(args.src_shape).c_str(),
                      ShapeToString(args.dst_shape).c_str(),
                      TypeUtils::DataTypeToSerialString(args.src_data_type).c_str());
    return ACL_ERROR_GE_SHAPE_INVALID;
  }
  GELOGD("Begin to trans format from %s to %s, src shape %s, dst shape %s, data type %s",
         TypeUtils::FormatToSerialString(args.src_format).c_str(),
         TypeUtils::FormatToSerialString(args.dst_format).c_str(), ShapeToString(args.src_shape).c_str(),
         ShapeToString(args.dst_shape).c_str(), TypeUtils::DataTypeToSerialString(args.src_data_type).c_str());

  ShapeVector hw_shape;
  Status ret = CheckShapeRelation(args, hw_shape);
  if (ret != SUCCESS) {
    return ret;
  }
  return TransFormatFromFracZzToNd(args, result, hw_shape);
}

Status FormatTransferFractalZzND::TransShape(Format src_format, const ShapeVector &src_shape, DataType data_type,
                                             Format dst_format, ShapeVector &dst_shape) {
  GELOGD("The shape derivation from %s to %s is not unique. Trans shape is not supported",
         TypeUtils::FormatToSerialString(src_format).c_str(), TypeUtils::FormatToSerialString(dst_format).c_str());
  return ACL_ERROR_GE_FORMAT_INVALID;
}

REGISTER_FORMAT_TRANSFER(FormatTransferFractalZz, FORMAT_ND, FORMAT_FRACTAL_ZZ)
REGISTER_FORMAT_TRANSFER(FormatTransferFractalZz, FORMAT_NCHW, FORMAT_FRACTAL_ZZ)
REGISTER_FORMAT_TRANSFER(FormatTransferFractalZz, FORMAT_NHWC, FORMAT_FRACTAL_ZZ)
REGISTER_FORMAT_TRANSFER(FormatTransferFractalZzND, FORMAT_FRACTAL_ZZ, FORMAT_ND)
REGISTER_FORMAT_TRANSFER(FormatTransferFractalZzND, FORMAT_FRACTAL_ZZ, FORMAT_NCHW)
REGISTER_FORMAT_TRANSFER(FormatTransferFractalZzND, FORMAT_FRACTAL_ZZ, FORMAT_NHWC)
}  // namespace formats
}  // namespace ge
