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

#include "common/formats/format_transfers/format_transfer_nhwc_nc1hwc0.h"

#include <securec.h>
#include <memory>

#include "common/formats/utils/formats_definitions.h"
#include "common/formats/utils/formats_trans_utils.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/debug/log.h"
#include "graph/utils/type_utils.h"

namespace ge {
namespace formats {
namespace {
bool CheckDataTypeSupported(const DataType &data_type) { return GetSizeByDataType(data_type) > 0; }

Status TransShapeNhwcToNc1hwc0(const std::vector<int64_t> &src_shape, DataType data_type,
                               std::vector<int64_t> &dst_shape) {
  int64_t c0 = GetCubeSizeByDataType(data_type);
  if (c0 <= 0) {
    GELOGE(ACL_ERROR_GE_DATATYPE_INVALID, "[Get][Cube]Failed, the data type %s is invalid",
           TypeUtils::DataTypeToSerialString(data_type).c_str());
    REPORT_CALL_ERROR("E19999", "Failed to get cube size, the data type %s is invalid",
                      TypeUtils::DataTypeToSerialString(data_type).c_str());
    return ACL_ERROR_GE_DATATYPE_INVALID;
  }
  dst_shape.clear();
  dst_shape.push_back(src_shape.at(kNhwcN));
  dst_shape.push_back(Ceil(src_shape.at(kNhwcC), c0));
  dst_shape.push_back(src_shape.at(kNhwcH));
  dst_shape.push_back(src_shape.at(kNhwcW));
  dst_shape.push_back(c0);
  if (!CheckShapeValid(dst_shape, kNc1hwc0DimsNum)) {
    GELOGE(ACL_ERROR_GE_SHAPE_INVALID, "[Check][Shape]Value is invalid, dst shape %s",
           ShapeToString(dst_shape).c_str());
    REPORT_CALL_ERROR("E19999", "Dst shape %s check invalid",
                      ShapeToString(dst_shape).c_str());
    return ACL_ERROR_GE_SHAPE_INVALID;
  }
  return SUCCESS;
}

Status CheckArgsForNhwcToNc1hwc0(const TransArgs &args) {
  if (args.src_format != FORMAT_NHWC || args.dst_format != FORMAT_NC1HWC0) {
    std::string error = "Dose not support trans format from " +
        FmtToStr(TypeUtils::FormatToSerialString(args.src_format)) + " to " +
        FmtToStr(TypeUtils::FormatToSerialString(args.dst_format));
    GE_ERRORLOG_AND_ERRORMSG(ACL_ERROR_GE_FORMAT_INVALID, error.c_str());
    return ACL_ERROR_GE_FORMAT_INVALID;
  }
  if (!CheckDataTypeSupported(args.src_data_type)) {
    GELOGE(ACL_ERROR_GE_DATATYPE_INVALID, "[Check][DataType]Failed from NHWC to NC1HWC0, "
           "invalid data type %s",
           TypeUtils::DataTypeToSerialString(args.src_data_type).c_str());
    REPORT_INNER_ERROR("E19999", "Failed to trans shape from NHWC to NC1HWC0, invalid data type %s",
                       TypeUtils::DataTypeToSerialString(args.src_data_type).c_str());
    return ACL_ERROR_GE_DATATYPE_INVALID;
  }
  if (!CheckShapeValid(args.src_shape, kNhwcDimsNum)) {
    GELOGE(ACL_ERROR_GE_SHAPE_INVALID, "[Check][Shape]Value is invalid, src shape %s",
           ShapeToString(args.src_shape).c_str());
    REPORT_CALL_ERROR("E19999", "Src shape %s check invalid",
                      ShapeToString(args.src_shape).c_str());
    return ACL_ERROR_GE_SHAPE_INVALID;
  }
  if (!CheckShapeValid(args.dst_shape, kNc1hwc0DimsNum)) {
    GELOGE(ACL_ERROR_GE_SHAPE_INVALID, "[Check][Shape]Value is invalid, dst shape %s",
           ShapeToString(args.dst_shape).c_str());
    REPORT_CALL_ERROR("E19999", "Dst shape %s check valid",
                      ShapeToString(args.dst_shape).c_str());
    return ACL_ERROR_GE_SHAPE_INVALID;
  }
  std::vector<int64_t> expect_dst_shape;
  auto ret = TransShapeNhwcToNc1hwc0(args.src_shape, args.src_data_type, expect_dst_shape);
  if (ret != SUCCESS) {
    return ret;
  }
  if (args.dst_shape != expect_dst_shape) {
    GELOGE(ACL_ERROR_GE_SHAPE_INVALID,
           "[Trans][Format]Failed , the src shape %s and dst shape %s are not compatible. "
           "expect dst shape %s",
           ShapeToString(args.src_shape).c_str(), ShapeToString(args.dst_shape).c_str(),
           ShapeToString(expect_dst_shape).c_str());
    REPORT_CALL_ERROR("E19999",  "Failed to trans format, the src shape %s and "
                      "dst shape %s are not compatible. expect dst shape %s",
                      ShapeToString(args.src_shape).c_str(), ShapeToString(args.dst_shape).c_str(),
                      ShapeToString(expect_dst_shape).c_str());
    return ACL_ERROR_GE_SHAPE_INVALID;
  }

  return SUCCESS;
}

Status GetDstDataAfterTrans(const TransArgs &args, TransResult &result, const int size, const int64_t total_size) {
  std::shared_ptr<uint8_t> dst(new (std::nothrow) uint8_t[total_size], std::default_delete<uint8_t[]>());
  if (dst == nullptr) {
    GELOGE(ACL_ERROR_GE_MEMORY_ALLOCATION, "[Allcoate][Memory]Failed, memory for dst buf %ld, "
           "shape %s when trans format from %s to %s",
           total_size, ShapeToString(args.dst_shape).c_str(),
           TypeUtils::FormatToSerialString(args.src_format).c_str(),
           TypeUtils::FormatToSerialString(args.dst_format).c_str());
    REPORT_CALL_ERROR("E19999", "Failed to alloc the memory for dst buf %ld, "
                      "shape %s when trans format from %s to %s",
                      total_size, ShapeToString(args.dst_shape).c_str(),
                      TypeUtils::FormatToSerialString(args.src_format).c_str(),
                      TypeUtils::FormatToSerialString(args.dst_format).c_str());
    return ACL_ERROR_GE_MEMORY_ALLOCATION;
  }

  auto n = args.src_shape.at(kNhwcN);
  auto h = args.src_shape.at(kNhwcH);
  auto w = args.src_shape.at(kNhwcW);
  auto c = args.src_shape.at(kNhwcC);
  auto c1 = args.dst_shape.at(kNc1hwc0C1);
  auto c0 = args.dst_shape.at(kNc1hwc0C0);
  int64_t wc = w * c;
  int64_t hwc = h * wc;
  int64_t wc0 = w * c0;
  int64_t hwc0 = h * wc0;
  int64_t c1hwc0 = c1 * hwc0;

  for (int64_t n_idx = 0; n_idx < n; n_idx++) {
    int64_t n_head_addr = n_idx * c1hwc0;
    for (int64_t c1_idx = 0; c1_idx < c1; c1_idx++) {
      int64_t c1_head_addr = n_head_addr + c1_idx * hwc0;
      for (int64_t h_idx = 0; h_idx < h; h_idx++) {
        int64_t h_head_addr = c1_head_addr + h_idx * wc0;
        for (int64_t w_idx = 0; w_idx < w; w_idx++) {
          int64_t w_head_addr = h_head_addr + w_idx * c0;
          for (int64_t c0_idx = 0; c0_idx < c0; c0_idx++) {
            int64_t dst_idx = c0_idx + w_head_addr;
            int64_t dst_offset = dst_idx * size;
            auto protected_size = total_size - dst_offset < static_cast<int64_t>(SECUREC_MEM_MAX_LEN)
                                      ? total_size - dst_offset
                                      : static_cast<int64_t>(SECUREC_MEM_MAX_LEN);
            GE_CHECK_GE(protected_size, 0);
            int64_t c_idx = c0_idx + c1_idx * c0;
            int64_t src_idx = n_idx * hwc + h_idx * wc + w_idx * c + c_idx;
            auto src_offset = src_idx * size;

            if (c_idx < c) {
              auto ret = memcpy_s(dst.get() + dst_offset, protected_size, args.data + src_offset, size);
              if (ret != EOK) {
                GELOGE(ACL_ERROR_GE_MEMORY_OPERATE_FAILED,
                       "[Operate][Memory]Failed to copy data from NHWC[%ld, %ld, %ld, %ld] "
                       "offset %ld to NC1HWC0[%ld, %ld, %ld, %ld, %ld] offset %ld err-code %d",
                       n_idx, h_idx, w_idx, c_idx, src_offset,
                       n_idx, c1_idx, h_idx, w_idx, c0_idx, dst_offset, ret);
                REPORT_CALL_ERROR("E19999", "Failed to copy data from NHWC[%ld, %ld, %ld, %ld] "
                                  "offset %ld to "
                                  "NC1HWC0[%ld, %ld, %ld, %ld, %ld] offset %ld err-code %d",
                                  n_idx, h_idx, w_idx, c_idx, src_offset,
                                  n_idx, c1_idx, h_idx, w_idx, c0_idx, dst_offset, ret);
                return ACL_ERROR_GE_MEMORY_OPERATE_FAILED;
              }
            } else {
              auto ret = memset_s(dst.get() + dst_offset, protected_size, 0, size);
              if (ret != EOK) {
                GELOGE(ACL_ERROR_GE_MEMORY_OPERATE_FAILED,
                       "[Operate][Memory]Failed to set 0 to "
                       "NC1HWC0[%ld, %ld, %ld, %ld, %ld] offset %ld base err-code %d",
                       n_idx, c1_idx, h_idx, w_idx, c0_idx, dst_offset, ret);
                REPORT_CALL_ERROR("E19999", "Failed to set 0 to "
                                  "NC1HWC0[%ld, %ld, %ld, %ld, %ld] offset %ld base err-code %d",
                                  n_idx, c1_idx, h_idx, w_idx, c0_idx, dst_offset, ret);
                return ACL_ERROR_GE_MEMORY_OPERATE_FAILED;
              }
            }
          }
        }
      }
    }
  }
  result.data = dst;
  result.length = static_cast<size_t>(total_size);
  return SUCCESS;
}
}  // namespace

Status FormatTransferNhwcNc1hwc0::TransFormat(const TransArgs &args, TransResult &result) {
  Status ret = CheckArgsForNhwcToNc1hwc0(args);
  if (ret != SUCCESS) {
    return ret;
  }
  int size = GetSizeByDataType(args.src_data_type);
  auto total_size = GetItemNumByShape(args.dst_shape) * size;
  if (total_size <= 0) {
    int64_t src_size = GetItemNumByShape(args.src_shape);
    if (total_size == 0 && src_size == 0) {
      result.length = static_cast<size_t>(total_size);
      return SUCCESS;
    }

    GELOGE(ACL_ERROR_GE_DATATYPE_INVALID, "[Get][ShapeSize]Failed, "
           "total size %ld from dst shape %s, src shape %s", total_size,
           ShapeToString(args.dst_shape).c_str(), ShapeToString(args.src_shape).c_str());
    REPORT_CALL_ERROR("E19999", "[Get][Shape]Failed, total size %ld from "
                      "dst shape %s, src shape %s", total_size,
                      ShapeToString(args.dst_shape).c_str(), ShapeToString(args.src_shape).c_str());
    return ACL_ERROR_GE_DATATYPE_INVALID;
  }
  GELOGD("Begin to trans format from NHWC to NC1HWC0, src shape %s, data type %s, dst shape %s, memory size %ld",
         ShapeToString(args.src_shape).c_str(), TypeUtils::DataTypeToSerialString(args.src_data_type).c_str(),
         ShapeToString(args.dst_shape).c_str(), total_size);

  ret = GetDstDataAfterTrans(args, result, size, total_size);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Get][Data]Failed, after trans, src shape %s, data type %s, "
           "dst shape %s, memory size %ld, error_code %u",
           ShapeToString(args.src_shape).c_str(),
           TypeUtils::DataTypeToSerialString(args.src_data_type).c_str(),
           ShapeToString(args.dst_shape).c_str(), total_size, ret);
    REPORT_CALL_ERROR("E19999", "Failed to get data after trans, src shape %s, data type %s, "
                      "dst shape %s, memory size %ld, error_code %u",
                      ShapeToString(args.src_shape).c_str(),
                      TypeUtils::DataTypeToSerialString(args.src_data_type).c_str(),
                      ShapeToString(args.dst_shape).c_str(), total_size, ret);
    return ret;
  }
  return SUCCESS;
}

Status FormatTransferNhwcNc1hwc0::TransShape(Format src_format, const std::vector<int64_t> &src_shape,
                                             DataType data_type, Format dst_format, std::vector<int64_t> &dst_shape) {
  if (src_format == FORMAT_NHWC && CheckDataTypeSupported(data_type)) {
    if (!CheckShapeValid(src_shape, kNhwcDimsNum)) {
      GELOGE(ACL_ERROR_GE_SHAPE_INVALID, "[Check][Shape]Value is invalid, src shape %s",
             ShapeToString(src_shape).c_str());
      REPORT_CALL_ERROR("E19999", "Src shape %s check invalid",
                        ShapeToString(src_shape).c_str());
      return ACL_ERROR_GE_SHAPE_INVALID;
    }
    return TransShapeNhwcToNc1hwc0(src_shape, data_type, dst_shape);
  } else if (src_format != FORMAT_NHWC) {
    return ACL_ERROR_GE_FORMAT_INVALID;
  } else {
    return ACL_ERROR_GE_DATATYPE_INVALID;
  }
}

REGISTER_FORMAT_TRANSFER(FormatTransferNhwcNc1hwc0, FORMAT_NHWC, FORMAT_NC1HWC0)
}  // namespace formats
}  // namespace ge
