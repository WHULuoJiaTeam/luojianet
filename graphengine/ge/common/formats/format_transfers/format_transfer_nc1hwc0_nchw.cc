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

#include "common/formats/format_transfers/format_transfer_nc1hwc0_nchw.h"

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

Status CheckArgsForNc1hwc0ToNchw(const TransArgs &args) {
  auto src_shape = args.src_shape;
  auto dst_shape = args.dst_shape;
  if (args.src_format != FORMAT_NC1HWC0 || args.dst_format != FORMAT_NCHW) {
    std::string error = "Dose not support trans format from " +
        FmtToStr(TypeUtils::FormatToSerialString(args.src_format)) + " to " +
        FmtToStr(TypeUtils::FormatToSerialString(args.dst_format));
    GE_ERRORLOG_AND_ERRORMSG(ACL_ERROR_GE_FORMAT_INVALID, error.c_str());
    return ACL_ERROR_GE_FORMAT_INVALID;
  }
  if (!CheckDataTypeSupported(args.src_data_type)) {
    GELOGE(ACL_ERROR_GE_DATATYPE_INVALID, "[Check][DataType]Failed, shape from NC1HWC0 to NCHW, "
           "invalid data type %s",
           TypeUtils::DataTypeToSerialString(args.src_data_type).c_str());
    REPORT_INNER_ERROR("E19999", "Failed to trans shape from NC1HWC0 to NCHW, invalid data type %s",
                       TypeUtils::DataTypeToSerialString(args.src_data_type).c_str());
    return ACL_ERROR_GE_DATATYPE_INVALID;
  }
  if (!CheckShapeValid(args.src_shape, kNc1hwc0DimsNum)) {
    GELOGE(ACL_ERROR_GE_SHAPE_INVALID, "[Check][Shape]Value is invalid, src shape %s",
           ShapeToString(args.src_shape).c_str());
    REPORT_CALL_ERROR("E19999", "Src shape %s check invalid",
                      ShapeToString(args.src_shape).c_str());
    return ACL_ERROR_GE_SHAPE_INVALID;
  }
  if (!CheckShapeValid(args.dst_shape, kNchwDimsNum)) {
    GELOGE(ACL_ERROR_GE_SHAPE_INVALID, "[Check][Shape]Value is invalid, dst shape %s",
           ShapeToString(args.dst_shape).c_str());
    REPORT_CALL_ERROR("E19999", "Dst shape %s check invalid",
                      ShapeToString(args.dst_shape).c_str());
    return ACL_ERROR_GE_SHAPE_INVALID;
  }
  int64_t c0 = GetCubeSizeByDataType(args.src_data_type);
  if (c0 <= 0) {
    GELOGE(ACL_ERROR_GE_DATATYPE_INVALID, "[Get][Cube]Failed, the data type %s is invalid",
           TypeUtils::DataTypeToSerialString(args.src_data_type).c_str());
    REPORT_CALL_ERROR("E19999", "Failed to get cube size, the data tyep %s is invalid",
                      TypeUtils::DataTypeToSerialString(args.src_data_type).c_str());
    return ACL_ERROR_GE_DATATYPE_INVALID;
  }
  if (src_shape.at(kNc1hwc0H) != dst_shape.at(kNchwH) || src_shape.at(kNc1hwc0W) != dst_shape.at(kNchwW) ||
      src_shape.at(kNc1hwc0N) != dst_shape.at(kNchwN) || src_shape.at(kNc1hwc0C0) != c0 ||
      src_shape.at(kNc1hwc0C1) != (Ceil(dst_shape.at(kNchwC), c0))) {
    GELOGE(ACL_ERROR_GE_SHAPE_INVALID, "[Check][Shape]Failed to check relationship between "
           "src shape %s and dst shape %s",
           ShapeToString(src_shape).c_str(), ShapeToString(dst_shape).c_str());
    REPORT_INNER_ERROR("E19999",  "Failed to check relationship between src shape %s "
                       "and dst shape %s",
                       ShapeToString(src_shape).c_str(), ShapeToString(dst_shape).c_str());
    return ACL_ERROR_GE_SHAPE_INVALID;
  }

  return SUCCESS;
}

Status GetDstDataAfterTrans(const TransArgs &args, TransResult &result, const int size, const int64_t total_size) {
  std::shared_ptr<uint8_t> dst(new (std::nothrow) uint8_t[total_size], std::default_delete<uint8_t[]>());
  if (dst == nullptr) {
    GELOGE(ACL_ERROR_GE_MEMORY_ALLOCATION, "[Allocate][DSTMemory]Failed, "
           "memory for dst buf %ld, shape %s when trans format from %s to %s",
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

  auto h = args.src_shape.at(kNc1hwc0H);
  auto w = args.src_shape.at(kNc1hwc0W);
  auto n = args.src_shape.at(kNc1hwc0N);
  auto c1 = args.src_shape.at(kNc1hwc0C1);
  auto c0 = args.src_shape.at(kNc1hwc0C0);
  auto c = args.dst_shape.at(kNchwC);
  int64_t hw = h * w;
  int64_t chw = c * hw;
  int64_t wc0 = w * c0;
  int64_t hwc0 = h * wc0;
  int64_t c1hwc0 = c1 * hwc0;

  for (int64_t n_idx = 0; n_idx < n; n_idx++) {
    int64_t n_head_addr = n_idx * chw;
    for (int64_t c_idx = 0; c_idx < c; c_idx++) {
      int64_t c_head_addr = n_head_addr + c_idx * hw;
      for (int64_t h_idx = 0; h_idx < h; h_idx++) {
        int64_t h_head_addr = c_head_addr + h_idx * w;
        for (int64_t w_idx = 0; w_idx < w; w_idx++) {
          int64_t dst_idx = h_head_addr + w_idx;
          int64_t c1_idx = c_idx / c0;
          int64_t c0_idx = c_idx % c0;
          int64_t src_idx = n_idx * c1hwc0 + c1_idx * hwc0 + h_idx * wc0 + w_idx * c0 + c0_idx;
          auto src_offset = src_idx * size;
          auto dst_offset = dst_idx * size;
          auto protected_size = total_size - dst_offset < static_cast<int64_t>(SECUREC_MEM_MAX_LEN)
                                    ? total_size - dst_offset
                                    : static_cast<int64_t>(SECUREC_MEM_MAX_LEN);
          GE_CHECK_GE(protected_size, 0);
          auto ret = memcpy_s(dst.get() + dst_offset, static_cast<size_t>(protected_size), args.data + src_offset,
                              static_cast<size_t>(size));
          if (ret != EOK) {
            GELOGE(ACL_ERROR_GE_MEMORY_OPERATE_FAILED, "[Copy][Data]Failed, data from "
                   "NC1HWC0[%ld, %ld, %ld, %ld, %ld] "
                   "src offset %ld to NCHW[%ld, %ld, %ld, %ld], dst offset %ld, err-code %d",
                   n_idx, c1_idx, h_idx, w_idx, c0_idx, src_offset, n_idx,
                   c_idx, h_idx, w_idx, dst_offset, ret);
            REPORT_CALL_ERROR("E19999", "Failed to copy data from NC1HWC0[%ld, %ld, %ld, %ld, %ld] "
                              "src offset %ld to NCHW[%ld, %ld, %ld, %ld], dst offset %ld, "
                              "err-code %d",
                              n_idx, c1_idx, h_idx, w_idx, c0_idx, src_offset, n_idx,
                              c_idx, h_idx, w_idx, dst_offset, ret);
            return ACL_ERROR_GE_MEMORY_OPERATE_FAILED;
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

Status FormatTransferNc1hwc0Nchw::TransFormat(const TransArgs &args, TransResult &result) {
  Status ret = CheckArgsForNc1hwc0ToNchw(args);
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

    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "[Get][ShapeSize]Failed, total size %ld from dst shape %s, "
           "src shape %s", total_size,
           ShapeToString(args.dst_shape).c_str(), ShapeToString(args.src_shape).c_str());
    REPORT_CALL_ERROR("E19999", "Failed to get total size %ld from dst shape %s, src shape %s",
                      total_size,
                      ShapeToString(args.dst_shape).c_str(), ShapeToString(args.src_shape).c_str());
    return ACL_ERROR_GE_PARAM_INVALID;
  }
  GELOGD("Begin to trans format from NC1HWC0 to NCHW, src shape %s, data type %s, dst shape %s, memory size %ld",
         ShapeToString(args.src_shape).c_str(), TypeUtils::DataTypeToSerialString(args.src_data_type).c_str(),
         ShapeToString(args.dst_shape).c_str(), total_size);
  ret = GetDstDataAfterTrans(args, result, size, total_size);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Get][Data]Failed, after trans, src shape %s, data type %s, "
           "dst shape %s, memory size %ld",
           ShapeToString(args.src_shape).c_str(),
           TypeUtils::DataTypeToSerialString(args.src_data_type).c_str(),
           ShapeToString(args.dst_shape).c_str(), total_size);
    REPORT_CALL_ERROR("E19999", "Failed to get data after trans, src shape %s, data type %s, "
                      "dst shape %s, memory size %ld",
                      ShapeToString(args.src_shape).c_str(),
                      TypeUtils::DataTypeToSerialString(args.src_data_type).c_str(),
                      ShapeToString(args.dst_shape).c_str(), total_size);
    return ret;
  }
  return SUCCESS;
}

Status FormatTransferNc1hwc0Nchw::TransShape(Format src_format, const std::vector<int64_t> &src_shape,
                                             DataType data_type, Format dst_format, std::vector<int64_t> &dst_shape) {
  GELOGD("The shape derivation from NC1HWC0 to NCHW is not unique. Trans shape in this direction is not supported");
  return ACL_ERROR_GE_FORMAT_INVALID;
}

REGISTER_FORMAT_TRANSFER(FormatTransferNc1hwc0Nchw, FORMAT_NC1HWC0, FORMAT_NCHW)
}  // namespace formats
}  // namespace ge
