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

#include "common/formats/format_transfers/format_transfer_hwcn_c1hwncoc0.h"

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
bool CheckDataTypeSupported(const DataType &data_type) {
  return (data_type == DT_FLOAT || data_type == DT_FLOAT16 || data_type == DT_INT8);
}

Status TransShapeHwcnToC1hwncoc0(const DataType &data_type, const std::vector<int64_t> &src_shape,
                                 std::vector<int64_t> &dst_shape) {
  auto cube_size = GetCubeSizeByDataType(data_type);
  dst_shape.clear();
  dst_shape.push_back(Ceil(src_shape.at(kHwcnC), static_cast<int64_t>(cube_size)));
  dst_shape.push_back(src_shape.at(kHwcnH));
  dst_shape.push_back(src_shape.at(kHwcnW));
  dst_shape.push_back(src_shape.at(kHwcnN));
  dst_shape.push_back(cube_size);
  dst_shape.push_back(cube_size);
  if (!CheckShapeValid(dst_shape, kC1hwncoc0DimsNum)) {
    GELOGE(ACL_ERROR_GE_SHAPE_INVALID, "[Check][Shape]Value is invalid, dst shape %s",
           ShapeToString(dst_shape).c_str());
    REPORT_CALL_ERROR("E19999", "Dst shape %s check invalid",
                      ShapeToString(dst_shape).c_str());
    return ACL_ERROR_GE_SHAPE_INVALID;
  }
  return SUCCESS;
}

Status CheckArgsForHwcnToC1hwncoc0(const TransArgs &args) {
  if (args.src_format != FORMAT_HWCN || args.dst_format != FORMAT_C1HWNCoC0) {
    std::string error = "Dose not support trans format from " +
        FmtToStr(TypeUtils::FormatToSerialString(args.src_format)) + " to " +
        FmtToStr(TypeUtils::FormatToSerialString(args.dst_format));
    GE_ERRORLOG_AND_ERRORMSG(ACL_ERROR_GE_FORMAT_INVALID, error.c_str());
    return ACL_ERROR_GE_FORMAT_INVALID;
  }
  if (!CheckDataTypeSupported(args.src_data_type)) {
    GELOGE(ACL_ERROR_GE_DATATYPE_INVALID, "[Trans][Shape]Failed, "
           "shape from HWCN to C1HWNCoC0, invalid data type %s",
           TypeUtils::DataTypeToSerialString(args.src_data_type).c_str());
    REPORT_INNER_ERROR("E19999", "Failed to trans shape from HWCN to C1HWNCoC0, "
                       "invalid data type %s",
                       TypeUtils::DataTypeToSerialString(args.src_data_type).c_str());
    return ACL_ERROR_GE_DATATYPE_INVALID;
  }
  if (!CheckShapeValid(args.src_shape, kHwcnDimsNum)) {
    GELOGE(ACL_ERROR_GE_SHAPE_INVALID, "[Check][Shape]Value is invalid, src shape %s",
           ShapeToString(args.src_shape).c_str());
    REPORT_CALL_ERROR("E19999", "Src shape %s check invalid",
                      ShapeToString(args.src_shape).c_str());
    return ACL_ERROR_GE_SHAPE_INVALID;
  }
  if (!CheckShapeValid(args.dst_shape, kC1hwncoc0DimsNum)) {
    GELOGE(ACL_ERROR_GE_SHAPE_INVALID, "[Check][Shape]Value is invalid, dst shape %s",
           ShapeToString(args.dst_shape).c_str());
    REPORT_CALL_ERROR("E19999", "Dst shape %s check invalid",
                      ShapeToString(args.dst_shape).c_str());
    return ACL_ERROR_GE_SHAPE_INVALID;
  }
  std::vector<int64_t> expect_dst_shape;
  auto ret = TransShapeHwcnToC1hwncoc0(args.src_data_type, args.src_shape, expect_dst_shape);
  if (ret != SUCCESS) {
    return ret;
  }
  if (args.dst_shape != expect_dst_shape) {
    GELOGE(ACL_ERROR_GE_SHAPE_INVALID,
           "[Trans][Shape]Failed, src shape %s and dst shape %s are not compatible. "
           "expect dst shape %s",
           ShapeToString(args.src_shape).c_str(), ShapeToString(args.dst_shape).c_str(),
           ShapeToString(expect_dst_shape).c_str());
    REPORT_INNER_ERROR("E19999", "Failed to trans format, src shape %s and dst shape %s "
                       "are not compatible. expect dst shape %s",
                       ShapeToString(args.src_shape).c_str(), ShapeToString(args.dst_shape).c_str(),
                       ShapeToString(expect_dst_shape).c_str());
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
    REPORT_CALL_ERROR("E19999", "Failed to alloc the memory for dst buf %ld, shape %s when trans format from %s to %s",
                      total_size, ShapeToString(args.dst_shape).c_str(),
                      TypeUtils::FormatToSerialString(args.src_format).c_str(),
                      TypeUtils::FormatToSerialString(args.dst_format).c_str());
    return ACL_ERROR_GE_MEMORY_ALLOCATION;
  }

  auto h = args.src_shape.at(kHwcnH);
  auto w = args.src_shape.at(kHwcnW);
  auto c = args.src_shape.at(kHwcnC);
  auto n = args.src_shape.at(kHwcnN);
  auto c1 = args.dst_shape.at(kC1hwncoc0C1);
  auto c0 = args.dst_shape.at(kC1hwncoc0C0);
  auto co = args.dst_shape.at(kC1hwncoc0Co);
  int64_t coc0 = co * c0;
  int64_t ncoc0 = n * coc0;
  int64_t wncoc0 = w * ncoc0;
  int64_t hwncoc0 = h * wncoc0;
  int64_t cn = c * n;
  int64_t wcn = w * cn;

  for (int64_t c1_idx = 0; c1_idx < c1; c1_idx++) {
    int64_t c1_head_addr = c1_idx * hwncoc0;
    for (int64_t h_idx = 0; h_idx < h; h_idx++) {
      int64_t h_head_addr = c1_head_addr + h_idx * wncoc0;
      for (int64_t w_idx = 0; w_idx < w; w_idx++) {
        int64_t w_head_addr = h_head_addr + w_idx * ncoc0;
        for (int64_t n_idx = 0; n_idx < n; n_idx++) {
          int64_t n_head_addr = w_head_addr + n_idx * coc0;
          for (int64_t co_idx = 0; co_idx < co; co_idx++) {
            int64_t co_head_addr = n_head_addr + co_idx * c0;
            for (int64_t c0_idx = 0; c0_idx < c0; c0_idx++) {
              int64_t dst_idx = c0_idx + co_head_addr;
              auto dst_offset = dst_idx * size;
              auto protected_size = total_size - dst_offset < static_cast<int64_t>(SECUREC_MEM_MAX_LEN)
                                        ? total_size - dst_offset
                                        : static_cast<int64_t>(SECUREC_MEM_MAX_LEN);
              GE_CHECK_GE(protected_size, 0);
              int64_t c_idx = c0_idx + c1_idx * c0;
              int64_t src_idx = h_idx * wcn + w_idx * cn + c_idx * n + n_idx;
              auto src_offset = src_idx * size;

              if (c_idx < c && c0_idx == co_idx) {
                auto ret = memcpy_s(dst.get() + dst_offset, static_cast<size_t>(protected_size), args.data + src_offset,
                                    static_cast<size_t>(size));
                if (ret != EOK) {
                  GELOGE(ACL_ERROR_GE_MEMORY_OPERATE_FAILED, "[Copy][Data]Failed, "
                         "data from HWCN[%ld, %ld, %ld, %ld] offset %ld to "
                         "C1HWNCoC0[%ld, %ld, %ld, %ld, %ld, %ld] offset %ld, err-code %d",
                         h_idx, w_idx, c_idx, n_idx, src_offset, c1_idx, h_idx, w_idx,
                         n_idx, co_idx, c0_idx, dst_offset, ret);
                  REPORT_CALL_ERROR("E19999", "Failed to copy data from "
                                    "HWCN[%ld, %ld, %ld, %ld] offset %ld "
                                    "to, C1HWNCoC0[%ld, %ld, %ld, %ld, %ld, %ld] "
                                    "offset %ld, err-code %d",
                                    h_idx, w_idx, c_idx, n_idx, src_offset, c1_idx, h_idx, w_idx,
                                    n_idx, co_idx, c0_idx, dst_offset, ret);
                  return ACL_ERROR_GE_MEMORY_OPERATE_FAILED;
                }
              } else {
                auto ret =
                    memset_s(dst.get() + dst_offset, static_cast<size_t>(protected_size), 0, static_cast<size_t>(size));
                if (ret != EOK) {
                  GELOGE(ACL_ERROR_GE_MEMORY_OPERATE_FAILED,
                         "[Operate][Memory]Failed to set to 0 to "
                         "C1HWNCoC0[%ld, %ld, %ld, %ld, %ld, %ld] offset %ld, err-code %d",
                         c1_idx, h_idx, w_idx, n_idx, co_idx, c0_idx, dst_offset, ret);
                  REPORT_CALL_ERROR("E19999",  "Failed to set to 0 to "
                                    "C1HWNCoC0[%ld, %ld, %ld, %ld, %ld, %ld] offset %ld, "
                                    "err-code %d",
                                    c1_idx, h_idx, w_idx, n_idx, co_idx, c0_idx, dst_offset, ret);
                  return ACL_ERROR_GE_MEMORY_OPERATE_FAILED;
                }
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

Status FormatTransferHwcnC1hwncoc0::TransFormat(const TransArgs &args, TransResult &result) {
  Status ret = CheckArgsForHwcnToC1hwncoc0(args);
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

    GELOGE(ACL_ERROR_GE_SHAPE_INVALID, "[Get][ShapeSize]Failed, total size %ld from dst shape %s, "
           "src shape %s", total_size,
           ShapeToString(args.dst_shape).c_str(), ShapeToString(args.src_shape).c_str());
    REPORT_CALL_ERROR("E19999",  "Failed to get total size %ld from dst shape %s, src shape %s",
                      total_size,
                      ShapeToString(args.dst_shape).c_str(), ShapeToString(args.src_shape).c_str());
    return ACL_ERROR_GE_SHAPE_INVALID;
  }
  GELOGD("Begin to trans format from HWCN to C1HWNCoC0, src shape %s, data type %s, dst shape %s, memory size %ld",
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

Status FormatTransferHwcnC1hwncoc0::TransShape(Format src_format, const std::vector<int64_t> &src_shape,
                                               DataType data_type, Format dst_format, std::vector<int64_t> &dst_shape) {
  if (src_format == FORMAT_HWCN && CheckDataTypeSupported(data_type)) {
    if (!CheckShapeValid(src_shape, kHwcnDimsNum)) {
      GELOGE(ACL_ERROR_GE_SHAPE_INVALID, "[Check][Shape]Value is invalid, src shape %s",
             ShapeToString(src_shape).c_str());
      REPORT_CALL_ERROR("E19999", "Src shape %s check invalid",
                        ShapeToString(src_shape).c_str());
      return ACL_ERROR_GE_SHAPE_INVALID;
    }
    return TransShapeHwcnToC1hwncoc0(data_type, src_shape, dst_shape);
  } else if (src_format != FORMAT_HWCN) {
    return ACL_ERROR_GE_FORMAT_INVALID;
  } else {
    return ACL_ERROR_GE_DATATYPE_INVALID;
  }
}

REGISTER_FORMAT_TRANSFER(FormatTransferHwcnC1hwncoc0, FORMAT_HWCN, FORMAT_C1HWNCoC0)
}  // namespace formats
}  // namespace ge
