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
#include "common/formats/format_transfers/format_transfer_dhwcn_fracz3D.h"

#include <securec.h>
#include <memory>

#include "common/formats/utils/formats_definitions.h"
#include "common/formats/utils/formats_trans_utils.h"
#include "framework/common/debug/ge_log.h"
#include "graph/utils/type_utils.h"

namespace ge {
namespace formats {
namespace {
Status CheckDataTypeSupport(DataType dtype) { return GetSizeByDataType(dtype) > 0 ? SUCCESS : UNSUPPORTED; }

Status TransShapeToFz(int64_t d, int64_t n, int64_t c, int64_t h, int64_t w, DataType data_type,
                      std::vector<int64_t> &dst_shape) {
  auto c0 = GetCubeSizeByDataType(data_type);
  if (c0 < 0) {
    return ACL_ERROR_GE_DATATYPE_INVALID;
  }

  auto c1 = Ceil(c, c0);
  auto no = Ceil(n, static_cast<int64_t>(kNiSize));

  dst_shape.clear();
  dst_shape.push_back(d * c1 * h * w);
  dst_shape.push_back(no);
  dst_shape.push_back(kNiSize);
  dst_shape.push_back(c0);

  return SUCCESS;
}

Status TransShapeDhwckToFz3D(const std::vector<int64_t> &src_shape, DataType data_type,
                             std::vector<int64_t> &dst_shape) {
  if (!CheckShapeValid(src_shape, kDhwcnDimsNum)) {
    return ACL_ERROR_GE_SHAPE_INVALID;
  }
  auto d = src_shape.at(kDhwcnD);
  auto h = src_shape.at(kDhwcnH);
  auto w = src_shape.at(kDhwcnW);
  auto c = src_shape.at(kDhwcnC);
  auto n = src_shape.at(kDhwcnN);

  return TransShapeToFz(d, n, c, h, w, data_type, dst_shape);
}
Status TransFormatDhwckToFz3D(const TransArgs &args, TransResult &result) {
  if (!CheckShapeValid(args.src_shape, kDhwcnDimsNum)) {
    return ACL_ERROR_GE_SHAPE_INVALID;
  }
  int64_t d = args.src_shape[kDhwcnD];
  int64_t h = args.src_shape[kDhwcnH];
  int64_t w = args.src_shape[kDhwcnW];
  int64_t c = args.src_shape[kDhwcnC];
  int64_t n = args.src_shape[kDhwcnN];
  int64_t n1n0 = Ceil(n, static_cast<int64_t>(kNiSize)) * kNiSize;
  int64_t c0 = GetCubeSizeByDataType(args.src_data_type);
  int64_t c1 = Ceil(c, c0);

  auto cn = c * n;
  auto wcn = w * cn;
  auto hwcn = h * wcn;
  auto n1n0c0 = n1n0 * c0;
  auto wn1n0c0 = w * n1n0c0;
  auto hwn1n0c0 = h * wn1n0c0;
  auto c1hwn1n0c0 = c1 * hwn1n0c0;

  int64_t data_size = GetSizeByDataType(args.src_data_type);
  int64_t dst_size = 1;
  for (auto dim : args.dst_shape) {
    dst_size *= dim;
  }
  dst_size *= data_size;
  if (dst_size == 0) {
    result.length = static_cast<size_t>(dst_size);
    return SUCCESS;
  }

  std::shared_ptr<uint8_t> dst(new (std::nothrow) uint8_t[dst_size], std::default_delete<uint8_t[]>());
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

  for (int64_t di = 0; di < d; di++) {
    for (int64_t c1i = 0; c1i < c1; c1i++) {
      for (int64_t hi = 0; hi < h; hi++) {
        for (int64_t wi = 0; wi < w; wi++) {
          for (int64_t n1n0i = 0; n1n0i < n1n0; n1n0i++) {
            for (int64_t c0i = 0; c0i < c0; c0i++) {
              int64_t dst_idx = di * c1hwn1n0c0 + c1i * hwn1n0c0 + hi * wn1n0c0 + wi * n1n0c0 + n1n0i * c0 + c0i;
              int64_t dst_offset = dst_idx * data_size;
              auto pad_zero = ((c1i * c0 + c0i) >= c) || (n1n0i >= n);
              auto protected_size = dst_size - dst_offset < static_cast<int64_t>(SECUREC_MEM_MAX_LEN)
                                      ? dst_size - dst_offset
                                      : static_cast<int64_t>(SECUREC_MEM_MAX_LEN);
              errno_t ret;
              if (pad_zero) {
                ret = memset_s(dst.get() + dst_offset, static_cast<size_t>(protected_size), 0,
                               static_cast<size_t>(data_size));
              } else {
                int64_t src_idx = di * hwcn + hi * wcn + wi * cn + (c1i * c0 + c0i) * n + n1n0i;
                ret = memcpy_s(dst.get() + dst_offset, static_cast<size_t>(protected_size),
                               args.data + src_idx * data_size, static_cast<size_t>(data_size));
              }
              if (ret != EOK) {
                GELOGE(ACL_ERROR_GE_MEMORY_OPERATE_FAILED, "[Operate][DSTMemory]Failed at "
                       "offset %ld, error-code %d, pad mode %d", dst_offset, ret, pad_zero);
                REPORT_CALL_ERROR("E19999", "Failed to operate dst memory at offset %ld, "
                                  "error-code %d, pad mode %d", dst_offset, ret, pad_zero);
                return ACL_ERROR_GE_MEMORY_OPERATE_FAILED;
              }
            }
          }
        }
      }
    }
  }
  result.data = dst;
  result.length = dst_size;
  return SUCCESS;
}
}  // namespace

Status FormatTransferDhwcnFractalZ3D::TransFormat(const TransArgs &args, TransResult &result) {
  GELOGD("Begin to trans format from %s to %s, src shape %s, data type %s, dst shape %s",
         TypeUtils::FormatToSerialString(args.src_format).c_str(),
         TypeUtils::FormatToSerialString(args.dst_format).c_str(), ShapeToString(args.src_shape).c_str(),
         TypeUtils::DataTypeToSerialString(args.src_data_type).c_str(), ShapeToString(args.dst_shape).c_str());
  std::vector<int64_t> expect_shape;
  auto ret = TransShape(args.src_format, args.src_shape, args.src_data_type, args.dst_format, expect_shape);
  if (ret != SUCCESS) {
    return ret;
  }
  if (!IsTransShapeDstCorrect(args, expect_shape)) {
    return ACL_ERROR_GE_SHAPE_INVALID;
  }

  if (args.src_format == FORMAT_DHWCN && args.dst_format == FORMAT_FRACTAL_Z_3D) {
    return TransFormatDhwckToFz3D(args, result);
  }

  return ACL_ERROR_GE_FORMAT_INVALID;
}

Status FormatTransferDhwcnFractalZ3D::TransShape(Format src_format, const std::vector<int64_t> &src_shape,
                                                 DataType data_type, Format dst_format,
                                                 std::vector<int64_t> &dst_shape) {
  if (CheckDataTypeSupport(data_type) != SUCCESS) {
    return ACL_ERROR_GE_DATATYPE_INVALID;
  }

  if (src_format == FORMAT_DHWCN && dst_format == FORMAT_FRACTAL_Z_3D) {
    return TransShapeDhwckToFz3D(src_shape, data_type, dst_shape);
  }

  return ACL_ERROR_GE_FORMAT_INVALID;
}

REGISTER_FORMAT_TRANSFER(FormatTransferDhwcnFractalZ3D, FORMAT_DHWCN, FORMAT_FRACTAL_Z_3D)
}  // namespace formats
}  // namespace ge
