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

#include "common/formats/format_transfers/format_transfer_nchw_fz_c04.h"
#include "common/formats/format_transfers/format_transfer_transpose.h"

#include <securec.h>
#include <memory>
#include <cstdlib>

#include "common/formats/utils/formats_definitions.h"
#include "common/formats/utils/formats_trans_utils.h"
#include "framework/common/util.h"
#include "framework/common/debug/ge_log.h"
#include "graph/utils/type_utils.h"

/** 【Explain about transfer from nchw to FZ_CO4】
 *  First Step: Padding in N and C axis. Here C must be less or equal than 4
 *      After Padding, it will be like (n = ceil(n,16)*16, 4, h, w)
 *  Second Step: transpose. It will be like (n = ceil(n,16)*16, h, w, 4)
 *  Third Step: View the 4D as 2D , first dim is N, second dim is h*w*c.
 *      Padding to (N, ceil(Z/16)*16)
 *  Last Step: View the (N, ceil(Z/16)*16) as 4D (N/16, 16, C/16, 16) and transpose to (C/16, N/16, 16, 16)
 */
namespace ge {
namespace formats {
namespace {
constexpr int64_t kMaxDimsNumC = 4;

Status CheckDataTypeSupport(DataType data_type) { return GetSizeByDataType(data_type) > 0 ? SUCCESS : UNSUPPORTED; }

Status TransShape(int64_t n, int64_t c, int64_t h, int64_t w, DataType data_type, std::vector<int64_t> &dst_shape) {
  auto c0 = GetCubeSizeByDataType(data_type);
  if (c0 < 0) {
    return ACL_ERROR_GE_DATATYPE_INVALID;
  }
  auto chw = c * h * w;

  auto first_dim = Ceil(chw, c0);
  auto no = Ceil(n, static_cast<int64_t>(c0));

  dst_shape.clear();
  dst_shape.push_back(first_dim);
  dst_shape.push_back(no);
  dst_shape.push_back(c0);
  dst_shape.push_back(c0);

  if (!IsShapeValid(dst_shape)) {
    GELOGE(ACL_ERROR_GE_SHAPE_INVALID, "[Check][Shape]Value is invalid, dst shape %s",
           ShapeToString(dst_shape).c_str());
    REPORT_CALL_ERROR("E19999", "Dst shape %s check invalid", ShapeToString(dst_shape).c_str());
    return ACL_ERROR_GE_SHAPE_INVALID;
  }
  return SUCCESS;
}

Status TransShapeNchwToFzC04(const std::vector<int64_t> &src_shape, DataType data_type,
                             std::vector<int64_t> &dst_shape) {
  if (!CheckShapeValid(src_shape, kNchwDimsNum)) {
    return ACL_ERROR_GE_SHAPE_INVALID;
  }

  auto n = src_shape.at(kNchwN);
  auto c = src_shape.at(kNchwC);
  auto h = src_shape.at(kNchwH);
  auto w = src_shape.at(kNchwW);
  return TransShape(n, c, h, w, data_type, dst_shape);
}

Status TransFormatFromNchwToFzC04(const TransArgs &args, TransResult &result) {
  int64_t n = args.src_shape.at(kNchwN);
  int64_t c = args.src_shape.at(kNchwC);
  int64_t h = args.src_shape.at(kNchwH);
  int64_t w = args.src_shape.at(kNchwW);

  int64_t c0 = GetCubeSizeByDataType(args.src_data_type);
  int size = GetSizeByDataType(args.src_data_type);

  auto data = args.data;
  TransResult trans_result_1;
  std::vector<int64_t> perm_arg_1 = {0, 2, 3, 1};
  std::vector<int64_t> expect_shape = {n, h, w, c};
  auto ret = ge::formats::Transpose(data, args.src_shape, args.src_data_type, perm_arg_1, trans_result_1);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Trans][Formats]Failed from NCHW to HWCN, src_shape %s, src_data_type %s",
           ShapeToString(args.src_shape).c_str(),
           TypeUtils::DataTypeToSerialString(args.src_data_type).c_str());
    REPORT_CALL_ERROR("E19999", "Failede to trans formats from NCHW to HWCN, src_shape %s, "
                      "src_data_type %s",
                      ShapeToString(args.src_shape).c_str(),
                      TypeUtils::DataTypeToSerialString(args.src_data_type).c_str());
    return ret;
  }

  TransArgs args_tmp = args;
  args_tmp.src_shape = expect_shape;
  args_tmp.data = trans_result_1.data.get();
  // check size it should be same with original
  size_t expect_size = n * c * h * w * size;  // before has do check about mul
  if (trans_result_1.length != expect_size) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "[Check][Shape]size %zu is not match expect size %zu "
           "after transpose",
           trans_result_1.length, expect_size);
    return ACL_ERROR_GE_PARAM_INVALID;
  }

  // prepare for padding in chw
  int64_t tmp = h * w * c;
  int64_t n_o = Ceil(n, static_cast<int64_t>(c0));
  int64_t c_o = c0;
  int64_t h_o = Ceil(tmp, c0);
  int64_t w_o = c0;
  std::vector<int64_t> shape_o = {n_o, c_o, h_o, w_o};

  // data overflow check totally
  GE_IF_BOOL_EXEC(!CheckInt64MulOverflow(h_o, w_o),
                  GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "[Check][Shape]Failed, "
                         "int64 mul overflow.A[%ld], B[%ld]", h_o, w_o);
                  REPORT_CALL_ERROR("E19999", "Check shape failed, int64 mul overflow.A[%ld], "
                                    "B[%ld]", h_o, w_o);
                  return ACL_ERROR_GE_INTERNAL_ERROR);
  GE_IF_BOOL_EXEC(!CheckInt64MulOverflow(n_o, c_o),
                  GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "[Check][Shape]Failed, "
                         "int64 mul overflow.A[%ld], B[%ld]", n_o, c_o);
                  REPORT_CALL_ERROR("E19999", "Check shape failed, int64 mul overflow.A[%ld], "
                                    "B[%ld]", n_o, c_o);
                  return ACL_ERROR_GE_INTERNAL_ERROR);
  auto t1 = h_o * w_o;
  auto t2 = n_o * c_o;
  GE_IF_BOOL_EXEC(!CheckInt64MulOverflow(t1, t2),
                  GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "[Check][Shape]Failed, "
                         "int64 mul overflow.A[%ld], B[%ld]", t1, t2);
                  REPORT_CALL_ERROR("E19999", "Check shape failed, "
                                    "int64 mul overflow.A[%ld], B[%ld]", t1, t2);
                  return ACL_ERROR_GE_INTERNAL_ERROR);

  int64_t total_ele_cnt = n_o * c_o * h_o * w_o;
  GE_IF_BOOL_EXEC(!CheckInt64MulOverflow(total_ele_cnt, size),
                  GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "[Check][Shape]Failed, "
                  "int64 mul overflow.A[%ld], B[%d]", total_ele_cnt, size);
                  REPORT_CALL_ERROR("E19999", "Check shape failed, int64 mul overflow.A[%ld], "
                  "B[%d]", total_ele_cnt, size);
                  return ACL_ERROR_GE_INTERNAL_ERROR);
  int64_t dst_size = total_ele_cnt * size;
  if (dst_size == 0) {
    result.length = 0;
    return SUCCESS;
  }

  std::shared_ptr<uint8_t> dst(new (std::nothrow) uint8_t[dst_size], std::default_delete<uint8_t[]>());
  if (dst == nullptr) {
    GELOGE(ACL_ERROR_GE_MEMORY_ALLOCATION, "Failed to alloc the memory for dst buf %ld "
           "when trans format from %s to %s",
           dst_size, TypeUtils::FormatToSerialString(args.src_format).c_str(),
           TypeUtils::FormatToSerialString(args.dst_format).c_str());
    REPORT_CALL_ERROR("E19999",  "Failed to alloc the memory for dst buf %ld "
                      "when trans format from %s to %s",
                      dst_size, TypeUtils::FormatToSerialString(args.src_format).c_str(),
                      TypeUtils::FormatToSerialString(args.dst_format).c_str());
    return ACL_ERROR_GE_MEMORY_ALLOCATION;
  }
  auto retMem = memset_s(dst.get(), dst_size, 0, dst_size);
  if (retMem != EOK) {
    GELOGE(ACL_ERROR_GE_MEMORY_OPERATE_FAILED, "[Set][Memory]Failed, dst buf %ld, error_code %d",
           dst_size, retMem);
    REPORT_CALL_ERROR("E19999", "Set memory failed, dst buf %ld, error_code %d", dst_size, retMem);
    return ACL_ERROR_GE_MEMORY_OPERATE_FAILED;
  }
  // copy data
  auto block = c * h * w * size;
  auto stride = h_o * w_o * size;
  auto p_s = trans_result_1.data.get();
  auto p_d = dst.get();
  auto protectSize = dst_size;
  for (auto k = 0; k < n; k++) {
    ret = memcpy_s(p_d + k * stride, protectSize, p_s + k * block, block);
    if (ret != EOK) {
      GELOGE(ACL_ERROR_GE_MEMORY_OPERATE_FAILED, "[Set][Memcpy]Failed, block %zu, stride %zu, "
             "protect_size %ld, error_code %d", block, stride, protectSize, ret);
      REPORT_CALL_ERROR("E19999", "[Set][Memcpy]Failed, block %zu, stride %zu, "
                        "protect_size %ld, error_code %d", block, stride, protectSize, ret);
      return ACL_ERROR_GE_MEMORY_OPERATE_FAILED;
    }
    protectSize = protectSize - block;
  }

  // transpose : 2,0,1,3
  std::vector<int64_t> perm_arg_2 = {2, 0, 1, 3};
  ret = ge::formats::Transpose(dst.get(), shape_o, args.src_data_type, perm_arg_2, result);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Trans][Formats]Failed from NCHW to HWCN, error_code %u", ret);
    REPORT_CALL_ERROR("E19999", "Failed to trans formats from NCHW to HWCN, error_code %u", ret);
    return ret;
  }

  return SUCCESS;
}

Status PaddingNC(const TransArgs &args, TransArgs &args_tmp, std::shared_ptr<uint8_t> &dst) {
  args_tmp = args;
  auto src_shape = args_tmp.src_shape;
  if (!CheckShapeValid(src_shape, kNchwDimsNum)) {
    return ACL_ERROR_GE_SHAPE_INVALID;
  }
  int64_t c0 = GetCubeSizeByDataType(args.src_data_type);

  auto n = src_shape.at(kNchwN);
  auto c = src_shape.at(kNchwC);
  auto h = src_shape.at(kNchwH);
  auto w = src_shape.at(kNchwW);

  if (c > kMaxDimsNumC) {
    GELOGE(ACL_ERROR_GE_SHAPE_INVALID, "[Check][Shape]Invalid dim c num[%lu]. "
           "It should be in (0,4]", c);
    return ACL_ERROR_GE_SHAPE_INVALID;
  }

  auto n_o = Ceil(n, c0) * c0;
  auto c_o = kMaxDimsNumC;
  auto h_o = h;
  auto w_o = w;
  args_tmp.src_shape.at(kNchwN) = n_o;
  args_tmp.src_shape.at(kNchwC) = c_o;
  args_tmp.src_shape.at(kNchwH) = h_o;
  args_tmp.src_shape.at(kNchwW) = w_o;

  // data overflow check
  GE_IF_BOOL_EXEC(!CheckInt64MulOverflow(h_o, w_o),
                  GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "[Check][Shape]Failed, "
                         "int64 mul overflow.A[%ld], B[%ld]", h_o, w_o);
                  REPORT_CALL_ERROR("E19999", "Check shape failed, int64 mul overflow.A[%ld], "
                                    "B[%ld]", h_o, w_o);
                  return ACL_ERROR_GE_INTERNAL_ERROR);
  GE_IF_BOOL_EXEC(!CheckInt64MulOverflow(n_o, c_o),
                  GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "[Check][Shape]Failed, "
                         "int64 mul overflow.A[%ld], B[%ld]", n_o, c_o);
                  REPORT_CALL_ERROR("E19999", "Check shape failed, int64 mul overflow.A[%ld], "
                                    "B[%ld]", n_o, c_o);
                  return ACL_ERROR_GE_INTERNAL_ERROR);
  auto t1 = h_o * w_o;
  auto t2 = n_o * c_o;
  GE_IF_BOOL_EXEC(!CheckInt64MulOverflow(t1, t2),
                  GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "[Check][Shape]Failed, "
                         "int64 mul overflow.A[%ld], B[%ld]", t1, t2);
                  REPORT_CALL_ERROR("E19999", "Check shape failed, int64 mul overflow.A[%ld], "
                                    "B[%ld]", t1, t2);
                  return ACL_ERROR_GE_INTERNAL_ERROR);

  int64_t total_ele_cnt = n_o * c_o * h_o * w_o;
  int size = GetSizeByDataType(args.src_data_type);
  GE_IF_BOOL_EXEC(!CheckInt64MulOverflow(total_ele_cnt, size),
                  GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "[Check][Shape]Failed, "
                         "int64 mul overflow.A[%ld], B[%d]", total_ele_cnt, size);
                  REPORT_CALL_ERROR("E19999", "Check shape failed, int64 mul overflow.A[%ld], "
                                    "B[%d]", total_ele_cnt, size);
                  return ACL_ERROR_GE_INTERNAL_ERROR);

  int64_t dst_size = total_ele_cnt * size;
  if (dst_size == 0) {
    return SUCCESS;
  }

  dst.reset(new (std::nothrow) uint8_t[dst_size], std::default_delete<uint8_t[]>());
  if (dst == nullptr) {
    GELOGE(ACL_ERROR_GE_MEMORY_ALLOCATION, "Failed to alloc the memory for dst buf %ld when "
           "trans format from %s to %s",
           dst_size, TypeUtils::FormatToSerialString(args.src_format).c_str(),
           TypeUtils::FormatToSerialString(args.dst_format).c_str());
    REPORT_CALL_ERROR("E19999", "Failed to alloc the memory for dst buf %ld when "
                      "trans format from %s to %s",
                      dst_size, TypeUtils::FormatToSerialString(args.src_format).c_str(),
                      TypeUtils::FormatToSerialString(args.dst_format).c_str());
    return ACL_ERROR_GE_MEMORY_ALLOCATION;
  }
  auto ret = memset_s(dst.get(), dst_size, 0, dst_size);
  if (ret != EOK) {
    GELOGE(ACL_ERROR_GE_MEMORY_OPERATE_FAILED, "[Set][Memory]Failed, dst buf %ld, error_code %d",
           dst_size, ret);
    REPORT_CALL_ERROR("E19999", "Set memory failed, dst buf %ld, error_code %d", dst_size, ret);
    return ACL_ERROR_GE_MEMORY_OPERATE_FAILED;
  }

  auto p_s = args.data;
  auto p_d = dst.get();
  auto block = h * w * size;
  auto protectSize = dst_size;

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < c; j++) {
      ret = memcpy_s(p_d + (i * c_o * h_o * w_o + j * h_o * w_o) * size, protectSize,
                     p_s + (i * c * h * w + j * h * w) * size, block);
      if (ret != EOK) {
        GELOGE(ACL_ERROR_GE_MEMORY_OPERATE_FAILED, "[Set][Memcpy]Failed, block %zu, "
               "protect_size %ld, error_code %d", block, protectSize, ret);
        REPORT_CALL_ERROR("E19999", "[Set][Memcpy]Failed, block %zu, protect_size %ld, "
                          "error_code %d", block, protectSize, ret);
        return ACL_ERROR_GE_MEMORY_OPERATE_FAILED;
      }
      protectSize = protectSize - block;
    }
  }
  args_tmp.data = dst.get();

  return SUCCESS;
}
}  // namespace

Status FormatTransferNchwToFZC04::TransFormat(const TransArgs &args, TransResult &result) {
  GELOGD("Begin to trans format from %s to %s, src shape %s, data type %s, dst shape %s",
         TypeUtils::FormatToSerialString(args.src_format).c_str(),
         TypeUtils::FormatToSerialString(args.dst_format).c_str(), ShapeToString(args.src_shape).c_str(),
         TypeUtils::DataTypeToSerialString(args.src_data_type).c_str(), ShapeToString(args.dst_shape).c_str());
  TransArgs args_tmp = args;
  std::shared_ptr<uint8_t> dst = nullptr;
  auto ret = PaddingNC(args, args_tmp, dst);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Padding][NCAxis]Failed, error_code %u", ret);
    REPORT_CALL_ERROR("E19999", "Padding in NC axis failed, error_code %u", ret);
    return ret;
  }

  std::vector<int64_t> expect_shape;
  ret = TransShape(args_tmp.src_format, args_tmp.src_shape, args_tmp.src_data_type,
                   args_tmp.dst_format, expect_shape);
  if (ret != SUCCESS) {
    return ret;
  }

  if (!IsTransShapeDstCorrect(args_tmp, expect_shape)) {
    return ACL_ERROR_GE_SHAPE_INVALID;
  }

  if (args_tmp.src_format == FORMAT_NCHW && args_tmp.dst_format == FORMAT_FRACTAL_Z_C04) {
    return TransFormatFromNchwToFzC04(args_tmp, result);
  }

  return ACL_ERROR_GE_FORMAT_INVALID;
}

Status FormatTransferNchwToFZC04::TransShape(Format src_format, const std::vector<int64_t> &src_shape,
                                             DataType data_type, Format dst_format, std::vector<int64_t> &dst_shape) {
  if (CheckDataTypeSupport(data_type) != SUCCESS) {
    return ACL_ERROR_GE_DATATYPE_INVALID;
  }
  if (src_format == FORMAT_NCHW && dst_format == FORMAT_FRACTAL_Z_C04) {
    return TransShapeNchwToFzC04(src_shape, data_type, dst_shape);
  }

  return ACL_ERROR_GE_FORMAT_INVALID;
}

REGISTER_FORMAT_TRANSFER(FormatTransferNchwToFZC04, FORMAT_NCHW, FORMAT_FRACTAL_Z_C04)
}  // namespace formats
}  // namespace ge
