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

#include "common/formats/format_transfers/format_transfer_fractal_z.h"

#include <securec.h>
#include <memory>

#include "framework/common/debug/log.h"
#include "common/formats/utils/formats_definitions.h"
#include "common/formats/utils/formats_trans_utils.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/debug/log.h"
#include "graph/utils/type_utils.h"

namespace ge {
namespace formats {
namespace {
constexpr int64_t kDim = 1;
static int64_t Measure(int64_t x, int64_t y) {
  int64_t z = y;
  while (x % y != 0) {
    z = x % y;
    x = y;
    y = z;
  }
  return z;
}
// least common multiple
static int64_t Lcm(int64_t a, int64_t b) {
  if (b == 0) {
    return -1;
  }
  int64_t temp = (a * b) / (Measure(a, b));
  return temp;
}

Status CheckDataTypeSupport(DataType data_type) { return GetSizeByDataType(data_type) > 0 ? SUCCESS : UNSUPPORTED; }

/**
 * FZ represents the weight of convolution,.
 * After the conversion to two-dimensional matrix, the memory arrangement is small n and large Z.
 * If 4D(eg.NCHW) is used to represent convolution kernel, N is width, HWC is height.
 *
 * frac_z axises: (C1*H*W, No, Ni, C0), which Ni = 16, C0 = 16/32, No = Ceil(N/Ni), C1 = Ceil(C/C0)
 * @return
 */
Status TransShapeToFz(int64_t n, int64_t c, int64_t h, int64_t w, DataType data_type, std::vector<int64_t> &dst_shape) {
  auto c0 = GetCubeSizeByDataType(data_type);
  if (c0 < 0) {
    return ACL_ERROR_GE_DATATYPE_INVALID;
  }

  auto c1 = Ceil(c, c0);
  auto no = Ceil(n, static_cast<int64_t>(kNiSize));

  dst_shape.clear();
  dst_shape.push_back(h * w * c1);
  dst_shape.push_back(no);
  dst_shape.push_back(kNiSize);
  dst_shape.push_back(c0);
  if (!IsShapeValid(dst_shape)) {
    GELOGE(ACL_ERROR_GE_SHAPE_INVALID, "[Check][Shape]Failed, dst shape %s",
           ShapeToString(dst_shape).c_str());
    REPORT_CALL_ERROR("E19999", "Failed to check dst shape %s", ShapeToString(dst_shape).c_str());
    return ACL_ERROR_GE_SHAPE_INVALID;
  }
  return SUCCESS;
}

Status TransShapeToFzWithGroups(int64_t n, int64_t c, int64_t h, int64_t w, DataType data_type, std::vector<int64_t> &dst_shape,
  int64_t groups) {
  auto c0 = GetCubeSizeByDataType(data_type);
  if (c0 < 0) {
    return ACL_ERROR_GE_DATATYPE_INVALID;
  }
  int64_t cin_ori = c;
  int64_t cout_ori = n / groups;
  int64_t cube_k = GetCubeSizeByDataType(data_type);
  int64_t e_mult = std::min(
      Lcm(Lcm(cin_ori, cube_k) / (cin_ori), Lcm(cout_ori, static_cast<int64_t>(kCubeSize)) / (cout_ori)),
      groups);
  int64_t cin_opt = Ceil(e_mult * cin_ori, cube_k) * cube_k;
  int64_t c1_dim = cin_opt / cube_k;
  int64_t g_dim = Ceil(groups, e_mult);
  auto n1 = Ceil(cout_ori * e_mult, static_cast<int64_t>(kCubeSize));
  dst_shape.clear();
  dst_shape.push_back(g_dim * c1_dim * h * w);
  dst_shape.push_back(n1);
  dst_shape.push_back(16);
  dst_shape.push_back(cube_k);
  if (!IsShapeValid(dst_shape)) {
    GELOGE(ACL_ERROR_GE_SHAPE_INVALID, "[Check][Shape]Failed, dst shape %s",
           ShapeToString(dst_shape).c_str());
    REPORT_CALL_ERROR("E19999", "Failed to check dst shape %s", ShapeToString(dst_shape).c_str());
    return ACL_ERROR_GE_SHAPE_INVALID;
  }
  return SUCCESS;
}

Status TransShapeNchwToFz(const std::vector<int64_t> &src_shape, DataType data_type, std::vector<int64_t> &dst_shape) {
  if (!CheckShapeValid(src_shape, kNchwDimsNum)) {
    return ACL_ERROR_GE_SHAPE_INVALID;
  }

  auto n = src_shape.at(kNchwN);
  auto c = src_shape.at(kNchwC);
  auto h = src_shape.at(kNchwH);
  auto w = src_shape.at(kNchwW);
  return TransShapeToFz(n, c, h, w, data_type, dst_shape);
}

Status TransShapeHwcnToFz(const std::vector<int64_t> &src_shape, DataType data_type, std::vector<int64_t> &dst_shape) {
  if (!CheckShapeValid(src_shape, kHwcnDimsNum)) {
    return ACL_ERROR_GE_SHAPE_INVALID;
  }

  auto h = src_shape.at(kHwcnH);
  auto w = src_shape.at(kHwcnW);
  auto c = src_shape.at(kHwcnC);
  auto n = src_shape.at(kHwcnN);

  return TransShapeToFz(n, c, h, w, data_type, dst_shape);
}

Status TransShapeHwcnToFzWithGroups(const std::vector<int64_t> &src_shape, DataType data_type, std::vector<int64_t> &dst_shape
, int64_t groups){
 if (!CheckShapeValid(src_shape, kHwcnDimsNum)) {
    return ACL_ERROR_GE_SHAPE_INVALID;
  }

  auto h = src_shape.at(kHwcnH);
  auto w = src_shape.at(kHwcnW);
  auto c = src_shape.at(kHwcnC);
  auto n = src_shape.at(kHwcnN);

  return TransShapeToFzWithGroups(n, c, h, w, data_type, dst_shape, groups);
}


Status TransShapeNhwcToFz(const std::vector<int64_t> &src_shape, DataType data_type, std::vector<int64_t> &dst_shape) {
  if (!CheckShapeValid(src_shape, kNhwcDimsNum)) {
    return ACL_ERROR_GE_SHAPE_INVALID;
  }

  auto n = src_shape.at(kNhwcN);
  auto h = src_shape.at(kNhwcH);
  auto w = src_shape.at(kNhwcW);
  auto c = src_shape.at(kNhwcC);

  return TransShapeToFz(n, c, h, w, data_type, dst_shape);
}

Status TransFormatFromNchwToFz(const TransArgs &args, TransResult &result) {
  int64_t n = args.src_shape.at(kNchwN);
  int64_t c = args.src_shape.at(kNchwC);
  int64_t h = args.src_shape.at(kNchwH);
  int64_t w = args.src_shape.at(kNchwW);

  int64_t c0 = GetCubeSizeByDataType(args.src_data_type);
  int64_t c1 = Ceil(c, c0);

  int64_t hw = h * w;
  int64_t chw = c * hw;
  int64_t nchw = n * chw;
  int64_t hwc0 = hw * c0;

  // horizontal fractal matrix count (N)
  int64_t hf_cnt = Ceil(n, static_cast<int64_t>(kNiSize));
  // vertical fractal matrix count (C1HWC0)
  int64_t vf_cnt = c1 * hw;
  // elements count in one fractal
  int64_t fractal_ele_cnt = c0 * kNiSize;
  int64_t total_ele_cnt = hf_cnt * vf_cnt * fractal_ele_cnt;
  int size = GetSizeByDataType(args.src_data_type);
  int64_t dst_size = total_ele_cnt * size;
  GE_CHK_BOOL_EXEC_NOLOG(dst_size != 0, result.length = static_cast<size_t>(dst_size); return SUCCESS;);

  std::shared_ptr<uint8_t> dst(new (std::nothrow) uint8_t[dst_size], std::default_delete<uint8_t[]>());
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(
      dst == nullptr,
      GELOGE(ACL_ERROR_GE_MEMORY_ALLOCATION, "[Allocate][DSTMemory]Failed to allcoate memory "
             "for dst buf %ld when trans format from %s to %s",
             dst_size, TypeUtils::FormatToSerialString(args.src_format).c_str(),
             TypeUtils::FormatToSerialString(args.dst_format).c_str());
      REPORT_CALL_ERROR("E19999", "Failed to allcoate memory for dst buf %ld "
                        "when trans format from %s to %s",
                        dst_size, TypeUtils::FormatToSerialString(args.src_format).c_str(),
                        TypeUtils::FormatToSerialString(args.dst_format).c_str());
      return ACL_ERROR_GE_MEMORY_ALLOCATION;);

  for (int64_t vfi = 0; vfi < vf_cnt; vfi++) {
    // vertical fractal matrix base index
    auto vf_base_i = vfi * hf_cnt;
    for (int64_t hfi = 0; hfi < hf_cnt; hfi++) {
      // global fractal matrix index
      auto gfi = vf_base_i + hfi;
      auto src_n_offset = hfi * chw * kNiSize;
      auto src_f_offset = src_n_offset + vfi % hw + vfi / hw * hwc0;
      for (int64_t row = 0; row < c0; row++) {
        auto src_ci = vfi / hw * c0 + row;
        auto src_row_offset = src_f_offset + row * hw;
        for (int col = 0; col < kNiSize; col++) {
          auto src_ni = hfi * kNiSize + col;
          auto src_offset = src_row_offset + chw * col;
          // pad 0
          // 1. src_ni grater than n
          // 2. src_ci grater than c
          // 3. source address grater than original array size
          auto need_pad_zero = src_ni >= n || src_offset >= nchw || src_ci >= c;
          auto idx = gfi * fractal_ele_cnt + col * c0 + row;
          auto offset = idx * size;
          auto protected_size = dst_size - offset < static_cast<int64_t>(SECUREC_MEM_MAX_LEN)
                                    ? dst_size - offset
                                    : static_cast<int64_t>(SECUREC_MEM_MAX_LEN);
          GE_CHECK_GE(protected_size, 0);
          errno_t ret = EOK;
          if (need_pad_zero) {
            ret = memset_s(dst.get() + offset, static_cast<size_t>(protected_size), 0, static_cast<size_t>(size));
          } else {
            if (protected_size < size) {
              std::string error = "Failed to operate the dst memory, protected_size is " +
                  FmtToStr(protected_size) + " and size is " + FmtToStr(size);
              GE_ERRORLOG_AND_ERRORMSG(ACL_ERROR_GE_PARAM_INVALID, error.c_str());
              return ACL_ERROR_GE_PARAM_INVALID;
            }
            char *dst_data = reinterpret_cast<char *>(dst.get() + offset);
            const char *src_data = reinterpret_cast<const char *>(args.data + src_offset * size);
            for (int64_t index = 0; index < size; index++) {
              *dst_data++ = *src_data++;
            }
          }
          if (ret != EOK) {
            GELOGE(ACL_ERROR_GE_MEMORY_OPERATE_FAILED,"[Operate][DSTMemory]Failed at offset %ld, "
                   "error-code %d pad mode %d",
                   offset, ret, need_pad_zero);
            REPORT_CALL_ERROR("E19999","Failed to operate dst memory at offset %ld, "
                              "error-code %d pad mode %d",
                              offset, ret, need_pad_zero);
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

Status TransFormatHwcnToFzWithGroups(const TransArgs &args, TransResult &result, int64_t groups){
  int64_t h_dim = args.src_shape[kHwcnH];
  int64_t w_dim = args.src_shape[kHwcnW];
  int64_t c_dim = args.src_shape[kHwcnC];
  int64_t n_dim = args.src_shape[kHwcnN];
  int64_t cin_ori = c_dim;
  int64_t cout_ori = n_dim / groups;
  if (cin_ori == 0 || cout_ori == 0) {
    GELOGE(GRAPH_FAILED, "[Check][Param]Failed, cin_ori, cout_ori must not be equal 0, "
           "and current cin_ori, cout_ori, groups are %ld %ld %ld", cin_ori, cout_ori, groups);
    REPORT_CALL_ERROR("E19999", "Check graph param failed, cin_ori, cout_ori must not be equal 0,"
                      "and current cin_ori, cout_ori, groups are %ld %ld %ld",
                      cin_ori, cout_ori, groups);
    return GRAPH_FAILED;
  }
  const int64_t cube_k = GetCubeSizeByDataType(args.src_data_type);
  int64_t e_mult = std::min(
      Lcm(Lcm(cin_ori, cube_k) / (cin_ori), Lcm(cout_ori, static_cast<int64_t>(kCubeSize)) / (cout_ori)),
      groups);
  int64_t cin_opt = Ceil(e_mult * cin_ori, cube_k) * cube_k;
  int64_t cout_opt = Ceil(e_mult * cout_ori, static_cast<int64_t>(kCubeSize)) * static_cast<int64_t>(kCubeSize);
  int64_t c1_dim = cin_opt / cube_k;
  int64_t g_dim = Ceil(groups, e_mult);
  int64_t dim_cin = cin_opt / cube_k;
  int64_t data_size = GetSizeByDataType(args.src_data_type);
  int64_t size_output_data = g_dim * kDim * dim_cin * h_dim * w_dim * cout_opt * cube_k * data_size;
  if (size_output_data == 0) {
      result.length = static_cast<size_t>(size_output_data);
      return SUCCESS;
  }
  errno_t ret = EOK;
  std::shared_ptr<uint8_t> dst(new (std::nothrow) uint8_t[size_output_data], std::default_delete<uint8_t[]>());
  if (dst == nullptr) {
      GELOGE(ACL_ERROR_GE_MEMORY_ALLOCATION, "[Allocate][DSTMemory]Failed to allcoate memory "
             "for dst buf %ld when trans format from %s to %s",
             size_output_data, TypeUtils::FormatToSerialString(args.src_format).c_str(),
             TypeUtils::FormatToSerialString(args.dst_format).c_str());
      REPORT_CALL_ERROR("E19999", "Failed to allcoate memory for dst buf %ld "
                        "when trans format from %s to %s",
                        size_output_data, TypeUtils::FormatToSerialString(args.src_format).c_str(),
                        TypeUtils::FormatToSerialString(args.dst_format).c_str());
      return ACL_ERROR_GE_MEMORY_ALLOCATION;
  }
  ret = memset_s(dst.get(), static_cast<size_t>(size_output_data), 0, static_cast<size_t>(size_output_data));
  if (ret != EOK) {
      GELOGE(ACL_ERROR_GE_MEMORY_OPERATE_FAILED, "[Operate][DSTMemory]Failed, ret is %d", ret);
      REPORT_CALL_ERROR("E19999", "Failed to operate dst memory, ret is %d", ret);
      return ACL_ERROR_GE_MEMORY_OPERATE_FAILED;
  }
  for (int64_t g = 0; g < groups; g++) {
    for (int64_t d = 0; d < kDim; d++) {
      for (int64_t c = 0; c < c_dim; c++) {
        for (int64_t h = 0; h < h_dim; h++) {
          for (int64_t w = 0; w < w_dim; w++) {
            for (int64_t n = 0; n < cout_ori; n++) {
              int64_t e_val = g % e_mult;
              int64_t dst_ci = e_val * cin_ori + c;
              int64_t dst_co = e_val * cout_ori + n;
              int64_t src_co = g * cout_ori + n;
              int64_t tempory = dst_ci % cube_k;
              int64_t srx_inx = 0;
              int64_t dst_inx = (g / e_mult) * kDim * c1_dim * h_dim * w_dim * cout_opt * cube_k +
                                d * c1_dim * h_dim * w_dim * cout_opt * cube_k +
                                (dst_ci / cube_k) * h_dim * w_dim * cout_opt * cube_k +
                                h * w_dim * cout_opt * cube_k + w * cout_opt * cube_k +
                                dst_co * cube_k + tempory;
              srx_inx = d * h_dim * w_dim * c_dim * n_dim + h * w_dim * c_dim * n_dim +
                        w * c_dim * n_dim + c * n_dim + src_co;
              char *dst_data = reinterpret_cast<char *>(dst.get() + dst_inx * data_size);
              const char *src_data = reinterpret_cast<const char *>(args.data + srx_inx * data_size);
              for (int64_t index = 0; index < data_size; index++) {
                *dst_data++ = *src_data++;
              }
            }
          }
        }
      }
    }
  }
  result.data = dst;
  result.length = static_cast<size_t>(size_output_data);
  return SUCCESS;
}
Status TransFormatHwcnToFz(const TransArgs &args, TransResult &result) {
  int64_t h = args.src_shape[kHwcnH];
  int64_t w = args.src_shape[kHwcnW];
  int64_t c = args.src_shape[kHwcnC];
  int64_t n = args.src_shape[kHwcnN];
  int64_t n1n0 = Ceil(n, static_cast<int64_t>(kNiSize)) * kNiSize;
  int64_t c0 = GetCubeSizeByDataType(args.src_data_type);
  int64_t c1 = Ceil(c, c0);

  auto cn = c * n;
  auto wcn = w * cn;
  auto n1n0c0 = n1n0 * c0;
  auto wn1n0c0 = w * n1n0c0;
  auto hwn1n0c0 = h * wn1n0c0;

  int64_t data_size = GetSizeByDataType(args.src_data_type);
  int64_t dst_size = 1;
  for (auto dim : args.dst_shape) {
    dst_size *= dim;
  }
  dst_size *= data_size;
  GE_CHK_BOOL_EXEC_NOLOG(dst_size != 0, result.length = static_cast<size_t>(dst_size); return SUCCESS;);

  std::shared_ptr<uint8_t> dst(new (std::nothrow) uint8_t[dst_size], std::default_delete<uint8_t[]>());
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(
      dst == nullptr,
      GELOGE(ACL_ERROR_GE_MEMORY_ALLOCATION, "[Allocate][DSTMemory]Failed to allcoate memory "
             "for dst buf %ld when trans format from %s to %s",
             dst_size, TypeUtils::FormatToSerialString(args.src_format).c_str(),
             TypeUtils::FormatToSerialString(args.dst_format).c_str());
      REPORT_CALL_ERROR("E19999", "Failed to allcoate memory for dst buf %ld "
                        "when trans format from %s to %s",
                        dst_size, TypeUtils::FormatToSerialString(args.src_format).c_str(),
                        TypeUtils::FormatToSerialString(args.dst_format).c_str());
      return ACL_ERROR_GE_MEMORY_ALLOCATION;);

  for (int64_t c1i = 0; c1i < c1; c1i++) {
    for (int64_t hi = 0; hi < h; hi++) {
      for (int64_t wi = 0; wi < w; wi++) {
        for (int64_t n1n0i = 0; n1n0i < n1n0; n1n0i++) {
          for (int64_t c0i = 0; c0i < c0; c0i++) {
            int64_t dst_idx = c1i * hwn1n0c0 + hi * wn1n0c0 + wi * n1n0c0 + n1n0i * c0 + c0i;
            int64_t dst_offset = dst_idx * data_size;
            auto protected_size = dst_size - dst_offset < static_cast<int64_t>(SECUREC_MEM_MAX_LEN)
                                      ? dst_size - dst_offset
                                      : static_cast<int64_t>(SECUREC_MEM_MAX_LEN);
            GE_CHECK_GE(protected_size, 0);
            auto pad_zero = ((c1i * c0 + c0i) >= c) || (n1n0i >= n);
            errno_t ret = EOK;
            if (pad_zero) {
              ret = memset_s(dst.get() + dst_offset, static_cast<size_t>(protected_size), 0,
                             static_cast<size_t>(data_size));
            } else {
              if (protected_size < data_size) {
                GELOGE(ACL_ERROR_GE_PARAM_INVALID,"[Operate][DSTMemory]Failed, protected_size "
                       "is %ld and size is %ld",
                       protected_size, data_size);
                return ACL_ERROR_GE_PARAM_INVALID;
              }
              int64_t src_idx = hi * wcn + wi * cn + (c1i * c0 + c0i) * n + n1n0i;
              char *dst_data = reinterpret_cast<char *>(dst.get() + dst_offset);
              const char *src_data = reinterpret_cast<const char *>(args.data + src_idx * data_size);
              for (int64_t index = 0; index < data_size; index++) {
                *dst_data++ = *src_data++;
              }
            }
            if (ret != EOK) {
              GELOGE(ACL_ERROR_GE_MEMORY_OPERATE_FAILED, "[Operate][DSTMemory]Failed, "
                     "at offset %ld, error-code %d, pad mode %d", dst_offset, ret, pad_zero);
              REPORT_CALL_ERROR("E19999", "Failed to operate dst memoery at offset %ld, "
                                "error-code %d, pad mode %d",
                                dst_offset, ret, pad_zero);
              return ACL_ERROR_GE_MEMORY_OPERATE_FAILED;
            }
          }
        }
      }
    }
  }

  result.data = dst;
  result.length = static_cast<size_t>(dst_size);
  return SUCCESS;
}

Status TransFormatNhwcToFz(const TransArgs &args, TransResult &result) {
  int64_t n = args.src_shape[kNhwcN];
  int64_t h = args.src_shape[kNhwcH];
  int64_t w = args.src_shape[kNhwcW];
  int64_t c = args.src_shape[kNhwcC];
  auto wc = w * c;
  auto hwc = h * w * c;

  int64_t n1n0 = Ceil(n, static_cast<int64_t>(kNiSize)) * kNiSize;
  int64_t c0 = GetCubeSizeByDataType(args.src_data_type);
  int64_t c1 = Ceil(c, c0);
  auto n1n0c0 = n1n0 * c0;
  auto wn1n0c0 = w * n1n0c0;
  auto hwn1n0c0 = h * wn1n0c0;

  int64_t data_size = GetSizeByDataType(args.src_data_type);
  int64_t dst_size = 1;
  for (auto dim : args.dst_shape) {
    dst_size *= dim;
  }
  dst_size *= data_size;
  GE_CHK_BOOL_EXEC_NOLOG(dst_size != 0, result.length = static_cast<size_t>(dst_size); return SUCCESS;);

  std::shared_ptr<uint8_t> dst(new (std::nothrow) uint8_t[dst_size], std::default_delete<uint8_t[]>());
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(
      dst == nullptr,
      GELOGE(ACL_ERROR_GE_MEMORY_ALLOCATION, "[Allocate][DSTMemory]Failed to allcoate memory "
             "for dst buf %ld when trans format from %s to %s",
             dst_size, TypeUtils::FormatToSerialString(args.src_format).c_str(),
             TypeUtils::FormatToSerialString(args.dst_format).c_str());
      REPORT_CALL_ERROR("E19999", "Failed to allcoate memory for dst buf %ld "
                        "when trans format from %s to %s",
                        dst_size, TypeUtils::FormatToSerialString(args.src_format).c_str(),
                        TypeUtils::FormatToSerialString(args.dst_format).c_str());
      return ACL_ERROR_GE_MEMORY_ALLOCATION;);

  for (int64_t c1i = 0; c1i < c1; c1i++) {
    for (int64_t hi = 0; hi < h; hi++) {
      for (int64_t wi = 0; wi < w; wi++) {
        for (int64_t n1n0i = 0; n1n0i < n1n0; n1n0i++) {
          for (int64_t c0i = 0; c0i < c0; c0i++) {
            int64_t dst_idx = c1i * hwn1n0c0 + hi * wn1n0c0 + wi * n1n0c0 + n1n0i * c0 + c0i;
            int64_t dst_offset = dst_idx * data_size;
            auto protected_size = dst_size - dst_offset < static_cast<int64_t>(SECUREC_MEM_MAX_LEN)
                                      ? dst_size - dst_offset
                                      : static_cast<int64_t>(SECUREC_MEM_MAX_LEN);
            GE_CHECK_GE(protected_size, 0);
            auto pad_zero = ((c1i * c0 + c0i) >= c) || (n1n0i >= n);
            errno_t ret = EOK;
            if (pad_zero) {
              ret = memset_s(dst.get() + dst_offset, static_cast<size_t>(protected_size), 0,
                             static_cast<size_t>(data_size));
            } else {
              if (protected_size < data_size) {
                GELOGE(ACL_ERROR_GE_PARAM_INVALID, "[Operate][DSTMemory]Failed, protected_size "
                       "is %ld and size is %ld",
                       protected_size, data_size);
                return ACL_ERROR_GE_PARAM_INVALID;
              }
              int64_t src_idx = n1n0i * hwc + hi * wc + wi * c + (c1i * c0 + c0i);
              char *dst_data = reinterpret_cast<char *>(dst.get() + dst_offset);
              const char *src_data = reinterpret_cast<const char *>(args.data + src_idx * data_size);
              for (int64_t index = 0; index < data_size; index++) {
                *dst_data++ = *src_data++;
              }
            }
            if (ret != EOK) {
              GELOGE(ACL_ERROR_GE_MEMORY_OPERATE_FAILED, "[Operate][DSTMemory]Failed at offset %ld,"
                     " error-code %d, pad mode %d", dst_offset, ret, pad_zero);
              REPORT_CALL_ERROR("E19999", "Failed to operate dst memory at offset %ld, "
                                "error-code %d, pad mode %d",
                                dst_offset, ret, pad_zero);
              return ACL_ERROR_GE_MEMORY_OPERATE_FAILED;
            }
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

Status FormatTransferFractalZ::TransFormat(const TransArgs &args, TransResult &result) {
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

  if (args.src_format == FORMAT_NHWC && args.dst_format == FORMAT_FRACTAL_Z) {
    return TransFormatNhwcToFz(args, result);
  }
  if ((args.src_format == FORMAT_HWCN) && (GetPrimaryFormat(args.dst_format) == FORMAT_FRACTAL_Z)) {
    if (GetSubFormat(args.dst_format) > 1) {
      return TransFormatHwcnToFzWithGroups(args, result, GetSubFormat(args.dst_format));
    }
    return TransFormatHwcnToFz(args, result);
  }

  if (args.src_format == FORMAT_NCHW && args.dst_format == FORMAT_FRACTAL_Z) {
    return TransFormatFromNchwToFz(args, result);
  }
  return ACL_ERROR_GE_FORMAT_INVALID;
}

Status FormatTransferFractalZ::TransShape(Format src_format, const std::vector<int64_t> &src_shape, DataType data_type,
                                          Format dst_format, std::vector<int64_t> &dst_shape) {
  if (CheckDataTypeSupport(data_type) != SUCCESS) {
    return ACL_ERROR_GE_DATATYPE_INVALID;
  }

  if (src_format == FORMAT_NHWC && dst_format == FORMAT_FRACTAL_Z) {
    return TransShapeNhwcToFz(src_shape, data_type, dst_shape);
  }
  if ((src_format == FORMAT_HWCN) && (GetPrimaryFormat(dst_format) == FORMAT_FRACTAL_Z)) {
    if (GetSubFormat(dst_format) > 1) {
        return TransShapeHwcnToFzWithGroups(src_shape, data_type, dst_shape, GetSubFormat(dst_format));
     }
    return TransShapeHwcnToFz(src_shape, data_type, dst_shape);
  }
  if (src_format == FORMAT_NCHW && dst_format == FORMAT_FRACTAL_Z) {
    return TransShapeNchwToFz(src_shape, data_type, dst_shape);
  }

  return ACL_ERROR_GE_FORMAT_INVALID;
}

REGISTER_FORMAT_TRANSFER(FormatTransferFractalZ, FORMAT_NCHW, FORMAT_FRACTAL_Z)
REGISTER_FORMAT_TRANSFER(FormatTransferFractalZ, FORMAT_HWCN, FORMAT_FRACTAL_Z)
REGISTER_FORMAT_TRANSFER(FormatTransferFractalZ, FORMAT_NHWC, FORMAT_FRACTAL_Z)
}  // namespace formats
}  // namespace ge
