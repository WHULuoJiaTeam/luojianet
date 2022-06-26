/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#ifndef AICPU_OPS_AICPU_COMMON_KERNEL_BASE_H_
#define AICPU_OPS_AICPU_COMMON_KERNEL_BASE_H_

#include <cstdint>
#include <vector>
#include <string>

#include "common/kernel_util.h"
#include "aicpu/common/aicpu_task_struct.h"
#include "securec/include/securec.h"
#include "common/tensor.h"
#include "cce/fwk_adpt_struct.h"
#include "common/kernel_log.h"
#include "proto/aicpu_tensor.pb.h"

namespace aicpu {
class KernelBase {
 public:
  explicit KernelBase(const std::string &kernel_name);

  ~KernelBase() = default;

  uint32_t Compute(void *param);
  size_t GetDataTypeSize(::aicpuops::DataType data_type);

 protected:
  virtual uint32_t ParseKernelParam() = 0;
  virtual uint32_t DoCompute() = 0;

  template <typename T>
  uint32_t ParseExtendParam(T *param_var, std::string param_name);

  uint32_t ParseNodeDef();

  uint32_t ParseExtInfo();

  uint32_t ParseExtShapeType(FWKAdapter::ExtInfo *ext_info);

  uint32_t ParseExtInputShape(FWKAdapter::ExtInfo *ext_info);

  uint32_t ParseExtOutputShape(FWKAdapter::ExtInfo *ext_info);

  uint32_t UpdateInputShape();

  uint32_t UpdateOutputShape();

 private:
  KernelBase(const KernelBase &) = delete;
  KernelBase &operator=(const KernelBase &) = delete;
  KernelBase(KernelBase &&) = delete;
  KernelBase &operator=(KernelBase &&) = delete;

  uint32_t ParseParam(void *param);

 protected:
  std::string kernel_name_;
  std::vector<uintptr_t> io_addrs_;
  uint32_t extend_param_len_;
  uint8_t *extend_param_base_;
  AicpuParamHead *param_head_;
  bool unknow_shape_;
  aicpuops::NodeDef node_def_;
  std::vector<FWKAdapter::ShapeAndType *> input_shape_and_type_;
  std::vector<FWKAdapter::ShapeAndType *> output_shape_and_type_;
};
}  // namespace aicpu
#endif  // AICPU_OPS_AICPU_COMMON_KERNEL_BASE_H_
