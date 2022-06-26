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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_SPARSITY_NHWC_SPMM_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_SPARSITY_NHWC_SPMM_H_

#include <vector>
#include "nnacl/matmul_parameter.h"
#include "src/inner_kernel.h"
#include "nnacl/fp32/transpose_fp32.h"

namespace mindspore::kernel {
struct SparsityWeight {
  uint32_t nnz;
  float *data;
  size_t *act_stride;
  uint32_t *non_zero_num;
};

using MatrixPackFun = void (*)(const float *src_ptr, float *dst_ptr, int row, int col);

class MatmulSparseCPUKernel : public InnerKernel {
 public:
  explicit MatmulSparseCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                                 const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : InnerKernel(parameter, inputs, outputs, ctx) {
    params_ = reinterpret_cast<MatMulParameter *>(op_parameter_);
  }
  ~MatmulSparseCPUKernel() override;
  int Prepare() override;
  int ReSize() override;
  int Run() override;
  int RunInstrinsics();

 private:
  void InitParameter();
  int PackInput();
  int PrepareWeight();
  int PrepareBias();

  MatMulParameter *params_ = nullptr;
  TransposeParameter trans_param_{};
  SparsityWeight *sparsity_weight_{nullptr};
  float *a_pack_ = nullptr;
  size_t matrix_a_pack_size_ = 0;
  float *bias_pack_ = nullptr;
};
}  // namespace mindspore::kernel
#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_SPARSITY_NHWC_SPMM_H_
