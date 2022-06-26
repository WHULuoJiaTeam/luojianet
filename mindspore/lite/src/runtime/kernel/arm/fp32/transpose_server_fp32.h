/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_KERNEL_CPU_ARM_FP32_TRANSPOSE_SERVER_FP32_H_
#define MINDSPORE_CCSRC_KERNEL_CPU_ARM_FP32_TRANSPOSE_SERVER_FP32_H_

#ifdef BFC_MEMORY
#include <vector>
#include "src/runtime/kernel/arm/base/transpose_base.h"
#include "nnacl/fp32/transpose_server_fp32.h"

namespace mindspore::kernel {
class TransposeServerCPUKernel : public TransposeBaseCPUKernel {
 public:
  explicit TransposeServerCPUKernel(OpParameter *param, const std::vector<lite::Tensor *> &inputs,
                                    const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : TransposeBaseCPUKernel(param, inputs, outputs, ctx) {}
  ~TransposeServerCPUKernel() override = default;

  int ReSize() override;

 private:
  void ComputeIndividualOfflineInfo();
  int ChooseThreadCuttingStrategy();
  int DoTransposeSingleThread() override;
  int DoTransposeMultiThread(int task_id) override;

  std::vector<int64_t> overflow_points_;
  std::vector<int64_t> strides_;
  std::vector<TransposeBlockBoundaryInfo> block_boundary_infos_;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_CCSRC_KERNEL_CPU_ARM_FP32_TRANSPOSE_SERVER_FP32_H_
#endif
