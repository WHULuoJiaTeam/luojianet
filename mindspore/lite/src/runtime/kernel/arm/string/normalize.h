/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_STRING_NORMALIZE_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_STRING_NORMALIZE_H_

#include <vector>
#include <string>
#include "src/inner_kernel.h"
#include "include/context.h"
#include "src/common/string_utils.h"

namespace mindspore::kernel {
class NormalizeCPUKernel : public InnerKernel {
 public:
  NormalizeCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                     const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : InnerKernel(parameter, inputs, outputs, ctx) {}
  ~NormalizeCPUKernel() = default;

  int Prepare() override;
  int ReSize() override;
  int Run() override;

 private:
  std::string Trim(const std::string &str, const std::string &pattern = " \t\n\v\f\r");
  std::string GlobalReplace(const std::string &str, const std::string &reg, const std::string &replace);
  std::string Normalize(const std::string &str);
  std::vector<char *> normalized_strs;
  void FreeBuffer();
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_STRING_NORMALIZE_H_
