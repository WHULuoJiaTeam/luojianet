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

#include "plugin/device/cpu/kernel/matrix_band_part_cpu_kernel.h"
#include <algorithm>
#include <utility>
#include "utils/ms_utils.h"

namespace mindspore {
namespace kernel {
void MatrixBandPartCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  shapes_ = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  dim_size_ = shapes_.size();
  if (shapes_.size() < kDim2) {
    MS_LOG(EXCEPTION) << "Wrong array shape, A should be a matrix max than 2.";
  }
  m_ = shapes_[dim_size_ - kDim2];
  n_ = shapes_[dim_size_ - kDim1];
  for (size_t i = 0; i < shapes_.size() - kDim2; i++) {
    out_range_size_ *= shapes_[i];
  }
  matrix_size_ = out_range_size_ * m_ * n_;

  auto kernel_attr = GetKernelAttrFromNode(kernel_node);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "MatrixBandPart does not support this kernel data type: " << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;
}

template <typename T>
bool MatrixBandPartCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                              const std::vector<kernel::AddressPtr> &outputs) {
  T *in_value = reinterpret_cast<T *>(inputs[0]->addr);
  const int64_t *lower = reinterpret_cast<int64_t *>(inputs[1]->addr);
  const int64_t *upper = reinterpret_cast<int64_t *>(inputs[2]->addr);
  T *out_value = reinterpret_cast<T *>(outputs[0]->addr);

  const size_t l = (*lower < 0 || *lower > static_cast<int64_t>(m_)) ? m_ : static_cast<size_t>(*lower);
  const size_t u = (*upper < 0 || *upper > static_cast<int64_t>(n_)) ? n_ : static_cast<size_t>(*upper);
  auto ret_s1 = memset_s(out_value, matrix_size_ * sizeof(T), 0, matrix_size_ * sizeof(T));
  if (ret_s1 != EOK) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', memset output to 0 failed. Error no: " << ret_s1;
  }
  if (l >= m_ && u >= n_) {
    auto ret_s2 = memcpy_s(out_value, matrix_size_ * sizeof(T), in_value, matrix_size_ * sizeof(T));
    if (ret_s2 != EOK) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', memcpy to output failed. Error no: " << ret_s2;
    }
    return true;
  }
  size_t diag_len = std::min(m_, l + n_);
  auto func = [matrix_size = matrix_size_, m = m_, n = n_, diag_len, l, u, in_value, out_value](size_t spos,
                                                                                                size_t epos) {
    for (size_t t = spos; t < epos; t++) {
      const size_t i = t / diag_len;
      const size_t j = t % diag_len;
      const size_t s = j < l ? 0 : j - l;
      // When i = n - u, end is n -1, because end pos is start from 0
      const size_t e = j >= n - u ? n - 1 : j + u;
      const size_t offset = i * m * n + j * n;
      auto ret_s3 =
        memcpy_s(out_value + offset + s, matrix_size * sizeof(T), in_value + offset + s, (e - s + 1) * sizeof(T));
      if (ret_s3 != EOK) {
        MS_LOG(EXCEPTION) << "memcpy in loop failed. Error no: " << ret_s3;
      }
    }
  };
  ParallelLaunch(func, out_range_size_ * diag_len);
  return true;
}

std::vector<std::pair<KernelAttr, MatrixBandPartCpuKernelMod::MatrixBandPartFunc>>
  MatrixBandPartCpuKernelMod::func_list_ = {{KernelAttr()
                                               .AddInputAttr(kNumberTypeInt32)
                                               .AddInputAttr(kNumberTypeInt64)
                                               .AddInputAttr(kNumberTypeInt64)
                                               .AddOutputAttr(kNumberTypeInt32),
                                             &MatrixBandPartCpuKernelMod::LaunchKernel<int32_t>},
                                            {KernelAttr()
                                               .AddInputAttr(kNumberTypeInt64)
                                               .AddInputAttr(kNumberTypeInt64)
                                               .AddInputAttr(kNumberTypeInt64)
                                               .AddOutputAttr(kNumberTypeInt64),
                                             &MatrixBandPartCpuKernelMod::LaunchKernel<int64_t>},
                                            {KernelAttr()
                                               .AddInputAttr(kNumberTypeFloat32)
                                               .AddInputAttr(kNumberTypeInt64)
                                               .AddInputAttr(kNumberTypeInt64)
                                               .AddOutputAttr(kNumberTypeFloat32),
                                             &MatrixBandPartCpuKernelMod::LaunchKernel<float>},
                                            {KernelAttr()
                                               .AddInputAttr(kNumberTypeFloat64)
                                               .AddInputAttr(kNumberTypeInt64)
                                               .AddInputAttr(kNumberTypeInt64)
                                               .AddOutputAttr(kNumberTypeFloat64),
                                             &MatrixBandPartCpuKernelMod::LaunchKernel<double>}};

std::vector<KernelAttr> MatrixBandPartCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, MatrixBandPartFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, MatrixBandPart, MatrixBandPartCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
