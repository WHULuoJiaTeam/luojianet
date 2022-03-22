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

#include "common/kernel_store.h"

namespace ge {

void KernelStore::AddKernel(const KernelBinPtr &kernel) {
  if (kernel != nullptr) {
    kernels_[kernel->GetName()] = kernel;
  }
}

bool KernelStore::Build() {
  buffer_.clear();
  size_t total_len = 0;
  for (const auto &item : kernels_) {
    auto kernel = item.second;
    total_len += sizeof(KernelStoreItemHead);
    total_len += kernel->GetName().length();
    total_len += kernel->GetBinDataSize();
  }

  try {
    buffer_.resize(total_len);
  } catch (std::bad_alloc &e) {
    GELOGE(ge::MEMALLOC_FAILED, "All build memory failed, memory size %zu", total_len);
    GELOGE(ge::MEMALLOC_FAILED, "[Malloc][Memmory]Resize buffer failed, memory size %zu, "
           "exception %s", total_len, e.what());
    REPORT_CALL_ERROR("E19999", "Resize buffer failed, memory size %zu, exception %s",
                      total_len, e.what());
    return false;
  }

  uint8_t *next_buffer = buffer_.data();
  size_t remain_len = total_len;
  errno_t mem_ret;
  for (const auto &item : kernels_) {
    auto kernel = item.second;
    KernelStoreItemHead kernel_head{};
    kernel_head.magic = kKernelItemMagic;
    kernel_head.name_len = static_cast<uint32_t>(kernel->GetName().length());
    kernel_head.bin_len = static_cast<uint32_t>(kernel->GetBinDataSize());

    GELOGD("get kernel bin name %s, addr %p, size %zu",
           kernel->GetName().c_str(), kernel->GetBinData(), kernel->GetBinDataSize());
    mem_ret = memcpy_s(next_buffer, remain_len, &kernel_head, sizeof(kernel_head));
    GE_CHK_BOOL_EXEC_NOLOG(mem_ret == EOK, return false);
    next_buffer += sizeof(kernel_head);

    mem_ret = memcpy_s(next_buffer, remain_len - sizeof(kernel_head), kernel->GetName().data(), kernel_head.name_len);
    GE_CHK_BOOL_EXEC_NOLOG(mem_ret == EOK, return false);
    next_buffer += kernel_head.name_len;

    mem_ret = memcpy_s(next_buffer, remain_len - sizeof(kernel_head) - kernel_head.name_len, kernel->GetBinData(),
                       kernel_head.bin_len);
    GE_CHK_BOOL_EXEC_NOLOG(mem_ret == EOK, return false);

    next_buffer += kernel_head.bin_len;
    remain_len = remain_len - sizeof(kernel_head) - kernel_head.name_len - kernel_head.bin_len;
  }
  kernels_.clear();
  return true;
}

const uint8_t *KernelStore::Data() const { return buffer_.data(); }

size_t KernelStore::DataSize() const { return buffer_.size(); }

bool KernelStore::Load(const uint8_t *data, const size_t &len) {
  if (data == nullptr || len == 0) {
    return false;
  }
  size_t buffer_len = len;
  while (buffer_len > sizeof(KernelStoreItemHead)) {
    const char *next_buffer = reinterpret_cast<const char *>(data) + (len - buffer_len);

    const auto *kernel_head = reinterpret_cast<const KernelStoreItemHead *>(next_buffer);
    if (buffer_len < kernel_head->name_len + kernel_head->bin_len + sizeof(KernelStoreItemHead)) {
      GELOGW("Invalid kernel block remain buffer len %zu, name len %u, bin len %u", buffer_len, kernel_head->name_len,
             kernel_head->bin_len);
      break;
    }

    next_buffer += sizeof(KernelStoreItemHead);
    std::string name(next_buffer, kernel_head->name_len);

    next_buffer += kernel_head->name_len;
    GELOGD("Load kernel from om:%s,%u,%u", name.c_str(), kernel_head->name_len, kernel_head->bin_len);
    std::vector<char> kernel_bin(next_buffer, next_buffer + kernel_head->bin_len);
    KernelBinPtr teb_kernel_ptr = ge::MakeShared<KernelBin>(name, std::move(kernel_bin));
    if (teb_kernel_ptr != nullptr) {
      kernels_.emplace(name, teb_kernel_ptr);
    }
    buffer_len -= sizeof(KernelStoreItemHead) + kernel_head->name_len + kernel_head->bin_len;
  }

  return true;
}

KernelBinPtr KernelStore::FindKernel(const std::string &name) const {
  auto it = kernels_.find(name);
  if (it != kernels_.end()) {
    return it->second;
  }
  return nullptr;
}

}  // namespace ge
