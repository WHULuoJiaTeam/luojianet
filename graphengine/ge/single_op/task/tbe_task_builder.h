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

#ifndef GE_SINGLE_OP_TASK_TBE_TASK_BUILDER_H_
#define GE_SINGLE_OP_TASK_TBE_TASK_BUILDER_H_

#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>

#include "graph/op_desc.h"
#include "single_op/single_op.h"
#include "single_op/single_op_model.h"

namespace ge {
class KernelHolder {
 public:
  KernelHolder(const char *stub_func, std::shared_ptr<ge::OpKernelBin> kernel_bin);
  ~KernelHolder();

  void SetBinHandle(void *bin_handle) { bin_handle_ = bin_handle; }

 private:
  friend class KernelBinRegistry;
  const char *stub_func_;
  void *bin_handle_;
  std::shared_ptr<ge::OpKernelBin> kernel_bin_;
};

class HandleHolder {
 public:
  HandleHolder(void *bin_handle);
  ~HandleHolder();

  void SetBinHandle(void *bin_handle) { bin_handle_ = bin_handle; }
  void *GetBinHandle() { return bin_handle_; }

 private:
  friend class HandleRegistry;
  void *bin_handle_ = nullptr;
};

class KernelBinRegistry {
 public:
  static KernelBinRegistry &GetInstance() {
    static KernelBinRegistry instance;
    return instance;
  }

  const char *GetUnique(const string &stub_func);

  const char *GetStubFunc(const std::string &stub_name);

  bool AddKernel(const std::string &stub_name, std::unique_ptr<KernelHolder> &&holder);

 private:
  std::map<std::string, std::unique_ptr<KernelHolder>> registered_bins_;
  std::set<std::string> unique_stubs_;
  std::mutex mutex_;
};

class HandleRegistry {
 public:
  static HandleRegistry &GetInstance() {
    static HandleRegistry instance;
    return instance;
  }

  bool AddHandle(std::unique_ptr<HandleHolder> &&holder);

 private:
  std::set<std::unique_ptr<HandleHolder>> registered_handles_;
};

class TbeTaskBuilder {
 public:
  TbeTaskBuilder(const std::string &model_name, const NodePtr &node, const domi::TaskDef &task_def);
  virtual ~TbeTaskBuilder() = default;

  Status BuildTask(TbeOpTask &task, const SingleOpModelParam &param);

 protected:
  virtual std::string GetKeyForOpParamSize() const;
  virtual std::string GetKeyForTvmMetaData() const;
  virtual TBEKernelPtr GetTbeKernel(const OpDescPtr &op_desc) const;
  virtual void GetKernelName(const OpDescPtr &op_desc, std::string &kernel_name) const;
  virtual Status InitKernelArgs(void *args_addr, size_t arg_size, const SingleOpModelParam &param);

 private:
  Status InitTilingInfo(TbeOpTask &task);
  Status SetKernelArgs(TbeOpTask &task, const SingleOpModelParam &param, const OpDescPtr &op_desc);
  Status GetSmDesc(void **sm_desc, const SingleOpModelParam &param) const;

  Status RegisterKernel(TbeOpTask &task, const SingleOpModelParam &param);
  Status RegisterKernelWithHandle(TbeOpTask &task, const SingleOpModelParam &param);
  Status DoRegisterKernel(const OpKernelBin &kernel_bin, const char *bin_file_key, void **bin_handle,
                          const SingleOpModelParam &param);
  Status DoRegisterBinary(const OpKernelBin &kernel_bin, void **bin_handle, const SingleOpModelParam &param) const;
  Status DoRegisterMeta(void *bin_handle);
  Status GetMagic(uint32_t &magic) const;

  static Status DoRegisterFunction(void *bin_handle, const char *stub_name, const char *kernel_name);

  const NodePtr node_;
  const OpDescPtr op_desc_;
  const domi::TaskDef &task_def_;
  const domi::KernelDef &kernel_def_;
  const domi::KernelDefWithHandle &kernel_def_with_handle_;
  const std::string model_name_;
  std::string stub_name_;
  void *handle_ = nullptr;
};

class AtomicAddrCleanTaskBuilder : public TbeTaskBuilder {
 public:
  AtomicAddrCleanTaskBuilder(const std::string &model_name, const NodePtr &node, const domi::TaskDef &task_def)
      : TbeTaskBuilder(model_name, node, task_def) {}
  ~AtomicAddrCleanTaskBuilder() override = default;

 protected:
  std::string GetKeyForOpParamSize() const override;
  std::string GetKeyForTvmMetaData() const override;
  TBEKernelPtr GetTbeKernel(const OpDescPtr &op_desc) const override;
  void GetKernelName(const OpDescPtr &op_desc, std::string &kernel_name) const override;
  Status InitKernelArgs(void *args_addr, size_t arg_size, const SingleOpModelParam &param) override;
};
}  // namespace ge

#endif  // GE_SINGLE_OP_TASK_TBE_TASK_BUILDER_H_
