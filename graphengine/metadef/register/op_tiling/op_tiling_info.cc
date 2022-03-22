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

#include "register/op_tiling_info.h"

namespace optiling {
using std::make_shared;

namespace utils {
class OpRunInfoImpl {
public:
  OpRunInfoImpl() = default;
  ~OpRunInfoImpl() = default;

  OpRunInfoImpl(const uint32_t &block_dim, const bool &clear_atomic, const uint64_t &tiling_key)
          : block_dim_(block_dim), clear_atomic_(clear_atomic), tiling_key_(tiling_key) {}

  void SetBlockDim(const uint32_t &block_dim) { block_dim_ = block_dim; }

  uint32_t GetBlockDim() const { return block_dim_; }

  void AddWorkspace(const int64_t &workspace) { workspaces_.push_back(workspace); }

  size_t GetWorkspaceNum() const { return workspaces_.size(); }

  ge::graphStatus GetWorkspace(const size_t &idx, int64_t &workspace) const {
    if (!workspaces_.empty() && idx < workspaces_.size()) {
      workspace = workspaces_[idx];
      return ge::GRAPH_SUCCESS;
    }
    return ge::GRAPH_FAILED;
  }

  void GetAllWorkspaces(std::vector<int64_t> &workspaces) const { workspaces = workspaces_; }

  const std::vector<int64_t> &GetAllWorkspaces() const { return workspaces_; }

  void SetWorkspaces(const std::vector<int64_t> &workspaces) { workspaces_ = workspaces; }

  void AddTilingData(const char *value, size_t size) {
    tiling_data_.write(value, size);
    tiling_data_.flush();
  }

  const ByteBuffer &GetAllTilingData() const { return tiling_data_; }

  ByteBuffer &GetAllTilingData() { return tiling_data_; }

  void SetAllTilingData(const ByteBuffer &value) {
    tiling_data_.clear();
    std::string temp = value.str();
    tiling_data_ << temp;
  }

  void SetClearAtomic(bool clear_atomic) { clear_atomic_ = clear_atomic; }

  bool GetClearAtomic() const { return clear_atomic_; }

  void SetTilingKey(const uint64_t &tiling_key) { tiling_key_ = tiling_key; }

  uint64_t GetTilingKey() const { return tiling_key_; }

private:
  uint32_t block_dim_;
  bool clear_atomic_;
  uint64_t tiling_key_;
  ByteBuffer tiling_data_;
  std::vector<int64_t> workspaces_;
};

OpRunInfo::OpRunInfo() {
  impl_ = make_shared<OpRunInfoImpl>();
}

OpRunInfo::OpRunInfo(const uint32_t &block_dim, const bool &clear_atomic, const uint64_t &tiling_key) {
  impl_ = make_shared<OpRunInfoImpl>(block_dim, clear_atomic, tiling_key);
}

OpRunInfo::OpRunInfo(const OpRunInfo &runinfo) {
  impl_ = make_shared<OpRunInfoImpl>(runinfo.GetBlockDim(), runinfo.GetClearAtomic(), runinfo.GetTilingKey());
  std::vector<int64_t> workspaces;
  runinfo.GetAllWorkspaces(workspaces);
  impl_->SetWorkspaces(workspaces);
  impl_->SetAllTilingData(runinfo.GetAllTilingData());
}

OpRunInfo::OpRunInfo(OpRunInfo &&runinfo) {
  impl_ = std::move(runinfo.impl_);
}

OpRunInfo &OpRunInfo::operator=(const OpRunInfo &runinfo) {
  if (&runinfo != this) {
    impl_ = make_shared<OpRunInfoImpl>(runinfo.GetBlockDim(), runinfo.GetClearAtomic(), runinfo.GetTilingKey());
    std::vector<int64_t> workspaces;
    runinfo.GetAllWorkspaces(workspaces);
    impl_->SetWorkspaces(workspaces);
    impl_->SetAllTilingData(runinfo.GetAllTilingData());
  }
  return *this;
}

OpRunInfo &OpRunInfo::operator=(OpRunInfo &&runinfo) {
  if (&runinfo != this) {
    impl_ = std::move(runinfo.impl_);
  }
  return *this;
}

void OpRunInfo::SetBlockDim(const uint32_t &block_dim) {
  impl_->SetBlockDim(block_dim);
}

uint32_t OpRunInfo::GetBlockDim() const {
  return impl_->GetBlockDim();
}

void OpRunInfo::AddWorkspace(const int64_t &workspace) {
  impl_->AddWorkspace(workspace);
}

size_t OpRunInfo::GetWorkspaceNum() const {
  return impl_->GetWorkspaceNum();
}

ge::graphStatus OpRunInfo::GetWorkspace(const size_t &idx, int64_t &workspace) const {
  return impl_->GetWorkspace(idx, workspace);
}

void OpRunInfo::GetAllWorkspaces(std::vector<int64_t> &workspaces) const {
  impl_->GetAllWorkspaces(workspaces);
}

const std::vector<int64_t> &OpRunInfo::GetAllWorkspaces() const {
  return impl_->GetAllWorkspaces();
}

void OpRunInfo::SetWorkspaces(const std::vector<int64_t> &workspaces) {
  impl_->SetWorkspaces(workspaces);
}

void OpRunInfo::InternelSetTiling(const ByteBuffer &value) {
  impl_->SetAllTilingData(value);
}

void OpRunInfo::AddTilingData(const char *_value, size_t _size) {
  impl_->AddTilingData(_value, _size);
}

ByteBuffer &OpRunInfo::GetAllTilingData() {
  return impl_->GetAllTilingData();
}

const ByteBuffer &OpRunInfo::GetAllTilingData() const {
  return impl_->GetAllTilingData();
}

void OpRunInfo::SetClearAtomic(bool clear_atomic_input) {
  impl_->SetClearAtomic(clear_atomic_input);
}

bool OpRunInfo::GetClearAtomic() const {
  return impl_->GetClearAtomic();
}

void OpRunInfo::SetTilingKey(const uint64_t &new_tiling_key) {
  impl_->SetTilingKey(new_tiling_key);
}

uint64_t OpRunInfo::GetTilingKey() const {
  return impl_->GetTilingKey();
}

class OpCompileInfoImpl {
public:
  OpCompileInfoImpl() : key_(), value_() {}
  ~OpCompileInfoImpl() = default;
  OpCompileInfoImpl(const ge::AscendString &key, const ge::AscendString &value) : key_(key), value_(value) {}
  OpCompileInfoImpl(const std::string &key, const std::string &value) : key_(key.c_str()), value_(value.c_str()) {}

  void SetKey(const ge::AscendString &key) { key_ = key; }

  void SetValue(const ge::AscendString &value) { value_ = value; }

  const ge::AscendString &GetKey() const { return key_; }

  const ge::AscendString &GetValue() const { return value_; }

private:
  ge::AscendString key_;
  ge::AscendString value_;
};

OpCompileInfo::OpCompileInfo() {
  impl_ = make_shared<OpCompileInfoImpl>();
}

OpCompileInfo::OpCompileInfo(const ge::AscendString &key, const ge::AscendString &value) {
  impl_ = make_shared<OpCompileInfoImpl>(key, value);
}

OpCompileInfo::OpCompileInfo(const std::string &key, const std::string &value) {
  impl_ = make_shared<OpCompileInfoImpl>(key, value);
}

OpCompileInfo::OpCompileInfo(const OpCompileInfo &compileinfo) {
  impl_ = make_shared<OpCompileInfoImpl>();
  *impl_ = *compileinfo.impl_;
}

OpCompileInfo::OpCompileInfo(OpCompileInfo &&compileinfo) {
  impl_ = std::move(compileinfo.impl_);
}

OpCompileInfo &OpCompileInfo::operator=(const OpCompileInfo &compileinfo) {
  if (&compileinfo != this) {
    impl_ = make_shared<OpCompileInfoImpl>();
    *impl_ = *compileinfo.impl_;
  }
  return *this;
}

OpCompileInfo &OpCompileInfo::operator=(OpCompileInfo &&compileinfo) {
  if (&compileinfo != this) {
    impl_ = std::move(compileinfo.impl_);
  }
  return *this;
}

void OpCompileInfo::SetKey(const ge::AscendString &_key) {
  impl_->SetKey(_key);
}

void OpCompileInfo::SetValue(const ge::AscendString &_value) {
  impl_->SetValue(_value);
}

const ge::AscendString &OpCompileInfo::GetKey() const {
  return impl_->GetKey();
}

const ge::AscendString &OpCompileInfo::GetValue() const {
  return impl_->GetValue();
}
}  // namespace utils
}  // namespace optiling
