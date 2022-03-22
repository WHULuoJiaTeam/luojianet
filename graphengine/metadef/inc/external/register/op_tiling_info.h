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

#ifndef INC_EXTERNAL_REGISTER_OP_TILING_INFO_H_
#define INC_EXTERNAL_REGISTER_OP_TILING_INFO_H_

#include <sstream>
#include <string>
#include <vector>
#include <map>
#include "external/graph/ge_error_codes.h"
#include "external/graph/ascend_string.h"
#include "external/graph/tensor.h"

namespace optiling {
using ByteBuffer = std::stringstream;

enum TensorArgType {
  TA_NONE,
  TA_SINGLE,
  TA_LIST,
};

class TeOpVarAttrArgsImpl;
class TeOpVarAttrArgs {
  friend class VarAttrHelper;

public:
  TeOpVarAttrArgs() = default;
  ~TeOpVarAttrArgs() = default;
  const uint8_t *GetData(const std::string &name, const std::string &dtype, size_t &size) const;

private:
  std::shared_ptr<TeOpVarAttrArgsImpl> impl_;
};

struct TeOpTensor {
  std::vector<int64_t> shape;
  std::vector<int64_t> ori_shape;
  std::string format;
  std::string ori_format;
  std::string dtype;
  std::string name;
  std::map<std::string, std::string> attrs;
};

struct TeOpTensorArg {
  TensorArgType arg_type;
  std::vector<TeOpTensor> tensor;
};

struct OpRunInfo {
  uint32_t block_dim;
  std::vector<int64_t> workspaces;
  ByteBuffer tiling_data;
  bool clear_atomic;
  uint64_t tiling_key;
};

using TeOpAttrArgs = std::vector<std::string>;
using TeConstTensorData = std::tuple<const uint8_t *, size_t, ge::Tensor>;

struct TeOpParas {
  std::vector<TeOpTensorArg> inputs;
  std::vector<TeOpTensorArg> outputs;
  std::map<std::string, TeConstTensorData> const_inputs;
  TeOpAttrArgs attrs;
  std::string op_type;
  TeOpVarAttrArgs var_attrs;
};

struct OpCompileInfo {
  std::string str;
  std::string key;
};

namespace utils {
class OpRunInfoImpl;
class OpRunInfo {
public:
  OpRunInfo();
  ~OpRunInfo() = default;

  OpRunInfo(const uint32_t &block_dim, const bool &clear_atomic, const uint64_t &tiling_key);
  // Copy
  OpRunInfo(const OpRunInfo &runinfo);
  // Move
  OpRunInfo(OpRunInfo &&runinfo);
  // Copy
  OpRunInfo &operator=(const OpRunInfo &runinfo);
  // Move
  OpRunInfo &operator=(OpRunInfo &&runinfo);

  void SetBlockDim(const uint32_t &block_dim);
  uint32_t GetBlockDim() const;

  void AddWorkspace(const int64_t &workspace);
  size_t GetWorkspaceNum() const;
  ge::graphStatus GetWorkspace(const size_t &idx, int64_t &workspace) const;
  void GetAllWorkspaces(std::vector<int64_t> &workspaces) const;
  const std::vector<int64_t> &GetAllWorkspaces() const;
  void SetWorkspaces(const std::vector<int64_t> &workspaces);

  template<class T>
  void AddTilingData(const T &value) {
    AddTilingData(reinterpret_cast<const char *>(&value), sizeof(value));
  }
  void AddTilingData(const char *value, size_t size);
  ByteBuffer &GetAllTilingData();
  const ByteBuffer &GetAllTilingData() const;
  void InternelSetTiling(const ByteBuffer &value);
  void SetClearAtomic(bool clear_atomic);
  bool GetClearAtomic() const;

  void SetTilingKey(const uint64_t &new_tiling_key);
  uint64_t GetTilingKey() const;

private:
  std::shared_ptr<OpRunInfoImpl> impl_;
};

class OpCompileInfoImpl;
class OpCompileInfo {
public:
  OpCompileInfo();
  ~OpCompileInfo() = default;
  OpCompileInfo(const ge::AscendString &key, const ge::AscendString &value);
  OpCompileInfo(const std::string &key, const std::string &value);
  // Copy
  OpCompileInfo(const OpCompileInfo &compileinfo);
  // Move
  OpCompileInfo(OpCompileInfo &&compileinfo);
  // Copy
  OpCompileInfo &operator=(const OpCompileInfo &compileinfo);
  // Move
  OpCompileInfo &operator=(OpCompileInfo &&compileinfo);

  void SetKey(const ge::AscendString &key);
  const ge::AscendString &GetKey() const;

  void SetValue(const ge::AscendString &value);
  const ge::AscendString &GetValue() const;

private:
  std::shared_ptr<OpCompileInfoImpl> impl_;
};
}
}  // namespace optiling
#endif  // INC_REGISTER_OP_TILING_REGISTRY_H_
