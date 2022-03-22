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

#ifndef GE_GRAPH_MANAGER_GRAPH_VAR_MANAGER_H_
#define GE_GRAPH_MANAGER_GRAPH_VAR_MANAGER_H_

#include <atomic>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/ge_types.h"
#include "framework/common/l2_cache_optimize.h"
#include "graph/ge_tensor.h"
#include "graph/op_desc.h"
#include "external/graph/tensor.h"
#include "runtime/mem.h"

namespace ge {
const size_t kGraphMemoryManagerMallocMaxSize = 26UL * 1024UL * 1024UL * 1024UL;
const size_t kMemoryVarManagerMallocSize = 5UL * 1024UL * 1024UL * 1024UL;
const size_t kMemoryVarLogicBase = 32UL * 1024UL * 1024UL * 1024UL;
const size_t kUseMaxMemorySize = kGraphMemoryManagerMallocMaxSize + kMemoryVarManagerMallocSize;
const size_t kGraphMemoryBuffer = 4UL * 1024UL * 1024UL * 1024UL;
const size_t kMaxMemorySize = 256UL * 1024UL * 1024UL * 1024UL;
const char kEnvGeuseStaticMemory[] = "GE_USE_STATIC_MEMORY";
const uint64_t kSessionMemAlignSize = 512;
const size_t kSessionMemAlignUnit = 2;
const double kGraphMemoryManagerMallocRatio = 26.0 / 32.0;
const double kVarMemoryManagerMallocRatio = 5.0 / 32.0;

enum MemStatus {
  NORMAL = 0,
  COMPILE_TASK = 1,
  RUN_TASK = 2,
};

enum SessionVersion {
  ClOUD_VERSION = 0,
  MINI_VERSION = 1,
  OTHER_VERSION = 2,
};

struct MemResourceCfg {
  uint32_t mem_status;
  size_t mem_res_size;
  MemResourceCfg() : mem_status(0), mem_res_size(0) {}
};

struct VarAddrMgr {
  ge::GeTensorDesc tensor_desc;
  uint8_t *address;
  uint64_t offset;
  rtMemType_t memory_type;
  VarAddrMgr() : address(nullptr), offset(0), memory_type(RT_MEMORY_HBM) {}
};

struct VarBroadCastInfo {
  std::string var_name;
  std::string broadcast_name;
  int idx;
  int64_t input_offset;
  uint64_t input_size;
  int64_t output_offset;
  uint64_t output_size;
};

struct VarFormatInfo {
  int format;
  int data_type;
  std::vector<int64_t> dims;
};

struct TransNodeInfo {
  std::string node_type;
  GeTensorDesc input;
  GeTensorDesc output;
};

using VarTransRoad = std::vector<TransNodeInfo>;

class VarResource {
 public:
  explicit VarResource(uint64_t session_id_);
  ~VarResource();

  ge::Status GetVarAddr(const std::string &var_name, const ge::GeTensorDesc &tensor_desc, uint8_t **dev_ptr,
                        rtMemType_t &memory_type);

  void GetAllVarAddrMgr(std::unordered_map<std::string, VarAddrMgr> &var_addr_mgr_map);

  void SetVarAddr(const std::string &var_name, const ge::GeTensorDesc &tensor_desc, uint8_t *dev_ptr,
                  rtMemType_t rtMemType_t);

  ge::Status SaveVarAddr(const std::string &var_name, const ge::GeTensorDesc &tensor_desc, uint8_t *address,
                         rtMemType_t memory_type);

  ge::Status GetCurVarDesc(const std::string &var_name, ge::GeTensorDesc &tensor_desc);

  ge::Status RenewCurVarDesc(const std::string &var_name, const ge::OpDescPtr &op_desc);

  void SaveBroadCastInfo(uint32_t graph_id, const VarBroadCastInfo &broad_cast_info);

  ge::Status GetBroadCastInfo(uint32_t graph_id, const string &var_name, VarBroadCastInfo &broad_cast_info);

  Status SetTransRoad(const std::string &var_name, const VarTransRoad &trans_road) {
    if (var_to_trans_road_.find(var_name) != var_to_trans_road_.end()) {
      GELOGW("Var name: %s has already set.", var_name.c_str());
      return GRAPH_SUCCESS;
    }
    var_to_trans_road_[var_name] = trans_road;
    return GRAPH_SUCCESS;
  }

  VarTransRoad *GetTransRoad(const std::string &var_name);

  Status SetChangedGraphId(const std::string &var_name, uint32_t graph_id) {
    var_names_to_changed_graph_id_[var_name] = graph_id;
    return SUCCESS;
  }

  Status GetChangedGraphId(const std::string &var_name, uint32_t &graph_id);

  void RemoveChangedGraphId(const std::string &var_name) { var_names_to_changed_graph_id_.erase(var_name); }

  Status SetAllocatedGraphId(const std::string &var_name, uint32_t graph_id);
  Status GetAllocatedGraphId(const std::string &var_name, uint32_t &graph_id);

  void RemoveAllocatedGraphId(const std::string &var_name) { var_names_to_allocated_graph_id_.erase(var_name); }

  bool IsVarExist(const std::string &var_name, const ge::GeTensorDesc &tensor_desc);

  bool IsVarExist(const std::string &var_name);

  bool IsVarAddr(const int64_t &offset);

  rtMemType_t GetVarMemType(const int64_t &offset);

  std::unordered_map<std::string, ge::GeTensorDesc> GetAllVarDesc() const { return cur_var_tensor_desc_map_; }

 private:
  std::string VarKey(const std::string &var_name, const ge::GeTensorDesc &tensor_desc);

  uint64_t session_id_;
  std::unordered_map<uint64_t, rtMemType_t> var_offset_map_;
  std::unordered_map<std::string, VarAddrMgr> var_addr_mgr_map_;
  std::unordered_map<std::string, ge::GeTensorDesc> cur_var_tensor_desc_map_;
  std::unordered_map<std::string, std::vector<TransNodeInfo>> var_to_trans_road_;
  std::map<std::string, uint32_t> var_names_to_changed_graph_id_;
  std::map<std::string, uint32_t> var_names_to_allocated_graph_id_;
  std::map<uint32_t, std::unordered_map<std::string, VarBroadCastInfo>> var_broad_cast_info_;
};

class MemResource {
 public:
  MemResource();
  virtual ~MemResource() = default;
  static MemResource *BuildMemResourceFromType(rtMemType_t mem_type);

  virtual Status AssignVarMem(const std::string &var_name, uint64_t size, uint64_t session_id, size_t &mem_offset) = 0;

  uint64_t GetVarMemSize() const;

  void UpdateVarMemSize(int64_t mem_size);

 protected:
  uint64_t total_size_;
  uint64_t var_mem_size_;
};

class HbmMemResource : public MemResource {
 public:
  HbmMemResource() = default;
  ~HbmMemResource() override = default;

  Status AssignVarMem(const std::string &var_name, uint64_t size, uint64_t session_id, size_t &address) override;
};

class RdmaMemResource : public MemResource {
 public:
  RdmaMemResource() = default;
  ~RdmaMemResource() override = default;

  Status AssignVarMem(const std::string &var_name, uint64_t size, uint64_t session_id, size_t &address) override;
};

class FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY VarManager {
 public:
  static VarManager *Instance(uint64_t session_id);
  explicit VarManager(uint64_t session_id);
  ~VarManager() = default;

  ge::Status Init(const uint32_t &version, const uint64_t &session_id, const uint32_t &device_id,
                  const uint64_t &job_id);

  void Destory();

  ge::Status AssignVarMem(const std::string &var_name, const ge::GeTensorDesc &tensor_desc, rtMemType_t memory_type);

  ge::Status SetVarAddr(const std::string &var_name, const ge::GeTensorDesc &tensor_desc, uint8_t *dev_ptr,
                        rtMemType_t memory_type);

  ge::Status SaveVarAddr(const std::string &var_name, const ge::GeTensorDesc &tensor_desc, uint8_t *address,
                         rtMemType_t memory_type);

  ge::Status GetVarAddr(const std::string &var_name, const ge::GeTensorDesc &tensor_desc, uint8_t **dev_ptr,
                        rtMemType_t &memory_type);

  ge::Status GetVarAddr(const std::string &var_name, const ge::GeTensorDesc &tensor_desc, uint8_t **dev_ptr);

  ge::Status SaveBroadCastInfo(uint32_t graph_id, const VarBroadCastInfo &broad_cast_info);

  ge::Status GetCurVarDesc(const std::string &var_name, ge::GeTensorDesc &tensor_desc);

  ge::Status RenewCurVarDesc(const std::string &var_name, ge::OpDescPtr op_desc);

  ge::Status MallocVarMemory(size_t memory_size = kMemoryVarManagerMallocSize);

  ge::Status FreeVarMemory();

  Status SetTransRoad(const std::string &var_name, const VarTransRoad &trans_road);

  VarTransRoad *GetTransRoad(const std::string &var_name);

  Status SetChangedGraphId(const std::string &var_name, uint32_t graph_id);

  Status GetChangedGraphId(const std::string &var_name, uint32_t &graph_id);

  Status SetMemoryMallocSize(const std::map<string, string> &options);

  const size_t &GetGraphMemoryMaxSize() const { return graph_mem_max_size_; }

  const size_t &GetVarMemMaxSize() const { return var_mem_max_size_; }

  const size_t &GetVarMemLogicBase() const { return var_mem_logic_base_; }

  const size_t &GetUseMaxMemorySize() const { return use_max_mem_size_; }

  void RemoveChangedGraphId(const std::string &var_name);

  Status SetAllocatedGraphId(const std::string &var_name, uint32_t graph_id);

  Status GetAllocatedGraphId(const std::string &var_name, uint32_t &graph_id);

  void RemoveAllocatedGraphId(const std::string &var_name);

  const uint64_t &SessionId() const;

  const uint32_t &DeviceId() const;

  const uint64_t &JobId() const;

  int64_t GetVarMemSize(rtMemType_t memory_type);

  bool IsVarExist(const std::string &var_name, const ge::GeTensorDesc &tensor_desc);

  bool IsVarExist(const std::string &var_name);

  bool IsVarAddr(const int64_t &offset);

  rtMemType_t GetVarMemType(const int64_t &offset);

  uint8_t *GetVarMemoryBase(rtMemType_t memory_type);

  uint8_t *GetVarMemoryAddr(uint8_t *logic_addr, rtMemType_t memory_type);

  Status GetAllVariables(std::map<std::string, GeTensorDesc> &all_variables);

 private:
  uint32_t version_;
  uint64_t session_id_;
  uint32_t device_id_;
  uint64_t job_id_;
  size_t graph_mem_max_size_;
  size_t var_mem_max_size_;
  size_t var_mem_logic_base_;
  size_t use_max_mem_size_;
  std::unique_ptr<ge::VarResource> var_resource_;
  map<rtMemType_t, MemResource *> mem_resource_map_;
  mutable std::recursive_mutex mutex_;

  Status ParseMemoryMallocSize(std::string &memory_size, size_t &my_size);
  Status GetTotalMemorySize(size_t &total_mem_size);
};

class VarManagerPool {
 public:
  virtual ~VarManagerPool();

  static VarManagerPool &Instance();

  VarManager *GetVarManager(uint64_t session_id);

  void RemoveVarManager(uint64_t session_id);

  void Destory() noexcept;

  ge::Status Init() const;

 private:
  VarManagerPool() = default;
  std::mutex var_manager_mutex_;
  map<uint64_t, VarManager *> var_manager_map_;
};
}  // namespace ge
#endif  // GE_GRAPH_MANAGER_GRAPH_VAR_MANAGER_H_
