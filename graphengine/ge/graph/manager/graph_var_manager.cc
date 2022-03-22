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

#include "graph/manager/graph_var_manager.h"

#include "graph/debug/ge_attr_define.h"
#include "graph/manager/graph_mem_manager.h"
#include "graph/manager/trans_var_data_utils.h"
#include "graph/utils/type_utils.h"
#include "graph/ge_context.h"

using std::map;
using std::string;
using std::vector;

namespace ge {
VarResource::VarResource(uint64_t session_id) : session_id_(session_id) {}

VarResource::~VarResource() {
  var_offset_map_.clear();
  var_addr_mgr_map_.clear();
  cur_var_tensor_desc_map_.clear();
  var_broad_cast_info_.clear();
}

ge::Status VarResource::GetVarAddr(const std::string &var_name, const ge::GeTensorDesc &tensor_desc, uint8_t **dev_ptr,
                                   rtMemType_t &memory_type) {
  if (dev_ptr == nullptr) {
    REPORT_INNER_ERROR("E19999", "Param dev_ptr is nullptr, var_name:%s, session_id:%lu, "
                       "check invalid", var_name.c_str(), session_id_);
    GELOGE(FAILED, "[Check][Param] Param dev_ptr is nullptr, var_name:%s, session_id:%lu",
           var_name.c_str(), session_id_);
    return FAILED;
  }
  std::string var_key = VarKey(var_name, tensor_desc);
  GELOGD("VarResource::GetVarAddr , var_key = %s", var_key.c_str());

  auto iter = var_addr_mgr_map_.find(var_key);
  if (iter == var_addr_mgr_map_.end()) {
    REPORT_INNER_ERROR("E19999", "var_key:%s can't find in var_addr_mgr_map_, var_name:%s, session_id:%lu, "
                       "check invalid", var_key.c_str(), var_name.c_str(),
                       session_id_);
    GELOGE(FAILED, "[Check][Param] var_key:%s can't find in var_addr_mgr_map_, var_name:%s, session_id:%lu",
           var_key.c_str(), var_name.c_str(), session_id_);
    return FAILED;
  }

  *dev_ptr = iter->second.address;
  memory_type = iter->second.memory_type;

  return SUCCESS;
}

void VarResource::GetAllVarAddrMgr(std::unordered_map<std::string, VarAddrMgr> &var_addr_mgr_map) {
  var_addr_mgr_map = var_addr_mgr_map_;
}

void VarResource::SetVarAddr(const std::string &var_name, const ge::GeTensorDesc &tensor_desc, uint8_t *dev_ptr,
                             rtMemType_t memory_type) {
  std::string var_key = VarKey(var_name, tensor_desc);
  GELOGI("VarResource::SetVarAddr , var_key = %s, mem_type:%u", var_key.c_str(), memory_type);
  if (var_addr_mgr_map_.count(var_key) == 0) {
    GELOGI("SetVarAddr node_name %s, tensor_desc type %s, format %s", var_name.c_str(),
           TypeUtils::DataTypeToSerialString(tensor_desc.GetDataType()).c_str(),
           TypeUtils::FormatToSerialString(tensor_desc.GetFormat()).c_str());

    VarAddrMgr var_addr_mgr;
    var_addr_mgr.address = dev_ptr;
    var_addr_mgr.tensor_desc = tensor_desc;
    var_addr_mgr_map_[var_key] = var_addr_mgr;
  }

  cur_var_tensor_desc_map_[var_name] = tensor_desc;
}

ge::Status VarResource::SaveVarAddr(const std::string &var_name, const ge::GeTensorDesc &tensor_desc, uint8_t *address,
                                    rtMemType_t memory_type) {
  std::string var_key = VarKey(var_name, tensor_desc);
  GELOGD("VarResource::SaveVarAddr, var_key = %s", var_key.c_str());
  if (var_addr_mgr_map_.count(var_key) == 0) {
    uint64_t logic_address = static_cast<uint64_t>(reinterpret_cast<std::uintptr_t>(address));
    if (memory_type != RT_MEMORY_RDMA_HBM) {
      logic_address += VarManager::Instance(session_id_)->GetVarMemLogicBase();
    }
    GELOGI("SaveVarAddr node_name %s, tensor_desc format %s, type %s.", var_name.c_str(),
           TypeUtils::FormatToSerialString(tensor_desc.GetFormat()).c_str(),
           TypeUtils::DataTypeToSerialString(tensor_desc.GetDataType()).c_str());
    VarAddrMgr var_addr_mgr;
    var_addr_mgr.address = reinterpret_cast<uint8_t *>(static_cast<std::uintptr_t>(logic_address));
    var_addr_mgr.offset = static_cast<uint64_t>(reinterpret_cast<std::uintptr_t>(address));
    var_addr_mgr.tensor_desc = tensor_desc;
    var_addr_mgr.memory_type = memory_type;
    var_addr_mgr_map_[var_key] = var_addr_mgr;
    var_offset_map_[logic_address] = memory_type;

    return SUCCESS;
  }

  REPORT_INNER_ERROR("E19999", "var_key:%s conflict in var_addr_mgr_map_, var_name:%s, session_id:%lu, "
                     "check invalid", var_key.c_str(), var_name.c_str(),
                     session_id_);
  GELOGE(FAILED, "[Check][Param] var_key:%s conflict in var_addr_mgr_map_, var_name:%s, session_id:%lu",
         var_key.c_str(), var_name.c_str(), session_id_);
  return FAILED;
}

bool VarResource::IsVarExist(const std::string &var_name, const ge::GeTensorDesc &tensor_desc) {
  std::string var_key = VarKey(var_name, tensor_desc);
  return var_addr_mgr_map_.count(var_key) != 0;
}

bool VarResource::IsVarExist(const std::string &var_name) { return cur_var_tensor_desc_map_.count(var_name) != 0; }

std::string VarResource::VarKey(const std::string &var_name, const ge::GeTensorDesc &tensor_desc) {
  std::string var_key(var_name);
  var_key.append(std::to_string(static_cast<int32_t>(tensor_desc.GetFormat())))
    .append("_")
    .append(std::to_string(static_cast<int32_t>(tensor_desc.GetDataType())));
  return var_key;
}

ge::Status VarResource::GetCurVarDesc(const std::string &var_name, ge::GeTensorDesc &tensor_desc) {
  if (cur_var_tensor_desc_map_.count(var_name) == 0) {
    return FAILED;
  }
  tensor_desc = cur_var_tensor_desc_map_[var_name];
  return SUCCESS;
}

ge::Status VarResource::RenewCurVarDesc(const std::string &var_name, const ge::OpDescPtr &op_desc) {
  if (cur_var_tensor_desc_map_.count(var_name) == 0) {
    GELOGI("There is no this node[%s] in var tensor_desc map. so no need renew!", var_name.c_str());
    return SUCCESS;
  }

  if (op_desc == nullptr) {
    REPORT_INNER_ERROR("E19999", "Param op_desc is nullptr, var_name:%s, session_id:%lu, check invalid",
                       var_name.c_str(), session_id_);
    GELOGE(FAILED, "[Check][Param] input opdesc is nullptr, var_name:%s, session_id:%lu",
           var_name.c_str(), session_id_);
    return FAILED;
  }

  ge::GeTensorDesc curr_desc;
  ge::Status ret = GetCurVarDesc(var_name, curr_desc);
  if (ret != SUCCESS) {
    GELOGE(FAILED, "[Get][CurVarDesc] fail, var_name:%s, session_id:%lu", var_name.c_str(), session_id_);
    return FAILED;
  }
  std::string key = VarKey(var_name, curr_desc);
  curr_desc.SetOriginFormat((op_desc->GetOutputDesc(0)).GetOriginFormat());
  curr_desc.SetFormat((op_desc->GetOutputDesc(0)).GetFormat());
  cur_var_tensor_desc_map_[var_name] = curr_desc;
  auto iter = var_addr_mgr_map_.find(key);
  if (iter == var_addr_mgr_map_.end()) {
    REPORT_INNER_ERROR("E19999", "var_key:%s can't find in var_addr_mgr_map_, var_name:%s, session_id:%lu, op:%s(%s), "
                       "check invalid", key.c_str(), var_name.c_str(),
                       session_id_, op_desc->GetName().c_str(), op_desc->GetType().c_str());
    GELOGE(FAILED, "[Check][Param] var_key:%s can't find in var_addr_mgr_map_, var_name:%s, session_id:%lu, op:%s(%s)",
           key.c_str(), var_name.c_str(), session_id_, op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return FAILED;
  }
  auto val = iter->second;
  val.tensor_desc.SetOriginFormat((op_desc->GetOutputDesc(0)).GetOriginFormat());
  val.tensor_desc.SetFormat((op_desc->GetOutputDesc(0)).GetFormat());
  var_addr_mgr_map_.erase(iter);
  key = VarKey(var_name, curr_desc);
  var_addr_mgr_map_[key] = val;

  return SUCCESS;
}

void VarResource::SaveBroadCastInfo(uint32_t graph_id, const VarBroadCastInfo &broad_cast_info) {
  var_broad_cast_info_[graph_id][broad_cast_info.var_name] = broad_cast_info;
}

ge::Status VarResource::GetBroadCastInfo(uint32_t graph_id, const string &var_name, VarBroadCastInfo &broad_cast_info) {
  if (var_broad_cast_info_.count(graph_id) == 0 || var_broad_cast_info_[graph_id].count(var_name) == 0) {
    return FAILED;
  }
  broad_cast_info = var_broad_cast_info_[graph_id][var_name];
  return SUCCESS;
}

bool VarResource::IsVarAddr(const int64_t &offset) { return var_offset_map_.count(offset) > 0; }

rtMemType_t VarResource::GetVarMemType(const int64_t &offset) {
  if (var_offset_map_.count(offset) > 0) {
    return var_offset_map_[offset];
  }
  return RT_MEMORY_RESERVED;
}

VarTransRoad *VarResource::GetTransRoad(const std::string &var_name) {
  auto iter = var_to_trans_road_.find(var_name);
  if (iter == var_to_trans_road_.end()) {
    return nullptr;
  } else {
    return &(iter->second);
  }
}

Status VarResource::GetChangedGraphId(const std::string &var_name, uint32_t &graph_id) {
  auto iter = var_names_to_changed_graph_id_.find(var_name);
  if (iter == var_names_to_changed_graph_id_.end()) {
    return FAILED;
  } else {
    graph_id = iter->second;
    return SUCCESS;
  }
}
Status VarResource::GetAllocatedGraphId(const std::string &var_name, uint32_t &graph_id) {
  auto iter = var_names_to_allocated_graph_id_.find(var_name);
  if (iter == var_names_to_allocated_graph_id_.end()) {
    return FAILED;
  } else {
    graph_id = iter->second;
    return SUCCESS;
  }
}

Status VarResource::SetAllocatedGraphId(const std::string &var_name, uint32_t graph_id) {
  if (GetAllocatedGraphId(var_name, graph_id) == SUCCESS) {
    GELOGW("VarManager var[%s] has been allocated in graph[%d]", var_name.c_str(), graph_id);
    return SUCCESS;
  }
  var_names_to_allocated_graph_id_[var_name] = graph_id;
  return SUCCESS;
}

MemResource::MemResource() : total_size_(0), var_mem_size_(0) {}

MemResource *MemResource::BuildMemResourceFromType(rtMemType_t mem_type) {
  switch (mem_type) {
    case RT_MEMORY_HBM:
      return new (std::nothrow) HbmMemResource();
    case RT_MEMORY_RDMA_HBM:
      return new (std::nothrow) RdmaMemResource();
    default:
      return nullptr;
  }
}

Status HbmMemResource::AssignVarMem(const std::string &var_name, uint64_t size, uint64_t session_id,
                                    size_t &mem_offset) {
  size = (size + kSessionMemAlignSize - 1) / kSessionMemAlignSize * kSessionMemAlignSize;
  uint64_t real_size = size;
  total_size_ = VarManager::Instance(session_id)->GetVarMemMaxSize();
  if (total_size_ < var_mem_size_) {
    REPORT_INNER_ERROR("E19999", "VarMemMaxSize:%lu < var_mem_size_:%lu, var_size:%lu, var_name:%s, check invalid"
                       "", total_size_, var_mem_size_, size, var_name.c_str());
    GELOGE(PARAM_INVALID, "[Check][Param] total_size_:%lu is smaller than var_mem_size_:%lu, var_name:%s",
           total_size_, var_mem_size_, var_name.c_str());
    return PARAM_INVALID;
  }
  uint64_t free_size = total_size_ - var_mem_size_;
  if (free_size < (size + kSessionMemAlignSize * kSessionMemAlignUnit)) {
    REPORT_INNER_ERROR("E19999", "free_size:%lu not enough, var_align_size:%lu, var_name:%s, check invalid",
                       free_size, size, var_name.c_str());
    GELOGE(PARAM_INVALID, "[Check][Param] Out of memory: current var size[%lu] exceeds total var size[%lu]",
           size + kSessionMemAlignSize * kSessionMemAlignUnit + var_mem_size_, total_size_);
    return PARAM_INVALID;
  }

  mem_offset = var_mem_size_;

  // offset for next, align 512 BYTE
  size = size + kSessionMemAlignSize;
  var_mem_size_ = var_mem_size_ + size;

  // align 512 BYTE
  var_mem_size_ = var_mem_size_ + kSessionMemAlignSize;
  GELOGI(
    "[IMAS]AssignVarMem Set session_%lu name[%s] output[%d]"
    "offset to [%zu] size[%lu] realsize[%lu].",
    session_id, var_name.c_str(), 0, mem_offset, (var_mem_size_ - mem_offset), real_size);
  return SUCCESS;
}

Status RdmaMemResource::AssignVarMem(const std::string &var_name, uint64_t size, uint64_t session_id, size_t &address) {
  uint8_t *buffer = MemManager::Instance().RdmaPoolInstance(RT_MEMORY_HBM).Malloc(size);
  if (buffer == nullptr) {
    REPORT_CALL_ERROR("E19999", "malloc rdma memory fail, var_size:%lu, var_name:%s",
                      size, var_name.c_str());
    GELOGE(MEMALLOC_FAILED, "[Malloc][RdmaMemory] for node %s failed, size = %lu", var_name.c_str(), size);
    return MEMALLOC_FAILED;
  }
  address = static_cast<size_t>(reinterpret_cast<uintptr_t>(buffer));
  var_mem_size_ += size;
  GELOGI("[IMAS]AssignVarMem Set session_%lu name[%s] output[%d] addr to [%p] size[%lu].",
         session_id, var_name.c_str(), 0, buffer, size);
  return SUCCESS;
}

uint64_t MemResource::GetVarMemSize() const { return var_mem_size_; }

void MemResource::UpdateVarMemSize(int64_t mem_size) { var_mem_size_ = mem_size; };

VarManager::VarManager(uint64_t session_id)
    : version_(SessionVersion::OTHER_VERSION),
      session_id_(session_id),
      device_id_(0),
      job_id_(0),
      graph_mem_max_size_(kGraphMemoryManagerMallocMaxSize),
      var_mem_max_size_(kMemoryVarManagerMallocSize),
      var_mem_logic_base_(kMemoryVarLogicBase),
      use_max_mem_size_(kUseMaxMemorySize) {}

VarManager *VarManager::Instance(uint64_t session_id) {
  GELOGD("VarManager::Instance, session id = %lu", session_id);
  return VarManagerPool::Instance().GetVarManager(session_id);
}

void VarManager::Destory() {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  GELOGI("VarManager::Destory, session id = %lu.", session_id_);
  version_ = SessionVersion::OTHER_VERSION;
  device_id_ = 0;
  session_id_ = 0;
  for (auto &memory_resource : mem_resource_map_) {
    if (memory_resource.second != nullptr) {
      delete memory_resource.second;
      memory_resource.second = nullptr;
    }
  }
  mem_resource_map_.clear();
}

ge::Status VarManager::Init(const uint32_t &version, const uint64_t &session_id, const uint32_t &device_id,
                            const uint64_t &job_id) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  GELOGI("VarManager::Init, session id = %lu.", session_id);
  if (var_resource_ == nullptr) {
    version_ = version;
    device_id_ = device_id;
    session_id_ = session_id;
    job_id_ = job_id;
    var_resource_ = std::unique_ptr<VarResource>(new (std::nothrow) VarResource(session_id_));
    if (var_resource_ == nullptr) {
      GELOGW("VarManager init failed session id = %lu.", session_id);
      return ge::INTERNAL_ERROR;
    }
  } else {
    GELOGW("VarManager::has been inited, session id = %lu.", session_id);
  }
  return SUCCESS;
}

const uint64_t &VarManager::SessionId() const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  return session_id_;
}

const uint32_t &VarManager::DeviceId() const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  return device_id_;
}

const uint64_t &VarManager::JobId() const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  return job_id_;
}

ge::Status VarManager::SetVarAddr(const std::string &var_name, const ge::GeTensorDesc &tensor_desc, uint8_t *dev_ptr,
                                  rtMemType_t memory_type) {
  GELOGI("VarManager::SetVarAddr var_name = %s, data_type = %s, data_format = %s.", var_name.c_str(),
         ge::TypeUtils::DataTypeToSerialString(tensor_desc.GetDataType()).c_str(),
         ge::TypeUtils::FormatToSerialString(tensor_desc.GetFormat()).c_str());

  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (var_resource_ == nullptr) {
    GELOGW("VarManager has not been init.");
    return ge::INTERNAL_ERROR;
  }
  var_resource_->SetVarAddr(var_name, tensor_desc, dev_ptr, memory_type);
  return ge::SUCCESS;
}

ge::Status VarManager::SaveVarAddr(const std::string &var_name, const ge::GeTensorDesc &tensor_desc, uint8_t *address,
                                   rtMemType_t memory_type) {
  GELOGI("VarManager::SaveVarAddr var_name = %s, data_type = %s, data_format = %s.", var_name.c_str(),
         ge::TypeUtils::DataTypeToSerialString(tensor_desc.GetDataType()).c_str(),
         ge::TypeUtils::FormatToSerialString(tensor_desc.GetFormat()).c_str());

  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (var_resource_ == nullptr) {
    GELOGW("VarManager has not been init.");
    return ge::INTERNAL_ERROR;
  }
  var_resource_->SaveVarAddr(var_name, tensor_desc, address, memory_type);
  return ge::SUCCESS;
}

ge::Status VarManager::GetVarAddr(const std::string &var_name, const ge::GeTensorDesc &tensor_desc, uint8_t **dev_ptr,
                                  rtMemType_t &memory_type) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  GELOGD("VarManager::GetVarAddr var_name = %s, data_type = %s, data_format = %s", var_name.c_str(),
         ge::TypeUtils::DataTypeToSerialString(tensor_desc.GetDataType()).c_str(),
         ge::TypeUtils::FormatToSerialString(tensor_desc.GetFormat()).c_str());

  if (var_resource_ == nullptr) {
    GELOGW("VarManager has not been init.");
    return ge::INTERNAL_ERROR;
  }
  auto ret = var_resource_->GetVarAddr(var_name, tensor_desc, dev_ptr, memory_type);
  if (ret != SUCCESS) {
    GELOGW("GetVarAddr fail.");
    return ge::INTERNAL_ERROR;
  }
  return SUCCESS;
}

ge::Status VarManager::GetVarAddr(const std::string &var_name, const ge::GeTensorDesc &tensor_desc, uint8_t **dev_ptr) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  rtMemType_t memory_type = RT_MEMORY_HBM;
  return GetVarAddr(var_name, tensor_desc, dev_ptr, memory_type);
}

int64_t VarManager::GetVarMemSize(rtMemType_t memory_type) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  MemResource *mem_resource = nullptr;
  auto iter = mem_resource_map_.find(memory_type);
  if (iter == mem_resource_map_.end()) {
    return 0;
  } else {
    mem_resource = iter->second;
  }

  if (mem_resource == nullptr) {
    REPORT_INNER_ERROR("E19999", "Find no mem_resource in map, memory_type:%d, session_id:%lu",
                       memory_type, session_id_);
    GELOGE(ge::INTERNAL_ERROR, "[Check][Param] MemResource is invalid, memory_type:%d, session_id:%lu",
           memory_type, session_id_);
    return 0;
  }
  return mem_resource->GetVarMemSize();
}

ge::Status VarManager::AssignVarMem(const std::string &var_name, const ge::GeTensorDesc &tensor_desc,
                                    rtMemType_t memory_type) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  GELOGI("VarManager::AssignVarMem var_name = %s, data_type = %s, data_format = %s.", var_name.c_str(),
         ge::TypeUtils::DataTypeToSerialString(tensor_desc.GetDataType()).c_str(),
         ge::TypeUtils::FormatToSerialString(tensor_desc.GetFormat()).c_str());

  int64_t tensor_desc_size = 0;
  size_t mem_offset = 0;
  ge::Status result = TensorUtils::GetSize(tensor_desc, tensor_desc_size);
  if (result != ge::SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Get size from tensor fail, var_name:%s, memory_type:%d, session_id:%lu",
                      var_name.c_str(), memory_type, session_id_);
    GELOGE(result, "[Get][Size] from tensor fail, var_name:%s, memory_type:%u, session_id:%lu",
           var_name.c_str(), memory_type, session_id_);
    return result;
  }

  MemResource *mem_resource = nullptr;
  auto it = mem_resource_map_.find(memory_type);
  if (it == mem_resource_map_.end()) {
    mem_resource = MemResource::BuildMemResourceFromType(memory_type);
    if (mem_resource == nullptr) {
      REPORT_CALL_ERROR("E19999", "memory_type:%d invalid or New MemResource fail, session_id:%lu",
                        memory_type, session_id_);
      GELOGE(ge::INTERNAL_ERROR, "[Alloc][MemResource] failed, memory_type:%u, session_id:%lu.",
             memory_type, session_id_);
      return ge::INTERNAL_ERROR;
    } else {
      mem_resource_map_[memory_type] = mem_resource;
    }
  } else {
    mem_resource = it->second;
  }

  if (mem_resource == nullptr) {
    REPORT_INNER_ERROR("E19999", "MemResource is invalid, memory_type:%d, session_id:%lu",
                       memory_type, session_id_);
    GELOGE(ge::INTERNAL_ERROR, "[Check][Param] MemResource is invalid, memory_type:%u, session_id:%lu.",
           memory_type, session_id_);
    return ge::INTERNAL_ERROR;
  }

  if (var_resource_ == nullptr) {
    REPORT_INNER_ERROR("E19999", "VarManager has not been init, memory_type:%d, session_id:%lu, "
                       "check invalid", memory_type, session_id_);
    GELOGW("VarManager has not been init.");
    return ge::INTERNAL_ERROR;
  }

  ge::GeTensorDesc cur_tensor_desc;
  int64_t cur_tensor_desc_size = 0;
  result = var_resource_->GetCurVarDesc(var_name, cur_tensor_desc);
  // reuse old format variable memory
  if (result == SUCCESS) {
    result = var_resource_->GetVarAddr(
      var_name, cur_tensor_desc, reinterpret_cast<uint8_t **>(reinterpret_cast<uintptr_t>(&mem_offset)), memory_type);
    if (result == SUCCESS) {
      result = TensorUtils::GetSize(cur_tensor_desc, cur_tensor_desc_size);
      GELOGD("tensor_desc_size is %ld, cur_tensor_desc_size is %ld, memoffset is %zu", tensor_desc_size,
             cur_tensor_desc_size, mem_offset);
    }
  }

  bool can_not_reuse_old_memory = (result != SUCCESS) || (tensor_desc_size > cur_tensor_desc_size);
  if (can_not_reuse_old_memory) {
    result = mem_resource->AssignVarMem(var_name, tensor_desc_size, session_id_, mem_offset);
    if (result != SUCCESS) {
      GELOGE(ge::INTERNAL_ERROR, "[Assign][VarMem] by offset failed, session_id:%lu.", session_id_);
      return ge::INTERNAL_ERROR;
    }

    result = var_resource_->SaveVarAddr(
      var_name, tensor_desc, reinterpret_cast<uint8_t *>(static_cast<uintptr_t>(mem_offset)), memory_type);
    if (result != SUCCESS) {
      GELOGE(ge::INTERNAL_ERROR, "[Save][VarAddr] by offset failed, memory type:%u, session_id:%lu.",
             memory_type, session_id_);
      return ge::INTERNAL_ERROR;
    }
  }
  // old not exist only save new tensor
  result = var_resource_->GetCurVarDesc(var_name, cur_tensor_desc);
  if (result != SUCCESS) {
    var_resource_->SetVarAddr(var_name, tensor_desc,
                              reinterpret_cast<uint8_t *>(static_cast<uintptr_t>(mem_offset)), memory_type);
    return SUCCESS;
  }
  bool format_changed = cur_tensor_desc.GetFormat() != tensor_desc.GetFormat() ||
                        cur_tensor_desc.GetDataType() != tensor_desc.GetDataType() ||
                        cur_tensor_desc.GetShape().GetDims() != tensor_desc.GetShape().GetDims();
  if (format_changed) {
    GELOGI("var %s assigned new memory (format, data type, shape)  (%s, %s, %zu) from (%s, %s, %zu)", var_name.c_str(),
           ge::TypeUtils::DataTypeToSerialString(tensor_desc.GetDataType()).c_str(),
           ge::TypeUtils::FormatToSerialString(tensor_desc.GetFormat()).c_str(),
           tensor_desc.GetShape().GetDims().size(),
           ge::TypeUtils::DataTypeToSerialString(cur_tensor_desc.GetDataType()).c_str(),
           ge::TypeUtils::FormatToSerialString(cur_tensor_desc.GetFormat()).c_str(),
           cur_tensor_desc.GetShape().GetDims().size());
    var_resource_->SetVarAddr(var_name, tensor_desc,
                              reinterpret_cast<uint8_t *>(static_cast<uintptr_t>(mem_offset)), memory_type);
  }

  return SUCCESS;
}

bool VarManager::IsVarExist(const std::string &var_name, const ge::GeTensorDesc &tensor_desc) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  GELOGD("VarManager::IsVarExist var_name = %s, data_type = %s, data_format = %s", var_name.c_str(),
         ge::TypeUtils::FormatToSerialString(tensor_desc.GetFormat()).c_str(),
         ge::TypeUtils::DataTypeToSerialString(tensor_desc.GetDataType()).c_str());

  if (var_resource_ == nullptr) {
    GELOGW("VarManager has not been init.");
    return false;
  }
  return var_resource_->IsVarExist(var_name, tensor_desc);
}

bool VarManager::IsVarExist(const std::string &var_name) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (var_resource_ == nullptr) {
    GELOGW("VarManager has not been init.");
    return false;
  }
  return var_resource_->IsVarExist(var_name);
}

ge::Status VarManager::GetCurVarDesc(const std::string &var_name, ge::GeTensorDesc &tensor_desc) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  GELOGI("VarManager::GetCurVarDesc var_name = %s.", var_name.c_str());

  if (var_resource_ == nullptr) {
    GELOGW("VarManager has not been init.");
    return ge::INTERNAL_ERROR;
  }
  return var_resource_->GetCurVarDesc(var_name, tensor_desc);
}

ge::Status VarManager::SaveBroadCastInfo(uint32_t graph_id, const VarBroadCastInfo &broad_cast_info) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  GELOGI(
    "VarManager::SaveBroadCastInfo var_name = %s, broadcast name = %s, "
    "idx = %d, input_offset = %ld, input_size = %lu, output_offset = %ld, "
    "output_size = %lu",
    broad_cast_info.var_name.c_str(), broad_cast_info.broadcast_name.c_str(), broad_cast_info.idx,
    broad_cast_info.input_offset, broad_cast_info.input_size, broad_cast_info.output_offset,
    broad_cast_info.output_size);
  if (var_resource_ == nullptr) {
    GELOGW("VarManager has not been init.");
    return ge::INTERNAL_ERROR;
  }
  var_resource_->SaveBroadCastInfo(graph_id, broad_cast_info);
  return SUCCESS;
}

ge::Status VarManager::RenewCurVarDesc(const std::string &var_name, ge::OpDescPtr op_desc) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  GELOGD("VarManager::RenewCurVarDesc var_name = %s.", var_name.c_str());

  if (var_resource_ == nullptr) {
    REPORT_INNER_ERROR("E19999", "VarManager has not been init, op:%s(%s), session_id:%lu, check invalid",
                       op_desc->GetName().c_str(), op_desc->GetType().c_str(),
                       session_id_);
    GELOGE(ge::INTERNAL_ERROR, "[Check][Param] VarManager has not been init, op:%s(%s), session_id:%lu",
           op_desc->GetName().c_str(), op_desc->GetType().c_str(), session_id_);
    return ge::INTERNAL_ERROR;
  }
  return var_resource_->RenewCurVarDesc(var_name, std::move(op_desc));
}

bool VarManager::IsVarAddr(const int64_t &offset) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (var_resource_ == nullptr) {
    GELOGD("VarManager has not been init.");
    return false;
  }
  return var_resource_->IsVarAddr(offset);
}

rtMemType_t VarManager::GetVarMemType(const int64_t &offset) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (var_resource_ == nullptr) {
    GELOGW("VarManager has not been init.");
    return RT_MEMORY_RESERVED;
  }
  return var_resource_->GetVarMemType(offset);
}

ge::Status VarManager::MallocVarMemory(size_t memory_size) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  uint8_t *var_mem_base = nullptr;
  string memory_key = std::to_string(session_id_);

  // malloc variable memory
  size_t var_memory_size = memory_size;

  // align 512 BYTE
  var_memory_size = (var_memory_size + kSessionMemAlignSize - 1) / kSessionMemAlignSize * kSessionMemAlignSize;

  const string purpose("variables and constant op memory in training network.");
  var_mem_base = MemManager::Instance().MemInstance(RT_MEMORY_HBM).MallocMemory(purpose, memory_key, var_memory_size);
  if (var_mem_base == nullptr) {
    GELOGE(ge::INTERNAL_ERROR, "[Malloc][VarMemory] failed, size:%zu, session_id:%s",
           var_memory_size, memory_key.c_str());
    return ge::INTERNAL_ERROR;
  }
  return SUCCESS;
}

uint8_t *VarManager::GetVarMemoryBase(rtMemType_t memory_type) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (memory_type == RT_MEMORY_RDMA_HBM) {
    return MemManager::Instance().RdmaPoolInstance(RT_MEMORY_HBM).GetRdmaBaseAddr();
  }
  string memory_key = std::to_string(session_id_);
  return MemManager::Instance().MemInstance(memory_type).GetMemoryAddr(memory_key);
}

uint8_t *VarManager::GetVarMemoryAddr(uint8_t *logic_addr, rtMemType_t memory_type) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (memory_type == RT_MEMORY_RDMA_HBM) {
    return logic_addr;
  }
  string mem_key = std::to_string(session_id_);
  uint8_t *mem_base = MemManager::Instance().MemInstance(memory_type).GetMemoryAddr(mem_key);
  if (mem_base == nullptr) {
    return nullptr;
  }
  uint8_t *mem_addr =
      logic_addr + reinterpret_cast<intptr_t>(mem_base) - VarManager::Instance(session_id_)->GetVarMemLogicBase();
  return mem_addr;
}

ge::Status VarManager::FreeVarMemory() {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  string memory_key = std::to_string(SessionId());
  return MemManager::Instance().MemInstance(RT_MEMORY_HBM).FreeMemory(memory_key);
}

ge::Status VarManager::SetTransRoad(const std::string &var_name, const VarTransRoad &trans_road) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (var_resource_ == nullptr) {
    GELOGW("VarManager has not been init.");
    return ge::INTERNAL_ERROR;
  }
  return var_resource_->SetTransRoad(var_name, trans_road);
}

VarTransRoad *VarManager::GetTransRoad(const std::string &var_name) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (var_resource_ == nullptr) {
    GELOGW("VarManager has not been init.");
    return nullptr;
  }
  return var_resource_->GetTransRoad(var_name);
}

Status VarManager::SetChangedGraphId(const std::string &var_name, uint32_t graph_id) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (var_resource_ == nullptr) {
    GELOGW("VarManager has not been init.");
    return INTERNAL_ERROR;
  }
  return var_resource_->SetChangedGraphId(var_name, graph_id);
}

Status VarManager::GetChangedGraphId(const std::string &var_name, uint32_t &graph_id) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (var_resource_ == nullptr) {
    GELOGW("VarManager has not been init.");
    return INTERNAL_ERROR;
  }
  return var_resource_->GetChangedGraphId(var_name, graph_id);
}

Status VarManager::GetTotalMemorySize(size_t &total_mem_size) {
  rtError_t rt_ret = rtSetDevice(GetContext().DeviceId());
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtSetDevice failed, device_id:%u, ret:0x%X",
                      GetContext().DeviceId(), rt_ret);
    GELOGE(RT_FAILED, "[Call][RtSetDevice] failed, device_id:%u, ret:0x%X", GetContext().DeviceId(), rt_ret);
    return RT_FAILED;
  }
  size_t free_mem = 0;
  rt_ret = rtMemGetInfoEx(RT_MEMORYINFO_HBM, &free_mem, &total_mem_size);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtMemGetInfo failed, ret:0x%X", rt_ret);
    GELOGE(RT_FAILED, "[Call][RtMemGetInfo] failed, ret:0x%X", rt_ret);
    return RT_FAILED;
  }
  rt_ret = rtDeviceReset(GetContext().DeviceId());
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtDeviceReset failed, device_id:%u, ret:0x%X",
                      GetContext().DeviceId(), rt_ret);
    GELOGE(RT_FAILED, "[Call][RtDeviceReset] failed, device_id:%u, ret:0x%X", GetContext().DeviceId(), rt_ret);
    return RT_FAILED;
  }
  return SUCCESS;
}

Status VarManager::SetMemoryMallocSize(const map<string, string> &options) {
  size_t total_mem_size = 0;
  GE_CHK_STATUS_RET_NOLOG(VarManager::GetTotalMemorySize(total_mem_size));
  GEEVENT("Total memory size is %zu", total_mem_size);

  graph_mem_max_size_ = floor(total_mem_size * kGraphMemoryManagerMallocRatio);
  var_mem_max_size_ = floor(total_mem_size * kVarMemoryManagerMallocRatio);

  auto it1 = options.find(GRAPH_MEMORY_MAX_SIZE);
  if (it1 != options.end()) {
    string graph_memory_manager_malloc_max_size = it1->second;
    ge::Status ret = ParseMemoryMallocSize(graph_memory_manager_malloc_max_size, graph_mem_max_size_);
    if (ret != SUCCESS) {
      GELOGE(ge::GE_GRAPH_OPTIONS_INVALID, "[Call][ParseMemoryMallocSize] failed, session id:%lu.", session_id_);
      return ge::GE_GRAPH_OPTIONS_INVALID;
    }
  }

  auto it2 = options.find(VARIABLE_MEMORY_MAX_SIZE);
  if (it2 != options.end()) {
    string memory_var_manager_malloc_size = it2->second;
    ge::Status ret = ParseMemoryMallocSize(memory_var_manager_malloc_size, var_mem_max_size_);
    if (ret != SUCCESS) {
      GELOGE(ge::GE_GRAPH_OPTIONS_INVALID, "[Call][ParseMemoryMallocSize] failed, session id:%lu.", session_id_);
      return ge::GE_GRAPH_OPTIONS_INVALID;
    }
  }

  GEEVENT("The graph_mem_max_size is %zu and the var_mem_max_size is %zu", graph_mem_max_size_, var_mem_max_size_);

  var_mem_logic_base_ = graph_mem_max_size_ + kGraphMemoryBuffer;
  if (var_mem_logic_base_ > kMaxMemorySize) {
    REPORT_INNER_ERROR("E19999", "var_login_base:%zu can not exeed limit:%zu, session_id:%lu, check invalid",
                       var_mem_logic_base_, kMaxMemorySize, session_id_);
    GELOGE(ge::GE_GRAPH_OPTIONS_INVALID, "[Check][Param] kMemoryVarLogicBase:%zu can not exceed "
           "max memory size:%zu, session_id:%lu.", var_mem_logic_base_, kMaxMemorySize, session_id_);
    return ge::GE_GRAPH_OPTIONS_INVALID;
  }

  use_max_mem_size_ = graph_mem_max_size_ + var_mem_max_size_;
  if (use_max_mem_size_ > kMaxMemorySize) {
    REPORT_INNER_ERROR("E19999", "all mem_use size:%zu can not exeed limit:%zu, session_id:%lu, check invalid",
                       use_max_mem_size_, kMaxMemorySize, session_id_);
    GELOGE(ge::GE_GRAPH_OPTIONS_INVALID, "[Check][Param] kUseMaxMemorySize:%zu can not exceed "
           "max memory size:%zu, session_id:%lu.", use_max_mem_size_, kMaxMemorySize, session_id_);
    return ge::GE_GRAPH_OPTIONS_INVALID;
  }
  GELOGI("Set memory malloc size successfully");
  return SUCCESS;
}

Status VarManager::ParseMemoryMallocSize(string &memory_size, size_t &result) {
  if (memory_size.empty()) {
    REPORT_INNER_ERROR("E19999", "Param memory_size is empty, session_id:%lu, check invalid",
                       session_id_);
    GELOGE(GE_GRAPH_OPTIONS_INVALID, "[Check][Param] Memory malloc size input is empty, session_id:%lu.", session_id_);
    return GE_GRAPH_OPTIONS_INVALID;
  }
  // split string by '*'
  vector<string> splits;
  std::istringstream str(memory_size);
  string str_split;
  while (getline(str, str_split, '*')) {
    splits.emplace_back(str_split);
  }

  result = 1;
  for (string split : splits) {
    // Trim
    auto it = split.find_first_not_of(" ");
    if (it != string::npos) {
      split.erase(0, it);
    }
    it = split.find_last_not_of(" ");
    if (it != string::npos) {
      split.erase(it + 1);
    }

    for (char c : split) {
      if (!isdigit(c)) {
        REPORT_INNER_ERROR("E19999", "Param memory_size:%s contains non digit, session_id:%lu, check invalid",
                           memory_size.c_str(), session_id_);
        GELOGE(GE_GRAPH_OPTIONS_INVALID,
               "[Check][Param] Memory malloc size:%s input contains non digit, session_id:%lu.",
               memory_size.c_str(), session_id_);
        return GE_GRAPH_OPTIONS_INVALID;
      }
    }
    uint64_t num = std::strtoul(split.c_str(), nullptr, 0);
    GE_IF_BOOL_EXEC(TypeUtils::CheckUint64MulOverflow(result, static_cast<uint32_t>(num)),
                    REPORT_INNER_ERROR("E19999", "Param memory_size:%s will overflow after multi all, session_id:%lu, "
                                       "check invalid", memory_size.c_str(),
                                       session_id_);
                    GELOGE(FAILED, "[Check][Param] Param memory_size:%s will overflow after multi all, session_id:%lu",
                           memory_size.c_str(), session_id_);
                    return FAILED);
    if ((num > kMaxMemorySize) || (result * static_cast<size_t>(num) > kMaxMemorySize)) {
      REPORT_INNER_ERROR("E19999", "Param memory_size:%s after multi will exceed limit:%lu, session_id:%lu, "
                         "check invalid", memory_size.c_str(), kMaxMemorySize,
                         session_id_);
      GELOGE(FAILED, "[Check][Param] Input memory size can not exceed max memory size:%zu, session_id:%lu.",
             kMaxMemorySize, session_id_);
      return FAILED;
    }
    result *= static_cast<size_t>(num);
  }

  return SUCCESS;
}

void VarManager::RemoveChangedGraphId(const std::string &var_name) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (var_resource_ == nullptr) {
    GELOGW("VarManager has not been init.");
    return;
  }
  var_resource_->RemoveChangedGraphId(var_name);
}

Status VarManager::SetAllocatedGraphId(const std::string &var_name, uint32_t graph_id) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (var_resource_ == nullptr) {
    GELOGW("VarManager has not been init.");
    return INTERNAL_ERROR;
  }
  return var_resource_->SetAllocatedGraphId(var_name, graph_id);
}

Status VarManager::GetAllocatedGraphId(const std::string &var_name, uint32_t &graph_id) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (var_resource_ == nullptr) {
    GELOGW("VarManager has not been init.");
    return INTERNAL_ERROR;
  }
  return var_resource_->GetAllocatedGraphId(var_name, graph_id);
}

void VarManager::RemoveAllocatedGraphId(const std::string &var_name) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (var_resource_ == nullptr) {
    GELOGW("VarManager has not been init.");
    return;
  }
  var_resource_->RemoveAllocatedGraphId(var_name);
}

Status VarManager::GetAllVariables(std::map<std::string, GeTensorDesc> &all_variables) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (var_resource_ == nullptr) {
    GELOGW("VarManager has not been inited.");
    return INTERNAL_ERROR;
  }
  auto new_variable_desc = var_resource_->GetAllVarDesc();
  if (new_variable_desc.size() == 0) {
    GELOGW("VarManager don't have variables.");
    return INTERNAL_ERROR;
  }

  for (auto iter = new_variable_desc.begin(); iter != new_variable_desc.end(); ++iter) {
    auto trans_road = var_resource_->GetTransRoad(iter->first);
    if (trans_road == nullptr || trans_road->empty()) {
      GELOGI("The variable %s does not have any trans road", iter->first.c_str());
      all_variables[iter->first] = iter->second;
      continue;
    }
    // get origin trans info : the first trans node info
    auto origin_trans_node_info = trans_road->at(0);
    all_variables[iter->first] = origin_trans_node_info.input;
  }
  return SUCCESS;
}

VarManagerPool::~VarManagerPool() { Destory(); }

VarManagerPool &VarManagerPool::Instance() {
  static VarManagerPool var_manager_pool;
  return var_manager_pool;
}

void VarManagerPool::Destory() noexcept {
  std::lock_guard<std::mutex> lock(var_manager_mutex_);
  for (auto &it : var_manager_map_) {
    VarManager *var_manager = it.second;
    if (var_manager != nullptr) {
      var_manager->Destory();
      delete var_manager;
      var_manager = nullptr;
    }
  }
  var_manager_map_.clear();
}

ge::Status VarManagerPool::Init() const { return SUCCESS; }

VarManager *VarManagerPool::GetVarManager(uint64_t session_id) {
  std::lock_guard<std::mutex> lock(var_manager_mutex_);
  auto it = var_manager_map_.find(session_id);
  if (it != var_manager_map_.end()) {
    GELOGD("VarManagerPool::GetVarManager");
    return it->second;
  }

  VarManager *var_manager = new (std::nothrow) VarManager(session_id);
  if (var_manager == nullptr) {
    REPORT_INNER_ERROR("E19999", "New VarManager fail, session_id:%lu", session_id);
    GELOGE(INTERNAL_ERROR, "[New][VarManager] fail, session_id:%lu", session_id);
    static VarManager new_var_manager(0);
    return &new_var_manager;
  }
  var_manager_map_[session_id] = var_manager;
  return var_manager;
}

void VarManagerPool::RemoveVarManager(uint64_t session_id) {
  VarManager *var_manager = nullptr;
  {
    std::lock_guard<std::mutex> lock(var_manager_mutex_);
    auto it = var_manager_map_.find(session_id);
    if (it != var_manager_map_.end()) {
      var_manager = it->second;
      var_manager_map_.erase(it);
    }
  }

  if (var_manager != nullptr) {
    var_manager->Destory();
    delete var_manager;
    var_manager = nullptr;
  }
}
}  // namespace ge
