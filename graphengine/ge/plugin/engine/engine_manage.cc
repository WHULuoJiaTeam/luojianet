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

#include "plugin/engine/engine_manage.h"

#include <map>
#include <string>
#include <utility>

#include "common/ge/ge_util.h"
#include "securec.h"
#include "framework/common/debug/ge_log.h"
#include "plugin/engine/dnnengines.h"

namespace ge {
std::unique_ptr<std::map<std::string, DNNEnginePtr>> EngineManager::engine_map_;

Status EngineManager::RegisterEngine(const std::string &engine_name, DNNEnginePtr engine_ptr) {
  if (engine_ptr == nullptr) {
    GELOGE(FAILED, "[Register][Engine] failed, as input engine_ptr is nullptr");
    REPORT_INNER_ERROR("E19999", "RegisterEngine failed for input engine_ptr is nullptr.");
    return FAILED;
  }

  if (engine_map_ == nullptr) {
    engine_map_.reset(new (std::nothrow) std::map<std::string, DNNEnginePtr>());
  }

  auto it = engine_map_->find(engine_name);
  if (it != engine_map_->end()) {
    GELOGW("engine %s already exist.", engine_name.c_str());
    return FAILED;
  }
  engine_map_->emplace(engine_name, engine_ptr);
  return SUCCESS;
}

DNNEnginePtr EngineManager::GetEngine(const std::string &engine_name) {
  auto it = engine_map_->find(engine_name);
  if (it == engine_map_->end()) {
    GELOGW("engine %s not exist.", engine_name.c_str());
    return nullptr;
  }

  auto engine = it->second;
  return engine;
}

void RegisterAiCoreEngine() {
  const std::string ai_core = "AIcoreEngine";
  std::vector<std::string> mem_type_aicore;
  mem_type_aicore.emplace_back(GE_ENGINE_ATTR_MEM_TYPE_HBM);
  DNNEngineAttribute attr_aicore = {ai_core, mem_type_aicore, COST_0, DEVICE, FORMAT_RESERVED, FORMAT_RESERVED};
  DNNEnginePtr aicore_engine_ptr = MakeShared<AICoreDNNEngine>(attr_aicore);
  if (aicore_engine_ptr == nullptr) {
    GELOGE(ge::FAILED, "[Register][AiCoreEngine] failed, as malloc shared_ptr failed.");
    REPORT_INNER_ERROR("E19999", "RegisterAiCoreEngine failed for new DNNEnginePtr failed.");
    return;
  }
  if (EngineManager::RegisterEngine(ai_core, aicore_engine_ptr) != SUCCESS) {
    GELOGW("register ai_core failed");
  }
}

void RegisterVectorEngine() {
  const std::string vector_core = "VectorEngine";
  std::vector<std::string> mem_type_aivcore;
  mem_type_aivcore.emplace_back(GE_ENGINE_ATTR_MEM_TYPE_HBM);
  DNNEngineAttribute attr_vector_core = {vector_core, mem_type_aivcore, COST_1,
                                         DEVICE,      FORMAT_RESERVED,  FORMAT_RESERVED};
  DNNEnginePtr vectorcore_engine_ptr = MakeShared<VectorCoreDNNEngine>(attr_vector_core);
  if (vectorcore_engine_ptr == nullptr) {
    GELOGE(ge::FAILED, "[Register][VectorEngine] failed, as malloc shared_ptr failed.");
    REPORT_INNER_ERROR("E19999", "RegisterVectorEngine failed for new DNNEnginePtr failed.");
    return;
  }
  if (EngineManager::RegisterEngine(vector_core, vectorcore_engine_ptr) != SUCCESS) {
    GELOGW("register vector_core failed");
  }
}

void RegisterAiCpuEngine() {
  const std::string vm_aicpu = "DNN_VM_AICPU_ASCEND";
  std::vector<std::string> mem_type_aicpu;
  mem_type_aicpu.emplace_back(GE_ENGINE_ATTR_MEM_TYPE_HBM);

  DNNEngineAttribute attr_aicpu = {vm_aicpu, mem_type_aicpu, COST_2, DEVICE, FORMAT_RESERVED, FORMAT_RESERVED};

  DNNEnginePtr vm_engine_ptr = MakeShared<AICpuDNNEngine>(attr_aicpu);
  if (vm_engine_ptr == nullptr) {
    GELOGE(ge::FAILED, "[Register][AiCpuEngine] failed, as malloc shared_ptr failed.");
    REPORT_INNER_ERROR("E19999", "RegisterAiCpuEngine failed for new DNNEnginePtr failed.");
    return;
  }
  if (EngineManager::RegisterEngine(vm_aicpu, vm_engine_ptr) != SUCCESS) {
    GELOGW("register vmAicpuEngine failed");
  }
}

void RegisterAiCpuTFEngine() {
  const std::string vm_aicpu_tf = "DNN_VM_AICPU";
  std::vector<std::string> mem_type_aicpu_tf;
  mem_type_aicpu_tf.emplace_back(GE_ENGINE_ATTR_MEM_TYPE_HBM);

  DNNEngineAttribute attr_aicpu_tf = {vm_aicpu_tf, mem_type_aicpu_tf, COST_3, DEVICE, FORMAT_RESERVED, FORMAT_RESERVED};

  DNNEnginePtr vm_engine_ptr = MakeShared<AICpuTFDNNEngine>(attr_aicpu_tf);
  if (vm_engine_ptr == nullptr) {
    GELOGE(ge::FAILED, "[Register][AiCpuTFEngine]make vm_engine_ptr failed");
    REPORT_INNER_ERROR("E19999", "RegisterAiCpuTFEngine failed for new DNNEnginePtr failed.");
    return;
  }
  if (EngineManager::RegisterEngine(vm_aicpu_tf, vm_engine_ptr) != SUCCESS) {
    GELOGW("register vmAicpuTFEngine failed");
  }
}

void RegisterGeLocalEngine() {
  const std::string vm_ge_local = "DNN_VM_GE_LOCAL";
  std::vector<std::string> mem_type_ge_local;
  mem_type_ge_local.emplace_back(GE_ENGINE_ATTR_MEM_TYPE_HBM);
  // GeLocal use minimum priority, set it as 9
  DNNEngineAttribute attr_ge_local = {vm_ge_local, mem_type_ge_local, COST_9, DEVICE, FORMAT_RESERVED, FORMAT_RESERVED};
  DNNEnginePtr ge_local_engine = MakeShared<GeLocalDNNEngine>(attr_ge_local);
  if (ge_local_engine == nullptr) {
    GELOGE(ge::FAILED, "[Register][GeLocalEngine] failed, as malloc shared_ptr failed.");
    REPORT_INNER_ERROR("E19999", "RegisterGeLocalEngine failed for new DNNEnginePtr failed.");
    return;
  }
  if (EngineManager::RegisterEngine(vm_ge_local, ge_local_engine) != SUCCESS) {
    GELOGW("register ge_local_engine failed");
  }
}

void RegisterHostCpuEngine() {
  const std::string vm_host_cpu = "DNN_VM_HOST_CPU";
  std::vector<std::string> mem_type_host_cpu;
  mem_type_host_cpu.emplace_back(GE_ENGINE_ATTR_MEM_TYPE_HBM);
  // HostCpu use minimum priority, set it as 10
  DNNEngineAttribute attr_host_cpu = {vm_host_cpu, mem_type_host_cpu, COST_10,
      HOST, FORMAT_RESERVED, FORMAT_RESERVED};
  DNNEnginePtr host_cpu_engine = MakeShared<HostCpuDNNEngine>(attr_host_cpu);
  if (host_cpu_engine == nullptr) {
    GELOGE(ge::FAILED, "[Register][HostCpuEngine] failed, as malloc shared_ptr failed.");
    REPORT_INNER_ERROR("E19999", "RegisterHostCpuEngine failed for new DNNEnginePtr failed.");
    return;
  }
  if (EngineManager::RegisterEngine(vm_host_cpu, host_cpu_engine) != SUCCESS) {
    GELOGW("register host_cpu_engine failed");
  }
}

void RegisterRtsEngine() {
  const std::string vm_rts = "DNN_VM_RTS";
  std::vector<std::string> mem_type_rts;
  mem_type_rts.emplace_back(GE_ENGINE_ATTR_MEM_TYPE_HBM);
  DNNEngineAttribute attr_rts = {vm_rts, mem_type_rts, COST_1, DEVICE, FORMAT_RESERVED, FORMAT_RESERVED};
  DNNEnginePtr rts_engine = MakeShared<RtsDNNEngine>(attr_rts);
  if (rts_engine == nullptr) {
    GELOGE(ge::FAILED, "[Register][RtsEngine] failed, as malloc shared_ptr failed.");
    REPORT_INNER_ERROR("E19999", "RegisterRtsEngine failed for new DNNEnginePtr failed.");
    return;
  }
  if (EngineManager::RegisterEngine(vm_rts, rts_engine) != SUCCESS) {
    GELOGW("register rts_engine failed");
  }
}

void RegisterHcclEngine() {
  const std::string dnn_hccl = "DNN_HCCL";
  std::vector<std::string> mem_type_hccl;
  mem_type_hccl.emplace_back(GE_ENGINE_ATTR_MEM_TYPE_HBM);
  DNNEngineAttribute attr_hccl = {dnn_hccl, mem_type_hccl, COST_1, DEVICE, FORMAT_RESERVED, FORMAT_RESERVED};
  DNNEnginePtr hccl_engine = MakeShared<HcclDNNEngine>(attr_hccl);
  if (hccl_engine == nullptr) {
    GELOGE(ge::FAILED, "[Register][HcclEngine] failed, as malloc shared_ptr failed.");
    REPORT_INNER_ERROR("E19999", "RegisterHcclEngine failed for new DNNEnginePtr failed.");
    return;
  }
  if (EngineManager::RegisterEngine(dnn_hccl, hccl_engine) != SUCCESS) {
    GELOGW("register hccl_engine failed");
  }
}

void GetDNNEngineObjs(std::map<std::string, DNNEnginePtr> &engines) {
  RegisterAiCoreEngine();
  RegisterVectorEngine();
  RegisterAiCpuTFEngine();
  RegisterAiCpuEngine();
  RegisterGeLocalEngine();
  RegisterHostCpuEngine();
  RegisterRtsEngine();
  RegisterHcclEngine();

  for (auto it = EngineManager::engine_map_->begin(); it != EngineManager::engine_map_->end(); ++it) {
    GELOGI("get engine %s from engine plugin.", it->first.c_str());
    engines.emplace(std::pair<std::string, DNNEnginePtr>(it->first, it->second));
  }

  GELOGI("after get engine, engine size: %zu", engines.size());
  return;
}
}  // namespace ge
