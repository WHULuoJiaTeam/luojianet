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

#include "plugin/engine/dnnengines.h"

#include <map>
#include <string>
#include <vector>

namespace ge {
AICoreDNNEngine::AICoreDNNEngine(const std::string &engine_name) {
  engine_attribute_.engine_name = engine_name;
  engine_attribute_.compute_cost = COST_0;
  engine_attribute_.runtime_type = DEVICE;
  engine_attribute_.engine_input_format = FORMAT_RESERVED;
  engine_attribute_.engine_output_format = FORMAT_RESERVED;
}

AICoreDNNEngine::AICoreDNNEngine(const DNNEngineAttribute &attrs) { engine_attribute_ = attrs; }

Status AICoreDNNEngine::Initialize(const std::map<std::string, std::string> &options) { return SUCCESS; }

Status AICoreDNNEngine::Finalize() { return SUCCESS; }

void AICoreDNNEngine::GetAttributes(DNNEngineAttribute &attrs) const { attrs = engine_attribute_; }

VectorCoreDNNEngine::VectorCoreDNNEngine(const std::string &engine_name) {
  engine_attribute_.engine_name = engine_name;
  engine_attribute_.compute_cost = COST_1;
  engine_attribute_.runtime_type = DEVICE;
  engine_attribute_.engine_input_format = FORMAT_RESERVED;
  engine_attribute_.engine_output_format = FORMAT_RESERVED;
}

VectorCoreDNNEngine::VectorCoreDNNEngine(const DNNEngineAttribute &attrs) { engine_attribute_ = attrs; }

Status VectorCoreDNNEngine::Initialize(const std::map<std::string, std::string> &options) { return SUCCESS; }

Status VectorCoreDNNEngine::Finalize() { return SUCCESS; }

void VectorCoreDNNEngine::GetAttributes(DNNEngineAttribute &attrs) const { attrs = engine_attribute_; }

AICpuDNNEngine::AICpuDNNEngine(const std::string &engine_name) {
  engine_attribute_.engine_name = engine_name;
  engine_attribute_.compute_cost = COST_2;
  engine_attribute_.runtime_type = DEVICE;
  engine_attribute_.engine_input_format = FORMAT_RESERVED;
  engine_attribute_.engine_output_format = FORMAT_RESERVED;
}

AICpuDNNEngine::AICpuDNNEngine(const DNNEngineAttribute &attrs) {  engine_attribute_ = attrs; }

Status AICpuDNNEngine::Initialize(const std::map<std::string, std::string> &options) { return SUCCESS; }

Status AICpuDNNEngine::Finalize() { return SUCCESS; }

void AICpuDNNEngine::GetAttributes(DNNEngineAttribute &attrs) const { attrs = engine_attribute_; }

AICpuTFDNNEngine::AICpuTFDNNEngine(const std::string &engine_name) {
  engine_attribute_.engine_name = engine_name;
  engine_attribute_.compute_cost = COST_3;
  engine_attribute_.runtime_type = DEVICE;
  engine_attribute_.engine_input_format = FORMAT_RESERVED;
  engine_attribute_.engine_output_format = FORMAT_RESERVED;
}

AICpuTFDNNEngine::AICpuTFDNNEngine(const DNNEngineAttribute &attrs) {  engine_attribute_ = attrs; }

Status AICpuTFDNNEngine::Initialize(const std::map<std::string, std::string> &options) { return SUCCESS; }

Status AICpuTFDNNEngine::Finalize() { return SUCCESS; }

void AICpuTFDNNEngine::GetAttributes(DNNEngineAttribute &attrs) const { attrs = engine_attribute_; }

GeLocalDNNEngine::GeLocalDNNEngine(const std::string &engine_name) {
  engine_attribute_.engine_name = engine_name;
  engine_attribute_.engine_input_format = FORMAT_RESERVED;
  engine_attribute_.engine_output_format = FORMAT_RESERVED;
}

GeLocalDNNEngine::GeLocalDNNEngine(const DNNEngineAttribute &attrs) { engine_attribute_ = attrs; }

Status GeLocalDNNEngine::Initialize(const std::map<std::string, std::string> &options) { return SUCCESS; }

Status GeLocalDNNEngine::Finalize() { return SUCCESS; }

void GeLocalDNNEngine::GetAttributes(DNNEngineAttribute &attrs) const { attrs = engine_attribute_; }

HostCpuDNNEngine::HostCpuDNNEngine(const std::string &engine_name) {
  engine_attribute_.engine_name = engine_name;
  engine_attribute_.compute_cost = COST_10;
  engine_attribute_.runtime_type = HOST;
  engine_attribute_.engine_input_format = FORMAT_RESERVED;
  engine_attribute_.engine_output_format = FORMAT_RESERVED;
}

HostCpuDNNEngine::HostCpuDNNEngine(const DNNEngineAttribute &attrs) { engine_attribute_ = attrs; }

Status HostCpuDNNEngine::Initialize(const std::map<std::string, std::string> &options) { return SUCCESS; }

Status HostCpuDNNEngine::Finalize() { return SUCCESS; }

void HostCpuDNNEngine::GetAttributes(DNNEngineAttribute &attrs) const { attrs = engine_attribute_; }

RtsDNNEngine::RtsDNNEngine(const std::string &engine_name) {
  engine_attribute_.engine_name = engine_name;
  engine_attribute_.engine_input_format = FORMAT_RESERVED;
  engine_attribute_.engine_output_format = FORMAT_RESERVED;
}

RtsDNNEngine::RtsDNNEngine(const DNNEngineAttribute &attrs) { engine_attribute_ = attrs; }

Status RtsDNNEngine::Initialize(const std::map<std::string, std::string> &options) { return SUCCESS; }

Status RtsDNNEngine::Finalize() { return SUCCESS; }

void RtsDNNEngine::GetAttributes(DNNEngineAttribute &attrs) const { attrs = engine_attribute_; }

HcclDNNEngine::HcclDNNEngine(const std::string &engine_name) {
  engine_attribute_.engine_name = engine_name;
  engine_attribute_.engine_input_format = FORMAT_RESERVED;
  engine_attribute_.engine_output_format = FORMAT_RESERVED;
}

HcclDNNEngine::HcclDNNEngine(const DNNEngineAttribute &attrs) { engine_attribute_ = attrs; }

Status HcclDNNEngine::Initialize(const std::map<std::string, std::string> &options) { return SUCCESS; }

Status HcclDNNEngine::Finalize() { return SUCCESS; }

void HcclDNNEngine::GetAttributes(DNNEngineAttribute &attrs) const { attrs = engine_attribute_; }
}  // namespace ge
