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

#ifndef GE_PLUGIN_ENGINE_DNNENGINES_H_
#define GE_PLUGIN_ENGINE_DNNENGINES_H_

#include <map>
#include <memory>
#include <string>

#include "framework/engine/dnnengine.h"
#include "plugin/engine/engine_manage.h"

namespace ge {
class GE_FUNC_VISIBILITY AICoreDNNEngine : public DNNEngine {
 public:
  AICoreDNNEngine() = default;
  explicit AICoreDNNEngine(const std::string &engine_name);
  explicit AICoreDNNEngine(const DNNEngineAttribute &attrs);
  ~AICoreDNNEngine() = default;

  Status Initialize(const std::map<std::string, std::string> &options);
  Status Finalize();
  void GetAttributes(DNNEngineAttribute &attr) const;

 private:
  DNNEngineAttribute engine_attribute_;
};

class GE_FUNC_VISIBILITY VectorCoreDNNEngine : public DNNEngine {
 public:
  VectorCoreDNNEngine() = default;
  explicit VectorCoreDNNEngine(const std::string &engine_name);
  explicit VectorCoreDNNEngine(const DNNEngineAttribute &attrs);
  ~VectorCoreDNNEngine() = default;

  Status Initialize(const std::map<std::string, std::string> &options);
  Status Finalize();
  void GetAttributes(DNNEngineAttribute &attr) const;

 private:
  DNNEngineAttribute engine_attribute_;
};


class GE_FUNC_VISIBILITY AICpuDNNEngine : public DNNEngine {
 public:
  AICpuDNNEngine() = default;
  explicit AICpuDNNEngine(const std::string &engine_name);
  explicit AICpuDNNEngine(const DNNEngineAttribute &attrs);
  ~AICpuDNNEngine() = default;

  Status Initialize(const std::map<std::string, std::string> &options);
  Status Finalize();
  void GetAttributes(DNNEngineAttribute &attr) const;

 private:
  DNNEngineAttribute engine_attribute_;
};

class GE_FUNC_VISIBILITY AICpuTFDNNEngine : public DNNEngine {
 public:
  AICpuTFDNNEngine() = default;
  explicit AICpuTFDNNEngine(const std::string &engine_name);
  explicit AICpuTFDNNEngine(const DNNEngineAttribute &attrs);
  ~AICpuTFDNNEngine() = default;

  Status Initialize(const std::map<std::string, std::string> &options);
  Status Finalize();
  void GetAttributes(DNNEngineAttribute &attr) const;

 private:
  DNNEngineAttribute engine_attribute_;
};

class GE_FUNC_VISIBILITY GeLocalDNNEngine : public DNNEngine {
 public:
  GeLocalDNNEngine() = default;
  explicit GeLocalDNNEngine(const std::string &engine_name);
  explicit GeLocalDNNEngine(const DNNEngineAttribute &attrs);
  ~GeLocalDNNEngine() = default;

  Status Initialize(const std::map<std::string, std::string> &options);
  Status Finalize();
  void GetAttributes(DNNEngineAttribute &attr) const;

 private:
  DNNEngineAttribute engine_attribute_;
};

class GE_FUNC_VISIBILITY HostCpuDNNEngine : public DNNEngine {
public:
  HostCpuDNNEngine() = default;
  explicit HostCpuDNNEngine(const std::string &engine_name);
  explicit HostCpuDNNEngine(const DNNEngineAttribute &attrs);
  ~HostCpuDNNEngine() = default;

  Status Initialize(const std::map<std::string, std::string> &options);
  Status Finalize();
  void GetAttributes(DNNEngineAttribute &attr) const;

private:
  DNNEngineAttribute engine_attribute_;
};

class GE_FUNC_VISIBILITY RtsDNNEngine : public DNNEngine {
 public:
  RtsDNNEngine() = default;
  explicit RtsDNNEngine(const std::string &engine_name);
  explicit RtsDNNEngine(const DNNEngineAttribute &attrs);
  ~RtsDNNEngine() = default;

  Status Initialize(const std::map<std::string, std::string> &options);
  Status Finalize();
  void GetAttributes(DNNEngineAttribute &attr) const;

 private:
  DNNEngineAttribute engine_attribute_;
};

class GE_FUNC_VISIBILITY HcclDNNEngine : public DNNEngine {
 public:
  HcclDNNEngine() = default;
  explicit HcclDNNEngine(const std::string &engine_name);
  explicit HcclDNNEngine(const DNNEngineAttribute &attrs);
  ~HcclDNNEngine() = default;

  Status Initialize(const std::map<std::string, std::string> &options);
  Status Finalize();
  void GetAttributes(DNNEngineAttribute &attr) const;

 private:
  DNNEngineAttribute engine_attribute_;
};
}  // namespace ge
#endif  // GE_PLUGIN_ENGINE_DNNENGINES_H_
