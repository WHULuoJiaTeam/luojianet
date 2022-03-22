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

#ifndef INC_REGISTER_GRAPH_OPTIMIZER_BUFFER_FUSION_PASS_REGISTRY_H_
#define INC_REGISTER_GRAPH_OPTIMIZER_BUFFER_FUSION_PASS_REGISTRY_H_
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "register/graph_optimizer/buffer_fusion/buffer_fusion_pass_base.h"

namespace fe {
class BufferFusionPassRegistry {
 public:
  using CreateFn = BufferFusionPassBase *(*)();
  ~BufferFusionPassRegistry();

  static BufferFusionPassRegistry &GetInstance();

  void RegisterPass(const BufferFusionPassType &pass_type, const std::string &pass_name, CreateFn create_fun);

  std::map<std::string, CreateFn> GetCreateFnByType(const BufferFusionPassType &pass_type);

 private:
  BufferFusionPassRegistry();
  class BufferFusionPassRegistryImpl;
  std::unique_ptr<BufferFusionPassRegistryImpl> impl_;
};

class BufferFusionPassRegistrar {
 public:
  BufferFusionPassRegistrar(const BufferFusionPassType &pass_type, const std::string &pass_name,
                            BufferFusionPassBase *(*create_fun)());
  ~BufferFusionPassRegistrar() {}
};

#define REGISTER_BUFFER_FUSION_PASS(pass_name, pass_type, pass_class) \
  REGISTER_BUFFER_FUSION_PASS_UNIQ_HELPER(__COUNTER__, pass_name, pass_type, pass_class)

#define REGISTER_BUFFER_FUSION_PASS_UNIQ_HELPER(ctr, pass_name, pass_type, pass_class) \
  REGISTER_BUFFER_FUSION_PASS_UNIQ(ctr, pass_name, pass_type, pass_class)

#define REGISTER_BUFFER_FUSION_PASS_UNIQ(ctr, pass_name, pass_type, pass_class)                     \
  static ::fe::BufferFusionPassRegistrar register_buffer_fusion_pass##ctr __attribute__((unused)) = \
      ::fe::BufferFusionPassRegistrar(                                                              \
          (pass_type), (pass_name), []()->::fe::BufferFusionPassBase * { return new (std::nothrow) pass_class();})

}  // namespace fe
#endif  // INC_REGISTER_GRAPH_OPTIMIZER_BUFFER_FUSION_PASS_REGISTRY_H_
