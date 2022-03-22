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

#ifndef GE_GRAPH_PASSES_LABEL_MAKER_FACTORY_H_
#define GE_GRAPH_PASSES_LABEL_MAKER_FACTORY_H_

#include <map>
#include <string>
#include <memory>
#include <functional>

#include "common/ge/ge_util.h"
#include "framework/common/debug/ge_log.h"

namespace ge {
class LabelMaker;
using LabelMakerPtr = std::shared_ptr<LabelMaker>;

class LabelMakerFactory {
 public:
  // TaskManagerCreator function def
  using LabelCreatorFun = std::function<LabelMakerPtr(const ComputeGraphPtr &, const NodePtr &)>;

  static LabelMakerFactory &Instance() {
    static LabelMakerFactory instance;
    return instance;
  }

  LabelMakerPtr Create(const std::string &node_type, const ComputeGraphPtr &graph, const NodePtr &node) {
    auto it = creator_map_.find(node_type);
    if (it == creator_map_.end()) {
      GELOGW("Cannot find node type %s in map.", node_type.c_str());
      return nullptr;
    }

    return it->second(graph, node);
  }

  // LabelInfo registerar
  class Registerar {
   public:
    Registerar(const std::string &node_type, const LabelCreatorFun func) {
      LabelMakerFactory::Instance().RegisterCreator(node_type, func);
    }

    ~Registerar() = default;
  };

 private:
  LabelMakerFactory() = default;
  ~LabelMakerFactory() = default;

  // register creator, this function will call in the constructor
  void RegisterCreator(const std::string &node_type, const LabelCreatorFun func) {
    auto it = creator_map_.find(node_type);
    if (it != creator_map_.end()) {
      GELOGD("LabelMarkFactory::RegisterCreator: %s creator already exist", node_type.c_str());
      return;
    }

    creator_map_[node_type] = func;
  }

  std::map<std::string, LabelCreatorFun> creator_map_;
};

#define REGISTER_LABEL_MAKER(type, clazz)                                                         \
  LabelMakerPtr Creator_##type##_Label_Maker(const ComputeGraphPtr &graph, const NodePtr &node) { \
    std::shared_ptr<clazz> maker = nullptr;                                                       \
    maker = MakeShared<clazz>(graph, node);                                                       \
    return maker;                                                                                 \
  }                                                                                               \
  LabelMakerFactory::Registerar g_##type##_Label_Maker_Creator(type, Creator_##type##_Label_Maker);
}  // namespace ge
#endif  // GE_GRAPH_PASSES_LABEL_MAKER_FACTORY_H_