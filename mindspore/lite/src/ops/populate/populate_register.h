/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_OPS_POPULATE_POPULATE_REGISTER_H_
#define MINDSPORE_LITE_SRC_OPS_POPULATE_POPULATE_REGISTER_H_

#include <map>
#include <vector>
#include "schema/model_generated.h"
#include "nnacl/op_base.h"
#include "src/common/common.h"
#include "src/common/log_adapter.h"
#include "src/common/prim_util.h"
#include "src/common/version_manager.h"
#include "src/common/utils.h"
#include "src/common/log_util.h"

namespace mindspore {
namespace lite {
constexpr int kOffsetTwo = 2;
constexpr int kOffsetThree = 3;
constexpr size_t kMinShapeSizeTwo = 2;
constexpr size_t kMinShapeSizeFour = 4;
typedef OpParameter *(*ParameterGen)(const void *prim);

static const std::vector<schema::PrimitiveType> string_op = {
  schema::PrimitiveType_CustomExtractFeatures, schema::PrimitiveType_CustomNormalize,
  schema::PrimitiveType_CustomPredict,         schema::PrimitiveType_HashtableLookup,
  schema::PrimitiveType_LshProjection,         schema::PrimitiveType_SkipGram};

class PopulateRegistry {
 public:
  static PopulateRegistry *GetInstance();

  void InsertParameterMap(int type, ParameterGen creator, int version) {
    parameters_[GenPrimVersionKey(type, version)] = creator;
  }

  ParameterGen GetParameterCreator(int type, int version) {
    ParameterGen param_creator = nullptr;
    auto iter = parameters_.find(GenPrimVersionKey(type, version));
    if (iter == parameters_.end()) {
#ifdef STRING_KERNEL_CLIP
      if (lite::IsContain(string_op, static_cast<schema::PrimitiveType>(type))) {
        MS_LOG(ERROR) << unsupport_string_tensor_log;
      } else {
#endif
        MS_LOG(ERROR) << "Unsupported parameter type in Create : "
                      << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(type));
#ifdef STRING_KERNEL_CLIP
      }
#endif
      return nullptr;
    }
    param_creator = iter->second;
    return param_creator;
  }

 protected:
  // key:type * 1000 + schema_version
  std::map<int, ParameterGen> parameters_;
};

class Registry {
 public:
  Registry(int primitive_type, ParameterGen creator, int version) {
    PopulateRegistry::GetInstance()->InsertParameterMap(primitive_type, creator, version);
  }
  ~Registry() = default;
};

#define REG_POPULATE(primitive_type, creator, version) \
  static Registry g_##primitive_type##version(primitive_type, creator, version);

}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_OPS_POPULATE_POPULATE_REGISTER_H_
