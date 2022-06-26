/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "tools/converter/adapter/acl/acl_pass.h"
#ifdef ENABLE_LITE_ACL
#include "mindspore/lite/tools/converter/adapter/acl/src/acl_pass_impl.h"
#endif

namespace mindspore {
namespace opt {
AclPass::AclPass(const converter::Flags &config) : Pass("ACL") {
#ifdef ENABLE_LITE_ACL
  impl_ = std::make_shared<AclPassImpl>(config);
#endif
}

bool AclPass::Run(const FuncGraphPtr &func_graph) {
#ifdef ENABLE_LITE_ACL
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "Impl is nullptr.";
    return false;
  }
  if (!impl_->Run(func_graph)) {
    MS_LOG(ERROR) << "Acl pass impl run failed.";
    return false;
  }
#endif
  return true;
}
}  // namespace opt
}  // namespace mindspore
