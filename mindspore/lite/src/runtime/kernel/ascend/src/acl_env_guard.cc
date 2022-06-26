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

#include "src/runtime/kernel/ascend/src/acl_env_guard.h"
#include "common/log_adapter.h"
#include "acl/acl.h"

namespace mindspore::kernel {
namespace acl {
std::shared_ptr<AclEnvGuard> AclEnvGuard::global_acl_env_ = nullptr;
std::mutex AclEnvGuard::global_acl_env_mutex_;

AclEnvGuard::AclEnvGuard(std::string_view cfg_file) : errno_(ACL_ERROR_NONE) {
  errno_ = aclInit(cfg_file.data());
  if (errno_ != ACL_ERROR_NONE && errno_ != ACL_ERROR_REPEAT_INITIALIZE) {
    MS_LOG(ERROR) << "Execute aclInit Failed";
    return;
  }
  MS_LOG(INFO) << "Acl init success";
}

AclEnvGuard::~AclEnvGuard() { (void)aclFinalize(); }

std::shared_ptr<AclEnvGuard> AclEnvGuard::GetAclEnv(std::string_view cfg_file) {
  std::shared_ptr<AclEnvGuard> acl_env;

  std::lock_guard<std::mutex> lock(global_acl_env_mutex_);
  acl_env = global_acl_env_;
  if (acl_env != nullptr) {
    MS_LOG(INFO) << "Acl has been initialized, skip.";
    if (!cfg_file.empty()) {
      MS_LOG(WARNING) << "Dump config file option " << cfg_file << " is ignored.";
    }
  } else {
    acl_env = std::make_shared<AclEnvGuard>(cfg_file);
    aclError ret = acl_env->GetErrno();
    if (ret != ACL_ERROR_NONE && ret != ACL_ERROR_REPEAT_INITIALIZE) {
      MS_LOG(ERROR) << "Execute aclInit Failed";
      return nullptr;
    }
    global_acl_env_ = acl_env;
    MS_LOG(INFO) << "Acl init success";
  }
  return acl_env;
}
}  // namespace acl
}  // namespace mindspore::kernel
