/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "cxx_api/graph/acl/acl_env_guard.h"
#include "utils/log_adapter.h"
#include "acl/acl.h"

namespace mindspore {
std::weak_ptr<AclEnvGuard> AclEnvGuard::global_acl_env_;
std::mutex AclEnvGuard::global_acl_env_mutex_;

AclEnvGuard::AclEnvGuard() {
  errno_ = aclInit(nullptr);
  if (errno_ != ACL_ERROR_NONE && errno_ != ACL_ERROR_REPEAT_INITIALIZE) {
    MS_LOG(ERROR) << "Execute aclInit Failed";
    return;
  }
  MS_LOG(INFO) << "Acl init success";
}

AclEnvGuard::~AclEnvGuard() {
  errno_ = aclFinalize();
  if (errno_ != ACL_ERROR_NONE && errno_ != ACL_ERROR_REPEAT_FINALIZE) {
    MS_LOG(ERROR) << "Finalize acl failed";
  }
  MS_LOG(INFO) << "Acl finalize success";
}

std::shared_ptr<AclEnvGuard> AclEnvGuard::GetAclEnv() {
  std::shared_ptr<AclEnvGuard> acl_env;

  std::lock_guard<std::mutex> lock(global_acl_env_mutex_);
  acl_env = global_acl_env_.lock();
  if (acl_env != nullptr) {
    MS_LOG(INFO) << "Acl has been initialized, skip.";
  } else {
    acl_env = std::make_shared<AclEnvGuard>();
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
}  // namespace mindspore
