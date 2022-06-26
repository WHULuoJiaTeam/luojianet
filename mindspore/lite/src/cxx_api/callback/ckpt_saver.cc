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
#include <cstddef>
#include <string>
#include <vector>
#include <memory>
#include "include/train/ckpt_saver.h"
#include "include/api/callback/ckpt_saver.h"
#include "src/cxx_api/callback/callback_impl.h"
#include "src/common/log_adapter.h"

namespace mindspore {
CkptSaver::CkptSaver(int save_every_n, const std::vector<char> &filename_prefix) {
  callback_impl_ =
    new (std::nothrow) CallbackImpl(new (std::nothrow) lite::CkptSaver(save_every_n, CharToString(filename_prefix)));
  if (callback_impl_ == nullptr) {
    MS_LOG(ERROR) << "Callback implement new failed";
  }
}

CkptSaver::~CkptSaver() {
  if (callback_impl_ == nullptr) {
    MS_LOG(ERROR) << "Callback implement is null.";
    return;
  }
  auto internal_call_back = callback_impl_->GetInternalCallback();
  if (internal_call_back != nullptr) {
    delete internal_call_back;
  }
  delete callback_impl_;
  callback_impl_ = nullptr;
}
}  // namespace mindspore
