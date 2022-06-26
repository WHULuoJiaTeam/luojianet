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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_COND_VAR_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_COND_VAR_H_

#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include "minddata/dataset/util/intrp_resource.h"
#include "minddata/dataset/util/intrp_service.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
class CondVar : public IntrpResource {
 public:
  CondVar();

  ~CondVar() noexcept;

  Status Wait(std::unique_lock<std::mutex> *lck, const std::function<bool()> &pred);

  /// Timed sleep.
  /// \param lck lock
  /// \param duration time to sleep in ms
  /// \return Status code
  Status WaitFor(std::unique_lock<std::mutex> *lck, int64_t duration);

  void Interrupt() override;

  void NotifyOne() noexcept;

  void NotifyAll() noexcept;

  Status Register(std::shared_ptr<IntrpService> svc);

  std::string my_name() const;

  Status Deregister();

 protected:
  std::condition_variable cv_;
  std::shared_ptr<IntrpService> svc_;

 private:
  std::string my_name_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_COND_VAR_H_
