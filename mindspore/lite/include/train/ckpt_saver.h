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
#ifndef MINDSPORE_LITE_INCLUDE_TRAIN_CKPT_SAVER_H_
#define MINDSPORE_LITE_INCLUDE_TRAIN_CKPT_SAVER_H_
#include <cstdio>
#include <vector>
#include <string>
#include <utility>
#include <unordered_map>
#include "include/train/train_loop.h"

namespace mindspore {
namespace lite {

class CkptSaver : public session::TrainLoopCallBack {
 public:
  CkptSaver(size_t save_every_n, std::string filename_prefix)
      : save_every_n_(save_every_n), filename_prefix_(std::move(filename_prefix)) {}

  ~CkptSaver() override = default;

  int EpochEnd(const session::TrainLoopCallBackData &cb_data) override {
    if ((cb_data.epoch_ + 1) % save_every_n_ == 0) {
      auto cpkt_fn = filename_prefix_ + "_trained_" + std::to_string(cb_data.epoch_ + 1) + ".ms";
      remove(cpkt_fn.c_str());
      cb_data.session_->Export(cpkt_fn);
    }
    return session::RET_CONTINUE;
  }

 private:
  size_t save_every_n_;
  std::string filename_prefix_;
};

}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_INCLUDE_TRAIN_CKPT_SAVER_H_
