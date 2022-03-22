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

#ifndef GE_GRAPH_LOAD_NEW_MODEL_MANAGER_DATA_INPUTER_H_
#define GE_GRAPH_LOAD_NEW_MODEL_MANAGER_DATA_INPUTER_H_

#include <memory>
#include <string>
#include <vector>

#include "common/blocking_queue.h"
#include "framework/common/ge_types.h"
#include "framework/common/types.h"

namespace ge {
///
/// @ingroup domi_ome
/// @brief wrapper input data
/// @author
///
class InputDataWrapper {
 public:
  InputDataWrapper() : is_init(false) {}

  ~InputDataWrapper() {}

  ///
  /// @ingroup domi_ome
  /// @brief init InputData
  /// @param [in] input use input to init InputData
  /// @param [in] output data copy dest address
  /// @return SUCCESS   success
  /// @return other             init failed
  ///
  domi::Status Init(const InputData &input, const OutputData &output);

  ///
  /// @ingroup domi_ome
  /// @brief init InputData
  /// @param [in] input use input to init InputData
  /// @param [in] output data copy dest address
  /// @return SUCCESS   success
  /// @return other             init failed
  ///
  OutputData *GetOutput() { return &output_; }

  ///
  /// @ingroup domi_ome
  /// @brief return InputData
  /// @return InputData
  ///
  const InputData &GetInput() const { return input_; }

 private:
  OutputData output_;
  InputData input_;
  bool is_init;
};

///
/// @ingroup domi_ome
/// @brief manage data input
/// @author
///
class DataInputer {
 public:
  ///
  /// @ingroup domi_ome
  /// @brief constructor
  ///
  DataInputer() {}

  ///
  /// @ingroup domi_ome
  /// @brief destructor
  ///
  ~DataInputer() {}

  ///
  /// @ingroup domi_ome
  /// @brief init
  /// @return SUCCESS init success
  ///
  domi::Status Init() { return domi::SUCCESS; }

  ///
  /// @ingroup domi_ome
  /// @brief is input data full
  /// @return true full
  /// @return false not full
  ///
  bool IsDataFull() { return queue_.IsFull(); }

  ///
  /// @ingroup domi_ome
  /// @brief add input data
  /// @param [int] input data
  /// @return SUCCESS add successful
  /// @return INTERNAL_ERROR  add failed
  ///
  domi::Status Push(const std::shared_ptr<InputDataWrapper> &data) {
    bool success = queue_.Push(data, false);
    return success ? domi::SUCCESS : domi::INTERNAL_ERROR;
  }

  ///
  /// @ingroup domi_ome
  /// @brief pop input data
  /// @param [out] save popped input data
  /// @return SUCCESS pop success
  /// @return INTERNAL_ERROR  pop fail
  ///
  domi::Status Pop(std::shared_ptr<InputDataWrapper> &data) {
    bool success = queue_.Pop(data);
    return success ? domi::SUCCESS : domi::INTERNAL_ERROR;
  }

  ///
  /// @ingroup domi_ome
  /// @brief stop receiving data, invoke thread at Pop
  ///
  void Stop() { queue_.Stop(); }

  uint32_t Size() { return queue_.Size(); }

 private:
  ///
  /// @ingroup domi_ome
  /// @brief save input data queue
  ///
  BlockingQueue<std::shared_ptr<InputDataWrapper>> queue_;
};
}  // namespace ge

#endif  // GE_GRAPH_LOAD_NEW_MODEL_MANAGER_DATA_INPUTER_H_
