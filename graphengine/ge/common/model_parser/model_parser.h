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

#ifndef GE_COMMON_MODEL_PARSER_BASE_H_
#define GE_COMMON_MODEL_PARSER_BASE_H_

#include <securec.h>
#include <memory>

#include "framework/common/debug/log.h"
#include "framework/common/ge_types.h"
#include "framework/common/types.h"
#include "framework/common/util.h"

namespace ge {
class ModelParserBase {
 public:
  /**
   * @ingroup hiai
   * @brief constructor
   */
  ModelParserBase();
  /**
   * @ingroup hiai
   * @brief destructor
   */
  ~ModelParserBase();

  /**
   * @ingroup hiai
   * @brief Parsing a model file
   * @param [in] model_file  model path
   * @param [in] priority    modle priority
   * @param [out] model_data model data
   * @return Status  result
   */
  static Status LoadFromFile(const char *model_file, int32_t priority,
                             ge::ModelData &model_data);

  /**
   * @ingroup domi_ome
   * @brief Parse model contents from the ModelData
   * @param [in] model  model data read from file
   * @param [out] model_data  address of the model data
   * @param [out] model_len  model actual length
   * If the input is an encrypted model, it needs to be deleted
   * @return SUCCESS success
   * @return others failure
   * @author
   */
  static Status ParseModelContent(const ge::ModelData &model, uint8_t *&model_data, uint32_t &model_len);
};
}  //  namespace ge
#endif  // GE_COMMON_MODEL_PARSER_BASE_H_
