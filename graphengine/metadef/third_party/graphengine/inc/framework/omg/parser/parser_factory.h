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

#ifndef INC_FRAMEWORK_OMG_PARSER_PARSER_FACTORY_H_
#define INC_FRAMEWORK_OMG_PARSER_PARSER_FACTORY_H_

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include "framework/omg/omg_inner_types.h"
#include "framework/omg/parser/parser_types.h"

using Status = domi::Status;

namespace domi {
class WeightsParser;
class ModelParser;

typedef std::shared_ptr<ModelParser> (*MODEL_PARSER_CREATOR_FUN)(void);

// Create modelparser for different frameworks
class GE_FUNC_VISIBILITY ModelParserFactory {
 public:
  static ModelParserFactory *Instance();

  /**
   * @ingroup domi_omg
   * @brief Create a modelparser based on the type entered
   * @param [in] type Framework type
   * @return Created modelparser
   */
  std::shared_ptr<ModelParser> CreateModelParser(const domi::FrameworkType type);

  /**
   * @ingroup domi_omg
   * @brief Register create function
   * @param [in] type Framework type
   * @param [in] fun ModelParser's create function
   */
  void RegisterCreator(const domi::FrameworkType type, MODEL_PARSER_CREATOR_FUN fun);

 protected:
  ModelParserFactory() {}
  ~ModelParserFactory();

 private:
  std::map<domi::FrameworkType, MODEL_PARSER_CREATOR_FUN> creator_map_;
};  // end class ModelParserFactory

class GE_FUNC_VISIBILITY ModelParserRegisterar {
 public:
  ModelParserRegisterar(const domi::FrameworkType type, MODEL_PARSER_CREATOR_FUN fun) {
    ModelParserFactory::Instance()->RegisterCreator(type, fun);
  }
  ~ModelParserRegisterar() {}
};

// Registration macros for model parsers
#define REGISTER_MODEL_PARSER_CREATOR(type, clazz)               \
  std::shared_ptr<ModelParser> Creator_##type##_Model_Parser() { \
    std::shared_ptr<clazz> ptr = nullptr;                        \
    try {                                                        \
      ptr = make_shared<clazz>();                                \
    } catch (...) {                                              \
      ptr = nullptr;                                             \
    }                                                            \
    return std::shared_ptr<ModelParser>(ptr);                    \
  }                                                              \
  ModelParserRegisterar g_##type##_Model_Parser_Creator(type, Creator_##type##_Model_Parser)

typedef std::shared_ptr<WeightsParser> (*WEIGHTS_PARSER_CREATOR_FUN)(void);

// Create weightsparser for different frameworks
class GE_FUNC_VISIBILITY WeightsParserFactory {
 public:
  static WeightsParserFactory *Instance();

  /**
   * @ingroup domi_omg
   * @brief Create weightsparser based on the type entered
   * @param [in] type Framework type
   * @return Created weightsparser
   */
  std::shared_ptr<WeightsParser> CreateWeightsParser(const domi::FrameworkType type);

  /**
   * @ingroup domi_omg
   * @brief Register create function
   * @param [in] type Framework type
   * @param [in] fun WeightsParser's create function
   */
  void RegisterCreator(const domi::FrameworkType type, WEIGHTS_PARSER_CREATOR_FUN fun);

 protected:
  WeightsParserFactory() {}
  ~WeightsParserFactory();

 private:
  std::map<domi::FrameworkType, WEIGHTS_PARSER_CREATOR_FUN> creator_map_;
};  // end class WeightsParserFactory

class GE_FUNC_VISIBILITY WeightsParserRegisterar {
 public:
  WeightsParserRegisterar(const domi::FrameworkType type, WEIGHTS_PARSER_CREATOR_FUN fun) {
    WeightsParserFactory::Instance()->RegisterCreator(type, fun);
  }
  ~WeightsParserRegisterar() {}
};

// Register macro of weight resolver
#define REGISTER_WEIGHTS_PARSER_CREATOR(type, clazz)                 \
  std::shared_ptr<WeightsParser> Creator_##type##_Weights_Parser() { \
    std::shared_ptr<clazz> ptr = nullptr;                            \
    try {                                                            \
      ptr = make_shared<clazz>();                                    \
    } catch (...) {                                                  \
      ptr = nullptr;                                                 \
    }                                                                \
    return std::shared_ptr<WeightsParser>(ptr);                      \
  }                                                                  \
  WeightsParserRegisterar g_##type##_Weights_Parser_Creator(type, Creator_##type##_Weights_Parser)
};  // namespace domi

#endif  // INC_FRAMEWORK_OMG_PARSER_PARSER_FACTORY_H_
