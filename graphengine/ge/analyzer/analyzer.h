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

#ifndef DOMI_ANALYZER_ANANLYZER_H_
#define DOMI_ANALYZER_ANANLYZER_H_

#include "nlohmann/json.hpp"

#include <map>
#include <string>
#include <mutex>
#include <memory>
#include <fstream>
#include <atomic>

#include "external/ge/ge_api_types.h"
#include "graph/compute_graph.h"
#include "graph/node.h"

namespace ge {
namespace analyzer {
enum AnalyzeType {
  PARSER         = 0,
  INFER_SHAPE    = 1,
  CHECKSUPPORT   = 2,
  GRAPH_OPTIMIZE = 3,
  GRAPH_PARTION  = 4,
  GRAPH_BUILDER  = 5,
};

struct TensorInfo {
  vector<int64_t> shape;
  string d_type;
  string layout;
};

struct OpInfo {
  string error_type;
  string op_name;
  string op_type;
  std::vector<TensorInfo> input_info;
  std::vector<TensorInfo> output_info;
  string reason;
};

struct GraphInfo {
  uint64_t session_id = 0;
  uint64_t graph_id = 0;
  std::vector<OpInfo> op_info;
};

struct DataInfo {
  DataInfo() = default;
  ~DataInfo() = default;

  DataInfo(uint64_t sess, uint64_t graph, AnalyzeType type,
           ge::NodePtr node, std::string error_info) {
    session_id = sess;
    graph_id = graph;
    analyze_type = type;
    node_ptr = node;
    reason = error_info;
  }
  uint64_t session_id;
  uint64_t graph_id;
  AnalyzeType analyze_type;
  ge::NodePtr node_ptr{nullptr};
  std::string reason;
};
}

class Analyzer {
public:
  /**
   * @ingroup ge
   * @brief: get analyzer instance.
   * @param [in]: None
   * @return: Analyzer instance ptr
   */
  static Analyzer *GetInstance();

  /**
   * @ingroup ge
   * @brief: check whether env var ENABLE_NETWORK_ANALYSIS_DEBUG is enabled.
   *     When enable env, it will keep adaptor sink geop graph even though fail.
   * @param [in]: None
   * @return: true: enable env   false : disable env
   */
  bool IsEnableNetAnalyzeDebug() { return std::getenv("ENABLE_NETWORK_ANALYSIS_DEBUG") != nullptr; }

  /**
   * @ingroup ge
   * @brief: build buff object by sess id and graph id .
   * @param [in]: session id & graph id
   * @return: 0: success other: failed
   */
  ge::Status BuildJsonObject(uint64_t session_id, uint64_t graph_id);

  /**
   * @ingroup ge
   * @brief: get buff object by sess id and graph id .
   * @param [in]: session id & graph id
   * @return: nullptr if failed
   */
  std::shared_ptr<analyzer::GraphInfo> GetJsonObject(uint64_t session_id, uint64_t graph_id);

  /**
   * @ingroup ge
   * @brief: analyzer globle init method.
   * @param [in]: None
   * @return: None
   */
  ge::Status Initialize();

  /**
   * @ingroup ge
   * @brief: DeConstruct method. Release all used resource of analyzer.
   * @param [in]: None
   * @return: None
   */
  void Finalize();

  /**
   * @ingroup ge
   * @brief: DeConstruct method. Only release resource about session id.
   * @param [in]: None
   * @return: None
   */
  void DestroySessionJsonObject(uint64_t session_id);

  /**
   * @ingroup ge
   * @brief: DeConstruct method. Only release resource about session id and graph id.
   * @param [in]: None
   * @return: None
   */
  void DestroyGraphJsonObject(uint64_t session_id, uint64_t graph_id);

  /**
   * @ingroup ge
   * @brief: main process method. Buff analyzed data and output to json file
   * @param [in]: DataInfo Object
   * @return: 0: SUCCESS other: FAILED
   */
  ge::Status DoAnalyze(analyzer::DataInfo &data_info);

  /**
   * @ingroup ge
   * @brief: Buff analyzed data and output to json file
   * @param [in]: session id , graph id
   * @return: 0: SUCCESS other: FAILED
   */
  ge::Status SaveAnalyzerDataToFile(uint64_t session_id, uint64_t graph_id);

  Analyzer(const Analyzer &) = delete;
  Analyzer& operator=(const Analyzer&) = delete;
  Analyzer(Analyzer &&) = delete;
  Analyzer& operator=(Analyzer &&) = delete;
private:
  void TensorInfoToJson(nlohmann::json& j, const analyzer::TensorInfo &tensor_info);
  void OpInfoToJson(nlohmann::json& j, const analyzer::OpInfo &op_info);
  void GraphInfoToJson(nlohmann::json& j, const analyzer::GraphInfo &graph_info);

  ge::Status SaveOpInfo(ge::OpDescPtr desc, analyzer::DataInfo &data_info,
                                  std::shared_ptr<analyzer::GraphInfo> graph_info);

  void ClearHistoryFile();
  ge::Status CreateAnalyzerFile();

  explicit Analyzer() {};
  ~Analyzer() = default;

private:
  std::map<uint64_t, std::map<uint64_t, std::shared_ptr<analyzer::GraphInfo>>> graph_infos_;
  std::recursive_mutex mutex_; // protect graph_infos_
  std::mutex file_mutex_; // protect json_file_
  std::ofstream json_file_;
  std::string json_file_name_;
  std::atomic_bool is_json_file_create_{false};
};
} // namespace ge
#endif // DOMI_ANALYZER_ANANLYZER_H_
