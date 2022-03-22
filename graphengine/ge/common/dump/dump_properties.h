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

#ifndef GE_COMMON_DUMP_DUMP_PROPERTIES_H_
#define GE_COMMON_DUMP_DUMP_PROPERTIES_H_

#include <map>
#include <set>
#include <string>
#include <vector>

namespace ge {
using Status = uint32_t;
class DumpProperties {
 public:
  DumpProperties() = default;

  ~DumpProperties() = default;

  DumpProperties(const DumpProperties &dump);

  DumpProperties &operator=(const DumpProperties &dump);

  Status InitByOptions();

  void AddPropertyValue(const std::string &model, const std::set<std::string> &layers);

  void DeletePropertyValue(const std::string &model);

  void ClearDumpPropertyValue();

  void ClearDumpInfo();

  std::set<std::string> GetAllDumpModel() const;

  std::set<std::string> GetPropertyValue(const std::string &model) const;

  bool IsLayerNeedDump(const std::string &model, const std::string &om_name, const std::string &op_name) const;

  void SetDumpPath(const std::string &path);

  const std::string &GetDumpPath() const;

  void SetDumpStep(const std::string &step);

  const std::string &GetDumpStep() const;

  void SetDumpMode(const std::string &mode);

  const std::string &GetDumpMode() const;

  void SetDumpStatus(const std::string &status);

  const std::string &GetDumpStatus() const;

  void InitInferOpDebug();

  bool IsInferOpDebug() const {
    return is_infer_op_debug_;
  }

  void SetDumpOpSwitch(const std::string &dump_op_switch);

  const std::string &GetDumpOpSwitch() const;

  bool IsOpDebugOpen() const {
    return is_train_op_debug_ || is_infer_op_debug_;
  }

  bool IsDumpOpen() const;

  bool IsSingleOpNeedDump() const;

  void SetOpDebugMode(const uint32_t &op_debug_mode);

  uint32_t GetOpDebugMode() const { return op_debug_mode_; }

  const std::string &GetEnableDump() const {return enable_dump_;}

  const std::string &GetEnableDumpDebug() const {return enable_dump_debug_;}


 private:
  void CopyFrom(const DumpProperties &other);

  Status SetDumpDebugOptions();

  Status SetDumpOptions();

  void Split(const std::string &s, std::vector<std::string> &result, const char *delchar);

  Status CheckDumpStep(const std::string &dump_step);

  Status CheckDumpMode(const std::string &dump_mode);

  Status CheckDumpPath(const std::string &input);

  Status CheckEnableDump(const std::string &input);

  std::string enable_dump_;
  std::string enable_dump_debug_;

  std::string dump_path_;
  std::string dump_step_;
  std::string dump_mode_;
  std::string dump_status_;
  std::string dump_op_switch_;
  std::map<std::string, std::set<std::string>> model_dump_properties_map_;

  bool is_train_op_debug_ = false;
  bool is_infer_op_debug_ = false;
  uint32_t op_debug_mode_ = 0;
};
}

#endif //GE_COMMON_DUMP_DUMP_PROPERTIES_H_