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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_CONFIG_PARSER_CONFIG_FILE_PARSER_H
#define MINDSPORE_LITE_TOOLS_CONVERTER_CONFIG_PARSER_CONFIG_FILE_PARSER_H
#include <string>
#include <map>
#include <vector>

namespace mindspore {
namespace lite {
struct DataPreProcessString {
  std::string calibrate_path;
  std::string calibrate_size;
  std::string input_type;
  std::string image_to_format;
  std::string normalize_mean;
  std::string normalize_std;
  std::string resize_width;
  std::string resize_height;
  std::string resize_method;
  std::string center_crop_width;
  std::string center_crop_height;
};

struct CommonQuantString {
  std::string quant_type;
  std::string bit_num;
  std::string min_quant_weight_size;
  std::string min_quant_weight_channel;
  std::string skip_quant_node;
  std::string debug_info_save_path;
};

struct MixedBitWeightQuantString {
  std::string init_scale;
  std::string auto_tune;
};

struct FullQuantString {
  std::string activation_quant_method;
  std::string bias_correction;
  std::string target_device;
};

struct RegistryInfoString {
  std::string plugin_path;
  std::string disable_fusion;
};

struct AclOptionCfgString {
  std::string device_id;
  std::string input_format;
  std::string input_shape_vector;
  std::string input_shape;
  std::string output_type;
  std::string precision_mode;
  std::string op_select_impl_mode;
  std::string fusion_switch_config_file_path;
  std::string dynamic_batch_size;
  std::string buffer_optimize;
  std::string insert_op_config_file_path;
  std::string dynamic_image_size;
};

struct MicroParamString {
  std::string codegen_mode;
  std::string target;
  std::string support_parallel;
  std::string debug_mode;
  std::string enable_micro;
};

class ConfigFileParser {
 public:
  int ParseConfigFile(const std::string &config_file_path);

  DataPreProcessString GetDataPreProcessString() const { return this->data_pre_process_string_; }
  CommonQuantString GetCommonQuantString() const { return this->common_quant_string_; }
  MixedBitWeightQuantString GetMixedBitWeightQuantString() const { return this->mixed_bit_quant_string_; }
  FullQuantString GetFullQuantString() const { return this->full_quant_string_; }
  RegistryInfoString GetRegistryInfoString() const { return this->registry_info_string_; }
  AclOptionCfgString GetAclOptionCfgString() { return this->acl_option_cfg_string_; }
  MicroParamString GetMicroParamString() { return this->micro_param_string_; }

 private:
  int ParseDataPreProcessString(const std::map<std::string, std::map<std::string, std::string>> &maps);
  int ParseCommonQuantString(const std::map<std::string, std::map<std::string, std::string>> &maps);
  int ParseMixedBitQuantString(const std::map<std::string, std::map<std::string, std::string>> &maps);
  int ParseFullQuantString(const std::map<std::string, std::map<std::string, std::string>> &maps);
  int ParseRegistryInfoString(const std::map<std::string, std::map<std::string, std::string>> &maps);
  int ParseAclOptionCfgString(const std::map<std::string, std::map<std::string, std::string>> &maps);
  int SetMapData(const std::map<std::string, std::string> &input_map,
                 const std::map<std::string, std::string &> &parse_map, const std::string &section);
  int ParseMicroParamString(const std::map<std::string, std::map<std::string, std::string>> &maps);

 private:
  DataPreProcessString data_pre_process_string_;
  CommonQuantString common_quant_string_;
  MixedBitWeightQuantString mixed_bit_quant_string_;
  FullQuantString full_quant_string_;
  RegistryInfoString registry_info_string_;
  AclOptionCfgString acl_option_cfg_string_;
  MicroParamString micro_param_string_;
};

}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_CONFIG_PARSER_CONFIG_FILE_PARSER_H
