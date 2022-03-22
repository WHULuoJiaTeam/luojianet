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

#ifndef GE_COMMON_AUTH_FILE_SAVER_H_
#define GE_COMMON_AUTH_FILE_SAVER_H_

#include <string>
#include <vector>

#include "framework/common/helper/om_file_helper.h"
#include "framework/common/types.h"
#include "external/ge/ge_ir_build.h"
#include "graph/buffer.h"
#include "mmpa/mmpa_api.h"

struct PROC_PARAM {
  uint8_t *model_name;

  // ISV Ek buffer
  uint8_t *model_key;
  uint32_t model_key_len;

  // ISV  root certificate buffer
  uint8_t *root_cert;
  uint32_t root_cert_len;

  // ISV private key buffer
  uint8_t *pri_key;
  uint32_t pri_key_len;

  // Raw AI Module Image buffer
  uint8_t *ai_image;
  uint32_t ai_image_len;

  // ISV HW key buffer
  uint8_t *hw_key;
  uint32_t hw_key_len;
};

struct ProcOut {
  uint8_t *passcode;
  uint32_t passcode_len;
  uint8_t *encrypted_img;
  uint32_t encrypted_img_len;
};

namespace ge {
using std::string;

class FileSaver {
 public:
  ///
  /// @ingroup domi_common
  /// @brief save model, no encryption
  /// @return Status  result
  ///
  static Status SaveToFile(const string &file_path, const ge::ModelData &model,
                           const ModelFileHeader *model_file_header = nullptr);

  static Status SaveToFile(const string &file_path, ModelFileHeader &model_file_header,
                           ModelPartitionTable &model_partition_table,
                           const std::vector<ModelPartition> &partition_datas);

  static Status SaveToFile(const string &file_path, ModelFileHeader &file_header,
                        vector<ModelPartitionTable *> &model_partition_tables,
                        const vector<vector<ModelPartition>> &all_partition_datas);

  static Status SaveToBuffWithFileHeader(const ModelFileHeader &file_header,
                                            ModelPartitionTable &model_partition_table,
                                            const std::vector<ModelPartition> &partition_datas,
                                            ge::ModelBufferData& model);

  static Status SaveToBuffWithFileHeader(const ModelFileHeader &file_header,
                                         const std::vector<ModelPartitionTable *> &model_partition_tables,
                                         const std::vector<std::vector<ModelPartition>> &all_partition_datas,
                                         ge::ModelBufferData &model);

  static Status SaveToFile(const string &file_path, const void *data, int len);

 protected:
  ///
  /// @ingroup domi_common
  /// @brief Check validity of the file path
  /// @return Status  result
  ///
  static Status CheckPath(const string &file_path);

  static Status WriteData(const void *data, uint32_t size, int32_t fd);

  static Status OpenFile(int32_t &fd, const std::string &file_path);

  ///
  /// @ingroup domi_common
  /// @brief save model to file
  /// @param [in] file_path  file output path
  /// @param [in] file_header  file header info
  /// @param [in] data  model data
  /// @param [in] len  model length
  /// @return Status  result
  ///
  static Status SaveWithFileHeader(const string &file_path, const ModelFileHeader &file_header, const void *data,
                                   int len);

  static Status SaveWithFileHeader(const std::string &file_path, const ModelFileHeader &file_header,
                                   ModelPartitionTable &model_partition_table,
                                   const std::vector<ModelPartition> &partition_datas);
  static Status SaveWithFileHeader(const std::string &file_path, const ModelFileHeader &file_header,
                                   std::vector<ModelPartitionTable *> &model_partition_tables,
                                   const std::vector<std::vector<ModelPartition>> &all_partition_datas);
};
}  // namespace ge
#endif  // GE_COMMON_AUTH_FILE_SAVER_H_
