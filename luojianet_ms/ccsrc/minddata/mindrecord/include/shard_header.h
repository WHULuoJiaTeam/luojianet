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

#ifndef LUOJIANET_MS_CCSRC_MINDDATA_MINDRECORD_INCLUDE_SHARD_HEADER_H_
#define LUOJIANET_MS_CCSRC_MINDDATA_MINDRECORD_INCLUDE_SHARD_HEADER_H_

#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include "minddata/mindrecord/include/common/shard_utils.h"
#include "minddata/mindrecord/include/shard_error.h"
#include "minddata/mindrecord/include/shard_index.h"
#include "minddata/mindrecord/include/shard_page.h"
#include "minddata/mindrecord/include/shard_schema.h"
#include "minddata/mindrecord/include/shard_statistics.h"

namespace luojianet_ms {
namespace mindrecord {
class __attribute__((visibility("default"))) ShardHeader {
 public:
  ShardHeader();

  ~ShardHeader() = default;

  Status BuildDataset(const std::vector<std::string> &file_paths, bool load_dataset = true);

  static Status BuildSingleHeader(const std::string &file_path, std::shared_ptr<json> *header_ptr);
  /// \brief add the schema and save it
  /// \param[in] schema the schema needs to be added
  /// \return the last schema's id
  int AddSchema(std::shared_ptr<Schema> schema);

  /// \brief add the statistic and save it
  /// \param[in] statistic the statistic needs to be added
  /// \return the last statistic's id
  void AddStatistic(std::shared_ptr<Statistics> statistic);

  /// \brief create index and add fields which from schema for each schema
  /// \param[in] fields the index fields needs to be added
  /// \return SUCCESS if add successfully, FAILED if not
  Status AddIndexFields(std::vector<std::pair<uint64_t, std::string>> fields);

  Status AddIndexFields(const std::vector<std::string> &fields);

  /// \brief get the schema
  /// \return the schema
  std::vector<std::shared_ptr<Schema>> GetSchemas();

  /// \brief get Statistics
  /// \return the Statistic
  std::vector<std::shared_ptr<Statistics>> GetStatistics();

  /// \brief add the statistic and save it
  /// \param[in] statistic info of slim size
  /// \return null
  int64_t GetSlimSizeStatistic(const json &slim_size_json);

  /// \brief get the fields of the index
  /// \return the fields of the index
  std::vector<std::pair<uint64_t, std::string>> GetFields();

  /// \brief get the index
  /// \return the index
  std::shared_ptr<Index> GetIndex();

  /// \brief get the schema by schemaid
  /// \param[in] schema_id the id of schema needs to be got
  /// \param[in] schema_ptr the schema obtained by schemaId
  /// \return Status
  Status GetSchemaByID(int64_t schema_id, std::shared_ptr<Schema> *schema_ptr);

  /// \brief get the filepath to shard by shardID
  /// \param[in] shardID the id of shard which filepath needs to be obtained
  /// \return the filepath obtained by shardID
  std::string GetShardAddressByID(int64_t shard_id);

  /// \brief get the statistic by statistic id
  /// \param[in] statistic_id the id of statistic needs to be get
  /// \param[in] statistics_ptr the statistics obtained by statistic id
  /// \return Status
  Status GetStatisticByID(int64_t statistic_id, std::shared_ptr<Statistics> *statistics_ptr);

  Status InitByFiles(const std::vector<std::string> &file_paths);

  void SetIndex(Index index) { index_ = std::make_shared<Index>(index); }

  Status GetPage(const int &shard_id, const int &page_id, std::shared_ptr<Page> *page_ptr);

  Status SetPage(const std::shared_ptr<Page> &new_page);

  Status AddPage(const std::shared_ptr<Page> &new_page);

  int64_t GetLastPageId(const int &shard_id);

  int GetLastPageIdByType(const int &shard_id, const std::string &page_type);

  Status GetPageByGroupId(const int &group_id, const int &shard_id, std::shared_ptr<Page> *page_ptr);

  std::vector<std::string> GetShardAddresses() const { return shard_addresses_; }

  int GetShardCount() const { return shard_count_; }

  int GetSchemaCount() const { return schema_.size(); }

  uint64_t GetHeaderSize() const { return header_size_; }

  uint64_t GetPageSize() const { return page_size_; }

  uint64_t GetCompressionSize() const { return compression_size_; }

  void SetHeaderSize(const uint64_t &header_size) { header_size_ = header_size; }

  void SetPageSize(const uint64_t &page_size) { page_size_ = page_size; }

  void SetCompressionSize(const uint64_t &compression_size) { compression_size_ = compression_size; }

  std::vector<std::string> SerializeHeader();

  Status PagesToFile(const std::string dump_file_name);

  Status FileToPages(const std::string dump_file_name);

  static Status Initialize(const std::shared_ptr<ShardHeader> *header_ptr, const json &schema,
                           const std::vector<std::string> &index_fields, std::vector<std::string> &blob_fields,
                           uint64_t &schema_id);

 private:
  Status InitializeHeader(const std::vector<json> &headers, bool load_dataset);

  /// \brief get the headers from all the shard data
  /// \param[in] the shard data real path
  /// \param[in] the headers which read from the shard data
  /// \return SUCCESS/FAILED
  Status GetHeaders(const vector<string> &real_addresses, std::vector<json> &headers);

  Status ValidateField(const std::vector<std::string> &field_name, json schema, const uint64_t &schema_id);

  /// \brief check the binary file status
  static Status CheckFileStatus(const std::string &path);

  static Status ValidateHeader(const std::string &path, std::shared_ptr<json> *header_ptr);

  void GetHeadersOneTask(int start, int end, std::vector<json> &headers, const vector<string> &realAddresses);

  Status ParseIndexFields(const json &index_fields);

  Status CheckIndexField(const std::string &field, const json &schema);

  Status ParsePage(const json &page, int shard_index, bool load_dataset);

  Status ParseStatistics(const json &statistics);

  Status ParseSchema(const json &schema);

  void ParseShardAddress(const json &address);

  std::string SerializeIndexFields();

  std::vector<std::string> SerializePage();

  std::string SerializeStatistics();

  std::string SerializeSchema();

  std::string SerializeShardAddress();

  std::shared_ptr<Index> InitIndexPtr();

  Status GetAllSchemaID(std::set<uint64_t> &bucket_count);

  uint32_t shard_count_;
  uint64_t header_size_;
  uint64_t page_size_;
  uint64_t compression_size_;

  std::shared_ptr<Index> index_;
  std::vector<std::string> shard_addresses_;
  std::vector<std::shared_ptr<Schema>> schema_;
  std::vector<std::shared_ptr<Statistics>> statistics_;
  std::vector<std::vector<std::shared_ptr<Page>>> pages_;
};
}  // namespace mindrecord
}  // namespace luojianet_ms

#endif  // LUOJIANET_MS_CCSRC_MINDDATA_MINDRECORD_INCLUDE_SHARD_HEADER_H_
