/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_MINDDATA_MINDRECORD_INCLUDE_SHARD_WRITER_H_
#define MINDSPORE_CCSRC_MINDDATA_MINDRECORD_INCLUDE_SHARD_WRITER_H_

#include <libgen.h>
#include <sys/file.h>
#include <unistd.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <exception>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>
#include "minddata/mindrecord/include/common/shard_utils.h"
#include "minddata/mindrecord/include/shard_column.h"
#include "minddata/mindrecord/include/shard_error.h"
#include "minddata/mindrecord/include/shard_header.h"
#include "minddata/mindrecord/include/shard_index.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace mindrecord {
class __attribute__((visibility("default"))) ShardWriter {
 public:
  ShardWriter();

  ~ShardWriter();

  /// \brief Open file at the beginning
  /// \param[in] paths the file names list
  /// \param[in] append new data at the end of file if true, otherwise try to overwrite file
  /// \param[in] overwrite a file with the same name if true
  /// \return Status
  Status Open(const std::vector<std::string> &paths, bool append = false, bool overwrite = false);

  /// \brief Open file at the ending
  /// \param[in] paths the file names list
  /// \return MSRStatus the status of MSRStatus
  Status OpenForAppend(const std::string &path);

  /// \brief Write header to disk
  /// \return MSRStatus the status of MSRStatus
  Status Commit();

  /// \brief Set file size
  /// \param[in] header_size the size of header, only (1<<N) is accepted
  /// \return MSRStatus the status of MSRStatus
  Status SetHeaderSize(const uint64_t &header_size);

  /// \brief Set page size
  /// \param[in] page_size the size of page, only (1<<N) is accepted
  /// \return MSRStatus the status of MSRStatus
  Status SetPageSize(const uint64_t &page_size);

  /// \brief Set shard header
  /// \param[in] header_data the info of header
  ///        WARNING, only called when file is empty
  /// \return MSRStatus the status of MSRStatus
  Status SetShardHeader(std::shared_ptr<ShardHeader> header_data);

  /// \brief write raw data by group size
  /// \param[in] raw_data the vector of raw json data, vector format
  /// \param[in] blob_data the vector of image data
  /// \param[in] sign validate data or not
  /// \return MSRStatus the status of MSRStatus to judge if write successfully
  Status WriteRawData(std::map<uint64_t, std::vector<json>> &raw_data, vector<vector<uint8_t>> &blob_data,
                      bool sign = true, bool parallel_writer = false);

  /// \brief write raw data by group size for call from python
  /// \param[in] raw_data the vector of raw json data, python-handle format
  /// \param[in] blob_data the vector of blob json data, python-handle format
  /// \param[in] sign validate data or not
  /// \return MSRStatus the status of MSRStatus to judge if write successfully
  Status WriteRawData(std::map<uint64_t, std::vector<py::handle>> &raw_data,
                      std::map<uint64_t, std::vector<py::handle>> &blob_data, bool sign = true,
                      bool parallel_writer = false);

  Status MergeBlobData(const std::vector<string> &blob_fields,
                       const std::map<std::string, std::unique_ptr<std::vector<uint8_t>>> &row_bin_data,
                       std::shared_ptr<std::vector<uint8_t>> *output);

  static Status Initialize(const std::unique_ptr<ShardWriter> *writer_ptr, const std::vector<std::string> &file_names);

 private:
  /// \brief write shard header data to disk
  Status WriteShardHeader();

  /// \brief erase error data
  void DeleteErrorData(std::map<uint64_t, std::vector<json>> &raw_data, std::vector<std::vector<uint8_t>> &blob_data);

  /// \brief populate error data
  void PopulateMutexErrorData(const int &row, const std::string &message, std::map<int, std::string> &err_raw_data);

  /// \brief check data
  void CheckSliceData(int start_row, int end_row, json schema, const std::vector<json> &sub_raw_data,
                      std::map<int, std::string> &err_raw_data);

  /// \brief write shard header data to disk
  Status ValidateRawData(std::map<uint64_t, std::vector<json>> &raw_data, std::vector<std::vector<uint8_t>> &blob_data,
                         bool sign, std::shared_ptr<std::pair<int, int>> *count_ptr);

  /// \brief fill data array in multiple thread run
  void FillArray(int start, int end, std::map<uint64_t, vector<json>> &raw_data,
                 std::vector<std::vector<uint8_t>> &bin_data);

  /// \brief serialized raw data
  Status SerializeRawData(std::map<uint64_t, std::vector<json>> &raw_data, std::vector<std::vector<uint8_t>> &bin_data,
                          uint32_t row_count);

  /// \brief write all data parallel
  Status ParallelWriteData(const std::vector<std::vector<uint8_t>> &blob_data,
                           const std::vector<std::vector<uint8_t>> &bin_raw_data);

  /// \brief write data shard by shard
  Status WriteByShard(int shard_id, int start_row, int end_row, const std::vector<std::vector<uint8_t>> &blob_data,
                      const std::vector<std::vector<uint8_t>> &bin_raw_data);

  /// \brief break image data up into multiple row groups
  Status CutRowGroup(int start_row, int end_row, const std::vector<std::vector<uint8_t>> &blob_data,
                     std::vector<std::pair<int, int>> &rows_in_group, const std::shared_ptr<Page> &last_raw_page,
                     const std::shared_ptr<Page> &last_blob_page);

  /// \brief append partial blob data to previous page
  Status AppendBlobPage(const int &shard_id, const std::vector<std::vector<uint8_t>> &blob_data,
                        const std::vector<std::pair<int, int>> &rows_in_group,
                        const std::shared_ptr<Page> &last_blob_page);

  /// \brief write new blob data page to disk
  Status NewBlobPage(const int &shard_id, const std::vector<std::vector<uint8_t>> &blob_data,
                     const std::vector<std::pair<int, int>> &rows_in_group,
                     const std::shared_ptr<Page> &last_blob_page);

  /// \brief shift last row group to next raw page for new appending
  Status ShiftRawPage(const int &shard_id, const std::vector<std::pair<int, int>> &rows_in_group,
                      std::shared_ptr<Page> &last_raw_page);

  /// \brief write raw data page to disk
  Status WriteRawPage(const int &shard_id, const std::vector<std::pair<int, int>> &rows_in_group,
                      std::shared_ptr<Page> &last_raw_page, const std::vector<std::vector<uint8_t>> &bin_raw_data);

  /// \brief generate empty raw data page
  Status EmptyRawPage(const int &shard_id, std::shared_ptr<Page> &last_raw_page);

  /// \brief append a row group at the end of raw page
  Status AppendRawPage(const int &shard_id, const std::vector<std::pair<int, int>> &rows_in_group, const int &chunk_id,
                       int &last_row_groupId, std::shared_ptr<Page> last_raw_page,
                       const std::vector<std::vector<uint8_t>> &bin_raw_data);

  /// \brief write blob chunk to disk
  Status FlushBlobChunk(const std::shared_ptr<std::fstream> &out, const std::vector<std::vector<uint8_t>> &blob_data,
                        const std::pair<int, int> &blob_row);

  /// \brief write raw chunk to disk
  Status FlushRawChunk(const std::shared_ptr<std::fstream> &out, const std::vector<std::pair<int, int>> &rows_in_group,
                       const int &chunk_id, const std::vector<std::vector<uint8_t>> &bin_raw_data);

  /// \brief break up into tasks by shard
  std::vector<std::pair<int, int>> BreakIntoShards();

  /// \brief calculate raw data size row by row
  Status SetRawDataSize(const std::vector<std::vector<uint8_t>> &bin_raw_data);

  /// \brief calculate blob data size row by row
  Status SetBlobDataSize(const std::vector<std::vector<uint8_t>> &blob_data);

  /// \brief populate last raw page pointer
  Status SetLastRawPage(const int &shard_id, std::shared_ptr<Page> &last_raw_page);

  /// \brief populate last blob page pointer
  Status SetLastBlobPage(const int &shard_id, std::shared_ptr<Page> &last_blob_page);

  /// \brief check the data by schema
  Status CheckData(const std::map<uint64_t, std::vector<json>> &raw_data);

  /// \brief check the data and type
  Status CheckDataTypeAndValue(const std::string &key, const json &value, const json &data, const int &i,
                               std::map<int, std::string> &err_raw_data);

  /// \brief Lock writer and save pages info
  Status LockWriter(bool parallel_writer, std::unique_ptr<int> *fd_ptr);

  /// \brief Unlock writer and save pages info
  Status UnlockWriter(int fd, bool parallel_writer = false);

  /// \brief Check raw data before writing
  Status WriteRawDataPreCheck(std::map<uint64_t, std::vector<json>> &raw_data, vector<vector<uint8_t>> &blob_data,
                              bool sign, int *schema_count, int *row_count);

  /// \brief Get full path from file name
  Status GetFullPathFromFileName(const std::vector<std::string> &paths);

  /// \brief Open files
  Status OpenDataFiles(bool append, bool overwrite);

  /// \brief Remove lock file
  Status RemoveLockFile();

  /// \brief Remove lock file
  Status InitLockFile();

 private:
  const std::string kLockFileSuffix = "_Locker";
  const std::string kPageFileSuffix = "_Pages";
  std::string lock_file_;   // lock file for parallel run
  std::string pages_file_;  // temporary file of pages info for parallel run

  int shard_count_;        // number of files
  uint64_t header_size_;   // header size
  uint64_t page_size_;     // page size
  uint32_t row_count_;     // count of rows
  uint32_t schema_count_;  // count of schemas

  std::vector<uint64_t> raw_data_size_;   // Raw data size
  std::vector<uint64_t> blob_data_size_;  // Blob data size

  std::vector<std::string> file_paths_;                      // file paths
  std::vector<std::shared_ptr<std::fstream>> file_streams_;  // file handles
  std::shared_ptr<ShardHeader> shard_header_;                // shard header
  std::shared_ptr<ShardColumn> shard_column_;                // shard columns

  std::map<uint64_t, std::map<int, std::string>> err_mg_;  // used for storing error raw_data info

  std::mutex check_mutex_;  // mutex for data check
  std::atomic<bool> flag_{false};
  std::atomic<int64_t> compression_size_;
};
}  // namespace mindrecord
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_MINDRECORD_INCLUDE_SHARD_WRITER_H_
