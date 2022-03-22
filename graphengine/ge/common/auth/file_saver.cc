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

#include "common/auth/file_saver.h"

#include <securec.h>
#include <cstdlib>
#include <fstream>
#include <vector>

#include "common/math/math_util.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/debug/log.h"
#include "framework/common/util.h"

namespace {
const int kFileOpSuccess = 0;
}  //  namespace

namespace ge {
Status FileSaver::OpenFile(int32_t &fd, const std::string &file_path) {
  if (CheckPath(file_path) != SUCCESS) {
    GELOGE(FAILED, "[Check][FilePath]Check output file failed, file_path:%s.",
           file_path.c_str());
    REPORT_CALL_ERROR("E19999", "Check output file failed, file_path:%s.",
                      file_path.c_str());
    return FAILED;
  }

  char real_path[MMPA_MAX_PATH] = {0};
  GE_IF_BOOL_EXEC(mmRealPath(file_path.c_str(), real_path, MMPA_MAX_PATH) != EN_OK,
                  GELOGI("File %s is not exist, it will be created.", file_path.c_str()));
  // Open file
  mmMode_t mode = M_IRUSR | M_IWUSR;
  fd = mmOpen2(real_path, M_RDWR | M_CREAT | O_TRUNC, mode);
  if (fd == EN_INVALID_PARAM || fd == EN_ERROR) {
    // -1: Failed to open file; - 2: Illegal parameter
    GELOGE(FAILED, "[Open][File]Failed. errno:%d, errmsg:%s", fd, strerror(errno));
    REPORT_INNER_ERROR("E19999", "Open file failed, errno:%d, errmsg:%s.",
                       fd, strerror(errno));
    return FAILED;
  }
  return SUCCESS;
}

Status FileSaver::WriteData(const void *data, uint32_t size, int32_t fd) {
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(size == 0 || data == nullptr, return PARAM_INVALID);
  mmSsize_t write_count;
  uint32_t size_2g = 2147483648;  // 0x1 << 31
  uint32_t size_1g = 1073741824;  // 0x1 << 30
  // Write data
  if (size > size_2g) {
    auto seek = reinterpret_cast<uint8_t *>(const_cast<void *>(data));
    while (size > size_1g) {
      write_count = mmWrite(fd, reinterpret_cast<void *>(seek), size_1g);
      if (write_count == EN_INVALID_PARAM || write_count == EN_ERROR) {
        GELOGE(FAILED, "[Write][Data]Failed, errno:%ld, errmsg:%s",
               write_count, strerror(errno));
        REPORT_INNER_ERROR("E19999", "Write data failed, errno:%ld, errmsg:%s.",
                           write_count, strerror(errno));
        return FAILED;
      }
      size -= size_1g;
      seek += size_1g;
    }
    write_count = mmWrite(fd, reinterpret_cast<void *>(seek), size);
  } else {
    write_count = mmWrite(fd, const_cast<void *>(data), size);
  }

  // -1: Failed to write to file; - 2: Illegal parameter
  if (write_count == EN_INVALID_PARAM || write_count == EN_ERROR) {
    GELOGE(FAILED, "[Write][Data]Failed. mmpa_errorno = %ld, error:%s",
           write_count, strerror(errno));
    REPORT_INNER_ERROR("E19999", "Write data failed, mmpa_errorno = %ld, error:%s.",
                       write_count, strerror(errno));
    return FAILED;
  }

  return SUCCESS;
}

Status FileSaver::SaveWithFileHeader(const std::string &file_path, const ModelFileHeader &file_header, const void *data,
                                     int len) {
  if (data == nullptr || len <= 0) {
    GELOGE(FAILED, "[Check][Param]Failed, model_data is null or the "
           "length[%d] is less than 1.", len);
    REPORT_INNER_ERROR("E19999", "Save file failed, model_data is null or the "
                       "length:%d is less than 1.", len);
    return FAILED;
  }

  // Open file
  int32_t fd = 0;
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(OpenFile(fd, file_path) != SUCCESS, return FAILED, "OpenFile FAILED");

  Status ret = SUCCESS;
  do {
    // Write file header
    GE_CHK_BOOL_EXEC(WriteData(static_cast<const void *>(&file_header), sizeof(ModelFileHeader), fd) == SUCCESS,
                     ret = FAILED;
                     break, "WriteData FAILED");
    // write data
    GE_CHK_BOOL_EXEC(WriteData(data, static_cast<uint32_t>(len), fd) == SUCCESS, ret = FAILED, "WriteData FAILED");
  } while (0);
  // Close file
  if (mmClose(fd) != 0) {  // mmClose 0: success
    GELOGE(FAILED, "[Close][File]Failed, error_code:%u errmsg:%s", ret, strerror(errno));
    REPORT_INNER_ERROR("E19999", "Close file failed, error_code:%u errmsg:%s",
                       ret, strerror(errno));
    ret = FAILED;
  }
  return ret;
}

Status FileSaver::SaveWithFileHeader(const std::string &file_path, const ModelFileHeader &file_header,
                                     ModelPartitionTable &model_partition_table,
                                     const std::vector<ModelPartition> &partition_datas) {
  GE_CHK_BOOL_RET_STATUS(!partition_datas.empty() && model_partition_table.num != 0
      && model_partition_table.num == partition_datas.size(), FAILED,
      "Invalid param:partition data size is (%u), model_partition_table.num is (%zu).",
      model_partition_table.num, partition_datas.size());
  // Open file
  int32_t fd = 0;
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(OpenFile(fd, file_path) != SUCCESS, return FAILED);
  Status ret = SUCCESS;
  do {
    // Write file header
    GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(
        WriteData(static_cast<const void *>(&file_header), sizeof(ModelFileHeader), fd) != SUCCESS, ret = FAILED;
        break);
    // Write model partition table
    uint32_t table_size = static_cast<uint32_t>(SIZE_OF_MODEL_PARTITION_TABLE(model_partition_table));
    GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(
        WriteData(static_cast<const void *>(&model_partition_table), table_size, fd) != SUCCESS, ret = FAILED; break);
    // Write partition data
    for (const auto &partitionData : partition_datas) {
      GELOGI("GC:size[%u]", partitionData.size);
      GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(
          WriteData(static_cast<const void *>(partitionData.data), partitionData.size, fd) != SUCCESS, ret = FAILED;
          break);
    }
  } while (0);
  // Close file
  if (mmClose(fd) != EN_OK) {
    GELOGE(FAILED, "[Close][File]Failed, error_code:%u errmsg:%s", ret, strerror(errno));
    REPORT_CALL_ERROR("E19999", "Close file failed, error_code:%u errmsg:%s",
                      ret, strerror(errno));
    ret = FAILED;
  }
  return ret;
}

Status FileSaver::SaveToBuffWithFileHeader(const ModelFileHeader &file_header,
                                           ModelPartitionTable &model_partition_table,
                                           const std::vector<ModelPartition> &partition_datas,
                                           ge::ModelBufferData &model) {
  const vector<ModelPartitionTable *> model_partition_tables = { &model_partition_table };
  const std::vector<std::vector<ModelPartition>> all_partition_datas = { partition_datas };
  return SaveToBuffWithFileHeader(file_header, model_partition_tables, all_partition_datas, model);
}

Status FileSaver::SaveToBuffWithFileHeader(const ModelFileHeader &file_header,
                                           const vector<ModelPartitionTable *> &model_partition_tables,
                                           const std::vector<std::vector<ModelPartition>> &all_partition_datas,
                                           ge::ModelBufferData &model) {
  GE_CHK_BOOL_RET_STATUS(model_partition_tables.size() == all_partition_datas.size(), PARAM_INVALID,
                         "Model table size %zu does not match partition size %zu.",
                         model_partition_tables.size(), all_partition_datas.size());
  for (size_t index = 0; index < model_partition_tables.size(); ++index) {
    auto &cur_partiton_data = all_partition_datas[index];
    auto &cur_model_partition_table = *model_partition_tables[index];
    GE_CHK_BOOL_RET_STATUS(!cur_partiton_data.empty() && cur_model_partition_table.num != 0
                           && cur_model_partition_table.num == cur_partiton_data.size(), FAILED,
                           "Invalid param: partition data size is (%zu), model_partition_table.num is (%u).",
                           cur_partiton_data.size(), cur_model_partition_table.num);
  }

  uint64_t model_header_size = sizeof(ModelFileHeader);
  uint64_t total_size = model_header_size;
  for (size_t index = 0; index < model_partition_tables.size(); ++index) {
    auto &cur_model_partition_table = *model_partition_tables[index];
    total_size += static_cast<uint64_t>(SIZE_OF_MODEL_PARTITION_TABLE(cur_model_partition_table));
    auto &cur_partition_data = all_partition_datas[index];
    for (const auto &partition_data : cur_partition_data) {
      auto ret = ge::CheckUint64AddOverflow(total_size, partition_data.size);
      GE_CHK_BOOL_RET_STATUS(ret == SUCCESS, FAILED, "Add uint64 overflow!");
      total_size += partition_data.size;
    }
  }
  // save to buff
  auto buff = reinterpret_cast<uint8_t *>(malloc(total_size));
  GE_CHK_BOOL_RET_STATUS(buff != nullptr, FAILED, "Malloc failed!");
  GE_PRINT_DYNAMIC_MEMORY(malloc, "File buffer.", total_size)
  model.data.reset(buff, [](uint8_t *buff) {
    GELOGD("Free online model memory.");
    free(buff);
    buff = nullptr;
  });
  model.length = total_size;
  uint64_t left_space = total_size;
  auto ret_mem = memcpy_s(buff, left_space, reinterpret_cast<void *>(const_cast<ModelFileHeader *>(&file_header)),
                          model_header_size);
  GE_CHK_BOOL_RET_STATUS(ret_mem == EOK, FAILED, "Memcpy_s failed!");
  buff += model_header_size;
  left_space -= model_header_size;

  for (size_t index = 0; index < model_partition_tables.size(); ++index) {
    auto &cur_tabel = *model_partition_tables[index];
    uint64_t table_size = static_cast<uint64_t>(SIZE_OF_MODEL_PARTITION_TABLE(cur_tabel));
    ret_mem = memcpy_s(buff, left_space, reinterpret_cast<void *>(&cur_tabel), table_size);
    GE_CHK_BOOL_RET_STATUS(ret_mem == EOK, FAILED, "Memcpy_s failed!");
    buff += table_size;
    left_space -= table_size;
    auto &cur_partition_data = all_partition_datas[index];
    for (const auto &partition_data : cur_partition_data) {
      ret_mem = memcpy_s(buff, left_space, reinterpret_cast<void *>(const_cast<uint8_t *>(partition_data.data)),
                         partition_data.size);
      GE_CHK_BOOL_RET_STATUS(ret_mem == EOK, FAILED, "Memcpy_s failed!");
      buff += partition_data.size;
      left_space -= partition_data.size;
    }
  }

  return SUCCESS;
}

Status FileSaver::CheckPath(const std::string &file_path) {
  // Determine file path length
  if (file_path.size() >= MMPA_MAX_PATH) {
    GELOGE(FAILED, "[Check][FilePath]Failed, file path's length:%zu > mmpa_max_path:%d",
           file_path.size(), MMPA_MAX_PATH);
    REPORT_INNER_ERROR("E19999", "Check file path failed, file path's length:%zu > "
                       "mmpa_max_path:%d", file_path.size(), MMPA_MAX_PATH);
    return FAILED;
  }

  // Find the last separator
  int path_split_pos = static_cast<int>(file_path.size() - 1);
  for (; path_split_pos >= 0; path_split_pos--) {
    if (file_path[path_split_pos] == '\\' || file_path[path_split_pos] == '/') {
      break;
    }
  }

  if (path_split_pos == 0) {
    return SUCCESS;
  }

  // If there is a path before the file name, create the path
  if (path_split_pos != -1) {
    if (CreateDirectory(std::string(file_path).substr(0, static_cast<size_t>(path_split_pos))) != kFileOpSuccess) {
      GELOGE(FAILED, "[Create][Directory]Failed, file path:%s.", file_path.c_str());
      return FAILED;
    }
  }

  return SUCCESS;
}

Status FileSaver::SaveToFile(const string &file_path, const ge::ModelData &model,
                             const ModelFileHeader *model_file_header) {
  if (file_path.empty() || model.model_data == nullptr || model.model_len == 0) {
    GELOGE(FAILED, "[Save][File]Incorrect input param, "
           "file_path is empty or model_data is nullptr or model_len is 0");
    REPORT_INNER_ERROR("E19999", "Save file failed, at least one of the "
                       "input parameters(file_path, model_data, model_len) is incorrect");
    return FAILED;
  }

  ModelFileHeader file_header;

  int32_t copy_header_ret = 0;
  GE_IF_BOOL_EXEC(model_file_header != nullptr, copy_header_ret = memcpy_s(&file_header, sizeof(ModelFileHeader),
                                                                           model_file_header, sizeof(ModelFileHeader)));
  GE_CHK_BOOL_RET_STATUS(copy_header_ret == 0, FAILED, "Copy ModelFileHeader failed, memcpy_s return: %d",
                         copy_header_ret);

  file_header.length = model.model_len;

  const Status ret = SaveWithFileHeader(file_path, file_header, model.model_data, file_header.length);
  if (ret != SUCCESS) {
    GELOGE(FAILED, "[Save][File]Failed, file_path:%s, file_header_len:%u, error_code:%u.",
		    file_path.c_str(), file_header.length, ret);
    return FAILED;
  }

  return SUCCESS;
}

Status FileSaver::SaveToFile(const string &file_path, ModelFileHeader &file_header,
                             ModelPartitionTable &model_partition_table,
                             const std::vector<ModelPartition> &partition_datas) {
  const Status ret = SaveWithFileHeader(file_path, file_header, model_partition_table, partition_datas);
  GE_CHK_BOOL_RET_STATUS(ret == SUCCESS, FAILED, "save file failed, file_path:%s, file header len:%u.",
                         file_path.c_str(), file_header.length);
  return SUCCESS;
}

Status FileSaver::SaveToFile(const string &file_path, ModelFileHeader &file_header,
                             vector<ModelPartitionTable *> &model_partition_tables,
                             const vector<vector<ModelPartition>> &all_partition_datas) {
  const Status ret = SaveWithFileHeader(file_path, file_header, model_partition_tables, all_partition_datas);
  GE_CHK_BOOL_RET_STATUS(ret == SUCCESS, FAILED, "save file failed, file_path:%s, file header len:%u.",
                         file_path.c_str(), file_header.length);
  return SUCCESS;
}

Status FileSaver::SaveWithFileHeader(const std::string &file_path, const ModelFileHeader &file_header,
                                     vector<ModelPartitionTable *> &model_partition_tables,
                                     const vector<vector<ModelPartition>> &all_partition_datas) {

  GE_CHK_BOOL_EXEC(model_partition_tables.size() == all_partition_datas.size(),
                   return PARAM_INVALID,
                   "model table size %zu does not match partition size %zu",
                   model_partition_tables.size(), all_partition_datas.size())
  for (size_t index = 0; index < model_partition_tables.size(); ++index) {
    auto &cur_partiton_data = all_partition_datas[index];
    auto &cur_model_partition_table = *model_partition_tables[index];
    GE_CHK_BOOL_RET_STATUS(!cur_partiton_data.empty() && cur_model_partition_table.num != 0
                           && cur_model_partition_table.num == cur_partiton_data.size(), FAILED,
                           "Invalid param:partition data size is (%u), model_partition_table.num is (%zu).",
                           cur_model_partition_table.num, cur_partiton_data.size());
  }

  // Open file
  int32_t fd = 0;
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(OpenFile(fd, file_path) != SUCCESS, return FAILED);
  Status ret = SUCCESS;
  do {
    // Write file header
    GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(
        WriteData(static_cast<const void *>(&file_header), sizeof(ModelFileHeader), fd) != SUCCESS, ret = FAILED;
        break);
    for (size_t index = 0; index < model_partition_tables.size(); ++index) {
      // Write model partition table
      auto &cur_tabel = *model_partition_tables[index];
      uint32_t table_size = static_cast<uint32_t>(SIZE_OF_MODEL_PARTITION_TABLE(cur_tabel));
      GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(
          WriteData(static_cast<const void *>(&cur_tabel), table_size, fd) != SUCCESS, ret = FAILED; break);
      // Write partition data
      auto &cur_partition_datas = all_partition_datas[index];
      for (const auto &partition_data : cur_partition_datas) {
        GELOGI("part_size[%u]", partition_data.size);
        GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(
            WriteData(static_cast<const void *>(partition_data.data), partition_data.size, fd) != SUCCESS, ret = FAILED;
            break);
      }
    }
  } while (0);
  // Close file
  if (mmClose(fd) != 0) {  // mmClose 0: success
    GELOGE(FAILED, "[Close][File]Failed, error_code:%u errmsg:%s", ret, strerror(errno));
    REPORT_CALL_ERROR("E19999", "Close file failed, error_code:%u errmsg:%s",
                      ret, strerror(errno));
    ret = FAILED;
  }
  return ret;
}

Status FileSaver::SaveToFile(const string &file_path, const void *data, int len) {
  if (data == nullptr || len <= 0) {
    GELOGE(FAILED, "[Check][Param]Failed, model_data is null or the "
           "length[%d] is less than 1.", len);
    REPORT_INNER_ERROR("E19999", "Save file failed, the model_data is null or "
                       "its length:%d is less than 1.", len);
    return FAILED;
  }

  // Open file
  int32_t fd = 0;
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(OpenFile(fd, file_path) != SUCCESS, return FAILED, "OpenFile FAILED");

  Status ret = SUCCESS;

  // write data
  GE_CHK_BOOL_EXEC(SUCCESS == WriteData(data, (uint32_t)len, fd), ret = FAILED, "WriteData FAILED");

  // Close file
  if (mmClose(fd) != 0) {  // mmClose 0: success
    GELOGE(FAILED, "[Close][File]Failed, error_code:%u errmsg:%s", ret, strerror(errno));
    REPORT_CALL_ERROR("E19999", "Close file failed, error_code:%u errmsg:%s",
                      ret, strerror(errno));
    ret = FAILED;
  }
  return ret;
}
}  //  namespace ge
