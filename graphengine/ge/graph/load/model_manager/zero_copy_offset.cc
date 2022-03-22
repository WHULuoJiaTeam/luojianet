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

#include "graph/load/model_manager/zero_copy_offset.h"

#include "framework/common/debug/ge_log.h"
#include "framework/common/util.h"
#include "graph/load/model_manager/model_utils.h"
#include "graph/load/model_manager/zero_copy_task.h"

namespace ge {
namespace {
const uint32_t kDataIndex = 0;
}  // namespace

ZeroCopyOffset::ZeroCopyOffset() {}

ZeroCopyOffset::~ZeroCopyOffset() {}

Status ZeroCopyOffset::InitInputDataInfo(int64_t output_size, void *virtual_addr, const OpDescPtr &op_desc,
                                         bool &fusion_flag) {
  GELOGI("[ZCPY] Start to InitInputDataInfo of %s, total_data_size is %ld, virtual_addr is %p",
         op_desc->GetName().c_str(), output_size, virtual_addr);
  basic_addr_ = virtual_addr;
  op_name_ = op_desc->GetName();
  (void)ge::AttrUtils::GetListInt(op_desc, ATTR_ZERO_COPY_BASIC_OFFSET, zero_copy_basic_offset_);
  (void)ge::AttrUtils::GetListInt(op_desc, ATTR_ZERO_COPY_RELATIVE_OFFSET, zero_copy_relative_offset_);
  GE_CHK_BOOL_EXEC(zero_copy_basic_offset_.size() == zero_copy_relative_offset_.size(),
                   REPORT_INNER_ERROR("E19999", "basic_offset_size:%zu not equal to relative_offset_size:%zu, "
                                      "check invalid", zero_copy_basic_offset_.size(),
                                      zero_copy_relative_offset_.size());
                   return PARAM_INVALID,
                   "[Check][Param] basic_offset_size:%zu should be equal to relative_offset_size:%zu",
                   zero_copy_basic_offset_.size(), zero_copy_relative_offset_.size());
  GELOGD("[ZCPY] zero_copy_basic_offset size is %zu", zero_copy_basic_offset_.size());

  int64_t virtual_addr_offset = op_desc->GetOutputOffset().at(kDataIndex);
  IsL2Fusion(zero_copy_basic_offset_, virtual_addr_offset, fusion_flag);

  uint32_t out_count = 0;
  data_size_ = output_size;
  if (!fusion_flag) {
    out_count++;
    data_info_.emplace_back(output_size, virtual_addr);
    relative_offset_.emplace_back(0);
    GELOGD("[ZCPY] %s size is %ld, virtual_addr is %p.", op_desc->GetName().c_str(), output_size, virtual_addr);
  } else {
    GELOGI("[ZCPY] set l2_fusion for %s.", op_desc->GetName().c_str());
    for (size_t index = 0; index < zero_copy_basic_offset_.size(); ++index) {
      if (zero_copy_basic_offset_.at(index) == virtual_addr_offset) {
        out_count++;
        uint64_t out_offset = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(virtual_addr)) +
                              zero_copy_relative_offset_.at(index);
        data_info_.emplace_back(output_size, reinterpret_cast<void *>(static_cast<uintptr_t>(out_offset)));
        relative_offset_.emplace_back(zero_copy_relative_offset_.at(index));
        GELOGI("[ZCPY] virtual_addr: %p has been l2-fusion to %lu, need copy data_size is %ld.", basic_addr_,
               out_offset, output_size);
      }
    }
  }
  data_count_ = out_count;
  return SUCCESS;
}

Status ZeroCopyOffset::InitOutputDataInfo(const vector<int64_t> &input_size_list,
                                          const vector<void *> &virtual_addr_list, const OpDescPtr &op_desc,
                                          const size_t &idx, bool &fusion_flag) {
  int64_t size = input_size_list[idx];
  auto tensor_desc = op_desc->GetInputDescPtr(idx);
  GE_CHECK_NOTNULL(tensor_desc);
  if (TensorUtils::GetTensorSizeInBytes(*tensor_desc, size) != GRAPH_SUCCESS) {
    REPORT_INNER_ERROR("E19999", "Get input TensorSize in op:%s(%s) failed, input_index:%zu",
                       op_desc->GetName().c_str(), op_desc->GetType().c_str(), idx);
    GELOGE(FAILED, "[Get][InputTensorSize] in op:%s(%s) failed, input_index:%zu",
           op_desc->GetName().c_str(), op_desc->GetType().c_str(), idx);
    return FAILED;
  }

  GELOGD("Tensor data size: GetSize=%ld, GetTensorSizeInBytes=%ld", input_size_list[idx], size);

  basic_addr_ = virtual_addr_list[idx];
  op_name_ = op_desc->GetName();
  (void)ge::AttrUtils::GetListInt(op_desc, ATTR_ZERO_COPY_BASIC_OFFSET, zero_copy_basic_offset_);
  (void)ge::AttrUtils::GetListInt(op_desc, ATTR_ZERO_COPY_RELATIVE_OFFSET, zero_copy_relative_offset_);
  GE_CHK_BOOL_EXEC(zero_copy_basic_offset_.size() == zero_copy_relative_offset_.size(),
                   REPORT_INNER_ERROR("E19999", "basic_offset_size:%zu not equal to relative_offset_size:%zu, "
                                      "check invalid",
                                      zero_copy_basic_offset_.size(), zero_copy_relative_offset_.size());
                   return PARAM_INVALID,
                   "[Check][Param] basic_offset_size:%zu should be equal to relative_offset_size:%zu",
                   zero_copy_basic_offset_.size(), zero_copy_relative_offset_.size());
  int64_t virtual_addr_offset = op_desc->GetInputOffset().at(idx);
  IsL2Fusion(zero_copy_basic_offset_, virtual_addr_offset, fusion_flag);

  uint32_t in_count = 0;
  data_size_ = size;
  if (!fusion_flag) {
    in_count++;
    data_info_.emplace_back(size, virtual_addr_list[idx]);
    // op_desc not set l2fusion when fusion_flag is false
    relative_offset_.emplace_back(0);
    GELOGI("[ZCPY] %s size is %ld, virtual_addr is %p.", op_desc->GetName().c_str(), size, virtual_addr_list[idx]);
  } else {
    GELOGI("[ZCPY] set l2-fusion for %s.", op_desc->GetName().c_str());
    for (size_t index = 0; index < zero_copy_basic_offset_.size(); ++index) {
      if (zero_copy_basic_offset_.at(index) == virtual_addr_offset) {
        in_count++;
        uint64_t in_offset = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(virtual_addr_list[idx])) +
                             zero_copy_relative_offset_.at(index);
        int64_t real_data_size = ModelUtils::GetInputSize(op_desc).at(idx);
        data_info_.emplace_back(real_data_size, reinterpret_cast<void *>(static_cast<uintptr_t>(in_offset)));
        relative_offset_.emplace_back(zero_copy_relative_offset_.at(index));
        GELOGI("[ZCPY] virtual_addr: %p has been l2-fusion from %lu, need copy data_size is %ld.", basic_addr_,
               in_offset, real_data_size);
      }
    }
  }
  data_count_ = in_count;
  return SUCCESS;
}

void ZeroCopyOffset::IsL2Fusion(const vector<int64_t> &fusion_basic_addrs, const int64_t &tensor_offset,
                                bool &fusion_flag) {
  for (size_t fusion_count = 0; fusion_count < fusion_basic_addrs.size(); ++fusion_count) {
    if (fusion_basic_addrs.at(fusion_count) == tensor_offset) {
      fusion_flag = true;
      break;
    }
  }
}

void ZeroCopyOffset::SetInputOutsideAddrs(int64_t output_offset, void *addr, bool fusion_flag,
                                          set<const void *> &real_virtual_addrs) {
  uint32_t out_count = 0;
  if (!fusion_flag) {
    out_count++;
    std::map<const void *, std::vector<void *>> addr_mapping;
    addr_mapping[addr] = {};
    outside_addrs_.emplace_back(addr_mapping);
    real_virtual_addrs.insert(addr);
  } else {
    GELOGI("[ZCPY] set l2-fusion for virtual_addr %p.", addr);
    for (size_t i = 0; i < zero_copy_basic_offset_.size(); ++i) {
      if (zero_copy_basic_offset_.at(i) == output_offset) {
        out_count++;
        void *virtual_addr =
            reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(addr) + zero_copy_relative_offset_.at(i));
        std::map<const void *, std::vector<void *>> addr_mapping;
        addr_mapping[virtual_addr] = {};
        outside_addrs_.emplace_back(addr_mapping);
        real_virtual_addrs.insert(virtual_addr);
        GELOGI("[ZCPY] virtual_addr %p has been fusion to virtual_addr %p.", addr, virtual_addr);
      }
    }
  }
  addr_count_ = out_count;
  valid_relative_offset_ = true;
}

void ZeroCopyOffset::SetOutputOutsideAddrs(const int64_t &input_offset, const bool &fusion_flag, void *addr,
                                           std::vector<void *> &tensor_addrs) {
  GELOGI("[ZCPY] Start to SetOutputOutsideAddrs for virtual_addr %p.", addr);
  uint32_t out_count = 0;
  if (!fusion_flag) {
    out_count++;
    std::map<const void *, std::vector<void *>> addr_mapping;
    addr_mapping[addr] = {};
    outside_addrs_.emplace_back(addr_mapping);
    tensor_addrs.emplace_back(addr);
  } else {
    GELOGI("[ZCPY] set l2-fusion for virtual_addr %p.", addr);
    for (size_t i = 0; i < zero_copy_basic_offset_.size(); ++i) {
      if (zero_copy_basic_offset_.at(i) == input_offset) {
        out_count++;
        void *virtual_addr =
            reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(addr) + zero_copy_relative_offset_.at(i));
        std::map<const void *, std::vector<void *>> addr_mapping;
        addr_mapping[virtual_addr] = {};
        outside_addrs_.emplace_back(addr_mapping);
        tensor_addrs.emplace_back(virtual_addr);
        GELOGI("[ZCPY] virtual_addr %p has been fusion to virtual_addr %p.", addr, virtual_addr);
      }
    }
  }
  addr_count_ = out_count;
  valid_relative_offset_ = true;
}

void ZeroCopyOffset::SetOutsideAddrsValue(ZeroCopyTask &zero_copy_task, void *outside_addr, void *args, size_t offset) {
  if (!valid_relative_offset_) {
    return;
  }
  const auto addr_val = reinterpret_cast<uintptr_t>(outside_addr);
  for (uint32_t out_count = 0; out_count < GetAddrCount(); ++out_count) {
    auto args_addrs = outside_addrs_[out_count].find(outside_addr);
    if (args_addrs != outside_addrs_[out_count].end()) {
      GE_CHK_STATUS(zero_copy_task.SetTaskArgsOffset(addr_val, offset),
                    "[Set][TaskArgsOffset] failed, Input args invalid, offset:%zu.", offset);
      void *args_val = static_cast<uint8_t *>(args) + offset;
      args_addrs->second.push_back(args_val);
      GELOGD("[ZCPY] set copy input: virtual_addr: 0x%lx, task_addr: %p, args: %p, offset: %zu.", addr_val, args_val,
             args, offset);
    }
  }
}

}  // namespace ge
