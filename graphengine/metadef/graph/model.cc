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

#include "graph/model.h"
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iomanip>
#include "graph/debug/ge_attr_define.h"
#include "debug/ge_util.h"
#include "framework/common/debug/ge_log.h"
#include "graph/model_serialize.h"
#include "mmpa/mmpa_api.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/ge_ir_utils.h"
#include "proto/ge_ir.pb.h"

namespace {
using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;

const int32_t DEFAULT_VERSION = 1U;
const int32_t ACCESS_PERMISSION_BITS = 0400;
static ge::ModelSerialize SERIALIZE;
}  // namespace

namespace ge {
static char_t *GetStrError() {
  constexpr size_t kMaxErrorStringLen = 128U;
  char_t err_buf[kMaxErrorStringLen + 1U] = {};
  const auto str_error = mmGetErrorFormatMessage(mmGetErrorCode(), &err_buf[0], kMaxErrorStringLen);
  return str_error;
}

void Model::Init() {
  (void)AttrUtils::SetInt(this, ATTR_MODEL_MEMORY_SIZE, 0U);
  (void)AttrUtils::SetInt(this, ATTR_MODEL_P2P_MEMORY_SIZE, 0U);
  (void)AttrUtils::SetInt(this, ATTR_MODEL_STREAM_NUM, 0U);
  (void)AttrUtils::SetInt(this, ATTR_MODEL_EVENT_NUM, 0U);
  (void)AttrUtils::SetInt(this, ATTR_MODEL_LABEL_NUM, 0U);
  (void)AttrUtils::SetInt(this, ATTR_MODEL_WEIGHT_SIZE, 0U);
  (void)AttrUtils::SetStr(this, ATTR_MODEL_TARGET_TYPE, TARGET_TYPE_MINI);
  version_ = 0;
}

Model::Model() :AttrHolder() {
  Init();
}

Model::Model(const std::string &name, const std::string &custom_version)
    : AttrHolder(), name_(name), version_(static_cast<uint32_t>(DEFAULT_VERSION)), platform_version_(custom_version) {
  Init();
}

std::string Model::GetName() const { return name_; }

void Model::SetName(const std::string &name) { name_ = name; }

uint32_t Model::GetVersion() const { return version_; }

std::string Model::GetPlatformVersion() const { return platform_version_; }

void Model::SetGraph(const ge::Graph &graph) { graph_ = graph; }

Graph Model::GetGraph() const { return graph_; }

graphStatus Model::Save(Buffer &buffer, bool is_dump) const {
  buffer = SERIALIZE.SerializeModel(*this, is_dump);
  return (buffer.GetSize() > 0U) ? GRAPH_SUCCESS : GRAPH_FAILED;
}

void Model::SetAttr(const ProtoAttrMap &attrs) { attrs_ = attrs; }

graphStatus Model::Load(const uint8_t *data, size_t len, Model &model) {
  return SERIALIZE.UnserializeModel(data, len, model) ? GRAPH_SUCCESS : GRAPH_FAILED;
}

graphStatus Model::Load(ge::proto::ModelDef &model_def) {
  return SERIALIZE.UnserializeModel(model_def, *this) ? GRAPH_SUCCESS : GRAPH_FAILED;
}

graphStatus Model::SaveToFile(const std::string &file_name) const {
  Buffer buffer;
  if ((*this).Save(buffer) != GRAPH_SUCCESS) {
    GE_LOGE("[Save][Data] to file:%s fail.", file_name.c_str());
    return GRAPH_FAILED;
  }
  // Write file
  ge::proto::ModelDef ge_proto;
  if (buffer.GetData() != nullptr) {
    const std::string str(reinterpret_cast<char_t *>(buffer.GetData()), buffer.GetSize());
    if (!ge_proto.ParseFromString(str)) {
      return GRAPH_FAILED;
    }
    char_t real_path[MMPA_MAX_PATH] = {};
    if (strnlen(file_name.c_str(), sizeof(real_path)) >= sizeof(real_path)) {
      return GRAPH_FAILED;
    }
    const INT32 result = mmRealPath(file_name.c_str(), &real_path[0], MMPA_MAX_PATH);
    if (result != EN_OK) {
      GELOGI("file %s does not exit, it will be created.", file_name.c_str());
    }
    const int32_t fd =
        mmOpen2(&real_path[0], static_cast<int32_t>(static_cast<uint32_t>(M_WRONLY) | static_cast<uint32_t>(M_CREAT) |
            static_cast<uint32_t>(O_TRUNC)), static_cast<uint32_t>(ACCESS_PERMISSION_BITS));
    if (fd < 0) {
      REPORT_CALL_ERROR("E19999", "open file:%s failed, error:%s ", &real_path[0], GetStrError());
      GELOGE(GRAPH_FAILED, "[Open][File] %s failed, error:%s ", &real_path[0], GetStrError());
      return GRAPH_FAILED;
    }
    const bool ret = ge_proto.SerializeToFileDescriptor(fd);
    if (!ret) {
      REPORT_CALL_ERROR("E19999", "SerializeToFileDescriptor failed, file:%s.", &real_path[0]);
      GELOGE(GRAPH_FAILED, "[Call][SerializeToFileDescriptor] failed, file:%s.", &real_path[0]);
      if (close(fd) != 0) {
        REPORT_CALL_ERROR("E19999", "close file:%s fail, error:%s.", &real_path[0], GetStrError());
        GELOGE(GRAPH_FAILED, "[Close][File] %s fail, error:%s.", &real_path[0], GetStrError());
        return GRAPH_FAILED;
      }
      return GRAPH_FAILED;
    }
    if (close(fd) != 0) {
      REPORT_CALL_ERROR("E19999", "close file:%s fail, error:%s.", &real_path[0], GetStrError());
      GELOGE(GRAPH_FAILED, "[Close][File] %s fail, error:%s.", &real_path[0], GetStrError());
      return GRAPH_FAILED;
    }
    if (!ret) {
      REPORT_CALL_ERROR("E19999", "SerializeToFileDescriptor failed, file:%s.", &real_path[0]);
      GELOGE(GRAPH_FAILED, "[Call][SerializeToFileDescriptor] failed, file:%s.", &real_path[0]);
      return GRAPH_FAILED;
    }
  }
  return GRAPH_SUCCESS;
}

bool Model::IsValid() const { return graph_.IsValid(); }

graphStatus Model::LoadFromFile(const std::string &file_name) {
  char_t real_path[MMPA_MAX_PATH] = {};
  if (strnlen(file_name.c_str(), sizeof(real_path)) >= sizeof(real_path)) {
    return GRAPH_FAILED;
  }
  const INT32 result = mmRealPath(file_name.c_str(), &real_path[0], MMPA_MAX_PATH);
  if (result != EN_OK) {
    REPORT_CALL_ERROR("E19999", "get realpath failed for %s, error:%s.", file_name.c_str(), GetStrError());
    GELOGE(GRAPH_FAILED, "[Get][RealPath] failed for %s, error:%s.", file_name.c_str(), GetStrError());
    return GRAPH_FAILED;
  }
  const int32_t fd = mmOpen(&real_path[0], M_RDONLY);
  if (fd < 0) {
    REPORT_CALL_ERROR("E19999", "open file:%s failed, error:%s", &real_path[0], GetStrError());
    GELOGE(GRAPH_FAILED, "[Open][File] %s failed, error:%s", &real_path[0], GetStrError());
    return GRAPH_FAILED;
  }

  ge::proto::ModelDef model_def;
  const bool ret = model_def.ParseFromFileDescriptor(fd);
  if (!ret) {
    REPORT_CALL_ERROR("E19999", "ParseFromFileDescriptor failed, file:%s.", &real_path[0]);
    GELOGE(GRAPH_FAILED, "[Call][ParseFromFileDescriptor] failed, file:%s.", &real_path[0]);
    if (mmClose(fd) != 0) {
      REPORT_CALL_ERROR("E19999", "close file:%s fail, error:%s.", &real_path[0], GetStrError());
      GELOGE(GRAPH_FAILED, "[Close][File] %s fail. error:%s", &real_path[0], GetStrError());
      return GRAPH_FAILED;
    }
    return GRAPH_FAILED;
  }
  if (mmClose(fd) != 0) {
    REPORT_CALL_ERROR("E19999", "close file:%s fail, error:%s.", &real_path[0], GetStrError());
    GELOGE(GRAPH_FAILED, "[Close][File] %s fail. error:%s", &real_path[0], GetStrError());
    return GRAPH_FAILED;
  }
  if (!ret) {
    REPORT_CALL_ERROR("E19999", "ParseFromFileDescriptor failed, file:%s.", &real_path[0]);
    GELOGE(GRAPH_FAILED, "[Call][ParseFromFileDescriptor] failed, file:%s.", &real_path[0]);
    return GRAPH_FAILED;
  }
  return Load(model_def);
}

ProtoAttrMap &Model::MutableAttrMap() { return attrs_; }

ConstProtoAttrMap &Model::GetAttrMap() const {
  return attrs_;
}
}  // namespace ge
