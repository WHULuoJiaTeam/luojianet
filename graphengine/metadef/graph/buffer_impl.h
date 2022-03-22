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

#ifndef GRAPH_BUFFER_IMPL_H_
#define GRAPH_BUFFER_IMPL_H_

#include <string>
#include "proto/ge_ir.pb.h"

namespace ge {
class BufferImpl {
 public:
  BufferImpl();
  ~BufferImpl();
  BufferImpl(const BufferImpl &other);
  BufferImpl(const std::size_t buffer_size, const std::uint8_t default_val);

  void CopyFrom(const std::uint8_t * const data, std::size_t buffer_size);
  BufferImpl(const std::shared_ptr<google::protobuf::Message> &proto_owner, proto::AttrDef * const buffer);
  BufferImpl(const std::shared_ptr<google::protobuf::Message> &proto_owner, std::string * const buffer);

  BufferImpl &operator=(const BufferImpl &other);
  const std::uint8_t *GetData() const;
  std::uint8_t *GetData();
  std::size_t GetSize() const;
  void ClearBuffer();
  uint8_t operator[](const size_t index) const;

 private:
  friend class GeAttrValueImp;
  GeIrProtoHelper<proto::AttrDef> data_;
  std::string *buffer_ = nullptr;
};
}  // namespace ge
#endif  // GRAPH_BUFFER_IMPL_H_
