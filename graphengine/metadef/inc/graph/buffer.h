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

#ifndef INC_GRAPH_BUFFER_H_
#define INC_GRAPH_BUFFER_H_

#include <graph/types.h>
#include <memory>
#include <string>
#include <vector>
#include "detail/attributes_holder.h"
#include "graph/compiler_options.h"

namespace ge {
class BufferImpl;
using BufferImplPtr = std::shared_ptr<BufferImpl>;

class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY Buffer {
 public:
  Buffer();
  Buffer(const Buffer &other);

  explicit Buffer(const std::size_t buffer_size, const std::uint8_t default_val = 0U);

  ~Buffer();

  Buffer &operator=(const Buffer &other);
  static Buffer CopyFrom(const std::uint8_t *const data, const std::size_t buffer_size);

  const std::uint8_t *GetData() const;
  std::uint8_t *GetData();
  std::size_t GetSize() const;
  void ClearBuffer();

  // For compatibility
  const std::uint8_t *data() const;
  std::uint8_t *data();
  std::size_t size() const;
  void clear();
  uint8_t operator[](const size_t index) const;

 private:
  BufferImplPtr impl_;

  // Create from protobuf obj
  Buffer(const ProtoMsgOwner &proto_owner, proto::AttrDef *const buffer);
  Buffer(const ProtoMsgOwner &proto_owner, std::string *const buffer);

  friend class GeAttrValueImp;
  friend class GeTensor;
  friend class BufferUtils;
};

class BufferUtils {
 public:
  static Buffer CreateShareFrom(const Buffer &other);
  static Buffer CreateCopyFrom(const Buffer &other);
  static Buffer CreateCopyFrom(const std::uint8_t *const data, const std::size_t buffer_size);
  static void ShareFrom(const Buffer &from, Buffer &to);
  static void CopyFrom(const Buffer &from, Buffer &to);
};
}  // namespace ge
#endif  // INC_GRAPH_BUFFER_H_
