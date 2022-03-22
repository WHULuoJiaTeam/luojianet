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

#include "graph/buffer.h"
#include "proto/ge_ir.pb.h"
#include "framework/common/debug/ge_log.h"
#include "graph/buffer_impl.h"
#include "graph/debug/ge_util.h"
#include "graph/utils/mem_utils.h"

namespace ge {
BufferImpl::BufferImpl() {
  data_.InitDefault();
  if (data_.GetProtoMsg() != nullptr) {
    buffer_ = data_.GetProtoMsg()->mutable_bt();
  }
}

BufferImpl::BufferImpl(const BufferImpl &other) {
  data_ = other.data_;
  buffer_ = other.buffer_;
}

BufferImpl::~BufferImpl() {}

BufferImpl::BufferImpl(const std::size_t buffer_size, const std::uint8_t default_val) : BufferImpl() {  // default
  auto const proto_msg = data_.GetProtoMsg();
  if (proto_msg != nullptr) {
    try {
      proto_msg->set_bt(std::string(buffer_size, static_cast<char_t>(default_val)));
      buffer_ = proto_msg->mutable_bt();
    } catch (std::bad_alloc &) {
      REPORT_CALL_ERROR("E19999", "failed to alloc buffer memory, buffer size %zu", buffer_size);
      GELOGE(MEMALLOC_FAILED, "[New][Memory] failed to alloc buffer memory, buffer size %zu", buffer_size);
      buffer_ = nullptr;
    }
  }
}

void BufferImpl::CopyFrom(const std::uint8_t * const data, const std::size_t buffer_size) {
  auto const proto_msg = data_.GetProtoMsg();
  if ((proto_msg != nullptr) && (data != nullptr)) {
    try {
      proto_msg->set_bt(data, buffer_size);
      buffer_ = proto_msg->mutable_bt();
    } catch (std::bad_alloc &) {
      REPORT_CALL_ERROR("E19999", "Failed to alloc buffer memory, buffer size %zu", buffer_size);
      GELOGE(MEMALLOC_FAILED, "[New][Memory] Failed to alloc buffer memory, buffer size %zu", buffer_size);
      buffer_ = nullptr;
    }
  }
}

BufferImpl::BufferImpl(const std::shared_ptr<google::protobuf::Message> &proto_owner,
                       proto::AttrDef * const buffer)
    : data_(proto_owner, buffer) {
  if (data_.GetProtoMsg() != nullptr) {
    buffer_ = data_.GetProtoMsg()->mutable_bt();
  }
}

BufferImpl::BufferImpl(const std::shared_ptr<google::protobuf::Message> &proto_owner, std::string * const buffer)
    : data_(proto_owner, nullptr) {
  buffer_ = buffer;
}

BufferImpl &BufferImpl::operator=(const BufferImpl &other) {
  if (&other != this) {
    // Share data
    data_ = other.data_;
    buffer_ = other.buffer_;
  }
  return *this;
}

const std::uint8_t *BufferImpl::GetData() const {
  if (buffer_ != nullptr) {
    return reinterpret_cast<const uint8_t*>(buffer_->data());
  }
  return nullptr;
}

std::uint8_t *BufferImpl::GetData() {
  if ((buffer_ != nullptr) && (!buffer_->empty())) {
    // Avoid copy on write
    (void)(*buffer_)[0UL];
    return reinterpret_cast<uint8_t *>(const_cast<char_t *>(buffer_->data()));
  }
  return nullptr;
}

std::size_t BufferImpl::GetSize() const {
  if (buffer_ != nullptr) {
    return buffer_->size();
  }
  return 0UL;
}

void BufferImpl::ClearBuffer() {
  if (buffer_ != nullptr) {
    buffer_->clear();
  }
}

uint8_t BufferImpl::operator[](const size_t index) const {
  if ((buffer_ != nullptr) && (index < buffer_->size())) {
    return static_cast<uint8_t>((*buffer_)[index]);
  }
  return 0xffU;
}

Buffer::Buffer() : impl_(MakeShared<BufferImpl>()) {}

Buffer::Buffer(const Buffer &other)
    : impl_(MakeShared<BufferImpl>(*(other.impl_))) {}

Buffer::Buffer(const std::size_t buffer_size, const std::uint8_t default_val)
    : impl_(MakeShared<BufferImpl>(buffer_size, default_val)) {}

Buffer::~Buffer() {}

Buffer Buffer::CopyFrom(const std::uint8_t * const data, const std::size_t buffer_size) {
  Buffer buffer;
  if (buffer.impl_ != nullptr) {
    buffer.impl_->CopyFrom(data, buffer_size);
  }
  return buffer;
}

Buffer::Buffer(const ProtoMsgOwner &proto_owner, proto::AttrDef * const buffer)
    : impl_(MakeShared<BufferImpl>(proto_owner, buffer)) {}

Buffer::Buffer(const ProtoMsgOwner &proto_owner, std::string * const buffer)
    : impl_(MakeShared<BufferImpl>(proto_owner, buffer)) {}

Buffer &Buffer::operator=(const Buffer &other) {
  if (&other != this) {
    if (impl_ != nullptr) {
      *impl_ = *(other.impl_);
    }
  }
  return *this;
}

const std::uint8_t *Buffer::GetData() const {
  return impl_->GetData();
}

std::uint8_t *Buffer::GetData() {
  return impl_->GetData();
}

std::size_t Buffer::GetSize() const {
  return impl_->GetSize();
}

void Buffer::ClearBuffer() {
  impl_->ClearBuffer();
}

const std::uint8_t *Buffer::data() const { return GetData(); }

std::uint8_t *Buffer::data() { return GetData(); }

std::size_t Buffer::size() const { return GetSize(); }

void Buffer::clear() { return ClearBuffer(); }

uint8_t Buffer::operator[](const size_t index) const {
  return (*impl_)[index];
}

Buffer BufferUtils::CreateShareFrom(const Buffer &other) {
  return other;
}

Buffer BufferUtils::CreateCopyFrom(const Buffer &other) {
  return BufferUtils::CreateCopyFrom(other.GetData(), other.GetSize());
}

Buffer BufferUtils::CreateCopyFrom(const std::uint8_t * const data, const std::size_t buffer_size) {
  return Buffer::CopyFrom(data, buffer_size);
}

void BufferUtils::ShareFrom(const Buffer &from, Buffer &to) {
  to = from;
}

void BufferUtils::CopyFrom(const Buffer &from, Buffer &to) {
  to = BufferUtils::CreateCopyFrom(from);
}
}  // namespace ge
