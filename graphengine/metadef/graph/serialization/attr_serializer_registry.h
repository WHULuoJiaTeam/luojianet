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

#ifndef METADEF_CXX_ATTR_SERIALIZER_REGISTRY_H
#define METADEF_CXX_ATTR_SERIALIZER_REGISTRY_H
#include <functional>
#include <memory>
#include <mutex>
#include <map>
#include <utility>

#include "graph/type_utils.h"
#include "attr_serializer.h"

#define REG_GEIR_SERIALIZER(cls, obj_type, bin_type)                                       \
    REG_GEIR_SERIALIZER_BUILDER_UNIQ_HELPER(__COUNTER__, cls, obj_type, bin_type)

#define REG_GEIR_SERIALIZER_BUILDER_UNIQ_HELPER(ctr, cls, obj_type, bin_type)              \
    REG_GEIR_SERIALIZER_BUILDER_UNIQ(ctr, cls, obj_type, bin_type)

#define REG_GEIR_SERIALIZER_BUILDER_UNIQ(ctr, cls, obj_type, bin_type)                     \
  static ::ge::AttrSerializerRegistrar register_serialize##ctr                             \
      __attribute__((unused)) =                                                            \
          ::ge::AttrSerializerRegistrar([]()->std::unique_ptr<ge::GeIrAttrSerializer>{     \
               return std::unique_ptr<ge::GeIrAttrSerializer>(new(std::nothrow)cls());     \
          }, obj_type, bin_type)

namespace ge {
using GeIrAttrSerializerBuilder = std::function<std::unique_ptr<GeIrAttrSerializer>()>;
class AttrSerializerRegistry {
 public:
  AttrSerializerRegistry(const AttrSerializerRegistry &) = delete;
  AttrSerializerRegistry(AttrSerializerRegistry &&) = delete;
  AttrSerializerRegistry &operator=(const AttrSerializerRegistry &) = delete;
  AttrSerializerRegistry &operator=(AttrSerializerRegistry &&) = delete;

  ~AttrSerializerRegistry() = default;

  static AttrSerializerRegistry &GetInstance();
  /**
   * 注册一个GE IR的序列化、反序列化handler
   * @param builder 调用该builder时，返回一个handler的实例
   * @param obj_type 内存中的数据类型，可以通过`GetTypeId<T>`函数获得
   * @param proto_type protobuf数据类型枚举值
   */
  void RegisterGeIrAttrSerializer(const GeIrAttrSerializerBuilder &builder,
                                  const TypeId obj_type,
                                  const proto::AttrDef::ValueCase proto_type);

  GeIrAttrSerializer *GetSerializer(const TypeId obj_type);
  GeIrAttrSerializer *GetDeserializer(const proto::AttrDef::ValueCase proto_type);

 private:
  AttrSerializerRegistry() = default;

  std::mutex mutex_;
  std::vector<std::unique_ptr<GeIrAttrSerializer>> serializer_holder_;
  std::map<TypeId, GeIrAttrSerializer *> serializer_map_;
  std::map<proto::AttrDef::ValueCase, GeIrAttrSerializer *> deserializer_map_;
};

class AttrSerializerRegistrar {
 public:
  AttrSerializerRegistrar(const GeIrAttrSerializerBuilder builder,
                          const TypeId obj_type,
                          const proto::AttrDef::ValueCase proto_type);
  ~AttrSerializerRegistrar() = default;
};
}  // namespace ge

#endif  //METADEF_CXX_ATTR_SERIALIZER_REGISTRY_H
