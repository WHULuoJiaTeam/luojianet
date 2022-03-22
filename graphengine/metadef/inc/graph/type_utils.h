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

#ifndef EXECUTE_GRAPH_TYPE_UTILS_H
#define EXECUTE_GRAPH_TYPE_UTILS_H
#include <type_traits>
#include "graph/types.h"

namespace ge {
class GeTensor;
class GeTensorDesc;
class Buffer;
class NamedAttrs;
namespace proto {
class GraphDef;
}

template<typename T>
struct GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeIdHolder {
  static char id;
};

template<typename T> char TypeIdHolder<T>::id = 0;

using TypeId = void *;
constexpr TypeId kInvalidTypeId = nullptr;

template<typename T>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId(const T &obj) {
  return GetTypeId<T>();
}

template<typename T>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId() {
  using PureT = typename std::remove_cv<typename std::remove_reference<T>::type>::type;
  return &(TypeIdHolder<PureT>::id);
}

template<>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<bool>() ;

template<>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<std::string>() ;

template<>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<float>() ;

template<>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<int64_t>() ;

template<>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<GeTensorDesc>() ;

template<>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<GeTensor>() ;

template<>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<Buffer>() ;

template<>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<proto::GraphDef>() ;

template<>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<NamedAttrs>() ;

template<>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<std::vector<std::vector<int64_t>>>() ;

template<>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<DataType>() ;

template<>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<std::vector<std::vector<float>>>() ;

template<>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<std::vector<std::string>>() ;

template<>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<std::vector<float>>() ;

template<>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<std::vector<bool>>() ;

template<>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<std::vector<int64_t>>() ;

template<>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<std::vector<GeTensorDesc>>() ;

template<>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<std::vector<GeTensor>>() ;

template<>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<std::vector<Buffer>>() ;

template<>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<std::vector<proto::GraphDef>>() ;

template<>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<std::vector<NamedAttrs>>() ;

template<>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<std::vector<DataType>>() ;
}
#endif  //EXECUTE_GRAPH_TYPE_UTILS_H
