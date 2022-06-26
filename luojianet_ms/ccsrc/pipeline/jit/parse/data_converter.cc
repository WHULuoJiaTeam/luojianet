/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
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

#include "pipeline/jit/parse/data_converter.h"
#include <utility>
#include <string>
#include <memory>
#include <vector>
#include "utils/hash_map.h"
#include "pipeline/jit/parse/resolve.h"
#include "pipeline/jit/pipeline.h"
#include "include/common/utils/python_adapter.h"
#include "frontend/operator/ops.h"
#include "frontend/operator/composite/composite.h"
#include "ir/func_graph_cloner.h"
#include "ir/cell.h"
#include "utils/symbolic.h"
#include "utils/ms_context.h"
#include "include/common/utils/utils.h"

namespace luojianet_ms {
namespace parse {
namespace {
struct PyDataToValueRegister {
  PyDataToValueRegister() { python_adapter::PyAdapterCallback::SetPyDataToValueHandler(data_converter::PyDataToValue); }
} callback_register;
}  // namespace
using Tensor = luojianet_ms::tensor::Tensor;
using TensorPtr = luojianet_ms::tensor::TensorPtr;
using MetaTensor = luojianet_ms::tensor::MetaTensor;
using MetaTensorPtr = luojianet_ms::tensor::MetaTensorPtr;
using CSRTensor = luojianet_ms::tensor::CSRTensor;
using CSRTensorPtr = luojianet_ms::tensor::CSRTensorPtr;
using COOTensor = luojianet_ms::tensor::COOTensor;
using COOTensorPtr = luojianet_ms::tensor::COOTensorPtr;

using InstanceCheckFunc = std::function<bool(const py::object &)>;
using InstanceConvertFunc = std::function<ValuePtr(const py::object &, bool, const TypePtr &)>;
static constexpr int kBit8 = 8;
static constexpr int kBit16 = 16;
static constexpr int kBit32 = 32;
static constexpr int kBit64 = 64;

class DataConverter {
 public:
  explicit DataConverter(InstanceConvertFunc convert_func) : convert_func_(std::move(convert_func)) {}

  virtual ~DataConverter() = default;

  virtual bool Matched(const py::object &obj) = 0;

  virtual ValuePtr ConvertPyObject(const py::object &obj, bool use_sig, const TypePtr &dtype) {
    if (convert_func_ == nullptr) {
      MS_LOG(EXCEPTION) << "convert func is null";
    }
    return convert_func_(obj, use_sig, dtype);
  }

 private:
  InstanceConvertFunc convert_func_ = nullptr;
};

using DataConverterPtr = std::shared_ptr<DataConverter>;

using ArgsObjConvertFunc = std::function<ValuePtr(const py::object &)>;
using ArgsObjSigConvertFunc = std::function<ValuePtr(const py::object &, bool)>;
using ArgsOjbTypeConvertFunc = std::function<ValuePtr(const py::object &, const TypePtr &)>;

// Convert the data according instance type
template <typename T>
class ByTypeDataConverter : public DataConverter {
 public:
  explicit ByTypeDataConverter(const InstanceConvertFunc &convert_func)
      : DataConverter(convert_func), check_func_(py::isinstance<T>) {}

  explicit ByTypeDataConverter(const ValuePtr &converted_type)
      : DataConverter(
          [converted_type](const py::object &, bool, const TypePtr &) -> ValuePtr { return converted_type; }),
        check_func_(py::isinstance<T>) {}

  explicit ByTypeDataConverter(const ArgsObjConvertFunc &convert_func)
      : DataConverter(
          [convert_func](const py::object &obj, bool, const TypePtr &) -> ValuePtr { return convert_func(obj); }),
        check_func_(py::isinstance<T>) {}

  explicit ByTypeDataConverter(const ArgsObjSigConvertFunc &convert_func)
      : DataConverter([convert_func](const py::object &obj, bool use_sig, const TypePtr &) -> ValuePtr {
          return convert_func(obj, use_sig);
        }),
        check_func_(py::isinstance<T>) {}

  explicit ByTypeDataConverter(const ArgsOjbTypeConvertFunc &convert_func)
      : DataConverter([convert_func](const py::object &obj, bool, const TypePtr &dtype) -> ValuePtr {
          return convert_func(obj, dtype);
        }),
        check_func_(py::isinstance<T>) {}

  ~ByTypeDataConverter() override = default;

  bool Matched(const py::object &obj) override { return check_func_ != nullptr ? check_func_(obj) : false; }

 private:
  InstanceCheckFunc check_func_ = nullptr;
};

// Convert the data according object attribute.
class ByAttrDataConverter : public DataConverter {
 public:
  ByAttrDataConverter(const std::string &attr_name, const ArgsObjConvertFunc &convert_func)
      : DataConverter(
          [convert_func](const py::object &obj, bool, const TypePtr &) -> ValuePtr { return convert_func(obj); }),
        attr_name_(attr_name) {}

  ByAttrDataConverter(const std::string &attr_name, const ArgsObjSigConvertFunc &convert_func)
      : DataConverter([convert_func](const py::object &obj, bool use_sig, const TypePtr &) -> ValuePtr {
          return convert_func(obj, use_sig);
        }),
        attr_name_(attr_name) {}

  ~ByAttrDataConverter() override = default;

  bool Matched(const py::object &obj) override { return py::hasattr(obj, attr_name_.c_str()); }

 private:
  std::string attr_name_;
};

FuncGraphPtr ConvertToBpropCut(const py::object &obj) {
  std::vector<std::string> results = data_converter::GetObjKey(obj);
  std::string obj_key = results[0];
  py::function bprop_func = py::getattr(obj, CUSTOM_BPROP_NAME);

  auto bprop_graph = std::make_shared<FuncGraph>();
  std::vector<AnfNodePtr> outputs;

  auto fake_bprop = std::make_shared<PrimitivePy>("bprop_cut");
  fake_bprop->AddBackwardHookFn(0, bprop_func);
  (void)fake_bprop->AddAttr(CUSTOM_BPROP_NAME, MakeValue(true));
  outputs.push_back(NewValueNode(fake_bprop));

  py::object code_obj = py::getattr(bprop_func, "__code__");
  // Three parameters self, out and dout need to be excluded
  constexpr auto kBpropExcludeParamNum = 3;
  size_t inputs_num = py::cast<int64_t>(py::getattr(code_obj, "co_argcount")) - kBpropExcludeParamNum;
  for (size_t i = 0; i < inputs_num; ++i) {
    auto param = bprop_graph->add_parameter();
    outputs.push_back(param);
  }
  auto p1 = bprop_graph->add_parameter();
  auto p2 = bprop_graph->add_parameter();
  outputs.push_back(p1);
  outputs.push_back(p2);

  bprop_graph->set_output(bprop_graph->NewCNode(std::move(outputs)));
  data_converter::SetObjGraphValue(obj_key, bprop_graph);
  return bprop_graph;
}

namespace {
ValuePtr ConvertTuple(const py::object &obj, bool use_signature) {
  MS_LOG(DEBUG) << "Converting python tuple";
  auto tuple = obj.cast<py::tuple>();
  std::vector<ValuePtr> value_list;
  for (size_t it = 0; it < tuple.size(); ++it) {
    ValuePtr out = nullptr;
    bool success = ConvertData(tuple[it], &out, use_signature);
    if (!success) {
      return nullptr;
    }
    value_list.push_back(out);
  }
  return std::make_shared<ValueTuple>(value_list);
}

ValuePtr ConvertList(const py::object &obj, bool use_signature) {
  MS_LOG(DEBUG) << "Converting python list";

  auto list = obj.cast<py::list>();
  std::vector<ValuePtr> value_list;
  for (size_t it = 0; it < list.size(); ++it) {
    ValuePtr out = nullptr;
    bool success = ConvertData(list[it], &out, use_signature);
    if (!success) {
      return nullptr;
    }
    value_list.push_back(out);
  }
  return std::make_shared<ValueList>(value_list);
}

ValuePtr ConvertCellList(const py::object &obj, bool use_signature) {
  MS_LOG(DEBUG) << "Converting cell list";
  py::sequence list = obj;
  std::vector<ValuePtr> value_list;
  for (size_t it = 0; it < list.size(); ++it) {
    ValuePtr out = nullptr;
    bool success = ConvertData(list[it], &out, use_signature);
    if (!success) {
      return nullptr;
    }
    value_list.push_back(out);
  }
  return std::make_shared<ValueTuple>(value_list);
}

ValuePtr ConvertDict(const py::object &obj, bool use_signature) {
  MS_LOG(DEBUG) << "Converting python dict";

  auto dict_values = obj.cast<py::dict>();
  std::vector<std::pair<std::string, ValuePtr>> key_values;
  for (auto item : dict_values) {
    if (!py::isinstance<py::str>(item.first)) {
      MS_LOG(ERROR) << "The key of dict is only support str.";
      return nullptr;
    }
    std::string key = py::str(item.first);
    ValuePtr out = nullptr;
    bool success = ConvertData(dict_values[item.first], &out, use_signature);
    if (!success) {
      return nullptr;
    }
    key_values.emplace_back(key, out);
  }
  return std::make_shared<ValueDictionary>(key_values);
}

ValuePtr ConvertModuleNameSpace(const py::object &obj) {
  MS_LOG(DEBUG) << "Converting python module";
  py::module mod = python_adapter::GetPyModule(PYTHON_MOD_PARSE_MODULE);
  py::object module_namespace = python_adapter::CallPyModFn(mod, PYTHON_MOD_GET_MODULE_NAMESPACE, obj);
  auto converted =
    std::make_shared<NameSpace>(RESOLVE_NAMESPACE_NAME_MODULE, py::cast<py::module>(module_namespace), obj);
  MS_LOG(DEBUG) << "name_space: " << converted->ToString();
  return converted;
}

ValuePtr ConvertDataClass(const py::object &obj) {
  MS_LOG(DEBUG) << "Converting dataclass";
  // Maybe the obj is dataclass define
  auto desc = py::cast<std::string>(python_adapter::CallPyObjMethod(obj, PYTHON_GET_OBJ_DESC, obj));
  // desc has format "<class xxxx>", strip the '<' and '>' by offset 1
  auto converted = std::make_shared<ClassObject>(obj, std::string(desc.begin() + 1, desc.end() - 1));
  return converted;
}

ValuePtr ConvertMsClass(const py::object &obj) {
  MS_LOG(DEBUG) << "Converting ms class";
  // Convert class instance decorated with ms_class.
  py::module mod = python_adapter::GetPyModule(PYTHON_MOD_PARSE_MODULE);
  py::object name = python_adapter::CallPyModFn(mod, PYTHON_MOD_GET_MS_CLASS_NAME, obj);
  auto cls_name = py::cast<std::string>(name);
  return std::make_shared<MsClassObject>(obj, cls_name);
}

ValuePtr ConvertPrimitive(const py::object &obj, bool use_signature = false) {
  MS_LOG(DEBUG) << "Converting primitive object" << use_signature;

  // need check the primitive is class type or instance
  auto obj_type = data_converter::GetObjType(obj);
  if (obj_type == RESOLVE_TYPE_CLASS_TYPE) {
    auto desc = py::cast<std::string>(python_adapter::CallPyObjMethod(obj, PYTHON_GET_OBJ_DESC, obj));
    // desc has format "<class xxxx>", strip the '<' and '>' by offset 1.
    return std::make_shared<ClassType>(obj, std::string(desc.begin() + 1, desc.end() - 1));
  }
  py::object adapter_obj = obj;
  if (py::hasattr(obj, "__setattr_flag__")) {
    if (py::hasattr(obj, "_clone")) {
      auto clone_fn = obj.attr("_clone");
      adapter_obj = clone_fn();
    }
  }
  auto prim_adapter = adapter_obj.cast<PrimitivePyAdapterPtr>();
  MS_EXCEPTION_IF_NULL(prim_adapter);
  auto primitive = prim_adapter->attached_primitive();
  if (primitive == nullptr) {
    primitive = std::make_shared<PrimitivePy>(adapter_obj, prim_adapter);
    prim_adapter->set_attached_primitive(primitive);
  }

  if (use_signature) {
    return std::make_shared<prim::DoSignaturePrimitive>(primitive->name(), primitive);
  }
  return primitive;
}

ValuePtr ConvertMetaFuncGraph(const py::object &obj, bool use_signature = false) {
  MS_LOG(DEBUG) << "Converting MetaFuncGraph object";
  auto meta = obj.cast<MetaFuncGraphPtr>();
  if (meta == nullptr) {
    MS_LOG(ERROR) << "Resolve MetaFuncGraph error, get ptr is null";
    return nullptr;
  }
  if (use_signature) {
    return std::make_shared<prim::DoSignaturePrimitive>(meta->name(), meta);
  }
  return meta;
}

ValuePtr ConvertFuncGraph(const py::object &obj) {
  MS_LOG(DEBUG) << "Converting FuncGraph object";
  auto func_graph = obj.cast<FuncGraphPtr>();
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "Resolve FuncGraph error, get ptr is null";
    return nullptr;
  }
  func_graph->set_attr("is_load", MakeValue(true));
  return func_graph;
}

ValuePtr ConvertSlice(const py::object &obj) {
  MS_LOG(DEBUG) << "Converting slice object";

  auto convert_func = [obj](const std::string &attr) -> ValuePtr {
    auto py_attr = py::getattr(obj, attr.c_str());
    if (py::isinstance<py::none>(py_attr)) {
      return kNone;
    }
    if (py::isinstance<py::int_>(py_attr)) {
      auto value = py::cast<int64_t>(py_attr);
      return MakeValue(value);
    }
    if (py::isinstance<Tensor>(py_attr)) {
      return py::cast<TensorPtr>(py_attr);
    }
    MS_LOG(EXCEPTION) << "Attribute '" << attr << "' of " << py::str(obj)
                      << " should be int or Tensor with Int type but got " << py::str(py_attr);
  };
  ValuePtr start = convert_func(kSliceStart);
  ValuePtr stop = convert_func(kSliceStop);
  ValuePtr step = convert_func(kSliceStep);
  return std::make_shared<ValueSlice>(start, stop, step);
}

ValuePtr ConvertCellObjToFuncGraph(const py::object &obj) {
  FuncGraphPtr func_graph = ConvertToFuncGraph(obj);
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "Parse resolve function error.";
    return nullptr;
  }
  // if the cell object has specified bprop, it has user-defined bprop function parse and record it
  if (py::hasattr(obj, CUSTOM_BPROP_NAME)) {
    bool enable_bprop_debug = py::cast<bool>(py::getattr(obj, "bprop_debug"));
    FuncGraphPtr bprop_graph =
      enable_bprop_debug ? ConvertToBpropCut(obj) : ConvertToFuncGraph(obj, PYTHON_MOD_GET_BPROP_METHOD);
    if (bprop_graph != nullptr) {
      (void)func_graph->transforms().emplace(CUSTOM_BPROP_NAME, FuncGraphTransform(bprop_graph));
      (void)bprop_graph->transforms().emplace("primal", FuncGraphTransform(func_graph));
      func_graph->set_flag(FUNC_GRAPH_FLAG_DEFER_INLINE, true);
    }
  }
  if (py::hasattr(obj, STAGE_NAME)) {
    auto stage = py::cast<int>(py::getattr(obj, STAGE_NAME));
    func_graph->set_stage(stage);
  }
  return func_graph;
}

ValuePtr ConvertOtherObj(const py::object &obj) {
  auto obj_type = data_converter::GetObjType(obj);
  MS_LOG(DEBUG) << "Converting the object(" << ((std::string)py::str(obj)) << ") detail type: " << obj_type << " ";
  if (obj_type == RESOLVE_TYPE_CLASS_TYPE) {
    MS_LOG(DEBUG) << "Resolve the class type, need create class instance.";
    std::string desc = py::str(obj);
    // desc has format "<class xxxx>", strip the '<' and '>' by offset 1.
    return std::make_shared<ClassType>(obj, std::string(desc.begin() + 1, desc.end() - 1));
  }
  if (obj_type == RESOLVE_TYPE_FUNCTION || obj_type == RESOLVE_TYPE_METHOD) {
    MS_LOG(DEBUG) << "Convert the obj to func graph, type is " << obj_type;
    FuncGraphPtr func_graph = ConvertToFuncGraph(obj);
    if (func_graph == nullptr) {
      MS_LOG(ERROR) << "Parse resolve function error.";
      return nullptr;
    }
    return func_graph;
  }
  if (obj_type == RESOLVE_TYPE_CLASS_INSTANCE) {
    // Create the namespace for common class instance
    // When the obj is Cell, default parse the 'construct'
    py::module mod = python_adapter::GetPyModule(PYTHON_MOD_PARSE_MODULE);
    py::object namespace_var = python_adapter::CallPyModFn(mod, PYTHON_MOD_GET_MEMBER_NAMESPACE_SYMBOL, obj);
    auto res = std::make_shared<NameSpace>(RESOLVE_NAMESPACE_NAME_CLASS_MEMBER, namespace_var);
    MS_LOG(DEBUG) << "name_space: " << res->ToString();
    return res;
  }
  // Start RESOLVE_TYPE_INVALID...
  // The fallback feature is enabled in default.
  // Not support change the flag during the process is alive.
  static const auto support_fallback = common::GetEnv("MS_DEV_ENABLE_FALLBACK");
  static const auto use_fallback = (support_fallback != "0");
  if (use_fallback) {
    auto res = std::make_shared<InterpretedObject>(obj, py::str(obj));
    MS_LOG(DEBUG) << "Get interpreted object: " << res->ToString();
    return res;
  }
  MS_LOG(ERROR) << "Resolve type is invalid, obj: " << py::str(obj);
  return nullptr;
}

template <typename T>
ValuePtr ConvertNumberWithType(const T &obj, const TypePtr &dtype) {
  ValuePtr data = nullptr;
  auto int_dypte = dyn_cast<Int>(dtype);
  if (int_dypte != nullptr) {
    switch (int_dypte->nbits()) {
      case kBit8:
        data = std::make_shared<Int8Imm>(obj);
        break;
      case kBit16:
        data = std::make_shared<Int16Imm>(obj);
        break;
      case kBit32:
        data = std::make_shared<Int32Imm>(obj);
        break;
      case kBit64:
        data = std::make_shared<Int64Imm>(obj);
        break;
      default:
        data = std::make_shared<Int64Imm>(obj);
    }
    return data;
  }

  auto uint_dypte = dyn_cast<UInt>(dtype);
  if (uint_dypte != nullptr) {
    switch (uint_dypte->nbits()) {
      case kBit8:
        data = std::make_shared<UInt8Imm>(obj);
        break;
      case kBit16:
        data = std::make_shared<UInt16Imm>(obj);
        break;
      case kBit32:
        data = std::make_shared<UInt32Imm>(obj);
        break;
      case kBit64:
        data = std::make_shared<UInt64Imm>(obj);
        break;
      default:
        data = std::make_shared<UInt32Imm>(obj);
    }
    return data;
  }

  auto float_dypte = dyn_cast<Float>(dtype);
  if (float_dypte != nullptr) {
    switch (float_dypte->nbits()) {
      case kBit32:
        data = std::make_shared<FP32Imm>(obj);
        break;
      case kBit64:
        data = std::make_shared<FP64Imm>(obj);
        break;
      default:
        data = std::make_shared<FP32Imm>(obj);
    }
    return data;
  }
  return nullptr;
}

ValuePtr ConvertIntegerWithType(const py::object &obj, const TypePtr &dtype = nullptr) {
  auto obj_int64 = py::cast<int64_t>(obj);
  if (dtype == nullptr) {
    return std::make_shared<Int64Imm>(obj_int64);
  }
  return ConvertNumberWithType<int64_t>(obj_int64, dtype);
}

ValuePtr ConvertFloatWithType(const py::object &obj, const TypePtr &dtype = nullptr) {
  auto obj_float64 = py::cast<float>(obj);
  if (dtype == nullptr) {
    return std::make_shared<FP32Imm>(obj_float64);
  }
  return ConvertNumberWithType<float>(obj_float64, dtype);
}

template <typename T, typename U>
ValuePtr PyCast(const py::object &obj) {
  return std::make_shared<T>(py::cast<U>(obj));
}

template <typename T>
ValuePtr ObjCast(const py::object &obj) {
  return obj.cast<T>();
}

static const std::vector<DataConverterPtr> &GetDataConverters() {
  static const std::vector<DataConverterPtr> data_converters{
    // Convert data by python object type.
    std::make_shared<ByTypeDataConverter<Tensor>>(ObjCast<TensorPtr>),
    std::make_shared<ByTypeDataConverter<MetaTensor>>(ObjCast<MetaTensorPtr>),
    std::make_shared<ByTypeDataConverter<CSRTensor>>(ObjCast<CSRTensorPtr>),
    std::make_shared<ByTypeDataConverter<COOTensor>>(ObjCast<COOTensorPtr>),
    std::make_shared<ByTypeDataConverter<py::tuple>>(ConvertTuple),
    std::make_shared<ByTypeDataConverter<py::list>>(ConvertList),
    std::make_shared<ByTypeDataConverter<py::bool_>>(PyCast<BoolImm, bool>),
    std::make_shared<ByTypeDataConverter<py::int_>>(ConvertIntegerWithType),
    std::make_shared<ByTypeDataConverter<py::float_>>(ConvertFloatWithType),
    std::make_shared<ByTypeDataConverter<py::str>>(PyCast<StringImm, string>),
    std::make_shared<ByTypeDataConverter<py::none>>(kNone),
    std::make_shared<ByTypeDataConverter<py::ellipsis>>(kEllipsis),
    std::make_shared<ByTypeDataConverter<py::module>>(ConvertModuleNameSpace),
    std::make_shared<ByAttrDataConverter>(PYTHON_DATACLASS_FIELDS, ConvertDataClass),
    std::make_shared<ByAttrDataConverter>(PYTHON_MS_CLASS, ConvertMsClass),
    std::make_shared<ByTypeDataConverter<Type>>(ObjCast<TypePtr>),
    std::make_shared<ByTypeDataConverter<UMonad>>(ObjCast<UMonadPtr>),
    std::make_shared<ByTypeDataConverter<IOMonad>>(ObjCast<IOMonadPtr>),
    std::make_shared<ByAttrDataConverter>(PYTHON_CLASS_MEMBER_NAMESPACE,
                                          [](const py::object &obj) -> ValuePtr {
                                            auto res =
                                              std::make_shared<NameSpace>(RESOLVE_NAMESPACE_NAME_CLASS_MEMBER, obj);
                                            MS_LOG(DEBUG) << "name_space: " << res->ToString();
                                            return res;
                                          }),
    std::make_shared<ByTypeDataConverter<py::dict>>(ConvertDict),
    std::make_shared<ByTypeDataConverter<py::slice>>(ConvertSlice),
    std::make_shared<ByAttrDataConverter>(PYTHON_CELL_AS_LIST, ConvertCellList),
    std::make_shared<ByTypeDataConverter<Cell>>(ConvertCellObjToFuncGraph),
    std::make_shared<ByAttrDataConverter>(PYTHON_PRIMITIVE_FLAG, ConvertPrimitive),
    std::make_shared<ByTypeDataConverter<MetaFuncGraph>>(ConvertMetaFuncGraph),
    std::make_shared<ByTypeDataConverter<FuncGraph>>(ConvertFuncGraph),
  };
  return data_converters;
}
}  // namespace

bool ConvertData(const py::object &obj, ValuePtr *data, bool use_signature, const TypePtr &dtype) {
  // Check parameter valid
  if (data == nullptr) {
    MS_LOG(ERROR) << "The value pointer should not be null.";
    return false;
  }
  ValuePtr converted = nullptr;
  bool matched = false;
  const auto &converters = GetDataConverters();
  for (auto &converter : converters) {
    if (converter->Matched(obj)) {
      converted = converter->ConvertPyObject(obj, use_signature, dtype);
      matched = true;
      break;
    }
  }
  if (!matched) {
    converted = ConvertOtherObj(obj);
  }
  *data = converted;
  return converted != nullptr;
}

// Convert data to graph
FuncGraphPtr ConvertToFuncGraph(const py::object &obj, const std::string &python_mod_get_parse_method) {
  std::vector<std::string> results = data_converter::GetObjKey(obj);
  std::string obj_id = results[0] + python_mod_get_parse_method;
  std::string obj_key = results[1];
  FuncGraphPtr func_graph = nullptr;
  ValuePtr value = nullptr;
  bool is_cache = data_converter::GetObjectValue(obj_id, &value);
  if (is_cache && value != nullptr && value->isa<FuncGraph>()) {
    MS_LOG(DEBUG) << "Get the cache data, obj: " << obj_id;
    func_graph = value->cast<FuncGraphPtr>();
    if (!func_graph->dropped()) {
      if (pipeline::GetJitLevel() == "o0") {
        return BasicClone(func_graph);
      }
      return func_graph;
    }
  }

  func_graph = ParsePythonCode(obj, python_mod_get_parse_method);
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "Parse resolve function error.";
    return nullptr;
  }

  data_converter::MakeProperNameToFuncGraph(func_graph, obj_id);
  data_converter::CacheObjectValue(obj_id, func_graph);
  if (!obj_key.empty()) {
    MS_LOG(DEBUG) << "Add graph: " << obj_key << ", func_graph: " << func_graph->ToString();
    data_converter::SetObjGraphValue(obj_key, func_graph);
  }

  return func_graph;
}

namespace data_converter {
static luojianet_ms::HashMap<std::string, ValuePtr> object_map_;

static luojianet_ms::HashMap<std::string, std::vector<FuncGraphPtr>> object_graphs_map_;

void SetObjGraphValue(const std::string &obj_key, const FuncGraphPtr &data) {
  object_graphs_map_[obj_key].push_back(data);
  MS_LOG(DEBUG) << "Set func graph size: " << object_graphs_map_.size();
}

const luojianet_ms::HashMap<std::string, std::vector<FuncGraphPtr>> &GetObjGraphs() {
  MS_LOG(DEBUG) << "Obj graphs size: " << object_graphs_map_.size();
  return object_graphs_map_;
}

void CacheObjectValue(const std::string &obj_key, const ValuePtr &data) { object_map_[obj_key] = data; }

bool GetObjectValue(const std::string &obj_key, ValuePtr *data) {
  if (object_map_.count(obj_key)) {
    *data = object_map_[obj_key];
    return true;
  }
  return false;
}

std::vector<std::string> GetObjKey(const py::object &obj) {
  py::module mod = python_adapter::GetPyModule(PYTHON_MOD_PARSE_MODULE);
  py::tuple obj_tuple = python_adapter::CallPyModFn(mod, PYTHON_MOD_RESOLVE_GET_OBJ_KEY, obj);
  if (obj_tuple.size() != 2) {
    MS_LOG(EXCEPTION) << "The function of \'get_obj_key()\' must return 2 elements";
  }
  return {py::cast<std::string>(obj_tuple[0]), py::cast<std::string>(obj_tuple[1])};
}

// Get obj detail type
ResolveTypeDef GetObjType(const py::object &obj) {
  try {
    py::module mod = python_adapter::GetPyModule(PYTHON_MOD_PARSE_MODULE);
    auto obj_type =
      ResolveTypeDef(python_adapter::CallPyModFn(mod, PYTHON_MOD_RESOLVE_GET_OBJ_TYPE, obj).cast<int32_t>());
    return obj_type;
  } catch (const py::error_already_set &ex) {
    MS_LOG(ERROR) << "Meet a exception from Python when get the type of \'" << py::str(obj) << "\'.\n" << ex.what();
    std::rethrow_exception(std::current_exception());
  } catch (const py::type_error &ex) {
    MS_LOG(ERROR) << "Meet a exception when get the type of \'" << py::str(obj) << "\'.\n" << ex.what();
    std::rethrow_exception(std::current_exception());
  }
}

// Get class instance detail type.
ClassInstanceTypeDef GetClassInstanceType(const py::object &obj) {
  py::module mod = python_adapter::GetPyModule(PYTHON_MOD_PARSE_MODULE);
  auto class_type =
    ClassInstanceTypeDef(python_adapter::CallPyModFn(mod, PYTHON_MOD_GET_CLASS_INSTANCE_TYPE, obj).cast<int32_t>());
  return class_type;
}

// Check the object is Cell Instance.
bool IsCellInstance(const py::object &obj) {
  auto class_type = GetClassInstanceType(obj);
  bool is_cell = (class_type == CLASS_INSTANCE_TYPE_CELL);
  return is_cell;
}

// Create the python class instance.
py::object CreatePythonObject(const py::object &type, const py::tuple &args_kwargs) {
  py::module mod = python_adapter::GetPyModule(PYTHON_MOD_PARSE_MODULE);
  // `args_kwargs` maybe a tuple(*args), tuple(**kwargs), or tuple(*args, **kwargs).
  return args_kwargs.empty() ? python_adapter::CallPyModFn(mod, PYTHON_MOD_CREATE_INSTANCE, type)
                             : python_adapter::CallPyModFn(mod, PYTHON_MOD_CREATE_INSTANCE, type, args_kwargs);
}

// Call the python script string.
py::object CallPythonScript(const py::object &script, const py::tuple &args_kwargs) {
  py::module mod = python_adapter::GetPyModule(PYTHON_MOD_PARSE_MODULE);
  // `args_kwargs` is a tuple(dict(global), dict(local)).
  return args_kwargs.empty() ? python_adapter::CallPyModFn(mod, PYTHON_MOD_EVAL_PY_SCRIPT, script)
                             : python_adapter::CallPyModFn(mod, PYTHON_MOD_EVAL_PY_SCRIPT, script, args_kwargs);
}

// Generate an appropriate name and set to graph debuginfo,
// character <> can not used in the dot file, so change to another symbol.
void MakeProperNameToFuncGraph(const FuncGraphPtr &func_graph, std::string name) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(func_graph->debug_info());
  // Set detail name info of function
  std::ostringstream oss;
  for (size_t i = 0; i < name.size(); i++) {
    if (name[i] == '<') {
      oss << "「";
    } else if (name[i] == '>') {
      oss << "」";
    } else {
      oss << name[i];
    }
  }
  func_graph->debug_info()->set_full_name(oss.str());
}

ValuePtr PyDataToValue(const py::object &obj) {
  py::object to_convert = obj;
  ValuePtr value = nullptr;
  (void)ConvertData(to_convert, &value);
  return value;
}

void ClearObjectCache() {
  object_map_.clear();
  object_graphs_map_.clear();
}
}  // namespace data_converter

static luojianet_ms::HashMap<std::string, ClassPtr> g_dataClassToClass = {};

// Parse dataclass to luojianet_ms Class type
ClassPtr ParseDataClass(const py::object &cls_obj) {
  std::string cls_name = py::cast<std::string>(python_adapter::GetPyObjAttr(cls_obj, "__name__"));
  std::string cls_module = py::cast<std::string>(python_adapter::GetPyObjAttr(cls_obj, "__module__"));
  std::string cls = cls_module + "." + cls_name;
  auto iterator = g_dataClassToClass.find(cls);
  if (iterator != g_dataClassToClass.end()) {
    return iterator->second;
  }

  py::module mod = python_adapter::GetPyModule(PYTHON_MOD_PARSE_MODULE);
  ClassAttrVector attributes;
  py::dict names = python_adapter::CallPyModFn(mod, PYTHON_MOD_GET_DATACLASS_ATTRS, cls_obj);
  for (auto &item : names) {
    auto type_value = item.second.cast<TypePtr>();
    MS_EXCEPTION_IF_NULL(type_value);
    MS_LOG(DEBUG) << "(Name: " << py::cast<std::string>(item.first) << ", type: " << type_value->ToString() << ")";
    attributes.push_back(std::make_pair(py::cast<std::string>(item.first), type_value));
  }

  luojianet_ms::HashMap<std::string, ValuePtr> methods_map;
  py::dict methods = python_adapter::CallPyModFn(mod, PYTHON_MOD_GET_DATACLASS_METHODS, cls_obj);
  for (auto &item : methods) {
    auto fun_name = item.first.cast<std::string>();
    auto obj = py::cast<py::object>(item.second);
    std::shared_ptr<PyObjectWrapper> method_obj = std::make_shared<PyObjectWrapper>(obj, fun_name);
    methods_map[fun_name] = method_obj;
  }

  std::shared_ptr<Class> me_class = std::make_shared<Class>(Named(cls_name), attributes, methods_map);
  // static Variable for cache
  // cppcheck-suppress unreadVariable
  g_dataClassToClass[cls] = me_class;

  return me_class;
}

void CleanDataClassToClassMap() { g_dataClassToClass.clear(); }
}  // namespace parse
}  // namespace luojianet_ms
