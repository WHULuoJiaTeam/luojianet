# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: checkpoint.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='checkpoint.proto',
  package='',
  syntax='proto2',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x10\x63heckpoint.proto\"b\n\nCheckpoint\x12 \n\x05value\x18\x01 \x03(\x0b\x32\x11.Checkpoint.Value\x1a\x32\n\x05Value\x12\x0b\n\x03tag\x18\x01 \x02(\t\x12\x1c\n\x06tensor\x18\x02 \x02(\x0b\x32\x0c.TensorProto\"H\n\x0bTensorProto\x12\x0c\n\x04\x64ims\x18\x01 \x03(\x03\x12\x13\n\x0btensor_type\x18\x02 \x02(\t\x12\x16\n\x0etensor_content\x18\x03 \x02(\x0c'
)




_CHECKPOINT_VALUE = _descriptor.Descriptor(
  name='Value',
  full_name='Checkpoint.Value',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='tag', full_name='Checkpoint.Value.tag', index=0,
      number=1, type=9, cpp_type=9, label=2,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='tensor', full_name='Checkpoint.Value.tensor', index=1,
      number=2, type=11, cpp_type=10, label=2,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=68,
  serialized_end=118,
)

_CHECKPOINT = _descriptor.Descriptor(
  name='Checkpoint',
  full_name='Checkpoint',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='value', full_name='Checkpoint.value', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[_CHECKPOINT_VALUE, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=20,
  serialized_end=118,
)


_TENSORPROTO = _descriptor.Descriptor(
  name='TensorProto',
  full_name='TensorProto',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='dims', full_name='TensorProto.dims', index=0,
      number=1, type=3, cpp_type=2, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='tensor_type', full_name='TensorProto.tensor_type', index=1,
      number=2, type=9, cpp_type=9, label=2,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='tensor_content', full_name='TensorProto.tensor_content', index=2,
      number=3, type=12, cpp_type=9, label=2,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=120,
  serialized_end=192,
)

_CHECKPOINT_VALUE.fields_by_name['tensor'].message_type = _TENSORPROTO
_CHECKPOINT_VALUE.containing_type = _CHECKPOINT
_CHECKPOINT.fields_by_name['value'].message_type = _CHECKPOINT_VALUE
DESCRIPTOR.message_types_by_name['Checkpoint'] = _CHECKPOINT
DESCRIPTOR.message_types_by_name['TensorProto'] = _TENSORPROTO
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Checkpoint = _reflection.GeneratedProtocolMessageType('Checkpoint', (_message.Message,), {

  'Value' : _reflection.GeneratedProtocolMessageType('Value', (_message.Message,), {
    'DESCRIPTOR' : _CHECKPOINT_VALUE,
    '__module__' : 'checkpoint_pb2'
    # @@protoc_insertion_point(class_scope:Checkpoint.Value)
    })
  ,
  'DESCRIPTOR' : _CHECKPOINT,
  '__module__' : 'checkpoint_pb2'
  # @@protoc_insertion_point(class_scope:Checkpoint)
  })
_sym_db.RegisterMessage(Checkpoint)
_sym_db.RegisterMessage(Checkpoint.Value)

TensorProto = _reflection.GeneratedProtocolMessageType('TensorProto', (_message.Message,), {
  'DESCRIPTOR' : _TENSORPROTO,
  '__module__' : 'checkpoint_pb2'
  # @@protoc_insertion_point(class_scope:TensorProto)
  })
_sym_db.RegisterMessage(TensorProto)


# @@protoc_insertion_point(module_scope)
