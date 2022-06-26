# Copyright 2020-2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Thr parser for parsing framework files."""
import csv
import os
import re
import struct
import json
from pathlib import Path
from typing import List
from collections import defaultdict
from collections import namedtuple

from mindspore import log as logger
from mindspore.profiler.parser.framework_struct import TASK_DESC_STRUCT
from mindspore.profiler.parser.framework_struct import TENSOR_DATA_STRUCT
from mindspore.profiler.parser.framework_struct import STEP_INFO_STRUCT
from mindspore.profiler.parser.framework_enum import VmDataType, VmFormat, FileDataType
from mindspore.profiler.common.struct_type import StructType
from mindspore.profiler.common.util import combine_stream_task_id
from mindspore.profiler.common.exceptions.exceptions import ProfilerDirNotFoundException
from mindspore.profiler.common.exceptions.exceptions import ProfilerFileNotFoundException
from mindspore.profiler.common.exceptions.exceptions import ProfilerParamValueErrorException


FILE_DATA_STRUCT_DICT = {
    FileDataType.STEP_INFO.value: STEP_INFO_STRUCT,
    FileDataType.TENSOR_DATA_INFO.value: TENSOR_DATA_STRUCT,
    FileDataType.TASK_DESC_INFO.value: TASK_DESC_STRUCT
}


COL_NAMES = ['task_id', 'stream_id', 'block_dim', 'full_op_name', 'op_name', 'op_type', 'subgraph', 'op_info']
OpData = namedtuple('OpData', field_names=COL_NAMES)


class FrameworkParser:
    """
    Thr parser for parsing framework files.

    Args:
        profiling_path (str): The profiling path which should contain CANN profiling data.
        rank_id (str): The rank ID.
        output_path (str): The directory of the parsed file. Default: `./`.
    """
    _regex_framework = r'Framework\.(?P<data_type>.+)\.(?P<device_id>\d).+'
    _graph_attr_name = [
        'input_format', 'input_data_type', 'input_shape', 'output_format',
        'output_data_type', 'output_shape'
    ]
    output_file_format = 'framework_raw_{rank_id}.csv'

    def __init__(self, profiling_path, rank_id, output_path='./'):
        self._profiling_path = profiling_path
        self._output_path = output_path
        self._rank_id = rank_id

        self._hash_dict = {}
        self._task_id_full_op_name_dict = {}
        self._point_info = {}

    @staticmethod
    def _check_output_path(path):
        if not os.path.exists(path) or not os.path.isdir(path):
            raise ProfilerDirNotFoundException(path)

    @property
    def save_path(self):
        """
        The property of save path.

        Returns:
            str, the save path.
        """
        return os.path.realpath(os.path.join(self._output_path, self.output_file_format.format(rank_id=self._rank_id)))

    @property
    def point_info(self):
        """
        The property of the framework point information.

        Returns:
            dict, the framework point information, key is tag, value is op name.
        """
        # Note: In the multi-subgraph or multi-tag scenario, op name is overwritten.
        return self._point_info

    def check_op_name(self, op_name, is_prefix=True):
        """
        Check whether the operator name exists.

        Args:
            op_name (str): The operator name or operator name prefix.
            is_prefix (bool): `True` if the op_name is prefix, else `False`.
                Default: True.

        Returns:
            bool, `True` if the operator name does exist in framework file, else
            `False`.
        """
        if not op_name:
            raise ProfilerParamValueErrorException('The op_name should exist.')
        for full_op_name in self._task_id_full_op_name_dict.values():
            if full_op_name:
                if is_prefix and full_op_name.startswith(op_name):
                    return True
                if not is_prefix and op_name == full_op_name:
                    return True
        return False

    def to_task_id_full_op_name_dict(self):
        """
        Get the task id and full operator name dict.

        Returns:
            dict, the task id and full operator name dict.
        """
        return self._task_id_full_op_name_dict

    def parse(self):
        """Parse the framework files."""
        framework_path_dict = self._search_file(self._profiling_path)
        self._hash_dict = self._parse_hash_dic(framework_path_dict)

        all_file_data = self._parse_binary_data(framework_path_dict)
        task_id_full_op_name_dict = self._construct_task_id_full_op_name_dict(
            all_file_data[FileDataType.TASK_DESC_INFO.value])
        point_info = self._construct_point_info(task_id_full_op_name_dict, all_file_data[FileDataType.STEP_INFO.value])
        task_id_op_attr_dict = self._construct_task_id_op_attr_dict(all_file_data[FileDataType.TENSOR_DATA_INFO.value])

        self._point_info = point_info
        self._task_id_full_op_name_dict = task_id_full_op_name_dict

        all_op_data = self._construct_op_data_to_file(all_file_data[FileDataType.TASK_DESC_INFO.value],
                                                      task_id_op_attr_dict)

        self._write_framework_to_file(all_op_data, output_file=self.save_path)

    def _search_file(self, profiling_path):
        """
        Search all framework files in raw profiling path.

        Args:
            profiling_path (str): This profiling path should contain data dir.

        Return:
            dict, return a dict container all framework file paths. Format is {FileDataType: [file paths]}.

        Raises:
            ProfilerFileNotFoundException: If the framework files are not found.
        """
        data_dir = os.path.join(profiling_path, 'data')
        if not os.path.isdir(data_dir):
            raise ProfilerDirNotFoundException(data_dir)

        framework_path_dict = defaultdict(list)
        for file in Path(data_dir).glob(r'Framework*[0-9]'):
            file_name = file.name
            match = re.search(self._regex_framework, file_name)
            if match is None:
                logger.warning("Profiler does not support to analyse file(%s), this file name format is not %s, "
                               "skip this file.", file.resolve(), self._regex_framework)
                continue

            if match['data_type'] not in FileDataType.members():
                logger.warning("Profiler does not support to analyse file(%s), this file data type is %s, "
                               "skip this file.", file.resolve(), match['data_type'])
                if match['data_type'].startswith('vm'):
                    raise RuntimeError("The current profiler file is generated by MindSpore 1.5 or earlier. Use "
                                       "MindSpore 1.5 or the matching MindSpore version to parse the profiler file.")
                continue

            framework_path_dict[match['data_type']].append(file.resolve())

        empty_files = [data_type for data_type, files in framework_path_dict.items() if not files]
        if not framework_path_dict or empty_files:
            if empty_files:
                logger.error("Can not find %s files when parse profiler framework file.", ','.join(empty_files))
            raise ProfilerFileNotFoundException('Framework')

        for data_type in FileDataType.members():
            if data_type not in framework_path_dict:
                logger.warning("Can not find %s file when parse profiler framework file.", data_type)
                continue
            framework_path_dict[data_type].sort()

        return framework_path_dict

    @staticmethod
    def _parse_hash_dic(framework_path_dict):
        """Parse the hash dic files, and return a hash value map op name dict."""
        hash_op_dict = {}
        for path in framework_path_dict[FileDataType.HASH_DIC.value]:
            with open(path, 'r') as file:
                for hash_str in file:
                    hash_value, op_name = hash_str.strip().split(':')
                    hash_op_dict[hash_value] = op_name
        return hash_op_dict

    def _parse_binary_data(self, framework_path_dict):
        """Parse binary data in the FILE_DATA_STRUCT_DICT from given files, such as task data, step point data"""
        all_file_data = defaultdict(list)
        for file_data_type, data_struct in FILE_DATA_STRUCT_DICT.items():
            line_size = StructType.sizeof(data_struct.values())
            for path in framework_path_dict[file_data_type]:
                with open(path, 'rb') as file_handler:
                    while True:
                        binary_data = file_handler.read(line_size)
                        if len(binary_data) < line_size:
                            break
                        line_data = StructType.unpack_binary_data(data_struct, binary_data,
                                                                  self._special_process_binary_data)
                        all_file_data[file_data_type].append(line_data)
        return all_file_data

    def _special_process_binary_data(self, item_binary_data, data_name, data_type, unpacked_data):
        """Specially processes binary data."""
        unpack_data = None
        success = False
        if isinstance(data_type, list):
            if data_name in ('opName', 'opType'):
                unpack_data = self._special_process_mixed_data(item_binary_data)
            elif data_name == 'tensorData':
                tensor_num = unpacked_data['tensorNum']
                unpack_data = self._special_process_tensor_data(item_binary_data, data_type, tensor_num)
            elif data_name == 'tensorNum':
                unpack_data = self._special_process_tensor_num(item_binary_data, data_type)
            else:
                # skip reserve data
                unpack_data = None
            success = True
        return unpack_data, success

    def _special_process_mixed_data(self, item_binary_data):
        """Specially processes mixed data, for example, opName and opType"""
        # The first byte is type flag, 0 means data is string, 1 means data is hash value
        cursor = 0
        data_size = len(item_binary_data)
        flag = struct.unpack(StructType.UINT8.value, item_binary_data[cursor:cursor + 1])[0]

        # skip rsv data, rsv has 7 bytes
        skip_size = 8
        remain_size = data_size - skip_size
        if flag == 0:
            unpack_data = struct.unpack(StructType.CHAR.value * remain_size,
                                        item_binary_data[cursor + skip_size:cursor + data_size])
            unpack_data = ''.join(list(map(lambda c: c.decode(), filter(lambda c: c != b'\x00', unpack_data))))
        else:
            size = StructType.sizeof(StructType.UINT64) + skip_size
            hash_value = struct.unpack(StructType.UINT64.value,
                                       item_binary_data[cursor + skip_size:cursor + size])[0]
            unpack_data = self._hash_dict[str(hash_value)]
        return unpack_data

    @staticmethod
    def _special_process_tensor_data(item_binary_data, data_type, tensor_num):
        """The tensor data depends tensor num, so need to special process."""
        start = 0
        op_attr_struct = data_type[0]
        op_attr_size = StructType.sizeof(op_attr_struct)
        unpack_data = []

        for _ in range(tensor_num):
            buffer = item_binary_data[start:start + op_attr_size]
            values = struct.unpack(StructType.format(op_attr_struct), buffer)
            one_data = dict(
                tensorType=values[0],
                format=values[1],
                dataType=values[2],
                shape=list(filter(lambda x: x != 0, values[3:]))
            )
            unpack_data.append(one_data)
            start += op_attr_size

        return unpack_data

    @staticmethod
    def _special_process_tensor_num(item_binary_data, data_type):
        """The memory of tensorNum is aligned, so here need to special process"""
        cursor = 0
        tensor_num_struct = data_type[0]
        size = StructType.sizeof(tensor_num_struct)
        unpack_data = struct.unpack(tensor_num_struct.value, item_binary_data[cursor:cursor + size])[0]
        return unpack_data

    @staticmethod
    def _construct_task_id_full_op_name_dict(task_desc_info):
        """The task desc info is a list[task_desc], task_desc is a dict, key is same as TASK_DESC_STRUCT."""
        task_id_full_op_name = {}
        for task_desc in task_desc_info:
            task_id = combine_stream_task_id(task_desc['streamId'], task_desc['taskId'])
            task_id_full_op_name[task_id] = task_desc['opName']
        return task_id_full_op_name

    @staticmethod
    def _construct_point_info(task_id_full_op_name_dict, step_point_data):
        """step_point_data is a list[step_data], step data is a dict, key is same as STEP_INFO_STRUCT."""
        point_info = {}
        for step_point in step_point_data:
            task_id = combine_stream_task_id(step_point['streamId'], step_point['taskId'])
            tag = step_point['tag']
            full_op_name = task_id_full_op_name_dict[task_id]
            point_info[tag] = full_op_name
        return point_info

    @staticmethod
    def _construct_task_id_op_attr_dict(prof_tensor_data):
        """prof_tensor_data is a list[tensor_data], tensor_data is a dict, key is same as TENSOR_DATA_STRUCT."""
        task_id_op_attr_dict = defaultdict(list)
        for tensor_data in prof_tensor_data:
            task_id = combine_stream_task_id(tensor_data['streamId'], tensor_data['taskId'])
            for tensor_attr in tensor_data['tensorData']:
                tensor_type = 'input' if tensor_attr['tensorType'] == 0 else 'output'
                tensor_format = VmFormat.get_format_name(tensor_attr['format'])
                op_attr = dict(
                    tensor_type=tensor_type,
                    format=tensor_format,
                    data_type=VmDataType.get_data_type_name(tensor_attr['dataType']),
                    shape=tensor_attr['shape']
                )
                task_id_op_attr_dict[task_id].append(op_attr)

        for task_id, op_attrs in task_id_op_attr_dict.items():
            input_count = 0
            output_count = 0
            new_op_attr = {}
            for op_attr in op_attrs:
                if op_attr['tensor_type'] == 'input':
                    op_attr.pop('tensor_type')
                    new_op_attr[f'input_{input_count}'] = op_attr
                    input_count += 1
                else:
                    op_attr.pop('tensor_type')
                    new_op_attr[f'output_{output_count}'] = op_attr
                    output_count += 1
            task_id_op_attr_dict[task_id] = new_op_attr

        return task_id_op_attr_dict

    def _construct_op_data_to_file(self, task_desc_info, task_id_op_attr_dict):
        """Build data written to a file."""
        all_op_data = []
        for task_desc in task_desc_info:
            task_id = task_desc['taskId']
            full_op_name = task_desc['opName']
            subgraph = self._get_subgraph_name(full_op_name)
            combined_task_id = combine_stream_task_id(task_desc['streamId'], task_id)
            op_data = OpData(task_id=task_id,
                             stream_id=task_desc['streamId'],
                             block_dim=task_desc['blockDims'],
                             full_op_name=full_op_name,
                             op_name=full_op_name.split('/')[-1],
                             op_type=task_desc['opType'],
                             subgraph=subgraph,
                             op_info=json.dumps(task_id_op_attr_dict.get(combined_task_id, {})))
            all_op_data.append(op_data)
        return all_op_data

    @staticmethod
    def _write_framework_to_file(all_op_data: List[OpData], output_file):
        with open(output_file, 'w') as file_handler:
            csv_writer = csv.writer(file_handler)
            csv_writer.writerow(COL_NAMES)
            csv_writer.writerows(all_op_data)

    @staticmethod
    def _get_subgraph_name(full_op_name):
        """
        Get subgraph name.

        Args:
            full_op_name (str): The full operator name.

        Returns:
            str, the subgraph name.
        """
        subgraph_name = full_op_name.split('/', 1)[0]
        if subgraph_name in ['Default', 'Gradients']:
            return subgraph_name
        return None
