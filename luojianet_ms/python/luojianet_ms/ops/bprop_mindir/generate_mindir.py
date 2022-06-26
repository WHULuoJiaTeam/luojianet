# Copyright 2021, 2022 LuoJiaNET Research and Development Group, Wuhan University
# Copyright 2021, 2022 Huawei Technologies Co., Ltd
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
"""Generate the mindir for bprop"""
import os
import shutil
import argparse

from luojianet_ms.ops import operations as P
from luojianet_ms.ops import _constants
import luojianet_ms.ops._grad as g
from luojianet_ms.ops.operations import _grad_ops as G
from luojianet_ms.ops.operations import _inner_ops as inner
from luojianet_ms._c_expression import _export_bprop_mindir

serializable_bprop_ops = [P.ReLU, P.Identity, inner.Range, P.OnesLike, P.ZerosLike, P.Argmax, P.Argmin, P.Broadcast,
                          P.AssignAdd, P.AssignSub, P.IsFinite, P.ApproximateEqual, P.Sign, P.LogicalNot, P.Round,
                          P.LinSpace, P.DropoutGenMask, P.OneHot, P.Assign, P.IOU, P.BNTrainingReduce, P.Equal,
                          P.NotEqual, P.Greater, P.GreaterEqual, P.Less, P.LessEqual, P.LogicalAnd, P.LogicalOr,
                          P.ReduceAll, P.ReduceAny, P.DropoutDoMask, P.DType, P.Shape, P.DynamicShape, P.Rank,
                          P.Select, P.ScatterMax, G.ReluGrad, _constants.kTupleGetItem, P.FloorDiv, P.TruncateDiv,
                          P.Minimum, P.Maximum, P.IsNan, P.IsInf, P.ReLUV2, "Depend", "stop_gradient", "Switch",
                          "UpdateState", "Load"]


def run_generate():
    for op in serializable_bprop_ops:
        if not isinstance(op, str):
            op = op.__name__
        _export_bprop_mindir(op)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bprop generator")
    parser.add_argument('--luojianet_ms_path', type=str, default=None,
                        help="The absolute path of the luojianet_ms root directory where the bprop source files has been \
                        modified. If not specified, it will find the bprop source files in your luojianet_ms installed \
                        path. Default: None.")

    args_opt = parser.parse_args()
    # luojianet_ms/ops/_grad/__init__.py
    bprop_path = g.__file__
    bprop_installed_dir = bprop_path[: bprop_path.rindex('/')]
    bprop_mindir_export_dir = bprop_installed_dir + "/../bprop_mindir"

    luojianet_ms_path = args_opt.luojianet_ms_path
    bprop_src_dir = None
    bprop_mindir_src_dir = None
    if not luojianet_ms_path is None:
        luojianet_ms_path = luojianet_ms_path.rstrip('/')
        bprop_src_dir = luojianet_ms_path + "/luojianet_ms/python/luojianet_ms/ops/_grad"
        bprop_mindir_src_dir = luojianet_ms_path + "/luojianet_ms/python/luojianet_ms/ops/bprop_mindir"

    copy_flag = not bprop_src_dir is None and bprop_src_dir != bprop_installed_dir
    # If the specified bprop source directory is not on the luojianet_ms installed path,
    # copy the bprop source files to the installed path.
    backup_suffix = "_generate_bak"
    if copy_flag:
        shutil.rmtree(bprop_installed_dir + backup_suffix, ignore_errors=True)
        os.rename(bprop_installed_dir, bprop_installed_dir + backup_suffix)
        os.mkdir(bprop_installed_dir)
        ls = os.listdir(bprop_src_dir)
        for line in ls:
            file_path = os.path.join(bprop_src_dir, line)
            if os.path.isfile(file_path):
                print("copy: " + file_path)
                shutil.copy(file_path, bprop_installed_dir)

    run_generate()

    # If the specified bprop source directory is not on the luojianet_ms installed path,
    # copy the generated mindir files to the mindir directory relative to the specified path.
    if copy_flag:
        shutil.rmtree(bprop_installed_dir)
        os.rename(bprop_installed_dir + backup_suffix, bprop_installed_dir)
        ls = os.listdir(bprop_mindir_export_dir)
        for line in ls:
            file_path = os.path.join(bprop_mindir_export_dir, line)
            if file_path.endswith(".mindir") and os.path.isfile(file_path):
                print("copy: " + file_path)
                shutil.copy(file_path, bprop_mindir_src_dir)

        print("The new bprop mindir files has been generated in the path \"" + bprop_mindir_src_dir +
              "\".")
    else:
        print("The new bprop mindir files has been generated in the path \"" + bprop_mindir_export_dir +
              "\", copy the *.mindir to your luojianet_ms path or PYTHONPATH if necessary.")
