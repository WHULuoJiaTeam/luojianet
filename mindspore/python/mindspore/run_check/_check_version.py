# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
"""version and config check"""
import os
import sys
import time
import subprocess
import glob
from pathlib import Path
from abc import abstractmethod, ABCMeta
from packaging import version
import numpy as np
from mindspore import log as logger
from ..version import __version__
from ..default_config import __package_name__

class EnvChecker(metaclass=ABCMeta):
    """basic class for environment check"""

    @abstractmethod
    def check_env(self, e):
        pass

    @abstractmethod
    def set_env(self):
        pass

    @abstractmethod
    def check_version(self):
        pass


class GPUEnvChecker(EnvChecker):
    """GPU environment check."""

    def __init__(self):
        self.version = ["10.1", "11.1"]
        self.lib_key_to_lib_name = {'libcu': 'libcuda.so'}
        # env
        self.path = os.getenv("PATH")
        self.ld_lib_path = os.getenv("LD_LIBRARY_PATH")

        # check
        self.v = "0"
        self.cuda_lib_path = self._get_lib_path("libcu")
        self.cuda_bin_path = self._get_bin_path("cuda")
        self.cudnn_lib_path = self._get_lib_path("libcudnn")

    def check_env(self, e):
        raise e

    def set_env(self):
        return

    def _get_bin_path(self, bin_name):
        """Get bin path by bin name."""
        if bin_name == "cuda":
            return self._get_cuda_bin_path()
        return []

    def _get_cuda_bin_path(self):
        """Get cuda bin path by lib path."""
        path_list = []
        for path in self.cuda_lib_path:
            path = os.path.abspath(path.strip()+"/bin/")
            if Path(path).is_dir():
                path_list.append(path)
        return np.unique(path_list)

    def _get_nvcc_version(self, is_set_env):
        """Get cuda version by nvcc command."""
        nvcc_result = subprocess.run(["nvcc", "--version | grep release"],
                                     timeout=3, text=True, capture_output=True, check=False)
        if nvcc_result.returncode:
            if not is_set_env:
                for path in self.cuda_bin_path:
                    if Path(path + "/nvcc").is_file():
                        os.environ['PATH'] = path + ":" + os.environ['PATH']
                        return self._get_nvcc_version(True)
            return ""
        result = nvcc_result.stdout
        for line in result.split('\n'):
            if line:
                return line.strip().split("release")[1].split(",")[0].strip()
        return ""

    def _get_cudnn_version(self):
        """Get cudnn version by libcudnn.so."""
        cudnn_version = []
        for path in self.cudnn_lib_path:
            real_path = glob.glob(path + "/lib*/libcudnn.so.*.*")
            if real_path == []:
                continue
            ls_cudnn = subprocess.run(["ls", real_path[0]], timeout=10, text=True,
                                      capture_output=True, check=False)
            if ls_cudnn.returncode == 0:
                cudnn_version = ls_cudnn.stdout.split('/')[-1].strip('libcudnn.so.').strip().split('.')
                if len(cudnn_version) == 2:
                    cudnn_version.append('0')
                break
        version_str = ''.join([n for n in cudnn_version])
        return version_str[0:3]

    def _get_cudart_version(self):
        """Get cuda runtime version by libcudart.so."""
        for path in self.cuda_lib_path:
            real_path = glob.glob(path + "/lib*/libcudart.so.*.*.*")
            if real_path == []:
                continue
            ls_cudart = subprocess.run(["ls", real_path[0]], timeout=10, text=True,
                                       capture_output=True, check=False)
            if ls_cudart.returncode == 0:
                self.v = ls_cudart.stdout.split('/')[-1].strip('libcudart.so.').strip()
                break
        return self.v

    def check_version(self):
        """Check cuda version."""
        version_match = False
        if self._check_version():
            version_match = True
        if not version_match:
            if self.v == "0":
                logger.warning("Can not found cuda libs, please confirm that the correct "
                               "cuda version has been installed, you can refer to the "
                               "installation guidelines: https://www.mindspore.cn/install")
            else:
                logger.warning(f"MindSpore version {__version__} and cuda version {self.v} does not match, "
                               "please refer to the installation guide for version matching "
                               "information: https://www.mindspore.cn/install")
        nvcc_version = self._get_nvcc_version(False)
        if nvcc_version and (nvcc_version not in self.version):
            logger.warning(f"MindSpore version {__version__} and nvcc(cuda bin) version {nvcc_version} "
                           "does not match, please refer to the installation guide for version matching "
                           "information: https://www.mindspore.cn/install")
        cudnn_version = self._get_cudnn_version()
        if cudnn_version and int(cudnn_version) < 760:
            logger.warning(f"MindSpore version {__version__} and cudDNN version {cudnn_version} "
                           "does not match, please refer to the installation guide for version matching "
                           "information: https://www.mindspore.cn/install. The recommended version is "
                           "CUDA10.1 with cuDNN7.6.x and CUDA11.1 with cuDNN8.0.x")
        if cudnn_version and int(cudnn_version) < 800 and int(str(self.v).split('.')[0]) > 10:
            logger.warning(f"CUDA version {self.v} and cuDNN version {cudnn_version} "
                           "does not match, please refer to the installation guide for version matching "
                           "information: https://www.mindspore.cn/install. The recommended version is "
                           "CUDA11.1 with cuDNN8.0.x")

    def _check_version(self):
        """Check cuda version"""
        v = self._get_cudart_version()
        v = version.parse(v)
        v_str = str(v.major) + "." + str(v.minor)
        if v_str not in self.version:
            return False
        return True

    def _get_lib_path(self, lib_name):
        """Get gpu lib path by ldd command."""
        path_list = []
        current_path = os.path.split(os.path.realpath(__file__))[0]
        mindspore_path = os.path.join(current_path, "../")
        try:
            real_path = glob.glob(mindspore_path + "/_c_expression*.so*")
            if real_path == []:
                logger.error(f"{self.lib_key_to_lib_name[lib_name]} (need by mindspore-gpu) is not found, please "
                             f"confirm that _c_expression.so is in directory:{mindspore_path} and the correct cuda "
                             "version has been installed, you can refer to the installation "
                             "guidelines: https://www.mindspore.cn/install")
                return path_list
            ldd_r = subprocess.Popen(['ldd', real_path[0]], stdout=subprocess.PIPE)
            ldd_result = subprocess.Popen(['grep', lib_name], stdin=ldd_r.stdout, stdout=subprocess.PIPE)
            result = ldd_result.communicate()[0].decode()
            for i in result.split('\n'):
                path = i.partition("=>")[2]
                if path.lower().find("not found") > 0:
                    logger.warning(f"Cuda {self.version} version(need by mindspore-gpu) is not found, please confirm "
                                   "that the path of cuda is set to the env LD_LIBRARY_PATH, please refer to the "
                                   "installation guidelines: https://www.mindspore.cn/install")
                    continue
                path = path.partition(lib_name)[0]
                if path:
                    path_list.append(os.path.abspath(path.strip() + "../"))
            return np.unique(path_list)
        except subprocess.TimeoutExpired:
            logger.warning("Failed to check cuda version due to the ldd command timeout, please confirm that "
                           "the correct cuda version has been installed, you can refer to the "
                           "installation guidelines: https://www.mindspore.cn/install")
            return path_list

    def _read_version(self, file_path):
        """Get gpu version info in version.txt."""
        with open(file_path, 'r') as f:
            all_info = f.readlines()
            for line in all_info:
                if line.startswith("CUDA Version"):
                    self.v = line.strip().split("CUDA Version")[1]
                    return self.v
        return self.v


class AscendEnvChecker(EnvChecker):
    """ascend environment check"""

    def __init__(self):
        self.version = ["1.81"]
        atlas_nnae_version = "/usr/local/Ascend/nnae/latest/fwkacllib/version.info"
        atlas_toolkit_version = "/usr/local/Ascend/ascend-toolkit/latest/fwkacllib/version.info"
        hisi_fwk_version = "/usr/local/Ascend/latest/fwkacllib/version.info"
        if os.path.exists(atlas_nnae_version):
            # atlas default path
            self.fwk_path = "/usr/local/Ascend/nnae/latest/fwkacllib"
            self.op_impl_path = "/usr/local/Ascend/nnae/latest/opp/op_impl/built-in/ai_core/tbe"
            self.tbe_path = self.fwk_path + "/lib64"
            self.cce_path = self.fwk_path + "/ccec_compiler/bin"
            self.fwk_version = atlas_nnae_version
            self.op_path = "/usr/local/Ascend/nnae/latest/opp"
            self.aicpu_path = "/usr/local/Ascend/nnae/latest"
        elif os.path.exists(atlas_toolkit_version):
            # atlas default path
            self.fwk_path = "/usr/local/Ascend/ascend-toolkit/latest/fwkacllib"
            self.op_impl_path = "/usr/local/Ascend/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe"
            self.tbe_path = self.fwk_path + "/lib64"
            self.cce_path = self.fwk_path + "/ccec_compiler/bin"
            self.fwk_version = atlas_toolkit_version
            self.op_path = "/usr/local/Ascend/ascend-toolkit/latest/opp"
            self.aicpu_path = "/usr/local/Ascend/ascend-toolkit/latest"
        elif os.path.exists(hisi_fwk_version):
            # hisi default path
            self.fwk_path = "/usr/local/Ascend/latest/fwkacllib"
            self.op_impl_path = "/usr/local/Ascend/latest/opp/op_impl/built-in/ai_core/tbe"
            self.tbe_path = self.fwk_path + "/lib64"
            self.cce_path = self.fwk_path + "/ccec_compiler/bin"
            self.fwk_version = hisi_fwk_version
            self.op_path = "/usr/local/Ascend/latest/opp"
            self.aicpu_path = "/usr/local/Ascend/latest"
        else:
            # custom or unknown environment
            self.fwk_path = ""
            self.op_impl_path = ""
            self.tbe_path = ""
            self.cce_path = ""
            self.fwk_version = ""
            self.op_path = ""
            self.aicpu_path = ""

        # env
        self.path = os.getenv("PATH")
        self.python_path = os.getenv("PYTHONPATH")
        self.ld_lib_path = os.getenv("LD_LIBRARY_PATH")
        self.ascend_opp_path = os.getenv("ASCEND_OPP_PATH")
        self.ascend_aicpu_path = os.getenv("ASCEND_AICPU_PATH")

        # check content
        self.path_check = "/fwkacllib/ccec_compiler/bin"
        self.python_path_check = "opp/op_impl/built-in/ai_core/tbe"
        self.ld_lib_path_check_fwk = "/fwkacllib/lib64"
        self.ld_lib_path_check_addons = "/add-ons"
        self.ascend_opp_path_check = "/op"
        self.v = ""

    def check_env(self, e):
        self._check_env()
        raise e

    def check_version(self):
        if not Path(self.fwk_version).is_file():
            logger.warning("Using custom Ascend AI software package (Ascend Data Center Solution) path, package "
                           "version checking is skipped, please make sure Ascend AI software package (Ascend Data "
                           "Center Solution) version is supported, you can reference to the installation guidelines "
                           "https://www.mindspore.cn/install")
            return

        v = self._read_version(self.fwk_version)
        if v not in self.version:
            v_list = str([x for x in self.version])
            logger.warning(f"MindSpore version {__version__} and Ascend AI software package (Ascend Data Center "
                           f"Solution)version {v} does not match, the version of software package expect one of "
                           f"{v_list}, please reference to the match info on: https://www.mindspore.cn/install")

    def check_deps_version(self):
        """
            te, topi, hccl wheel package version check
            in order to update the change of 'LD_LIBRARY_PATH' env, run a sub process
        """
        input_args = ["--mindspore_version=" + __version__]
        for v in self.version:
            input_args.append("--supported_version=" + v)
        deps_version_checker = os.path.join(os.path.split(os.path.realpath(__file__))[0],
                                            "_check_deps_version.py")
        call_cmd = [sys.executable, deps_version_checker] + input_args
        try:
            process = subprocess.run(call_cmd, timeout=3, text=True, capture_output=True, check=False)
            if process.stdout.strip() != "":
                logger.warning(process.stdout.strip())
                warning_countdown = 3
                for i in range(warning_countdown, 0, -1):
                    logger.warning(f"Please pay attention to the above warning, countdown: {i}")
                    time.sleep(1)
        except subprocess.TimeoutExpired:
            logger.info("Package te, topi, hccl version check timed out, skip.")

    def set_env(self):
        if not self.tbe_path:
            self._check_env()
            return

        try:
            import te  # pylint: disable=unused-import
        # pylint: disable=broad-except
        except Exception:
            if Path(self.tbe_path).is_dir():
                if os.getenv('LD_LIBRARY_PATH'):
                    os.environ['LD_LIBRARY_PATH'] = self.tbe_path + ":" + os.environ['LD_LIBRARY_PATH']
                else:
                    os.environ['LD_LIBRARY_PATH'] = self.tbe_path
            else:
                raise EnvironmentError(
                    f"No such directory: {self.tbe_path}, Please check if Ascend AI software package (Ascend Data "
                    "Center Solution) is installed correctly.")

        # check te version after set te env
        self.check_deps_version()

        if Path(self.op_impl_path).is_dir():
            # python path for sub process
            if os.getenv('PYTHONPATH'):
                os.environ['PYTHONPATH'] = self.op_impl_path + ":" + os.environ['PYTHONPATH']
            else:
                os.environ['PYTHONPATH'] = self.op_impl_path
            # sys path for this process
            sys.path.append(self.op_impl_path)

            os.environ['TBE_IMPL_PATH'] = self.op_impl_path
        else:
            raise EnvironmentError(
                f"No such directory: {self.op_impl_path}, Please check if Ascend AI software package (Ascend Data "
                "Center Solution) is installed correctly.")

        if Path(self.cce_path).is_dir():
            os.environ['PATH'] = self.cce_path + ":" + os.environ['PATH']
        else:
            raise EnvironmentError(
                f"No such directory: {self.cce_path}, Please check if Ascend AI software package (Ascend Data Center "
                "Solution) is installed correctly.")

        if self.op_path is None:
            pass
        elif Path(self.op_path).is_dir():
            os.environ['ASCEND_OPP_PATH'] = self.op_path
        else:
            raise EnvironmentError(
                f"No such directory: {self.op_path}, Please check if Ascend AI software package (Ascend Data Center "
                "Solution) is installed correctly.")

        if self.aicpu_path is None:
            pass
        elif Path(self.aicpu_path).is_dir():
            os.environ['ASCEND_AICPU_PATH'] = self.aicpu_path
        else:
            raise EnvironmentError(
                f"No such directory: {self.aicpu_path}, Please check if Ascend AI software package (Ascend Data Center"
                " Solution) is installed correctly.")

    def _check_env(self):
        """ascend dependence path check"""
        if self.path is None or self.path_check not in self.path:
            logger.warning("Can not find ccec_compiler(need by mindspore-ascend), please check if you have set env "
                           "PATH, you can reference to the installation guidelines https://www.mindspore.cn/install")

        if self.python_path is None or self.python_path_check not in self.python_path:
            logger.warning(
                "Can not find tbe op implement(need by mindspore-ascend), please check if you have set env "
                "PYTHONPATH, you can reference to the installation guidelines "
                "https://www.mindspore.cn/install")

        if self.ld_lib_path is None or not (self.ld_lib_path_check_fwk in self.ld_lib_path and
                                            self.ld_lib_path_check_addons in self.ld_lib_path):
            logger.warning("Can not find driver so(need by mindspore-ascend), please check if you have set env "
                           "LD_LIBRARY_PATH, you can reference to the installation guidelines "
                           "https://www.mindspore.cn/install")

        if self.ascend_opp_path is None or self.ascend_opp_path_check not in self.ascend_opp_path:
            logger.warning(
                "Can not find opp path (need by mindspore-ascend), please check if you have set env ASCEND_OPP_PATH, "
                "you can reference to the installation guidelines https://www.mindspore.cn/install")

    def _read_version(self, file_path):
        """get ascend version info"""
        with open(file_path, 'r') as f:
            all_info = f.readlines()
            for line in all_info:
                if line.startswith("Version="):
                    full_version = line.strip().split("=")[1]
                    self.v = '.'.join(full_version.split('.')[0:2])
                    return self.v
        return self.v


def check_version_and_env_config():
    """check version and env config"""
    if __package_name__.lower() == "mindspore-ascend":
        env_checker = AscendEnvChecker()
        # Note: pre-load libgomp.so to solve error like "cannot allocate memory in statis TLS block"
        try:
            import ctypes
            ctypes.cdll.LoadLibrary("libgomp.so.1")
        except OSError:
            logger.warning(
                "Pre-Load Lirary libgomp.so.1 failed, this might cause cannot allocate TLS memory problem, "
                "if so find solution in FAQ in https://www.mindspore.cn/docs/faq/en/r1.7/index.html.")
    elif __package_name__.lower() == "mindspore-gpu":
        env_checker = GPUEnvChecker()
    else:
        logger.info(f"Package version {__package_name__} does not need to check any environment variable, skipping.")
        return
    if os.getenv("MS_DEV_CLOSE_VERSION_CHECK") == "ON":
        return
    os.environ["MS_DEV_CLOSE_VERSION_CHECK"] = "ON"

    try:
        # check version of ascend site or cuda
        env_checker.check_version()
        from .. import _c_expression  # pylint: disable=unused-import
        env_checker.set_env()
    except ImportError as e:
        env_checker.check_env(e)


def _set_pb_env():
    """Set env variable `PROTOCOL_BUFFERS` to prevent memory overflow."""
    if os.getenv("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION") == "cpp":
        logger.info("Current env variable `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=cpp`. "
                    "When the checkpoint file is too large, "
                    "it may cause memory limit error during load checkpoint file. "
                    "This can be solved by set env `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`.")
    elif os.getenv("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION") is None:
        logger.info("Setting the env `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` to prevent memory overflow "
                    "during save or load checkpoint file.")
        os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"


check_version_and_env_config()
_set_pb_env()
