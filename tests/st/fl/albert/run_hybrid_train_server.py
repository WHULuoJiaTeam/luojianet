# Copyright 2021 Huawei Technologies Co., Ltd
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

import argparse
import subprocess
import os
import ast

parser = argparse.ArgumentParser(description="Run train_cloud.py case")
parser.add_argument("--device_target", type=str, default="CPU")
parser.add_argument("--server_mode", type=str, default="FEDERATED_LEARNING")
parser.add_argument("--worker_num", type=int, default=0)
parser.add_argument("--server_num", type=int, default=2)
parser.add_argument("--scheduler_ip", type=str, default="127.0.0.1")
parser.add_argument("--scheduler_port", type=int, default=8113)
parser.add_argument("--fl_server_port", type=int, default=6666)
parser.add_argument("--start_fl_job_threshold", type=int, default=1)
parser.add_argument("--start_fl_job_time_window", type=int, default=3000)
parser.add_argument("--update_model_ratio", type=float, default=1.0)
parser.add_argument("--update_model_time_window", type=int, default=3000)
parser.add_argument("--fl_name", type=str, default="Lenet")
parser.add_argument("--fl_iteration_num", type=int, default=25)
parser.add_argument("--client_epoch_num", type=int, default=20)
parser.add_argument("--client_batch_size", type=int, default=32)
parser.add_argument("--client_learning_rate", type=float, default=1e-5)
# The number of servers that this script will launch.
parser.add_argument("--local_server_num", type=int, default=-1)
parser.add_argument("--encrypt_type", type=str, default="NOT_ENCRYPT")
# parameters for encrypt_type='DP_ENCRYPT'
parser.add_argument("--dp_eps", type=float, default=50.0)
parser.add_argument("--dp_delta", type=float, default=0.01)  # 1/worker_num
parser.add_argument("--dp_norm_clip", type=float, default=1.0)
# parameters for encrypt_type='PW_ENCRYPT'
parser.add_argument("--share_secrets_ratio", type=float, default=1.0)
parser.add_argument("--cipher_time_window", type=int, default=300000)
parser.add_argument("--reconstruct_secrets_threshold", type=int, default=3)
parser.add_argument("--client_password", type=str, default="")
parser.add_argument("--server_password", type=str, default="")
parser.add_argument("--enable_ssl", type=ast.literal_eval, default=False)
parser.add_argument("--config_file_path", type=str, default="")
parser.add_argument("--pki_verify", type=ast.literal_eval, default=False)
# parameters used for pki_verify=True
# You can download root_first_ca, root_second_ca and equip_crl
# from https://pki.consumer.huawei.com/ca/
parser.add_argument("--root_first_ca_path", type=str, default="")
parser.add_argument("--root_second_ca_path", type=str, default="")
parser.add_argument("--equip_crl_path", type=str, default="")
parser.add_argument("--replay_attack_time_diff", type=int, default=600000)
# parameters for 'SIGNDS'
parser.add_argument("--sign_k", type=float, default=0.01)
parser.add_argument("--sign_eps", type=float, default=100)
parser.add_argument("--sign_thr_ratio", type=float, default=0.6)
parser.add_argument("--sign_global_lr", type=float, default=0.1)
parser.add_argument("--sign_dim_out", type=int, default=0)
parser.add_argument("--global_iteration_time_window", type=int, default=3600000)
# parameters for "compression"
parser.add_argument("--upload_compress_type", type=str, default="NO_COMPRESS",
                    choices=["NO_COMPRESS", "DIFF_SPARSE_QUANT"])
parser.add_argument("--upload_sparse_rate", type=float, default=0.5)
parser.add_argument("--download_compress_type", type=str, default="NO_COMPRESS",
                    choices=["NO_COMPRESS", "QUANT"])

args, _ = parser.parse_known_args()
device_target = args.device_target
server_mode = args.server_mode
worker_num = args.worker_num
server_num = args.server_num
scheduler_ip = args.scheduler_ip
scheduler_port = args.scheduler_port
fl_server_port = args.fl_server_port
start_fl_job_threshold = args.start_fl_job_threshold
start_fl_job_time_window = args.start_fl_job_time_window
update_model_ratio = args.update_model_ratio
update_model_time_window = args.update_model_time_window
fl_name = args.fl_name
fl_iteration_num = args.fl_iteration_num
client_epoch_num = args.client_epoch_num
client_batch_size = args.client_batch_size
client_learning_rate = args.client_learning_rate
local_server_num = args.local_server_num
dp_eps = args.dp_eps
dp_delta = args.dp_delta
dp_norm_clip = args.dp_norm_clip
encrypt_type = args.encrypt_type
share_secrets_ratio = args.share_secrets_ratio
cipher_time_window = args.cipher_time_window
reconstruct_secrets_threshold = args.reconstruct_secrets_threshold
client_password = args.client_password
server_password = args.server_password
enable_ssl = args.enable_ssl
config_file_path = args.config_file_path
pki_verify = args.pki_verify
root_first_ca_path = args.root_first_ca_path
root_second_ca_path = args.root_second_ca_path
equip_crl_path = args.equip_crl_path
replay_attack_time_diff = args.replay_attack_time_diff
sign_k = args.sign_k
sign_eps = args.sign_eps
sign_thr_ratio = args.sign_thr_ratio
sign_global_lr = args.sign_global_lr
sign_dim_out = args.sign_dim_out
global_iteration_time_window = args.global_iteration_time_window
upload_compress_type = args.upload_compress_type
upload_sparse_rate = args.upload_sparse_rate
download_compress_type = args.download_compress_type

if local_server_num == -1:
    local_server_num = server_num

assert local_server_num <= server_num, "The local server number should not be bigger than total server number."

for i in range(local_server_num):
    os.environ['MS_NODE_ID'] = str(i)
    cmd_server = "execute_path=$(pwd) && self_path=$(dirname \"${script_self}\") && "
    cmd_server += "rm -rf ${execute_path}/server_" + str(i) + "/ &&"
    cmd_server += "mkdir ${execute_path}/server_" + str(i) + "/ &&"
    cmd_server += "cd ${execute_path}/server_" + str(i) + "/ || exit && export GLOG_v=1 &&"
    cmd_server += "python ${self_path}/../cloud_train.py"
    cmd_server += " --device_target=" + device_target
    cmd_server += " --server_mode=" + server_mode
    cmd_server += " --ms_role=MS_SERVER"
    cmd_server += " --worker_num=" + str(worker_num)
    cmd_server += " --server_num=" + str(server_num)
    cmd_server += " --scheduler_ip=" + scheduler_ip
    cmd_server += " --scheduler_port=" + str(scheduler_port)
    cmd_server += " --fl_server_port=" + str(fl_server_port + i)
    cmd_server += " --start_fl_job_threshold=" + str(start_fl_job_threshold)
    cmd_server += " --start_fl_job_time_window=" + str(start_fl_job_time_window)
    cmd_server += " --update_model_ratio=" + str(update_model_ratio)
    cmd_server += " --update_model_time_window=" + str(update_model_time_window)
    cmd_server += " --fl_name=" + fl_name
    cmd_server += " --fl_iteration_num=" + str(fl_iteration_num)
    cmd_server += " --client_epoch_num=" + str(client_epoch_num)
    cmd_server += " --client_batch_size=" + str(client_batch_size)
    cmd_server += " --client_learning_rate=" + str(client_learning_rate)
    cmd_server += " --dp_eps=" + str(dp_eps)
    cmd_server += " --dp_delta=" + str(dp_delta)
    cmd_server += " --dp_norm_clip=" + str(dp_norm_clip)
    cmd_server += " --encrypt_type=" + str(encrypt_type)
    cmd_server += " --share_secrets_ratio=" + str(share_secrets_ratio)
    cmd_server += " --cipher_time_window=" + str(cipher_time_window)
    cmd_server += " --client_password=" + str(client_password)
    cmd_server += " --server_password=" + str(server_password)
    cmd_server += " --enable_ssl=" + str(enable_ssl)
    cmd_server += " --reconstruct_secrets_threshold=" + str(reconstruct_secrets_threshold)
    cmd_server += " --config_file_path=" + config_file_path
    cmd_server += " --pki_verify=" + str(pki_verify)
    cmd_server += " --root_first_ca_path=" + str(root_first_ca_path)
    cmd_server += " --root_second_ca_path=" + str(root_second_ca_path)
    cmd_server += " --equip_crl_path=" + str(equip_crl_path)
    cmd_server += " --replay_attack_time_diff=" + str(replay_attack_time_diff)
    cmd_server += " --sign_k=" + str(sign_k)
    cmd_server += " --sign_eps=" + str(sign_eps)
    cmd_server += " --sign_thr_ratio=" + str(sign_thr_ratio)
    cmd_server += " --sign_global_lr=" + str(sign_global_lr)
    cmd_server += " --sign_dim_out=" + str(sign_dim_out)
    cmd_server += " --global_iteration_time_window=" + str(global_iteration_time_window)
    cmd_server += " --upload_compress_type=" + str(upload_compress_type)
    cmd_server += " --upload_sparse_rate=" + str(upload_sparse_rate)
    cmd_server += " --download_compress_type=" + str(download_compress_type)
    cmd_server += " > server.log 2>&1 &"

    import time
    time.sleep(0.3)
    subprocess.call(['bash', '-c', cmd_server])
