#!/bin/bash

# === NCCL 설정 (멀티노드 RoCE) ===
export NCCL_SOCKET_IFNAME=enp1s0f1np1
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5
export NCCL_DEBUG=WARN
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=3600
export NCCL_SOCKET_TIMEOUT_MS=360000

# === Blackwell sm_121 JIT 문제 방지 ===
export TORCH_CUDA_ARCH_LIST="12.1+PTX"
export TORCHINDUCTOR_DISABLE=1

# === 학습 설정 ===
Framework_name=QwenOFT
freeze_module_list=''
base_vlm=/mnt/synology/PretrainedModel/Qwen2.5-VL-3B-Instruct-Action
config_yaml=./examples/LIBERO/train_files/starvla_cotrain_libero.yaml
libero_data_root=/mnt/synology/RobotData/LEROBOT_LIBERO_DATA
data_mix=libero_all
run_root_dir=./results/Checkpoints
run_id=1229_libero4in1_qwen25oft

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
cp $0 ${output_dir}/

# === 노드2 학습 시작 (백그라운드 SSH) ===
echo "[INFO] Starting training on node2..."
ssh 192.168.200.13 "cd $(pwd) && bash run_train_node2.sh" &
SSH_PID=$!

# === 노드1 학습 시작 ===
echo "[INFO] Starting training on node1 (master)..."
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --main_process_ip 192.168.200.12 \
  --main_process_port 29500 \
  --machine_rank 0 \
  --num_machines 2 \
  --num_processes 2 \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --datasets.vla_data.data_root_dir ${libero_data_root} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 16 \
  --trainer.vla_data.video_backend torchvision_av \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 80000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 100 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA_Libero \
  --wandb_entity jinhuiye

# 노드1 종료 시 노드2도 정리
wait $SSH_PID