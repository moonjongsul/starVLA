#!/usr/bin/env bash
set -euo pipefail

################################################################################
# Host config (노드1 호스트에서 설정)
################################################################################
export MASTER_ADDR=${SPARK_NODE_1}
export MASTER_PORT=29500
export NNODES=2

################################################################################
# 노드1 호스트에서 실행: rsync
################################################################################
echo "[SYNC] Syncing 'workspace' folder to node2..."
rsync -avz --delete \
  --exclude 'results/' \
  ~/workspace/ \
  ${USER}@${SPARK_NODE_2}:~/workspace/
echo "[SYNC] Done."

################################################################################
# 컨테이너 내부에서 실행할 스크립트 생성
################################################################################
cat > /tmp/train_container.sh << 'SCRIPT'
#!/usr/bin/env bash
set -euo pipefail

################################################################################
# User config
################################################################################
Framework_name="QwenOFT"
freeze_module_list=""
base_vlm="/mnt/synology/PretrainedModel/Qwen2.5-VL-3B-Instruct-Action"
config_yaml="./examples/LIBERO/train_files/starvla_cotrain_libero.yaml"
libero_data_root="/mnt/synology/RobotData/LEROBOT_LIBERO_DATA"
data_mix="libero_all"
run_root_dir="./results/Checkpoints"
run_id="multinode_libero_qwen25oft"

# training options
per_device_batch_size=24
max_train_steps=10001000
save_interval=10000
logging_frequency=100
eval_interval=100

# wandb
wandb_project="starVLA_Libero"
wandb_entity="jinhuiye"
################################################################################

: "${MASTER_ADDR:?Need MASTER_ADDR}"
: "${MASTER_PORT:?Need MASTER_PORT}"
: "${NODE_RANK:?Need NODE_RANK}"
: "${NNODES:?Need NNODES}"

export WANDB_MODE="${WANDB_MODE:-disabled}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1
export TORCH_CUDA_ARCH_LIST=12.1+PTX
export TORCHINDUCTOR_DISABLE=1
export NCCL_SOCKET_IFNAME=enp1s0f1np1
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5
export NCCL_TIMEOUT=3600

echo "============================================================"
echo "[starVLA multinode launch]"
echo "HOSTNAME              = $(hostname)"
echo "MASTER_ADDR           = ${MASTER_ADDR}"
echo "MASTER_PORT           = ${MASTER_PORT}"
echo "NODE_RANK             = ${NODE_RANK}"
echo "NNODES                = ${NNODES}"
echo "NCCL_SOCKET_IFNAME    = ${NCCL_SOCKET_IFNAME:-<unset>}"
echo "CUDA_VISIBLE_DEVICES  = ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "============================================================"

python -c "import torch; print('torch=', torch.__version__)"
python -c "import torch; print('cuda_available=', torch.cuda.is_available(), 'device_count=', torch.cuda.device_count())"
python -c "import torch; print('nccl=', torch.cuda.nccl.version())"

cd /workspace/starVLA

output_dir="${run_root_dir}/${run_id}"
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_machines "${NNODES}" \
  --num_processes "${NNODES}" \
  --machine_rank "${NODE_RANK}" \
  --main_process_ip "${MASTER_ADDR}" \
  --main_process_port "${MASTER_PORT}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${config_yaml}" \
  --framework.name "${Framework_name}" \
  --framework.qwenvl.base_vlm "${base_vlm}" \
  --datasets.vla_data.data_root_dir "${libero_data_root}" \
  --datasets.vla_data.data_mix "${data_mix}" \
  --datasets.vla_data.per_device_batch_size "${per_device_batch_size}" \
  --trainer.vla_data.video_backend torchvision_av \
  --trainer.freeze_modules "${freeze_module_list}" \
  --trainer.max_train_steps "${max_train_steps}" \
  --trainer.save_interval "${save_interval}" \
  --trainer.logging_frequency "${logging_frequency}" \
  --trainer.eval_interval "${eval_interval}" \
  --run_root_dir "${run_root_dir}" \
  --run_id "${run_id}" \
  --wandb_project "${wandb_project}" \
  --wandb_entity "${wandb_entity}"
SCRIPT

chmod +x /tmp/train_container.sh
scp /tmp/train_container.sh ${USER}@${SPARK_NODE_2}:/tmp/train_container.sh

################################################################################
# 노드2 컨테이너에서 실행 (백그라운드)
################################################################################
echo "[INFO] Starting worker on node2 (starVLA-node2)..."
# 노드2 (백그라운드, 출력 버림)
ssh ${USER}@${SPARK_NODE_2} \
  "docker exec \
     -e NODE_RANK=1 \
     -e MASTER_ADDR=${MASTER_ADDR} \
     -e MASTER_PORT=${MASTER_PORT} \
     -e NNODES=${NNODES} \
   starVLA-node2 bash /tmp/train_container.sh" > /dev/null 2>&1 &
SSH_PID=$!

# 노드1 (tty 연결로 tqdm 정상 출력)
docker exec -it \
  -e NODE_RANK=0 \
  -e MASTER_ADDR=${MASTER_ADDR} \
  -e MASTER_PORT=${MASTER_PORT} \
  -e NNODES=${NNODES} \
  starVLA-node1 bash /tmp/train_container.sh

wait $SSH_PID
echo "[INFO] Training finished."