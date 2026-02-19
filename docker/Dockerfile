# syntax=docker/dockerfile:1.7

ARG CUDA_VER=12.4.1
ARG UBUNTU_VER=ubuntu24.04

# ---- build stage (needs nvcc for flash-attn) ----
FROM nvidia/cuda:${CUDA_VER}-cudnn-devel-${UBUNTU_VER} AS build

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev python3-venv \
    git curl ca-certificates \
    build-essential ninja-build cmake pkg-config \
    libstdc++-14-dev \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python
RUN python -m pip install --upgrade pip setuptools wheel

# ---- Pin versions (edit only if you intentionally change stack) ----
# PyTorch cu124 index
ARG TORCH_VER=2.6.0
ARG TRANSFORMERS_VER=4.49.0
ARG ACCELERATE_VER=1.1.1
ARG HFHUB_VER=0.26.2
ARG FLASH_ATTN_VER=2.8.3

# Install torch first (important)
RUN pip install --no-cache-dir \
    torch==${TORCH_VER} torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

# Qwen3-VL needs recent transformers; pin for reproducibility
RUN pip install --no-cache-dir \
    transformers==${TRANSFORMERS_VER} \
    accelerate==${ACCELERATE_VER} \
    huggingface_hub==${HFHUB_VER}

# Build flash-attn against THIS torch + THIS nvcc (CUDA 12.4)
# Key: build inside image so ABI matches torch.
ENV MAX_JOBS=8
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"
RUN pip install --no-cache-dir \
    flash-attn==${FLASH_ATTN_VER} --no-build-isolation

# ---- runtime stage (lighter) ----
FROM nvidia/cuda:${CUDA_VER}-cudnn-runtime-${UBUNTU_VER} AS runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# Basic runtime deps + ROS2 Jazzy (Ubuntu 24.04)
RUN apt-get update && apt-get install -y \
    python3 python3-pip \
    git curl ca-certificates \
    locales \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python
RUN python -m pip install --upgrade pip setuptools wheel

# Locale (ROS tooling sometimes expects UTF-8 locale)
RUN apt-get update && apt-get install -y locales && \
    locale-gen en_US.UTF-8 && \
    update-locale LANG=en_US.UTF-8 && \
    rm -rf /var/lib/apt/lists/*
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# ROS2 Jazzy install (24.04 native)
RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository universe && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y curl gnupg && rm -rf /var/lib/apt/lists/*
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    | gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg

RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
    http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
    > /etc/apt/sources.list.d/ros2.list

RUN apt-get update && apt-get install -y \
    ros-jazzy-ros-base \
    python3-argcomplete \
    && rm -rf /var/lib/apt/lists/*

# Copy python deps from build stage (torch/transformers/flash-attn pinned)
# This avoids rebuilding flash-attn in runtime.
COPY --from=build /usr/local/lib/python3.*/dist-packages /usr/local/lib/python3.*/dist-packages
COPY --from=build /usr/local/bin /usr/local/bin

# Workspace
WORKDIR /workspace/starVLA

# Hugging Face cache location (compose에서 볼륨 마운트 추천)
ENV HF_HOME=/root/.cache/huggingface
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface/transformers
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# ROS default middleware (환경에 따라 변경 가능)
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# Entry
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/e]()

