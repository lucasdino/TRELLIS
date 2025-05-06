# --- 0  base image -----------------------------------------------------------
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Set CUDA archs (Pascal to Ada)
ENV DEBIAN_FRONTEND=noninteractive \
    TORCH_CUDA_ARCH_LIST="6.1;7.5;8.0;8.6;8.9+PTX" \
    MAX_JOBS=8 \
    PYTHONPATH="/opt/trellis"

# --- 1  system build essentials ---------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev git curl ninja-build build-essential cmake unzip git-lfs && \
    rm -rf /var/lib/apt/lists/*

# --- 2  Python + core libraries ----------------------------------------------
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3
RUN pip install --no-cache-dir \
    torch==2.4.0+cu121 torchvision==0.19.0+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121

# --- 3  install TRELLIS dependencies -----------------------------------------
RUN pip install --no-cache-dir \
    git+https://github.com/EasternJournalist/utils3d.git@9a4eb15 \
    spconv-cu121==2.3.8 \
    kaolin==0.17.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html

# Build xformers from source
RUN FORCE_CUDA=1 MAX_JOBS=2 pip install --no-cache-dir --no-binary xformers xformers==0.0.27.post2

# --- 3b  flash-attn ----------------------------------------------------------
RUN git clone --recursive https://github.com/Dao-AILab/flash-attention.git /tmp/flash-attn && \
    cd /tmp/flash-attn && \
    pip install --no-cache-dir . && \
    cd / && rm -rf /tmp/flash-attn

# --- 4  diff-gaussian-rasterization ------------------------------------------
RUN git clone --depth 1 https://github.com/autonomousvision/mip-splatting.git /tmp/mip && \
    pip install --no-cache-dir /tmp/mip/submodules/diff-gaussian-rasterization && \
    rm -rf /tmp/mip

# --- 5  clone & install TRELLIS ---------------------------------------------
WORKDIR /opt
RUN git clone --depth 1 https://github.com/lucasdino/TRELLIS.git trellis
WORKDIR /opt/trellis
RUN chmod +x setup.sh && \
    ./setup.sh --diffoctreerast --nvdiffrast

# --- 6  final entry ----------------------------------------------------------
CMD ["/bin/bash"]    