# --- 0  base image -----------------------------------------------------------
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# --- 1  system build essentials ---------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-dev git curl ninja-build build-essential cmake && \
    rm -rf /var/lib/apt/lists/*

# --- 2  Python + core libraries ----------------------------------------------
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3
RUN pip install --no-cache-dir torch==2.4.0 torchvision==0.19 \
        --index-url https://download.pytorch.org/whl/cu121

# --- 3  install TRELLIS dependencies -----------------------------------------
RUN pip install --no-cache-dir \
        git+https://github.com/EasternJournalist/utils3d.git@9a4eb15 \
        xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121 \
        flash-attn --no-build-isolation \
        spconv-cu121 \
        kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html

# --- 4  diff-gaussian-rasterization ------------------------------------------
RUN git clone --depth 1 https://github.com/autonomousvision/mip-splatting.git /tmp/mip && \
    pip install --no-cache-dir /tmp/mip/submodules/diff-gaussian-rasterization && \
    rm -rf /tmp/mip

# --- 5  clone & install TRELLIS ---------------------------------------------
WORKDIR /opt
RUN git clone --depth 1 https://github.com/lucasdino/TRELLIS.git trellis
WORKDIR /opt/trellis
RUN pip install --no-cache-dir -e .

# --- 6  final entry ----------------------------------------------------------
CMD ["/bin/bash"]
    