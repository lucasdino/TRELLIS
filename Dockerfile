# ---- Base image -------------------------------------------------------------
FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV HF_HOME=/root/.cache/huggingface
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

# ---- OS packages ------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates sudo bzip2 \
    libjpeg-dev libgl1 libglib2.0-0 libxrender1 libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# ---- Miniconda (needed because setup.sh can create a conda env) -------------
RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-py310_24.1.2-0-Linux-x86_64.sh -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p $CONDA_DIR \
    && rm /tmp/miniconda.sh \
    && $CONDA_DIR/bin/conda init bash \
    && echo "conda activate base" >> /root/.bashrc \
    && $CONDA_DIR/bin/conda clean -afy

# ---- Copy repo & helper script ----------------------------------------------
WORKDIR /opt/trellis
RUN git clone --depth=1 https://github.com/lucasdino/TRELLIS.git /opt/trellis && \
    cd /opt/trellis && git rev-parse HEAD
COPY setup.sh /opt/trellis/setup.sh
RUN chmod +x /opt/trellis/setup.sh

# ---- Install additional packages ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim unzip && \
    rm -rf /var/lib/apt/lists/*

# ---- Add onstart.sh to be executed on startup ----
COPY onstart.sh /root/onstart.sh
RUN chmod +x /root/onstart.sh
# RUN /root/onstart.sh

# ---- Add debug logs to Dockerfile ----
RUN echo "[INFO] Starting Dockerfile build process"

# Verify Miniconda installation
RUN $CONDA_DIR/bin/conda --version || { echo "[ERROR] Miniconda installation failed"; exit 1; }

# Verify setup.sh script is executable
RUN ls -l /opt/trellis/setup.sh || { echo "[ERROR] setup.sh script is not found or not executable"; exit 1; }

# Verify onstart.sh script is executable
RUN ls -l /root/onstart.sh || { echo "[ERROR] onstart.sh script is not found or not executable"; exit 1; }

# Log successful build
RUN echo "[INFO] Dockerfile build process completed successfully"

CMD ["/bin/bash"]