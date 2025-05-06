#!/usr/bin/env bash
set -e
set -x  # Enable debug mode to print each command before execution

export PYTHONPATH=/opt/trellis:$PYTHONPATH

source /opt/conda/etc/profile.d/conda.sh
conda activate base || { echo "[ERROR] Failed to activate conda environment 'base'"; exit 1; }

cd /opt/trellis || { echo "[ERROR] Failed to change directory to /opt/trellis"; exit 1; }

# Log the start of setup.sh execution
echo "[INFO] Starting setup.sh script with all flags except HELP and NEW_ENV"
./setup.sh --basic --xformers --flash-attn --diffoctreerast --vox2seq --spconv --mipgaussian --kaolin --nvdiffrast --demo || {
    echo "[ERROR] setup.sh script failed";
    exit 1;
}

# Log successful completion
echo "[INFO] onstart.sh completed successfully"