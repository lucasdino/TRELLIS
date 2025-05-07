#!/usr/bin/env bash
set -e
set -x


source /opt/conda/etc/profile.d/conda.sh
conda activate base
cd /opt/trellis

# Log the start of setup.sh execution
echo "[INFO] Starting setup.sh script with all flags except HELP and NEW_ENV"
# ./setup.sh --basic --xformers --flash-attn --diffoctreerast --vox2seq --spconv --mipgaussian --kaolin --nvdiffrast --demo

# Add to python path for trellis
export PYTHONPATH=/opt/trellis:$PYTHONPATH

# Log successful completion
echo "[INFO] onstart.sh completed successfully"