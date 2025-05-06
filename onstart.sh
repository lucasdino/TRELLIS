#!/usr/bin/env bash
set -e

source /opt/conda/etc/profile.d/conda.sh
cd /opt/trellis
./setup.sh --new-env --basic --xformers --flash-attn --diffoctreerast --vox2seq --spconv --mipgaussian --kaolin --nvdiffrast --demo

# Reload conda context in case a new env was just created
source /opt/conda/etc/profile.d/conda.sh
conda activate trellis