#!/usr/bin/env bash
set -e

source /opt/conda/etc/profile.d/conda.sh
conda activate base

cd /opt/trellis
./setup.sh --basic --xformers --flash-attn --diffoctreerast --vox2seq --spconv --mipgaussian --kaolin --nvdiffrast --demo