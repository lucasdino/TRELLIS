#!/usr/bin/env bash
set -e

# Properly initialize Conda and activate base
eval "$(/opt/conda/bin/conda shell.bash hook)"
conda activate base

# Run setup
cd /opt/trellis
./setup.sh --basic --xformers --flash-attn --diffoctreerast --vox2seq --spconv --mipgaussian --kaolin --nvdiffrast --demo