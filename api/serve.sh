#!/usr/bin/env bash
# chmod +x api/serve.sh
set -e
export PORT=${PORT:-8080}              # Vast.ai will inject its own $PORT
python3 api/server.py