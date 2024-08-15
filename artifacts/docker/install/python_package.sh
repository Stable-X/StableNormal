#!/bin/bash

set -exo pipefail

eval "$(conda shell.bash hook)"

conda activate py311
# pytorch
pip install --no-cache-dir \
    torch==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu118
pip install --no-cache-dir \
    pytest \
    omegaconf \
    loguru \
    pre-commit \
    pytorch_memlab \
    memory_profiler
