#!/bin/bash

module load openmpi/4.0.2rc3-ucx1.6.1-gdrcopy2.0-3-cuda10.1.2-intel2019.5.281 cmake
export CUDA_VISIBLE_DEVICES=0,1 # ,2,3,4,5,6,7
nvidia-smi -i 0,1 -c EXCLUSIVE_PROCESS
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
nvidia-cuda-mps-control -d

$@