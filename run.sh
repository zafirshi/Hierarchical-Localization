#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate hloc

# switch to current project root dir
CURRENT_DIR=$(cd $(dirname $0); pwd)

# template run command
#CUDA_VISIBLE_DEVICES='' python3 -m hloc.pipelines.Aachen.pipeline

# GPU_rz640_topk30_//woMR_nSGV
##### woMR or wMR
python3 -m hloc.pipelines.Aachen_v1_1.pipeline --outputs outputs/aachen_v1.1/GPU_rz640_topk30_woMR_SGV --feature_conf zippypoint_aachen --num_loc 30 --matcher_conf zippypoint-matcher_wothd --skip_geometric_verification=True
python3 -m hloc.pipelines.Aachen_v1_1.pipeline --outputs outputs/aachen_v1.1/GPU_rz640_topk30_wMR_SGV --feature_conf zippypoint_aachen --num_loc 30 --matcher_conf zippypoint-matcher --skip_geometric_verification=True
#### SGV or nSGV
python3 -m hloc.pipelines.Aachen_v1_1.pipeline --outputs outputs/aachen_v1.1/GPU_rz640_topk30_woMR_nSGV --feature_conf zippypoint_aachen --num_loc 30 --matcher_conf zippypoint-matcher_wothd --skip_geometric_verification=False
python3 -m hloc.pipelines.Aachen_v1_1.pipeline --outputs outputs/aachen_v1.1/GPU_rz640_topk30_wMR_nSGV --feature_conf zippypoint_aachen --num_loc 30 --matcher_conf zippypoint-matcher --skip_geometric_verification=False

### topk30 or topk50
python3 -m hloc.pipelines.Aachen_v1_1.pipeline --outputs outputs/aachen_v1.1/GPU_rz640_topk50_woMR_SGV --feature_conf zippypoint_aachen --num_loc 50 --matcher_conf zippypoint-matcher_wothd --skip_geometric_verification=True
python3 -m hloc.pipelines.Aachen_v1_1.pipeline --outputs outputs/aachen_v1.1/GPU_rz640_topk50_wMR_SGV --feature_conf zippypoint_aachen --num_loc 50 --matcher_conf zippypoint-matcher --skip_geometric_verification=True
python3 -m hloc.pipelines.Aachen_v1_1.pipeline --outputs outputs/aachen_v1.1/GPU_rz640_topk50_woMR_nSGV --feature_conf zippypoint_aachen --num_loc 50 --matcher_conf zippypoint-matcher_wothd --skip_geometric_verification=False
python3 -m hloc.pipelines.Aachen_v1_1.pipeline --outputs outputs/aachen_v1.1/GPU_rz640_topk50_wMR_nSGV --feature_conf zippypoint_aachen --num_loc 50 --matcher_conf zippypoint-matcher --skip_geometric_verification=False

## rz640 or rz1024
python3 -m hloc.pipelines.Aachen_v1_1.pipeline --outputs outputs/aachen_v1.1/GPU_rz1024_topk30_woMR_SGV --feature_conf zippypoint_aachen_n4096_r1024 --num_loc 30 --matcher_conf zippypoint-matcher_wothd --skip_geometric_verification=True
python3 -m hloc.pipelines.Aachen_v1_1.pipeline --outputs outputs/aachen_v1.1/GPU_rz1024_topk30_wMR_SGV --feature_conf zippypoint_aachen_n4096_r1024 --num_loc 30 --matcher_conf zippypoint-matcher --skip_geometric_verification=True
python3 -m hloc.pipelines.Aachen_v1_1.pipeline --outputs outputs/aachen_v1.1/GPU_rz1024_topk30_woMR_nSGV --feature_conf zippypoint_aachen_n4096_r1024 --num_loc 30 --matcher_conf zippypoint-matcher_wothd --skip_geometric_verification=False
python3 -m hloc.pipelines.Aachen_v1_1.pipeline --outputs outputs/aachen_v1.1/GPU_rz1024_topk30_wMR_nSGV --feature_conf zippypoint_aachen_n4096_r1024 --num_loc 30 --matcher_conf zippypoint-matcher --skip_geometric_verification=False
python3 -m hloc.pipelines.Aachen_v1_1.pipeline --outputs outputs/aachen_v1.1/GPU_rz1024_topk50_woMR_SGV --feature_conf zippypoint_aachen_n4096_r1024 --num_loc 50 --matcher_conf zippypoint-matcher_wothd --skip_geometric_verification=True
python3 -m hloc.pipelines.Aachen_v1_1.pipeline --outputs outputs/aachen_v1.1/GPU_rz1024_topk50_wMR_SGV --feature_conf zippypoint_aachen_n4096_r1024 --num_loc 50 --matcher_conf zippypoint-matcher --skip_geometric_verification=True
python3 -m hloc.pipelines.Aachen_v1_1.pipeline --outputs outputs/aachen_v1.1/GPU_rz1024_topk50_woMR_nSGV --feature_conf zippypoint_aachen_n4096_r1024 --num_loc 50 --matcher_conf zippypoint-matcher_wothd --skip_geometric_verification=False
python3 -m hloc.pipelines.Aachen_v1_1.pipeline --outputs outputs/aachen_v1.1/GPU_rz1024_topk50_wMR_nSGV --feature_conf zippypoint_aachen_n4096_r1024 --num_loc 50 --matcher_conf zippypoint-matcher --skip_geometric_verification=False

# CPU version
CUDA_VISIBLE_DEVICES='' python3 -m hloc.pipelines.Aachen_v1_1.pipeline --outputs outputs/aachen_v1.1/GPU_rz640_topk30_woMR_SGV --feature_conf zippypoint_aachen --num_loc 30 --matcher_conf zippypoint-matcher_wothd --skip_geometric_verification=True
CUDA_VISIBLE_DEVICES='' python3 -m hloc.pipelines.Aachen_v1_1.pipeline --outputs outputs/aachen_v1.1/GPU_rz640_topk30_wMR_SGV --feature_conf zippypoint_aachen --num_loc 30 --matcher_conf zippypoint-matcher --skip_geometric_verification=True

CUDA_VISIBLE_DEVICES='' python3 -m hloc.pipelines.Aachen_v1_1.pipeline --outputs outputs/aachen_v1.1/GPU_rz640_topk30_woMR_nSGV --feature_conf zippypoint_aachen --num_loc 30 --matcher_conf zippypoint-matcher_wothd --skip_geometric_verification=False
CUDA_VISIBLE_DEVICES='' python3 -m hloc.pipelines.Aachen_v1_1.pipeline --outputs outputs/aachen_v1.1/GPU_rz640_topk30_wMR_nSGV --feature_conf zippypoint_aachen --num_loc 30 --matcher_conf zippypoint-matcher --skip_geometric_verification=False

CUDA_VISIBLE_DEVICES='' python3 -m hloc.pipelines.Aachen_v1_1.pipeline --outputs outputs/aachen_v1.1/GPU_rz640_topk50_woMR_SGV --feature_conf zippypoint_aachen --num_loc 50 --matcher_conf zippypoint-matcher_wothd --skip_geometric_verification=True
CUDA_VISIBLE_DEVICES='' python3 -m hloc.pipelines.Aachen_v1_1.pipeline --outputs outputs/aachen_v1.1/GPU_rz640_topk50_wMR_SGV --feature_conf zippypoint_aachen --num_loc 50 --matcher_conf zippypoint-matcher --skip_geometric_verification=True
CUDA_VISIBLE_DEVICES='' python3 -m hloc.pipelines.Aachen_v1_1.pipeline --outputs outputs/aachen_v1.1/GPU_rz640_topk50_woMR_nSGV --feature_conf zippypoint_aachen --num_loc 50 --matcher_conf zippypoint-matcher_wothd --skip_geometric_verification=False
CUDA_VISIBLE_DEVICES='' python3 -m hloc.pipelines.Aachen_v1_1.pipeline --outputs outputs/aachen_v1.1/GPU_rz640_topk50_wMR_nSGV --feature_conf zippypoint_aachen --num_loc 50 --matcher_conf zippypoint-matcher --skip_geometric_verification=False

CUDA_VISIBLE_DEVICES='' python3 -m hloc.pipelines.Aachen_v1_1.pipeline --outputs outputs/aachen_v1.1/GPU_rz1024_topk30_woMR_SGV --feature_conf zippypoint_aachen_n4096_r1024 --num_loc 30 --matcher_conf zippypoint-matcher_wothd --skip_geometric_verification=True
CUDA_VISIBLE_DEVICES='' python3 -m hloc.pipelines.Aachen_v1_1.pipeline --outputs outputs/aachen_v1.1/GPU_rz1024_topk30_wMR_SGV --feature_conf zippypoint_aachen_n4096_r1024 --num_loc 30 --matcher_conf zippypoint-matcher --skip_geometric_verification=True
CUDA_VISIBLE_DEVICES='' python3 -m hloc.pipelines.Aachen_v1_1.pipeline --outputs outputs/aachen_v1.1/GPU_rz1024_topk30_woMR_nSGV --feature_conf zippypoint_aachen_n4096_r1024 --num_loc 30 --matcher_conf zippypoint-matcher_wothd --skip_geometric_verification=False
CUDA_VISIBLE_DEVICES='' python3 -m hloc.pipelines.Aachen_v1_1.pipeline --outputs outputs/aachen_v1.1/GPU_rz1024_topk30_wMR_nSGV --feature_conf zippypoint_aachen_n4096_r1024 --num_loc 30 --matcher_conf zippypoint-matcher --skip_geometric_verification=False
CUDA_VISIBLE_DEVICES='' python3 -m hloc.pipelines.Aachen_v1_1.pipeline --outputs outputs/aachen_v1.1/GPU_rz1024_topk50_woMR_SGV --feature_conf zippypoint_aachen_n4096_r1024 --num_loc 50 --matcher_conf zippypoint-matcher_wothd --skip_geometric_verification=True
CUDA_VISIBLE_DEVICES='' python3 -m hloc.pipelines.Aachen_v1_1.pipeline --outputs outputs/aachen_v1.1/GPU_rz1024_topk50_wMR_SGV --feature_conf zippypoint_aachen_n4096_r1024 --num_loc 50 --matcher_conf zippypoint-matcher --skip_geometric_verification=True
CUDA_VISIBLE_DEVICES='' python3 -m hloc.pipelines.Aachen_v1_1.pipeline --outputs outputs/aachen_v1.1/GPU_rz1024_topk50_woMR_nSGV --feature_conf zippypoint_aachen_n4096_r1024 --num_loc 50 --matcher_conf zippypoint-matcher_wothd --skip_geometric_verification=False
CUDA_VISIBLE_DEVICES='' python3 -m hloc.pipelines.Aachen_v1_1.pipeline --outputs outputs/aachen_v1.1/GPU_rz1024_topk50_wMR_nSGV --feature_conf zippypoint_aachen_n4096_r1024 --num_loc 50 --matcher_conf zippypoint-matcher --skip_geometric_verification=False


