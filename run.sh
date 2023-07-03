#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate hloc

# switch to current project root dir
CURRENT_DIR=$(cd $(dirname $0); pwd)

# template run command
#CUDA_VISIBLE_DEVICES='' python3 -m hloc.pipelines.Aachen.pipeline

## GPU_rz640_topk30_//woMR_nSGV
###### woMR or wMR
#python3 -m hloc.pipelines.Aachen_v1_1.pipeline --outputs outputs/aachen_v1.1/GPU_rz640_topk30_woMR_SGV --feature_conf zippypoint_aachen --num_loc 30 --matcher_conf zippypoint-matcher_wothd --skip_geometric_verification=True
#python3 -m hloc.pipelines.Aachen_v1_1.pipeline --outputs outputs/aachen_v1.1/GPU_rz640_topk30_wMR_SGV --feature_conf zippypoint_aachen --num_loc 30 --matcher_conf zippypoint-matcher --skip_geometric_verification=True
##### SGV or nSGV
#python3 -m hloc.pipelines.Aachen_v1_1.pipeline --outputs outputs/aachen_v1.1/GPU_rz640_topk30_woMR_nSGV --feature_conf zippypoint_aachen --num_loc 30 --matcher_conf zippypoint-matcher_wothd --skip_geometric_verification=False
#python3 -m hloc.pipelines.Aachen_v1_1.pipeline --outputs outputs/aachen_v1.1/GPU_rz640_topk30_wMR_nSGV --feature_conf zippypoint_aachen --num_loc 30 --matcher_conf zippypoint-matcher --skip_geometric_verification=False
#
#### topk30 or topk50
#python3 -m hloc.pipelines.Aachen_v1_1.pipeline --outputs outputs/aachen_v1.1/GPU_rz640_topk50_woMR_SGV --feature_conf zippypoint_aachen --num_loc 50 --matcher_conf zippypoint-matcher_wothd --skip_geometric_verification=True
#python3 -m hloc.pipelines.Aachen_v1_1.pipeline --outputs outputs/aachen_v1.1/GPU_rz640_topk50_wMR_SGV --feature_conf zippypoint_aachen --num_loc 50 --matcher_conf zippypoint-matcher --skip_geometric_verification=True
#python3 -m hloc.pipelines.Aachen_v1_1.pipeline --outputs outputs/aachen_v1.1/GPU_rz640_topk50_woMR_nSGV --feature_conf zippypoint_aachen --num_loc 50 --matcher_conf zippypoint-matcher_wothd --skip_geometric_verification=False
#python3 -m hloc.pipelines.Aachen_v1_1.pipeline --outputs outputs/aachen_v1.1/GPU_rz640_topk50_wMR_nSGV --feature_conf zippypoint_aachen --num_loc 50 --matcher_conf zippypoint-matcher --skip_geometric_verification=False
#
### rz640 or rz1024
#python3 -m hloc.pipelines.Aachen_v1_1.pipeline --outputs outputs/aachen_v1.1/GPU_rz1024_topk30_woMR_SGV --feature_conf zippypoint_aachen_n4096_r1024 --num_loc 30 --matcher_conf zippypoint-matcher_wothd --skip_geometric_verification=True
#python3 -m hloc.pipelines.Aachen_v1_1.pipeline --outputs outputs/aachen_v1.1/GPU_rz1024_topk30_wMR_SGV --feature_conf zippypoint_aachen_n4096_r1024 --num_loc 30 --matcher_conf zippypoint-matcher --skip_geometric_verification=True
#python3 -m hloc.pipelines.Aachen_v1_1.pipeline --outputs outputs/aachen_v1.1/GPU_rz1024_topk30_woMR_nSGV --feature_conf zippypoint_aachen_n4096_r1024 --num_loc 30 --matcher_conf zippypoint-matcher_wothd --skip_geometric_verification=False
#python3 -m hloc.pipelines.Aachen_v1_1.pipeline --outputs outputs/aachen_v1.1/GPU_rz1024_topk30_wMR_nSGV --feature_conf zippypoint_aachen_n4096_r1024 --num_loc 30 --matcher_conf zippypoint-matcher --skip_geometric_verification=False
#python3 -m hloc.pipelines.Aachen_v1_1.pipeline --outputs outputs/aachen_v1.1/GPU_rz1024_topk50_woMR_SGV --feature_conf zippypoint_aachen_n4096_r1024 --num_loc 50 --matcher_conf zippypoint-matcher_wothd --skip_geometric_verification=True
#python3 -m hloc.pipelines.Aachen_v1_1.pipeline --outputs outputs/aachen_v1.1/GPU_rz1024_topk50_wMR_SGV --feature_conf zippypoint_aachen_n4096_r1024 --num_loc 50 --matcher_conf zippypoint-matcher --skip_geometric_verification=True
#python3 -m hloc.pipelines.Aachen_v1_1.pipeline --outputs outputs/aachen_v1.1/GPU_rz1024_topk50_woMR_nSGV --feature_conf zippypoint_aachen_n4096_r1024 --num_loc 50 --matcher_conf zippypoint-matcher_wothd --skip_geometric_verification=False
#python3 -m hloc.pipelines.Aachen_v1_1.pipeline --outputs outputs/aachen_v1.1/GPU_rz1024_topk50_wMR_nSGV --feature_conf zippypoint_aachen_n4096_r1024 --num_loc 50 --matcher_conf zippypoint-matcher --skip_geometric_verification=False

# CPU version
#CUDA_VISIBLE_DEVICES='' python3 -m hloc.pipelines.Aachen_v1_1.pipeline --outputs outputs/aachen_v1.1/CPU_rz640_topk30_woMR_SGV --feature_conf zippypoint_aachen --num_loc 30 --matcher_conf zippypoint-matcher_wothd --skip_geometric_verification=True
#CUDA_VISIBLE_DEVICES='' python3 -m hloc.pipelines.Aachen_v1_1.pipeline --outputs outputs/aachen_v1.1/CPU_rz640_topk30_wMR_SGV --feature_conf zippypoint_aachen --num_loc 30 --matcher_conf zippypoint-matcher --skip_geometric_verification=True
#
#CUDA_VISIBLE_DEVICES='' python3 -m hloc.pipelines.Aachen_v1_1.pipeline --outputs outputs/aachen_v1.1/CPU_rz640_topk30_woMR_nSGV --feature_conf zippypoint_aachen --num_loc 30 --matcher_conf zippypoint-matcher_wothd --skip_geometric_verification=False
#CUDA_VISIBLE_DEVICES='' python3 -m hloc.pipelines.Aachen_v1_1.pipeline --outputs outputs/aachen_v1.1/CPU_rz640_topk30_wMR_nSGV --feature_conf zippypoint_aachen --num_loc 30 --matcher_conf zippypoint-matcher --skip_geometric_verification=False
#
#CUDA_VISIBLE_DEVICES='' python3 -m hloc.pipelines.Aachen_v1_1.pipeline --outputs outputs/aachen_v1.1/CPU_rz640_topk50_woMR_SGV --feature_conf zippypoint_aachen --num_loc 50 --matcher_conf zippypoint-matcher_wothd --skip_geometric_verification=True
#CUDA_VISIBLE_DEVICES='' python3 -m hloc.pipelines.Aachen_v1_1.pipeline --outputs outputs/aachen_v1.1/CPU_rz640_topk50_wMR_SGV --feature_conf zippypoint_aachen --num_loc 50 --matcher_conf zippypoint-matcher --skip_geometric_verification=True
#CUDA_VISIBLE_DEVICES='' python3 -m hloc.pipelines.Aachen_v1_1.pipeline --outputs outputs/aachen_v1.1/CPU_rz640_topk50_woMR_nSGV --feature_conf zippypoint_aachen --num_loc 50 --matcher_conf zippypoint-matcher_wothd --skip_geometric_verification=False
#CUDA_VISIBLE_DEVICES='' python3 -m hloc.pipelines.Aachen_v1_1.pipeline --outputs outputs/aachen_v1.1/CPU_rz640_topk50_wMR_nSGV --feature_conf zippypoint_aachen --num_loc 50 --matcher_conf zippypoint-matcher --skip_geometric_verification=False
#
#CUDA_VISIBLE_DEVICES='' python3 -m hloc.pipelines.Aachen_v1_1.pipeline --outputs outputs/aachen_v1.1/CPU_rz1024_topk30_woMR_SGV --feature_conf zippypoint_aachen_n4096_r1024 --num_loc 30 --matcher_conf zippypoint-matcher_wothd --skip_geometric_verification=True
#CUDA_VISIBLE_DEVICES='' python3 -m hloc.pipelines.Aachen_v1_1.pipeline --outputs outputs/aachen_v1.1/CPU_rz1024_topk30_wMR_SGV --feature_conf zippypoint_aachen_n4096_r1024 --num_loc 30 --matcher_conf zippypoint-matcher --skip_geometric_verification=True
#CUDA_VISIBLE_DEVICES='' python3 -m hloc.pipelines.Aachen_v1_1.pipeline --outputs outputs/aachen_v1.1/CPU_rz1024_topk30_woMR_nSGV --feature_conf zippypoint_aachen_n4096_r1024 --num_loc 30 --matcher_conf zippypoint-matcher_wothd --skip_geometric_verification=False
#CUDA_VISIBLE_DEVICES='' python3 -m hloc.pipelines.Aachen_v1_1.pipeline --outputs outputs/aachen_v1.1/CPU_rz1024_topk30_wMR_nSGV --feature_conf zippypoint_aachen_n4096_r1024 --num_loc 30 --matcher_conf zippypoint-matcher --skip_geometric_verification=False
#CUDA_VISIBLE_DEVICES='' python3 -m hloc.pipelines.Aachen_v1_1.pipeline --outputs outputs/aachen_v1.1/CPU_rz1024_topk50_woMR_SGV --feature_conf zippypoint_aachen_n4096_r1024 --num_loc 50 --matcher_conf zippypoint-matcher_wothd --skip_geometric_verification=True
#CUDA_VISIBLE_DEVICES='' python3 -m hloc.pipelines.Aachen_v1_1.pipeline --outputs outputs/aachen_v1.1/CPU_rz1024_topk50_wMR_SGV --feature_conf zippypoint_aachen_n4096_r1024 --num_loc 50 --matcher_conf zippypoint-matcher --skip_geometric_verification=True
#CUDA_VISIBLE_DEVICES='' python3 -m hloc.pipelines.Aachen_v1_1.pipeline --outputs outputs/aachen_v1.1/CPU_rz1024_topk50_woMR_nSGV --feature_conf zippypoint_aachen_n4096_r1024 --num_loc 50 --matcher_conf zippypoint-matcher_wothd --skip_geometric_verification=False
#CUDA_VISIBLE_DEVICES='' python3 -m hloc.pipelines.Aachen_v1_1.pipeline --outputs outputs/aachen_v1.1/CPU_rz1024_topk50_wMR_nSGV --feature_conf zippypoint_aachen_n4096_r1024 --num_loc 50 --matcher_conf zippypoint-matcher --skip_geometric_verification=False

# refine run
###-> topk30 and matching ratio 0.9
#CUDA_VISIBLE_DEVICES='' python3 -m hloc.pipelines.Aachen_v1_1.pipeline --outputs outputs/aachen_v1.1/CPU_rz1024_topk30_wMR.9_SGV --feature_conf zippypoint_aachen_n4096_r1024 --num_loc 30 --matcher_conf zippypoint-matcher.9 --skip_geometric_verification=True
###-> topk50 and matching ratio 0.9
#CUDA_VISIBLE_DEVICES='' python3 -m hloc.pipelines.Aachen_v1_1.pipeline --outputs outputs/aachen_v1.1/CPU_rz1024_topk50_wMR.9_SGV --feature_conf zippypoint_aachen_n4096_r1024 --num_loc 50 --matcher_conf zippypoint-matcher.9 --skip_geometric_verification=True

# ChunXiRoad Experiment
#CUDA_VISIBLE_DEVICES='' python3 -m hloc.pipelines.ChunXiRoad.pipeline --outputs outputs/ChunXiRoad/CPU_rz640_topk30_wMR_SGV --feature_conf zippypoint_aachen --num_loc 30 --matcher_conf zippypoint-matcher --skip_geometric_verification=True
# add new sift model in CPU_rz640_topk30_wMR_SGV_nDC5
#CUDA_VISIBLE_DEVICES='' python3 -m hloc.pipelines.ChunXiRoad.pipeline --outputs outputs/ChunXiRoad/CPU_rz640_topk30_wMR_SGV_nDC5 --feature_conf zippypoint_aachen --num_loc 30 --matcher_conf zippypoint-matcher --skip_geometric_verification=True
# add new colmap model from hy CPU_rz640_topk30_wMR_SGV_nDC6
#CUDA_VISIBLE_DEVICES='' python3 -m hloc.pipelines.ChunXiRoad.pipeline --outputs outputs/ChunXiRoad/CPU_rz640_topk30_wMR_SGV_nDC6 --feature_conf zippypoint_aachen --num_loc 30 --matcher_conf zippypoint-matcher --skip_geometric_verification=True
### just change topk30 -> topk10
#CUDA_VISIBLE_DEVICES='' python3 -m hloc.pipelines.ChunXiRoad.pipeline --outputs outputs/ChunXiRoad/CPU_rz640_topk10_wMR_SGV --feature_conf zippypoint_aachen --num_loc 10 --matcher_conf zippypoint-matcher --skip_geometric_verification=True
#CUDA_VISIBLE_DEVICES='' python3 -m hloc.pipelines.ChunXiRoad.pipeline --outputs outputs/ChunXiRoad/CPU_rz640_topk10_wMR_SGV_kThs.01_tf --feature_conf zippypoint_aachen.01 --num_loc 10 --matcher_conf zippypoint-matcher --skip_geometric_verification=True

# use onnxruntime
#CUDA_VISIBLE_DEVICES='' python3 -m hloc.pipelines.ChunXiRoad.pipeline --outputs outputs/ChunXiRoad/CPU_rz640_topk10_wMR_SGV_kThs.01_onnx --feature_conf zippypoint_aachen.01_onnx --num_loc 10 --matcher_conf zippypoint-matcher --skip_geometric_verification=True


# kThd & onnx/tf
#CUDA_VISIBLE_DEVICES='' python3 -m hloc.pipelines.ChunXiRoad.pipeline --outputs outputs/ChunXiRoad/CPU_rz640_topk10_wMR_SGV --feature_conf zippypoint_aachen --num_loc 10 --matcher_conf zippypoint-matcher --skip_geometric_verification=True
#CUDA_VISIBLE_DEVICES='' python3 -m hloc.pipelines.ChunXiRoad.pipeline --outputs outputs/ChunXiRoad/CPU_rz640_topk10_wMR_SGV_kThd.001 --feature_conf zippypoint_aachen.001 --num_loc 10 --matcher_conf zippypoint-matcher --skip_geometric_verification=True
#CUDA_VISIBLE_DEVICES='' python3 -m hloc.pipelines.ChunXiRoad.pipeline --outputs outputs/ChunXiRoad/CPU_rz640_topk10_wMR_SGV_kThd.01 --feature_conf zippypoint_aachen.01 --num_loc 10 --matcher_conf zippypoint-matcher --skip_geometric_verification=True

#CUDA_VISIBLE_DEVICES='' python3 -m hloc.pipelines.ChunXiRoad.pipeline --outputs outputs/ChunXiRoad/CPU_rz640_topk10_wMR_SGV_onnx --feature_conf zippypoint_aachen_onnx --num_loc 10 --matcher_conf zippypoint-matcher --skip_geometric_verification=True
#CUDA_VISIBLE_DEVICES='' python3 -m hloc.pipelines.ChunXiRoad.pipeline --outputs outputs/ChunXiRoad/CPU_rz640_topk10_wMR_SGV_kThd.001_onnx --feature_conf zippypoint_aachen.001_onnx --num_loc 10 --matcher_conf zippypoint-matcher --skip_geometric_verification=True
#CUDA_VISIBLE_DEVICES='' python3 -m hloc.pipelines.ChunXiRoad.pipeline --outputs outputs/ChunXiRoad/CPU_rz640_topk10_wMR_SGV_kThd.01_onnx --feature_conf zippypoint_aachen.01_onnx --num_loc 10 --matcher_conf zippypoint-matcher --skip_geometric_verification=True

# matching ratio
#CUDA_VISIBLE_DEVICES='' python3 -m hloc.pipelines.ChunXiRoad.pipeline --outputs outputs/ChunXiRoad/CPU_rz640_topk10_wMR.9_SGV --feature_conf zippypoint_aachen --num_loc 10 --matcher_conf zippypoint-matcher.9 --skip_geometric_verification=True
#CUDA_VISIBLE_DEVICES='' python3 -m hloc.pipelines.ChunXiRoad.pipeline --outputs outputs/ChunXiRoad/CPU_rz640_topk10_wMR.9_SGV_kThd.001 --feature_conf zippypoint_aachen.001 --num_loc 10 --matcher_conf zippypoint-matcher.9 --skip_geometric_verification=True
#CUDA_VISIBLE_DEVICES='' python3 -m hloc.pipelines.ChunXiRoad.pipeline --outputs outputs/ChunXiRoad/CPU_rz640_topk10_wMR.9_SGV_kThd.01 --feature_conf zippypoint_aachen.01 --num_loc 10 --matcher_conf zippypoint-matcher.9 --skip_geometric_verification=True

#CUDA_VISIBLE_DEVICES='' python3 -m hloc.pipelines.ChunXiRoad.pipeline --outputs outputs/ChunXiRoad/CPU_rz640_topk10_wMR.9_SGV_onnx --feature_conf zippypoint_aachen_onnx --num_loc 10 --matcher_conf zippypoint-matcher.9 --skip_geometric_verification=True
#CUDA_VISIBLE_DEVICES='' python3 -m hloc.pipelines.ChunXiRoad.pipeline --outputs outputs/ChunXiRoad/CPU_rz640_topk10_wMR.9_SGV_kThd.001_onnx --feature_conf zippypoint_aachen.001_onnx --num_loc 10 --matcher_conf zippypoint-matcher.9 --skip_geometric_verification=True
#CUDA_VISIBLE_DEVICES='' python3 -m hloc.pipelines.ChunXiRoad.pipeline --outputs outputs/ChunXiRoad/CPU_rz640_topk10_wMR.9_SGV_kThd.01_onnx --feature_conf zippypoint_aachen.01_onnx --num_loc 10 --matcher_conf zippypoint-matcher.9 --skip_geometric_verification=True

# hfnet on ChunXiRoad
CUDA_VISIBLE_DEVICES='' python3 -m hloc.pipelines.ChunXiRoad.pipeline --outputs outputs/ChunXiRoad/hfnet_onnx_wnms --feature_conf hfnet_onnx --num_loc 10 --matcher_conf NN-mutual --skip_geometric_verification=True

