<div align="center"> 

## Custom Hloc with Zippypoint, SiLK and HF-net interface

</div>

<p align="center">
  Ze&nbsp;Shi*</a> <b>&middot;</b>
  Zhiguo&nbsp;Chen</a> <b>&middot;</b>
  Yang&nbsp;He</a> <b>&middot;</b>
  Chen&nbsp;Wang</a> <b></b>
</p>

<p align="center">
    <a href="https://github.com/cvg/Hierarchical-Localization">
        <img src="https://img.shields.io/badge/Repo-hloc-red" /></a>
    <a href="https://pytorch.org/">
        <img src="https://img.shields.io/badge/Framework-PyTorch | Tensorflow-yellow.svg" /></a>
    <a href="https://paperswithcode.com/task/visual-localization">
        <img src="https://img.shields.io/badge/Task-Visual%20Localization-green.svg" /></a>
    <a href="https://github.com/zafirshi/PanoVPR/blob/master/LICENSE">
        <img src="https://img.shields.io/badge/License-MIT-blue.svg" /></a>
</p>

This repo is a **Custom** **[Hloc](https://github.com/cvg/Hierarchical-Localization)** **framework** with
[Zippypoint](https://github.com/menelaoskanakis/ZippyPoint), [SiLK](https://github.com/facebookresearch/silk) and
[HF-net](https://github.com/ethz-asl/hfnet) interface.
This repo is more of engineering integration and does not contain any framework-based innovation.

### Features

The project features are as follows：

- [x] [Zippypoint](https://github.com/menelaoskanakis/ZippyPoint)  -> Fast like ORB but accurate like SuperPoint binary
  descriptor <b>&middot;</b> ETH <b>&middot;</b> CVPRW IMC2020
- [x] [SiLK](https://github.com/facebookresearch/silk)  -> Simple local self-supervised descriptor <b>&middot;</b>
  MetaAI <b>&middot;</b> IMC2022
- [x] [HF-net](https://github.com/ethz-asl/hfnet)   -> Proven visual localization methods
- [x] [MixVPR](https://github.com/amaralibey/MixVPR)   -> SOTA method on Visual Place Recognition  <b>&middot;</b>
  Retrieval  <b>&middot;</b> WACV2023

### Usages

We evaluate our integrated method into AachenV1.1 and Aachen Datasets, usage about these datasets can check
out `hloc/pipelines/Aachen` for more details.
Using the command and url in Aachen dataset readme file is faster and more efficient when transfer to AachenV1.1 by
test.

All the third-party repo's code mentioned above should be put into `third_party` directory under root path,
model-weights file should also put in there.

In addition, we also evaluate these methods into our private business datasets, _ChunXiRoad_.
For quantitative evaluation like [hloc-benchmark](https://www.visuallocalization.net/benchmark/), we also
add `evaluate.py` script. 

### Results

ZippyPoint Results on AachenV1.1 dataset with different inference configures:

|          Experiment name          | Device | Input resize | Topk from retrieval | Matching ratio | Geometric verification |  Status   |        Day         |       Night        |
|:---------------------------------:|:------:|:------------:|:-------------------:|:--------------:|:----------------------:|:---------:|:------------------:|:------------------:|
| ssp_rz1024_m4096+nn-ssp+NetVLAD30 |  CPU   |     1024     |      netvald30      |       w        |           no           | Baseline  | 86.5 / 93.4 / 97.6 | 64.9 / 81.7 / 95.8 |
|     CPU_rz640_topk30_woMR_SGV     |  CPU   |     640      |      netvlad30      |       wo       |           no           |           | 77.5 / 83.3 / 90.7 | 38.2 / 45.5 / 61.8 |
|     CPU_rz640_topk30_wMR_SGV      |  CPU   |     640      |      netvlad30      |       w        |           no           | Mini Best | 80.0 / 87.6 / 94.1 | 44.5 / 57.1 / 75.4 |
|     CPU_rz640_topk50_woMR_SGV     |  CPU   |     640      |      netvlad50      |       wo       |           no           |           | 73.2 / 79.1 / 87.1 | 28.3 / 35.1 / 52.9 |
|     CPU_rz640_topk50_wMR_SGV      |  CPU   |     640      |      netvlad50      |       w        |           no           |           | 81.3 / 88.6 / 93.9 | 41.4 / 53.4 / 70.2 |
|    CPU_rz1024_topk30_woMR_SGV     |  CPU   |     1024     |      netvlad30      |       wo       |           no           |           | 80.7 / 86.5 / 92.5 | 41.4 / 53.4 / 69.1 |
|     CPU_rz1024_topk30_wMR_SGV     |  CPU   |     1024     |      netvlad30      |       w        |           no           |           | 84.7 / 91.5 / 95.8 | 56.5 / 71.7 / 84.3 |
|    CPU_rz1024_topk50_woMR_SGV     |  CPU   |     1024     |      netvlad50      |       wo       |           no           |           | 76.8 / 82.4 / 88.6 | 37.2 / 43.5 / 56.5 |
|     CPU_rz1024_topk50_wMR_SGV     |  CPU   |     1024     |      netvlad50      |       w        |           no           |           | 84.2 / 90.9 / 95.3 | 49.2 / 64.9 / 79.1 |
|    CPU_rz1024_topk50_wMR.9_SGV    |  CPU   |     1024     |      netvlad50      |     w_0.9      |           no           | Our Best  | 84.3 / 91.7 / 96.8 | 58.6 / 75.4 / 88.0 |
|    CPU_rz1024_topk30_wMR.9_SGV    |  CPU   |     1024     |      netvlad30      |     w_0.9      |           no           |           | 84.3 / 91.4 / 96.6 | 56.0 / 74.9 / 88.5 |

Our Best results on above table is quite similar to the reported results in original paper, besides by deconstructing
the
parameters in experiments, we find in our pipeline without **geometric verification** or not does't matter to final
results,
**retrieval TopK** and **matching ratio** compared more influence.

Compared with the original paper, the accuracy reproduction is basically the same in the daytime scene,
the accuracy of the night scene reproduction baseline is higher than paper,
and the reproduction zippypoint value is slightly lower (which has a lot of space for fine-tune hyperparameters).
Overall, the accuracy effect of the paper is basically reproduced.

ZippyPoint Results on ChunXiRoad dataset with different inference configures:

|         Experiment name          |    Status     | Input resize | Topk from retrieval | Matching ratio | KeyPoint thd |      Metric      |
|:--------------------------------:|:-------------:|:------------:|:-------------------:|:--------------:|:------------:|:----------------:|
|         sp+sg_netvlad10          |  Unknown Cfg  |              |                     |                |              | 96.4/ 97.9/ 98.8 |
|         hfnet+nn_hfnet10         |  Unknown Cfg  |     640      |      netvald10      |                |              | 64.0/ 76.8/ 86.9 |
|         hfnet+nn_hfnet10         |    wo nms     |     640      |      netvald10      |                |              | 78.7/ 91.3/ 97.3 |
|         hfnet+nn_hfnet10         |    w nms_3    |     640      |      netvald10      |                |              | 77.5/ 90.6/ 97.4 |
|     CPU_rz640_topk30_wMR_SGV     | Mini Best Cfg |     640      |      netvald30      |     w_0.95     |    0.0001    | 83.6/ 93.6/ 97.9 |
|     CPU_rz640_topk10_wMR_SGV     |               |     640      |      netvald10      |     w_0.95     |    0.0001    | 81.0/ 92.5/ 97.6 |
| CPU_rz640_topk10_wMR_SGV_kThd.01 |               |     640      |      netvald10      |     w_0.95     |     0.01     | 81.3/ 92.5/ 97.5 |

### Thanks

https://github.com/tsattler/visloc_pseudo_gt_limitations/blob/main/evaluate_estimates.py
