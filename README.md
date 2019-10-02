# UPSNet on RC

## Introduction 

UPSNet is initially described in a [CVPR 2019 oral](https://arxiv.org/abs/1901.03784) paper. Here I introduce the settings on RC Harvard Cluster and potential problems you may meet.

The performance for my pretrained model(use upsnet_resnet50_cityscapes_4gpu.yaml) test log on RC is shown in [my_model_test_cityscapes.log](https://github.com/charlotte12l/UPSNet/blob/master/outputs/my_model_test_cityscapes.log).
The performance of the author's pretrained model test results on RC is shown in [author_model_test_cityscapes.log](https://github.com/charlotte12l/UPSNet/blob/master/outputs/author_model_test_cityscapes.log).

|                | PQ   | SQ   | RQ   | PQ<sup>Th</sup> | PQ<sup>St</sup> | mIOU | AP |
|----------------|------|------|------|-----------------|-----------------|------|----|
| UPSNet-50 (My) | 58.6 | 79.8 | 72.1 | 53.8            | 62.1            | 75.5 | 34.4|
| UPSNet-50(Author's test on RC) | 59.4 | 79.7 | 73.1 | 54.6 | 62.8 | 75.3 | 33.3|
| UPSNet-50(Author's report ) | 59.3 | 79.7 | 73.0 | 54.6    | 62.7 |  75.2  |33.3       |

For COCO, use the author's model can also reproduce its results. They are shown in [author_model_test_coco.log](https://github.com/charlotte12l/UPSNet/blob/master/outputs/author_model_test_coco.log)





## Requirements: Software

We recommend using Anaconda3 as it already includes many common packages.


## Requirements: Hardware

We recommend using 4~16 GPUs with at least 11 GB memory to train our model.

## Modules I load on RC

module load Anaconda3/5.0.1-fasrc01

module load cuda/9.0-fasrc02 cudnn/7.4.1.5_cuda9.0-fasrc01

module load gcc/4.9.3-fasrc01

Please use torchvision == 0.4.0

## Installation

Please use gcc4.9 when building the operators.

Clone this repo to `$UPSNet_ROOT`

Run `init.sh` to build essential C++/CUDA modules and download pretrained model.

For Cityscapes:

Assuming you already downloaded Cityscapes dataset at `$CITYSCAPES_ROOT` and TrainIds label images are generated, please create a soft link by `ln -s $CITYSCAPES_ROOT data/cityscapes` under `UPSNet_ROOT`, and run `init_cityscapes.sh` to prepare Cityscapes dataset for UPSNet.

For COCO:

Assuming you already downloaded COCO dataset at `$COCO_ROOT` and have `annotations` and `images` folders under it, please create a soft link by `ln -s $COCO_ROOT data/coco` under `UPSNet_ROOT`, and run `init_coco.sh` to prepare COCO dataset for UPSNet.

Training:

`python upsnet/upsnet_end2end_train.py --cfg upsnet/experiments/$EXP.yaml`

Test:

`python upsnet/upsnet_end2end_test.py --cfg upsnet/experiments/$EXP.yaml`

We provide serveral config files (16/4 GPUs for Cityscapes/COCO dataset) under upsnet/experiments folder.

## Model Weights

The model weights that can reproduce numbers in our paper are available now. Please follow these steps to use them:

Run `download_weights.sh` to get trained model weights for Cityscapes and COCO.

For Cityscapes:

```shell
python upsnet/upsnet_end2end_test.py --cfg upsnet/experiments/upsnet_resnet50_cityscapes_16gpu.yaml --weight_path ./model/upsnet_resnet_50_cityscapes_12000.pth
```

```shell
python upsnet/upsnet_end2end_test.py --cfg upsnet/experiments/upsnet_resnet101_cityscapes_w_coco_16gpu.yaml --weight_path ./model/upsnet_resnet_101_cityscapes_w_coco_3000.pth
```

For COCO:

```shell
python upsnet/upsnet_end2end_test.py --cfg upsnet/experiments/upsnet_resnet50_coco_16gpu.yaml --weight_path model/upsnet_resnet_50_coco_90000.pth
```

```shell
python upsnet/upsnet_end2end_test.py --cfg upsnet/experiments/upsnet_resnet101_dcn_coco_3x_16gpu.yaml --weight_path model/upsnet_resnet_101_dcn_coco_270000.pth
```

## My Modification

Because the previous codes use many relative path and would cause error when you sbatch a job on RC. I modify many paths to let it run on RC, which you can see in my commit. You can change them to your path.

## Potential Problems You May Meet

- When Building the operators: 
    - Error:  #error -- unsupported GNU version! gcc versions later than 6 are not supported!
              ^~~~~
            error: command '/n/helmod/apps/centos7/Core/cuda/9.0-fasrc02/bin/nvcc' failed with exit status 
    - Solution: Downgrade the gcc to 4.9

- When runing, can't import cv2
    - Error: ImportError: /n/helmod/apps/centos7/Core/gcc/4.9.3-fasrc01/lib64/libstdc++.so.6: version \`GLIBCXX_3.4.21\' not found (required by /n/home05/xingyu/.conda/envs/ups/lib/python3.6/site-packages/cv2.cpython-36m-x86_64-linux-gnu.so)
    - Solution: Change gcc version to 7+ 
    
- Undefined Symbol:
    - Error: 
    Traceback (most recent call last): File "upsnet/upsnet_end2end_test.py", line 44, in <module> from upsnet.models import * File "upsnet/../upsnet/models/__init__.py", line 1, in <module> from .resnet_upsnet import resnet_50_upsnet, resnet_101_upsnet File "upsnet/../upsnet/models/resnet_upsnet.py", line 22, in <module> from upsnet.models.resnet import get_params, resnet_rcnn, ResNetBackbone File "upsnet/../upsnet/models/resnet.py", line 21, in <module> from upsnet.operators.modules.deform_conv import DeformConv File "upsnet/../upsnet/operators/modules/deform_conv.py", line 22, in <module> from upsnet.operators.functions.deform_conv import DeformConvFunction File "upsnet/../upsnet/operators/functions/deform_conv.py", line 21, in <module> from .._ext.deform_conv import deform_conv_cuda ImportError: upsnet/../upsnet/operators/_ext/deform_conv/deform_conv_cuda.cpython-36m-x86_64-linux-gnu.so: undefined symbol: _ZN6caffe26detail37_typeMetaDataInstance_preallocated_32E
    
    - Solution:
    pip install -U torchvision==0.4.0

- Segmentation Error:
    - Error: Core Dump, Segmentation Error
    
    - Solution:
    Maybe pytorch version you used to build the operator is different from the pytorch version you used to run experiments. Please double check the python env/pytorch version and try to rebuild the operators (don't forget to delete upsnet/operators/build folder first)
    
- Dataset:

    - Error: Division Zero
    
    - Solution:
    Please make sure the data are in data/cityscapes/images/,data/cityscapes/annotations/, data/cityscapes/labels/, data/cityscapes/panoptic/, and the path to the dataset is right 


------------------------------------------------------------------------------------------------------------------
# UPSNet: A Unified Panoptic Segmentation Network

# Introduction
UPSNet is initially described in a [CVPR 2019 oral](https://arxiv.org/abs/1901.03784) paper.

# Disclaimer

This repository is tested under Python 3.6, PyTorch 0.4.1. And model training is done with 16 GPUs by using [horovod](https://github.com/horovod/horovod). It should also work under Python 2.7 / PyTorch 1.0 and with 4 GPUs.

# License
© Uber, 2018-2019. Licensed under the Uber Non-Commercial License.

# Citing UPSNet

If you find UPSNet is useful in your research, please consider citing:
```
@inproceedings{xiong19upsnet,
    Author = {Yuwen Xiong, Renjie Liao, Hengshuang Zhao, Rui Hu, Min Bai, Ersin Yumer, Raquel Urtasun},
    Title = {UPSNet: A Unified Panoptic Segmentation Network},
    Conference = {CVPR},
    Year = {2019}
}
```


# Main Results

COCO 2017 (trained on train-2017 set)

|                | test split | PQ   | SQ   | RQ   | PQ<sup>Th</sup> | PQ<sup>St</sup> |
|----------------|------------|------|------|------|-----------------|-----------------|
| UPSNet-50      | val        | 42.5 | 78.0 | 52.4 | 48.5            | 33.4            |
| UPSNet-101-DCN | test-dev   | 46.6 | 80.5 | 56.9 | 53.2            | 36.7            |

Cityscapes

|                | PQ   | SQ   | RQ   | PQ<sup>Th</sup> | PQ<sup>St</sup> |
|----------------|------|------|------|-----------------|-----------------|
| UPSNet-50      | 59.3 | 79.7 | 73.0 | 54.6            | 62.7            |
| UPSNet-101-COCO (ms test) | 61.8 | 81.3 | 74.8 | 57.6 | 64.8 |








