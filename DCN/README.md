# Deformable Convolutional Networks


**[A third-party improvement](https://github.com/bharatsingh430/Deformable-ConvNets) of Deformable R-FCN + Soft NMS**


## Introduction

**Deformable ConvNets** is initially described in an [ICCV 2017 oral paper](https://arxiv.org/abs/1703.06211). (Slides at [ICCV 2017 Oral](http://www.jifengdai.org/slides/Deformable_Convolutional_Networks_Oral.pdf))



## Disclaimer

This is an official implementation for [Deformable Convolutional Networks](https://arxiv.org/abs/1703.06211) (Deformable ConvNets) based on MXNet. It is worth noticing that:

  * The original implementation is based on our internal Caffe version on Windows. There are slight differences in the final accuracy and running time due to the plenty details in platform switch.
  * The code is tested on official [MXNet@(commit 62ecb60)](https://github.com/dmlc/mxnet/tree/62ecb60) with the extra operators for Deformable ConvNets.
  	* After [MXNet@(commit ce2bca6)](https://github.com/dmlc/mxnet/tree/ce2bca6) the offical MXNet support all operators for Deformable ConvNets.
  * We trained our model based on the ImageNet pre-trained [ResNet-v1-101](https://github.com/KaimingHe/deep-residual-networks) using a [model converter](https://github.com/dmlc/mxnet/tree/430ea7bfbbda67d993996d81c7fd44d3a20ef846/tools/caffe_converter). The converted model produces slightly lower accuracy (Top-1 Error on ImageNet val: 24.0% v.s. 23.6%).
  * This repository used code from [MXNet rcnn example](https://github.com/dmlc/mxnet/tree/master/example/rcnn) and [mx-rfcn](https://github.com/giorking/mx-rfcn).
  
## License

© Microsoft, 2017. Licensed under an MIT license.

## Citing Deformable ConvNets

If you find Deformable ConvNets useful in your research, please consider citing:
```
@article{dai17dcn,
    Author = {Jifeng Dai, Haozhi Qi, Yuwen Xiong, Yi Li, Guodong Zhang, Han Hu, Yichen Wei},
    Title = {Deformable Convolutional Networks},
    Journal = {arXiv preprint arXiv:1703.06211},
    Year = {2017}
}
@inproceedings{dai16rfcn,
    Author = {Jifeng Dai, Yi Li, Kaiming He, Jian Sun},
    Title = {{R-FCN}: Object Detection via Region-based Fully Convolutional Networks},
    Conference = {NIPS},
    Year = {2016}
}
```

## Requirements: Software

1. MXNet from [the offical repository](https://github.com/dmlc/mxnet). We tested our code on [MXNet@(commit 62ecb60)](https://github.com/dmlc/mxnet/tree/62ecb60). Due to the rapid development of MXNet, it is recommended to checkout this version if you encounter any issues. We may maintain this repository periodically if MXNet adds important feature in future release.

2. Python 2.7. We recommend using Anaconda2 as it already includes many common packages. We do not support Python 3 yet, if you want to use Python 3 you need to modify the code to make it work.


3. Python packages might missing: cython, opencv-python >= 3.2.0, easydict. If `pip` is set up on your system, those packages should be able to be fetched and installed by running
	```
	pip install -r requirements.txt
	```
4. For Windows users, Visual Studio 2015 is needed to compile cython module.


## Requirements: Hardware

Any NVIDIA GPUs with at least 4GB memory should be OK.

## Installation

1. Clone the Deformable ConvNets repository, and we'll call the directory that you cloned Deformable-ConvNets as ${DCN_ROOT}.

2. For Windows users, run ``cmd .\init.bat``. For Linux user, run `sh ./init.sh`. The scripts will build cython module automatically and create some folders.

3. Install MXNet:
	
	**Note: The MXNet's Custom Op cannot execute parallelly using multi-gpus after this [PR](https://github.com/apache/incubator-mxnet/pull/6928). We strongly suggest the user rollback to version [MXNet@(commit 998378a)](https://github.com/dmlc/mxnet/tree/998378a) for training (following Section 3.2 - 3.5).**

	***Quick start***

	3.1 Install MXNet and all dependencies by 
	```
	pip install -r requirements.txt
	```
	If there is no other error message, MXNet should be installed successfully. 
	
	***Build from source (alternative way)***

	3.2 Clone MXNet and checkout to [MXNet@(commit 998378a)](https://github.com/dmlc/mxnet/tree/998378a) by
	```
	git clone --recursive https://github.com/dmlc/mxnet.git
	git checkout 998378a
	git submodule update
	# if it's the first time to checkout, just use: git submodule update --init --recursive
	```
	3.3 Compile MXNet
	```
	cd ${MXNET_ROOT}
	make -j $(nproc) USE_OPENCV=1 USE_BLAS=openblas USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda USE_CUDNN=1
	```
	3.4 Install the MXNet Python binding by
	
	***Note: If you will actively switch between different versions of MXNet, please follow 3.5 instead of 3.4***
	```
	cd python
	sudo python setup.py install
	```
	3.5 For advanced users, you may put your Python packge into `./external/mxnet/$(YOUR_MXNET_PACKAGE)`, and modify `MXNET_VERSION` in `./experiments/rfcn/cfgs/*.yaml` to `$(YOUR_MXNET_PACKAGE)`. Thus you can switch among different versions of MXNet quickly.

4. For Deeplab, we use the argumented VOC 2012 dataset. The argumented annotations are provided by [SBD](http://home.bharathh.info/pubs/codes/SBD/download.html) dataset. For convenience, we provide the converted PNG annotations and the lists of train/val images, please download them from [OneDrive](https://1drv.ms/u/s!Am-5JzdW2XHzhqMRhVImMI1jRrsxDg).


## Preparation for Training & Testing


1. Please download COCO and VOC 2007+2012 datasets, and make sure it looks like this:

	```
	./data/VOCdevkit/power_plant_dataset/
	```

2. Please download ImageNet-pretrained ResNet-v1-101 model manually from [OneDrive](https://1drv.ms/u/s!Am-5JzdW2XHzhqMEtxf1Ciym8uZ8sg), and put it under folder `./model`. Make sure it looks like this:
	```
	./model/pretrained_model/resnet_v1_101-0000.params
	```


## Usage

1. All of our experiment settings (GPU #, dataset, etc.) are kept in yaml config files at folder `./experiments/rfcn/cfgs`, `./experiments/faster_rcnn/cfgs` and `./experiments/deeplab/cfgs/`.
2. Eight config files have been provided so far, namely, R-FCN for COCO/VOC, Deformable R-FCN for COCO/VOC, Faster R-CNN(2fc) for COCO/VOC, Deformable Faster R-CNN(2fc) for COCO/VOC, Deeplab for Cityscapes/VOC and Deformable Deeplab for Cityscapes/VOC, respectively. We use 8 and 4 GPUs to train models on COCO and on VOC for R-FCN, respectively. For deeplab, we use 4 GPUs for all experiments.

3. To perform experiments, run the python scripts with the corresponding config file as input. For example, to train and test deformable convnets on COCO with ResNet-v1-101, use the following command
    ```
    python experiments\rfcn\rfcn_end2end_train_test.py --cfg experiments\rfcn\cfgs\resnet_v1_101_coco_trainval_rfcn_dcn_end2end_ohem.yaml
    ```
    A cache folder would be created automatically to save the model and the log under `output/rfcn_dcn_coco/`.
4. Please find more details in config files and in our code.

