# DSSD : Deconvolutional Single Shot Detector

*=Equal Contribution

### Introduction

Deconvolutional SSD brings additional context into state-of-the-art general object detection by adding extra deconvolution structures. The DSSD achieves much better accuracy on small objects compared to SSD.

The code is based on [SSD](https://github.com/weiliu89/caffe/tree/ssd). For more details, please refer to our [arXiv paper](https://arxiv.org/abs/1701.06659). 


### Installation
1. Download the code from github. We call this directory as `$CAFFE_ROOT` later.

	```Shell
	git clone https://github.com/chengyangfu/caffe.git
	cd $CAFFE_ROOT
	git checkout dssd
	```
	
2. Build the code. Please follow [Caffe instruction](http://caffe.berkeleyvision.org/installation.html) to install all necessary packages and build it.

	```Shell
  	# Modify Makefile.config according to your Caffe installation.
  	cp Makefile.config.example Makefile.config
  	make -j8
  	# Make sure to include $CAFFE_ROOT/python to your PYTHONPATH.
  	make py
  	make test -j8
  	# (Optional)
  	make runtest -j
  	```

### Preparation
1.  Please Follow the Orginal [SSD](https://github.com/weiliu89/caffe/tree/ssd) to do all the preparation works. You should have lmdb fils for VOC2007. Check the following two links exist or not. 
   
   	```Shell
   	ls $CAFFE_ROOT/examples
   	# $CAFFE_ROOT/examples/VOC0712/VOC0712_trainval_lmdb
   	# $CAFFE_ROOT/examples/VOC0712/VOC0712_test_lmdb
   	```
   
2.  Download the Resnet-101 models from the [Deep-Residual-Network](https://github.com/KaimingHe/deep-residual-networks).
    
	```Shell
	# creat the directory for ResNet-101
	cd $CAFFE_ROOT/models
	mkdir ResNet-101
	# Rename the Resnet-101 models and put in the ResNet-101 direcotry
	ls $CAFFE_ROOT/models/ResNet-101
	# $CAFFE_ROOT/models/ResNet-101/ResNet-101-model.caffemodel
	# $CAFFE_ROOT/models/ResNet-101/ResNet-101-deploy.prototxt
	```

### Train/Eval
1. Train and Eval the SSD model 

	```Shell
	# Train the SSD-ResNet-101 321x321
	python examples/ssd/ssd_pascal_resnet_321.py
	# GPU setting may need be change according to the numbers of gpu 
	# models are generated in:
	# $CAFFE_ROOT/models/ResNet-101/VOC0712/SSD_VOC07_321x321
	# Evaluate the model
	cd $CAFFE_ROOT
	./build/tools/caffe train --solver="./models/ResNet-101/VOC0712/SSD_VOC07_321x321/test_solver.prototxt"  \
	--weights="./models/ResNet-101/VOC0712/SSD_VOC07_321x321/ResNet-101_VOC0712_SSD_VOC07_321x321_iter_80000.caffemodel" \
	--gpu=0
	# batch size in the test.prototxt may need be changed.
	# If the batch size is changed, remeber to change the test_iter in test_solver.prototxt at same time. 
	# It should reach 77.5* mAP at 80k iterations.
	```
   
2. Train and Evaluate the DSSD model. In this script, Resnet-101 and SSD related layers are frozen and only the DSSD related layers are trained.

	```Shell
	# Use the SSD-ResNet-101 321x321 as the pretrained model
	python examples/ssd/ssd_pascal_resnet_deconv_321.py
	# Evaluate the model
	cd $CAFFE_ROOT
	./build/tools/caffe train --solver="./models/ResNet-101/VOC0712/DSSD_VOC07_321x321/test_solver.prototxt"  \
	--weights="./models/ResNet-101/VOC0712/DSSD_VOC07_321x321/ResNet-101_VOC0712_DSSD_VOC07_321x321_iter_30000.caffemodel" \
	--gpu=0
	# It should reach 78.6* mAP at 30k iterations.
	```
	
3. Train and Evalthe DSSD model. In this script, we try to fine-tune the entire network. In order to sucessfully finetune the network, we need to freeze all the batch norm related layers in Caffe.

	```Shell
	# Use the DSSD-ResNet-101 321x321 as the pretrained model
	python examples/ssd/ssd_pascal_resnet_deconv_ft_321.py
	# Evaluate the model
	cd $CAFFE_ROOT
	./build/tools/caffe train --solver="./models/ResNet-101/VOC0712/DSSD_VOC07_FT_321x321/test_solver.prototxt"  \
	--weights="./models/ResNet-101/VOC0712/DSSD_VOC07_FT_321x321/ResNet-101_VOC0712_DSSD_VOC07_FT_321x321_iter_40000.caffemodel" \
	--gpu=0
	# Finetuning the entire network only works for the model with 513x513 inputs not 321x321. 
	```
  

     		
	
