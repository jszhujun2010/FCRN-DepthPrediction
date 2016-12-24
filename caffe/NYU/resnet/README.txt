# Train and Test Guide

## About training
Training is quite simple here. Just Run:
```
python solve.py gpu_id [2>&1 | tee log.txt]
```
Make sure you have pretrianed `ResNet-50-model.caffemodel`
in this folder.


## About testing
I have not combined training and testing process. For testing,
we have to run:
```
python checkcaffeIO_batch.py proto_file model_file 1 testNum path
```
proto_file is just train/test network prototxt.
model_file is model file.
testNum is the number of files to be tested.
path is where you want your testing result be stored.

This will generate a list of files, containg:
1. original image files
2. original label files
3. predicted depth files

This can be used for:
1. visualiaze result(image, gt, prediction tuple)
2. error metrics. e.g. `../data/calcRMS.py`
