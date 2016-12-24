# NYU training and testing guide

## Training and testing
See README.txt in resnet/README.txt for details.

## data related
See README.txt in data/README.txt for details.


## Some Notifications
To use train/test, we need caffe support python layer(
one of the make options in configuration).


Note that this network is not the same as what the article
describes. I added some batch normalizations after each
up-conv layers.

For now, the training MSE is 0.36, testing MSE is 1.13.
The training data is just 795 images(without data agumentation).
