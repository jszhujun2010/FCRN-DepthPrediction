#!/data1/zhujun_data/local/python27/bin/python2.7

import sys
import numpy as np
from skimage import io as io
from skimage import transform as tf
import scipy.io as scio
import scipy
import h5py
import os


'''
this file is used to transform original
.mat file to the format we need(I did
image resize as well as paper describes).

we should set data_path(main NYU data)
and split.mat properly.

future it should no longer be hard codes.
'''

def writeData(image_path, label_path, image, label):
	assert(image.shape[0] == label.shape[0])
	for i in range(image.shape[0]):
		print i
		image_ = image[i].transpose((1,2,0))
		image_ = scipy.ndimage.zoom(image_, (320/640.0, 240/480.0, 1), order=0)
		image_ = image_[8:312, 6:234]
		assert(image_.shape == (304, 228, 3))
		io.imsave(os.path.join(image_path, str(i)+'.png'), image_)
		label_ = scipy.ndimage.zoom(label[i], (160/640.0, 128/480.0), order=0)
		scio.savemat(os.path.join(label_path, str(i)+'.mat'), {'label': label_})


data_path = 'nyu_depth_v2_labeled.mat'
split_path = 'splits.mat'
data = h5py.File(data_path)
split = scio.loadmat(split_path)

print 'loading data...'

image = data['images']
label = data['depths']
image = image[:]
label = label[:]

print 'image.shape: ', image.shape
print 'label.shape', label.shape

x = split['trainNdxs']
y = split['testNdxs']
x = x.flatten()
y = y.flatten()
x = x-1
y = y-1
train_image = image[x,:,:,:]
test_image = image[y,:,:,:]
train_value = label[x,:,:]
test_value = label[y,:,:]
#train_value = train_value/np.max(label)
#test_value = test_value/np.max(label)

print np.max(train_image), np.min(train_image)
print np.max(test_image), np.min(test_image)
print np.max(train_value), np.min(train_value)
print np.max(test_value), np.min(test_value)

print 'loading data done...'


train_image_path = 'train_image'
train_label_path = 'train_label'
test_image_path = 'test_image'
test_label_path = 'test_label'

paths = [train_image_path, train_label_path, test_image_path, test_label_path]
for path in paths:
	if not os.path.exists(path):
		os.mkdir(path)

print 'writing training data'
writeData(train_image_path, train_label_path, train_image, train_value)
print 'writing testing data'
writeData(test_image_path, test_label_path, test_image, test_value)
