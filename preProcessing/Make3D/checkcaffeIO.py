#!/home/zhujun/python27/bin/python

import caffe
import skimage.io as io
import sys

test_proto = sys.argv[1]
model_file = sys.argv[2]
'''
mode 1: check input
mode 2: check output
mode 3: check internal
'''
mode = int(sys.argv[3])

net = caffe.Net(test_proto, model_file, caffe.TEST)
out = net.forward()

if mode == 1:
	image = net.blobs['train_data'].data
	image = image[0,:,:].transpose((1,2,0))
	io.imsave('test_image.png', image)
	label = net.blobs['train_label'].data
	label = label[0,0,:,:]
	io.imsave('test_label.png', label)
elif mode == 2:
	label = net.blobs['train_label'].data
	label = label[0,0,:,:]
	io.imsave('test_label.png', label)
	score = net.blobs['score'].data
	score = score[0,0,:,:]
	io.imsave('test_score.png', score)
else:
	layername = sys.argv[4]
	out = net.blobs[layername].data
	print out

