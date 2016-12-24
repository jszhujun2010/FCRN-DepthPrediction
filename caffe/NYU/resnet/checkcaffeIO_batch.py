#!/data1/zhujun_data/local/python27/bin/python2.7

import caffe
import skimage.io as io
import sys
import numpy as np
import os
import scipy

def normalize(score):
	'''
	Normalize score matrix.
	Not used now.
	'''
	min = np.min(score)
	max = np.max(score)
	return (score-min)/(max-min)

def checkInput(net, path, i):
	'''
	input args:
		net: caffe model net
		path: target saving location
		i: test index

	return: None

	function: save images and labels from net(to check 
	whether the network is working properly). 
	'''
	image = net.blobs['train_data'].data
        image = image[0,:,:].transpose((1,2,0))
        mean = np.array((103.939, 116.779, 123.68), dtype=np.float32)
        image += mean
        image = image.astype(np.uint8)
	image_name = 'test_image_{}.png'.format(i)
        io.imsave(os.path.join(path,image_name), image[:,:,::-1])
        label = net.blobs['train_label'].data
        label = label[0,0,:,:]
	label_name = 'test_label_{}.mat'.format(i)
        io.imsave(os.path.join(path, label_name), {'label': label})

def checkOutput(net, path, i):
	'''
	input args:
                net: caffe model net
                path: target saving location
                i: test index

        return: None

	function: save images and labels and predicted scores 
	from net(for error metrics and visulization--combined
	with matlab maybe).
	'''
	image = net.blobs['train_data'].data
        image = image[0,:,:].transpose((1,2,0))
        mean = np.array((103.939, 116.779, 123.68), dtype=np.float32)
        image += mean
        image = image.astype(np.uint8)
        image_name = 'test_image_{}.png'.format(i)
        io.imsave(os.path.join(path, image_name), image[:,:,::-1])
        label = net.blobs['train_label'].data
        label = label[0,0,:,:]
        label_name = 'test_label_{}.mat'.format(i)
        scipy.io.savemat(os.path.join(path, label_name), {'label': label})
        score = net.blobs['last_conv'].data
        score = score[0,0,:,:]
        #print score
        #score = normalize(score)
	score_name = 'test_score_{}.png'.format(i)
        scipy.io.savemat(os.path.join(path, score_name), {'label': score})


def checkInternal(net, path, i):
	'''
	to be implemented...
	'''
	#layername = sys.argv[4]
        #out = net.blobs[layername].data
        #print out[0,0,:,:]
        #xx = np.transpose(out, (0,2,3,1))
        #io.imsave(os.join('y0.png'), out[0,0,:,:])
        #io.imsave('y1.png', out[1,0,:,:])
	pass



def main():
	test_proto = sys.argv[1]
	model_file = sys.argv[2]
	'''
	mode 0: check input
	mode 1: check output
	mode 2: check internal
	'''
	mode = int(sys.argv[3])
	func = [checkInput, checkOutput, checkInternal]

	testNum = int(sys.argv[4])

	net = caffe.Net(test_proto, model_file, caffe.TEST)
	path = sys.argv[5]
	if not os.path.exists(path):
		os.mkdir(path)	

	loss = 0.0
	for i in range(testNum):
		out = net.forward()
		print out
		loss += out['loss']
		func[mode](net, path, i)
	print '>>> avergae loss: ', loss/testNum

if __name__ == '__main__':
	main()

