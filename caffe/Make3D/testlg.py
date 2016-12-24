from __future__ import division
import caffe
import numpy as np
import os
import sys
from datetime import datetime
from skimage import io 

def test(net, dataset, layer, save):
	loss = 0
	print dataset
	for idx in dataset:
		net.forward()
		if save:
			score = net.blobs[layer].data[0,0,:,:]
			score = score.astype(np.uint8)
			io.imsave(os.path.join(save, 'score_'+idx+'.png'), score)
	print '>>>test done...'


def run_test(solver, dataset, layer, save):
	print '>>>', datetime.now(), 'Begin test...'
	solver.test_nets[0].share_with(solver.net)
	test(solver.test_nets[0], dataset, layer, save)
	

if __name__ == '__main__':
	run_test()
