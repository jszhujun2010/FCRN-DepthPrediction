#!/data1/zhujun_data/local/python27/bin/python2.7

import os
import sys
import numpy as np
import skimage.io as io



def calcRMS(depth_path):
	'''
	args:
		depth_path: path where we put depth map
		and score(depth prediction) files
	
	returns:
		[loss, loss_ignore]
		loss is RMSE, loss_ignore is RMSE but ignore those
		absolute diff value larger than 20.
		Note that 20 is hard coded here, future it can be taken
		as a parameter.

	this can omly be tested after running checkcaffeIO.py
	'''
	loss = 0.0
	loss_ignore = 0.0
	cnt = 0
	for root, dirs, filenames in os.walk(depth_path):
		for filename in filenames:
			train_depth = io.imread(os.path.join(root, filename))
			try:
				prefix = filename.split('label_')[1].split('.')[0]
			except:
				continue
			cnt += 1
			gt_file = os.path.join(depth_path, 'test_score_'+prefix+'.png')
			gt_depth = io.imread(gt_file)
			assert train_depth.shape == gt_depth.shape
			loss += np.sum((train_depth-gt_depth)**2)
			## 20 as a threshold, can be parametered. I'm too lazy to do it...
			ignore = gt_depth[(gt_depth <= 0) | (gt_depth >= 20)]
			ignore_loss = train_depth-gt_depth
			ignore_loss[ignore] = 0.0
			loss_ignore += np.sum(ignore_loss**2)
	print cnt
	div = cnt*160*128.0
	return (np.sqrt(loss/div), np.sqrt(loss_ignore/div))

def main():
	train_score_path = sys.argv[1]
	test_score_path = sys.argv[2]
	print calcRMS(train_score_path)
	print calcRMS(test_score_path)

if __name__ == '__main__':
	main()
