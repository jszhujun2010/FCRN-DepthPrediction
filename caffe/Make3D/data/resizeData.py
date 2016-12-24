#!/home/zhujun/python27/bin/python

import sys
import numpy as np
from skimage import io as io
from skimage import transform as tf
import skimage
import os
from os import walk
import scipy.io as scio

def processData(path):
	data = io.imread(path)
	standard = (230, 173, 3)
	if data.shape != standard:
		data = tf.resize(data, standard)
	data = skimage.img_as_ubyte(data)
	return data


def processLabel_mat(path):
	depth_name_matrix = scio.loadmat(path)
        data = depth_name_matrix['Position3DGrid'][:,:,3]
	standard = (128, 96)
	if data.shape != standard:
		data = tf.resize(data, standard, order=0)
	mask = data.copy()
	mask[(mask >= 0) & (mask <= 70)] = 1
	mask[(mask < 0) | (mask > 70)] = 0
	return [data, mask]

	

data_path = sys.argv[1]
label_path = sys.argv[2]
target_data = sys.argv[3]
target_label = sys.argv[4]
target_mask = sys.argv[5]

if not os.path.exists(target_data):
	os.makedirs(target_data)

if not os.path.exists(target_label):
	os.makedirs(target_label)

if not os.path.exists(target_mask):
	os.makedirs(target_mask)

item_id = 0
for dirpath, dirnames, filenames in walk(data_path):
	for filename in filenames:
		print filename
		item_id += 1
		prefix = '-'.join('.'.join(filename.split('.')[0:-1]).split('-')[1:])
		label_name = 'depth_sph_corr-'+prefix+'.mat'
		data_file = os.path.join(dirpath, filename)
		label_file = os.path.join(label_path, label_name)
		data = processData(data_file)
		label, mask = processLabel_mat(label_file)
		io.imsave(target_data+'/'+str(item_id)+'.png', data)
		scio.savemat(target_label+'/'+str(item_id)+'.mat', {'label':label})
		scio.savemat(target_mask+'/'+str(item_id)+'.mat', {'label':mask})
