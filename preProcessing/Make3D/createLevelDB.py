#!/home/zhujun/python27/bin/python

import sys
import numpy as np
import leveldb
import caffe
from caffe.proto import caffe_pb2
from skimage import io as io
from skimage import transform as tf
import os
from os import walk

def processData(path):
	data = io.imread(path)/255.0
	standard = (172, 230, 3)
	#print data.shape
	if data.shape != standard:
		data = tf.resize(data, standard)
#	io.imsave('data1.png', data)
	data = np.swapaxes(np.swapaxes(data, 1, 2), 0, 1)
	return data

def processLabel(path):
	data = io.imread(path)/70.0
	standard = (172, 230)
	if data.shape != standard:
		data = tf.resize(data, standard)
#	io.imsave('label1.png', data)
	data = np.expand_dims(data, axis=0)
	return data


leveldb_file = sys.argv[1]
leveldb_label_file = sys.argv[2]
batch_size = 1024

db = leveldb.LevelDB(leveldb_file)
db_l = leveldb.LevelDB(leveldb_label_file)
batch = leveldb.WriteBatch()
batch_l = leveldb.WriteBatch()
datum = caffe_pb2.Datum()
datum_l = caffe_pb2.Datum()

item_id = -1
data_path = sys.argv[3]
label_path = sys.argv[4]

#data_path = 'test_path_output'
#label_path = 'test_depth_path_output'

for dirpath, dirnames, filenames in walk(data_path):
	for filename in filenames:
		#print filename
		if item_id == int(sys.argv[5])-1:
			break
		item_id += 1
		prefix = '-'.join('.'.join(filename.split('.')[0:-1]).split('-')[1:])
		label_name = 'depth_sph_corr-'+prefix+'.png'
		data_file = os.path.join(dirpath, filename)
		label_file = os.path.join(label_path, label_name)
		#print data_file, label_file
		data = processData(data_file)
		label = processLabel(label_file)
		
		datum = caffe.io.array_to_datum(data)
		datum_l = caffe.io.array_to_datum(label)
		keystr = '{:0>8d}'.format(item_id)
		batch.Put( keystr, datum.SerializeToString() )
		batch_l.Put( keystr, datum_l.SerializeToString() )

		if(item_id + 1) % batch_size == 0:
			db.Write(batch, sync=True)
			db_l.Write(batch_l, sync=True)
			batch = leveldb.WriteBatch()
			batch_l = leveldb.WriteBatch()
			print (item_id + 1)
		
	if (item_id+1) % batch_size != 0:
		db.Write(batch, sync=True)
		db_l.Write(batch_l, sync=True)
		print 'last batch'
		print (item_id + 1)
