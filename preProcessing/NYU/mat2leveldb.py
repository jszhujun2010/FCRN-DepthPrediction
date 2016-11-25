#!/home/zhujun/local/python27/bin/python2.7

import sys
import numpy as np
import leveldb
import caffe
from caffe.proto import caffe_pb2
from skimage import io as io
from skimage import transform as tf
import scipy.io as scio
import h5py


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
train_value = train_value/np.max(label)
test_value = test_value/np.max(label)

#print np.max(train_image), np.min(train_image)
#print np.max(test_image), np.min(test_image)
#print np.max(train_value), np.min(train_value)
#print np.max(test_value), np.min(test_value)

print 'loading data done...'

leveldb_file_train = sys.argv[1]
leveldb_label_file_train = sys.argv[2]
leveldb_file_test = sys.argv[3]
leveldb_label_file_test = sys.argv[4]
batch_size = 1024

db_train = leveldb.LevelDB(leveldb_file_train)
db_l_train = leveldb.LevelDB(leveldb_label_file_train)
batch_train = leveldb.WriteBatch()
batch_l_train = leveldb.WriteBatch()
datum_train = caffe_pb2.Datum()
datum_l_train = caffe_pb2.Datum()

item_id = -1

print 'train_image.shape', train_image.shape

for i in range(0, train_image.shape[0]):
	item_id += 1
	ti = train_image[i,:,:,:]
	ti = ti.transpose((1,2,0))
	ti = tf.rescale(ti, 0.5, order=0)
	ti = ti[8:312,7:235,::-1]
	ti = ti.transpose((2,0,1))

	te = train_value[i,:,:]
	te = tf.rescale(te, 0.5, order=0)
	te = te[8:312,7:235]
	te = tf.resize(te, (160, 128), order=0)
	te = np.expand_dims(te, axis=0)
	
	datum = caffe.io.array_to_datum(ti)
        datum_l = caffe.io.array_to_datum(te)
        keystr = '{:0>8d}'.format(item_id)
        batch_train.Put( keystr, datum.SerializeToString() )
        batch_l_train.Put( keystr, datum_l.SerializeToString() )

        if(item_id + 1) % batch_size == 0:
        	db_train.Write(batch_train, sync=True)
                db_l_train.Write(batch_l_train, sync=True)
                batch_train = leveldb.WriteBatch()
                batch_l_train = leveldb.WriteBatch()
                print 'a', (item_id + 1)
	

if (item_id+1) % batch_size != 0:
	db_train.Write(batch_train, sync=True)
        db_l_train.Write(batch_l_train, sync=True)
        print 'train last batch'
        print (item_id + 1)


db_test = leveldb.LevelDB(leveldb_file_test)
db_l_test = leveldb.LevelDB(leveldb_label_file_test)
batch_test = leveldb.WriteBatch()
batch_l_test = leveldb.WriteBatch()
datum_test = caffe_pb2.Datum()
datum_l_test = caffe_pb2.Datum()

item_id = -1

for i in range(0, test_image.shape[0]):
	item_id += 1
	ti = test_image[i,:,:,:]
	ti = ti.transpose((1,2,0))
	ti = tf.rescale(ti, 0.5, order=0)
	ti = ti[8:312,7:235,::-1]
	ti = ti.transpose((2,0,1))

	te = test_value[i,:,:]
	te = tf.rescale(te, 0.5, order=0)
	te = te[8:312,7:235]
	te = tf.resize(te, (160, 128), order=0)
	te = np.expand_dims(te, axis=0)

	datum = caffe.io.array_to_datum(ti)
        datum_l = caffe.io.array_to_datum(te)
        keystr = '{:0>8d}'.format(item_id)
        batch_test.Put( keystr, datum.SerializeToString() )
        batch_l_test.Put( keystr, datum_l.SerializeToString() )

        if(item_id + 1) % batch_size == 0:
        	db_test.Write(batch_test, sync=True)
                db_l_test.Write(batch_l_test, sync=True)
                batch_test = leveldb.WriteBatch()
                batch_l_test = leveldb.WriteBatch()
                print (item_id + 1)


if (item_id+1) % batch_size != 0:
	db_test.Write(batch_test, sync=True)
        db_l_test.Write(batch_l_test, sync=True)
        print 'test last batch'
        print (item_id + 1)
