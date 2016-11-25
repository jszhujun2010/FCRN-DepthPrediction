#!/home/zhujun/python27/bin/python

import leveldb
import numpy as np
import caffe
from caffe.proto import caffe_pb2
import skimage.io as io
import sys

leveldb_file = sys.argv[1]
out_image = sys.argv[2]
mode = int(sys.argv[3])

db = leveldb.LevelDB(leveldb_file)
datum = caffe_pb2.Datum()

i = 0

for key, value in db.RangeIter():
	i += 1
	if i == mode:
		datum.ParseFromString(value)
		data = caffe.io.datum_to_array(datum)
		image = np.transpose(data, (1,2,0))
		print image.shape
		if image.shape[2] == 1:
			image = image[:,:,0]
		io.imsave(out_image, image)
		break
