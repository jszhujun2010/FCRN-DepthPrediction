import caffe
import surgery, score
import testlg

import numpy as np
import os
import sys
import setproctitle
setproctitle.setproctitle(os.path.basename(os.getcwd()))

weights = 'ResNet-50-model.caffemodel'

caffe.set_device(int(sys.argv[1]))
caffe.set_mode_gpu()

solver = caffe.SGDSolver('solver.prototxt')
solver.net.copy_from(weights)


# scoring
train = np.loadtxt('../data/train.txt', dtype=str)
test = np.loadtxt('../data/test.txt', dtype=str)

for _ in range(20):
    solver.step(15091)

train_res = '../data/train_log_res'
test_res = '../data/test_log_res'
if not os.path.exists(train_res):
	os.mkdir(train_res)
if not os.path.exists(test_res):
	os.mkdir(test_res)
testlg.run_test(solver, train, 'last_conv', train_res)
testlg.run_test(solver, test, 'last_conv', test_res)

