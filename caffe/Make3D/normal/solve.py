import caffe
import testRes

import numpy as np
import os
import sys
import setproctitle
setproctitle.setproctitle(os.path.basename(os.getcwd()))

weights = 'ResNet-50-model.caffemodel'

# init
caffe.set_device(int(sys.argv[1]))
caffe.set_mode_gpu()

solver = caffe.SGDSolver('solver.prototxt')
solver.net.copy_from(weights)

# scoring
test = np.loadtxt('../data/test.txt', dtype=str)
# test result output
path = '../data/testRes'

for _ in range(100):
    solver.step(12091)
    testRes.run_test(solver, test, 'last_conv', False)
testRes.run_test(solver, test, 'last_conv', path)
