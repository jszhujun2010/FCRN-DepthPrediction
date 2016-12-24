import caffe
import surgery, score

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
test = np.loadtxt('../data/train.txt', dtype=str)

for _ in range(150):
    solver.step(2000)
