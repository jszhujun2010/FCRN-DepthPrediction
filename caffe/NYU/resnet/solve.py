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

# surgeries
#interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
#print interp_layers
#surgery.interp(solver.net, interp_layers)

# scoring
test = np.loadtxt('../data/train.txt', dtype=str)

for _ in range(150):
    solver.step(2000)
    # N.B. metrics on the semantic labels are off b.c. of missing classes;
    # score manually from the histogram instead for proper evaluation
    #score.seg_tests(solver, False, test, layer='last_conv', gt='label')
