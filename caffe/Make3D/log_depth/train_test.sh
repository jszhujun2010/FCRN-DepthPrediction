#!/bin/bash

python checkcaffeIO_batch.py train.prototxt snapshot/train-_iter_300000.caffemodel 1 12091 ../data/train_log_res
python checkcaffeIO_batch.py test.prototxt snapshot/train-_iter_300000.caffemodel 1 3023 ../data/test_log_res 
