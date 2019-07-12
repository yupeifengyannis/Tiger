#!/bin/bash

/home/yupefieng/Documents/dl_framework/Tiger/build/sample/mnist/convert_mnist --image_filename \
    /home/yupefieng/Documents/dl_framework/caffe/data/mnist/train-images-idx3-ubyte \
    --label_filename \
    /home/yupefieng/Documents/dl_framework/caffe/data/mnist/train-labels-idx1-ubyte \
    --db_path /home/yupefieng/Documents/dl_framework/Tiger/test_data/mnist \
    --backend leveldb


