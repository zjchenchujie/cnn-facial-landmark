#!/bin/bash
GPU_ID=0
cur_date=`date +%Y-%m-%d-%H-%M-%S`
log_file_name=./log/${cur_date}
/home/aia1/ccj/caffe-master/build/tools/caffe train \
    -solver ./solver.prototxt \
    -gpu $GPU_ID 2>&1 | tee -a ${log_file_name}
