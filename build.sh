#!/bin/bash
set -x
SOURCE_DIR=`pwd`
BUILD_DIR=./build

rm ${BUILD_DIR} -rf

mkdir ${BUILD_DIR} -p\
    && cd ${BUILD_DIR}\
    && cmake \
     -DCMAKE_BUILD_TYPE=Debug\
     ${SOURCE_DIR}\
     && make -j8
