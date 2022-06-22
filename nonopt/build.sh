#! /bin/bash

# Compile

current_dir=$(pwd)

cd ./port
make clean
export CFLAGS_EXTRA=-ta=multicore
make cpu
unset CFLAGS_EXTRA
make clean_cpu
make
cd $current_dir
