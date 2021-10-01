#!/bin/sh -x
export DIR=tmp
export VER=Orig
mkdir -p $DIR
rm -rf $DIR/*$VER*
. ./build/xmipp.bashrc
(time xmipp_cuda_movie_alignment_correlation -i phantom.mrc -o $DIR/shifts$VER.xmd --oavg $DIR/mic$VER.mrc --skipAutotuning) | tee $DIR/$VER.log
xsj $DIR/mic$VER.mrc
