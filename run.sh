#!/bin/sh -x
export DIR=tmp
export VER=Umpalumpa
mkdir -p $DIR
. ./build/xmipp.bashrc
(time xmipp_cuda_flexalign -i phantom.mrc -o $DIR/shifts$VER.xmd --oavg $DIR/mic$VER.mrc --processLocalShifts --skipAutotuning) | tee $DIR/$VER.log
xsj $DIR/mic$VER.mrc
