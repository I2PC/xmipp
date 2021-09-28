#pragma once


template<typename T>
void performFFTAndScale(T* inOutData, int noOfImgs, int inX, int inY,
        int inBatch, int outFFTX, int outY,  T *filter);

