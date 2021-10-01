#pragma once


template<typename T>
void performFFTAndScale(T* inOutData, int noOfImgs, int inX, int inY,
        int inBatch, int outFFTX, int outY,  T *filter);

template<typename T>
void performFFT(T* inOutData, int noOfImgs, int inX, int inY,
        int inBatch, int outFFTX, int outY,  T *filter);