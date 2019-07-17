#include "core/xmipp_macros.h"
#include "assert.h"

#include "cuda_gpu_multidim_array.cu"

template<typename T, int degree, bool wrap>
__global__
void applyGeometryKernel_2D_wrap(const T* trInv, T minxpp, T maxxpp, T minypp,
        T maxypp, T minxp, T maxxp, T minyp, T maxyp, T* data, int xdim,
        int ydim, T* coefs, int coefsXDim, int coefsYDim) {
    // assign output pixel to thread
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j >= xdim || i >= ydim)
        return;

    // Calculate this position in the input image according to the
    // geometrical transformation
    // they are related by
    // coords_output(=x,y) = A * coords_input (=xp,yp)
    T xp = j * trInv[0] + i * trInv[1] + trInv[2];
    T yp = j * trInv[3] + i * trInv[4] + trInv[5];

    if (wrap) {
        bool x_isOut = XMIPP_RANGE_OUTSIDE_FAST(xp, minxpp, maxxpp);
        bool y_isOut = XMIPP_RANGE_OUTSIDE_FAST(yp, minypp, maxypp);

        if (x_isOut) {
            xp = realWRAP(xp, minxp - 0.5, maxxp + 0.5); // FIXME specialize for float/double
        }

        if (y_isOut) {
            yp = realWRAP(yp, minyp - 0.5, maxyp + 0.5);
        }

        switch (degree) {
        case 0:
        case 1:
        case 2:
            assert("degree 0..2 not implemented");
            break;
        case 3: {
			T res = interpolatedElementBSpline2D_Degree3(xp, yp, coefsXDim,
                    coefsYDim, coefs);
            size_t index = i * xdim + j;
            data[index] = res;
        }
            break;
        default:
            printf("Degree %d is not supported\n", degree);
        }
    } else {
        assert("non-wrap not implemented");
    }
}

// FIXME unify with C++ implementation
template<typename T>
__device__
void getShift(int lX, int lY, int lN, int xdim, int ydim, int ndim, int x,
        int y, int curFrame, T &shiftY, T &shiftX, const T* coefsX,
        const T* coefsY) {
    T delta = 0.0001;
    // take into account end poits
    T hX = (lX == 3) ? xdim : (xdim / (T) ((lX - 3)));
    T hY = (lY == 3) ? ydim : (ydim / (T) ((lY - 3)));
    T hT = (lN == 3) ? ndim : (ndim / (T) ((lN - 3)));
    // index of the 'cell' where pixel is located (<0, N-3> for N control points)
    T xPos = x / hX;
    T yPos = y / hY;
    T tPos = curFrame / hT;
    // indices of the control points are from -1 .. N-2 for N points
    // pixel in 'cell' 0 may be influenced by points with indices <-1,2>
    for (int idxT = max(-1, (int) (tPos) - 1);
            idxT <= min((int) (tPos) + 2, lN - 2); ++idxT) {
        T tmpT = bspline03(tPos - idxT);
        for (int idxY = max(-1, (int) (yPos) - 1);
                idxY <= min((int) (yPos) + 2, lY - 2); ++idxY) {
            T tmpY = bspline03(yPos - idxY);
            for (int idxX = max(-1, (int) (xPos) - 1);
                    idxX <= min((int) (xPos) + 2, lX - 2); ++idxX) {
                T tmpX = bspline03(xPos - idxX);
                T tmp = tmpX * tmpY * tmpT;
                if (fabsf(tmp) > delta) {
                    size_t coeffOffset = (idxT + 1) * (lX * lY)
                            + (idxY + 1) * lX + (idxX + 1);
                    shiftX += coefsX[coeffOffset] * tmp;
                    shiftY += coefsY[coeffOffset] * tmp;
                }
            }
        }
    }
}

template<typename T, int degree>
__global__
void applyLocalShiftGeometryKernel(const T* coefsX, const T *coefsY,
	T* output, int xdim, int ydim, int ndim,
	T* input, int curFrame,
	int lX, int lY, int lN) { // number of control points in each dim
    // assign output pixel to thread
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x >= xdim || y >= ydim)
        return;

    // Calculate this position in the input image according to the
    // geometrical transformation
	T shiftX = 0;
	T shiftY = 0;
    getShift(lX, lY, lN, xdim, ydim, ndim, x, y, curFrame, shiftY, shiftX,
            coefsX, coefsY);

	switch (degree) {
        case 0:
        case 1:
        case 2:
            assert("degree 0..2 not implemented");
            break;
        case 3: {
            T res = interpolatedElementBSpline2D_Degree3New(x, y, shiftX, shiftY, xdim, ydim, input);
		    size_t index = y * xdim + x;
		    output[index] = res;
	    }
            break;
        default:
            printf("Degree %d is not supported\n", degree);
        }
}

template<typename T, int pixels_per_thread>
__device__
void getShiftMorePixels(int lX, int lY, int lN, int x,
        int y, T* __restrict__ shiftY, T* __restrict__ shiftX, const T* __restrict__ coefsX,
        const T* __restrict__ coefsY, T hX, T hY, T tPos) {
    T imax = 1.5; // inverted maximum value of bspline03 function

    T delta = 0.0001;
    T deltaX = delta * imax;  //0.00015000015
    T deltaT = deltaX * imax; //0.00022500045

    // index of the 'cell' where pixel is located (<0, N-3> for N control points)
    T xPos = x / hX;

    int tEnd = min((int) (tPos) + 2, lN - 2);
    int xEnd = min((int) (xPos) + 2, lX - 2);

    // indices of the control points are from -1 .. N-2 for N points
    // pixel in 'cell' 0 may be influenced by points with indices <-1,2>
    // for loops in different order
    for (int idxT = (int) (tPos) - 1; idxT <= tEnd; ++idxT) {
        T tmpT = bspline03(tPos - idxT);

        if (tmpT < deltaT) {
            continue;
        }

         for (int idxX = (int) (xPos) - 1; idxX <= xEnd; ++idxX) {
            T tmpX = bspline03(xPos - idxX) * tmpT;

            if (tmpX < deltaX) {
                continue;
            }

            for (int i = 0; i < pixels_per_thread; ++i) {
                T yPos = (y + i) / hY;
                int yEnd = min((int) (yPos) + 2, lY - 2);
                for (int idxY = (int) (yPos) - 1; idxY <= yEnd; ++idxY) {
                    T tmp = bspline03(yPos - idxY) * tmpX;

                    if (tmp > delta) {
                        int coeffOffset = (idxT + 1) * lX * lY
                                + (idxY + 1) * lX + (idxX + 1);
                        shiftX[i] += coefsX[coeffOffset] * tmp;
                        shiftY[i] += coefsY[coeffOffset] * tmp;
                    }
                }
            }
        }
    }
}

__device__
bool isEdge(int x, int y, int xdim, int ydim, int edge_dist = 32) {
    return (x < edge_dist) || (x >= xdim - edge_dist) || (y < edge_dist) || (y >= ydim - edge_dist);
}

/*
 * One thread computes more pixels in its column
*/
template<typename T, int degree, int pixels_per_thread>
__global__
void applyLocalShiftGeometryKernelMorePixels(const T* __restrict__ coefsX, const T * __restrict__ coefsY,
    T* __restrict__ output, int xdim, int ydim, int ndim,
    const T* __restrict__ input, int curFrame,
    int lX, int lY, int lN, // number of control points in each dim
    T hX, T hY, T tPos) {
    // assign output pixel to thread
    int y = pixels_per_thread * (blockIdx.y * blockDim.y + threadIdx.y);
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x >= xdim || y >= ydim)
        return;

    // Calculate this position in the input image according to the
    // geometrical transformation
    T shiftX[pixels_per_thread] = { 0 };
    T shiftY[pixels_per_thread] = { 0 };
    getShiftMorePixels<T, pixels_per_thread>(lX, lY, lN, x, y, shiftY, shiftX,
            coefsX, coefsY, hX, hY, tPos);

    switch (degree) {
        case 0:
        case 1:
        case 2:
            assert("degree 0..2 not implemented");
            break;
        case 3: {
            #pragma unroll
            for (int i = 0; i < pixels_per_thread; ++i) {
                if ( y + i >= ydim ) {
                    continue;
                }
                T res;
                if ( isEdge( x - shiftX[i], y + i - shiftY[i], xdim, ydim, 32 ) ) {
                    res = interpolatedElementBSpline2D_Degree3MorePixelsEdge< T >(x, y + i, shiftX[i], shiftY[i], xdim,
                                ydim, input);
                } else {
                    res = interpolatedElementBSpline2D_Degree3MorePixelsInner< T >(x, y + i, shiftX[i], shiftY[i], xdim,
                                ydim, input);
                }
                size_t index = (y + i) * xdim + x;
                output[index] = res;
            }
        }
            break;
        default:
            printf("Degree %d is not supported\n", degree);
        }
}