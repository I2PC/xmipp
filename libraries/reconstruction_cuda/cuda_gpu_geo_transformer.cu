#include "core/xmipp_macros.h"
#include "assert.h"

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
    
    T delta = 0.0001;
    T hX = xdim / (T)(lX-1);
    T hY = ydim / (T)(lY-1);
    T hT = ndim / (T)(lN-1);
    
	T shiftX = 0;
	T shiftY = 0;
	// compute influence of each control point
	for (int j = 0; j < (lN+2)*(lY+2)*(lX+2); ++j) {
	    int controlIdxT = j/((lY+2)*(lX+2))-1;
	    int XY=j%((lX+2)*(lY+2));
	    int controlIdxY = (XY/(lX+2)) -1;
	    int controlIdxX = (XY%(lX+2)) -1;
	    // note: if control point is not in the tile vicinity, val == 0 and can be skipped
	    T tmp = bspline03((x / (T)hX) - controlIdxX) *
	            bspline03((y / (T)hY) - controlIdxY) *
	            bspline03((curFrame / (T)hT) - controlIdxT);
	    if (fabsf(tmp) > delta) {
	        size_t coeffOffset = (controlIdxT+1) * (lX+2)*(lY+2) + (controlIdxY+1) * (lX+2) + (controlIdxX+1);
	        shiftX += coefsX[coeffOffset] * tmp;
	        shiftY += coefsY[coeffOffset] * tmp;
	    }
	}
	
	switch (degree) {
        case 0:
        case 1:
        case 2:
            assert("degree 0..2 not implemented");
            break;
        case 3: {
			T res = interpolatedElementBSpline2D_Degree3(x - shiftX, y - shiftY, xdim,
			                    ydim, input);
		    size_t index = y * xdim + x;
		    output[index] = res;
	    }
            break;
        default:
            printf("Degree %d is not supported\n", degree);
        }
}

