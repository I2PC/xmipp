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
        case 3:
            T res = interpolatedElementBSpline2D_Degree3(xp, yp, coefsXDim,
                    coefsYDim, coefs);
            size_t index = i * xdim + j;
            data[index] = res;
            break;
        default:
            printf("Degree %d is not supported\n", degree);
        }
    } else {
        assert("non-wrap not implemented");
    }
}
