// #include "cuda_utils.h"
#include "cuda_asserts.h"
#include "cuda_basic_math.h"

namespace iirConvolve2D_Cardinal_BSpline_3_MirrorOffBoundKernels {

const int WARP_SIZE = 32;

/*
    Changing these parameters can affect speed on different devices
    Values in curly brackets are sets of allowed values that can
    be used
    BLOCK_SIZE = Xdim and Ydim of block in row pass
    COL_BLOCK_SIZe = Xdim of block in column pass
*/

const int BLOCK_SIZE = 64; // { 16, 32, 64 }
const int COL_BLOCK_SIZE = 128; // { 32, 64, 128, 256, 512, 1024 } ... above 128 only if it is multiple of dim size
const int UNROLL = 8;

template< typename T >
using shared_type = T[BLOCK_SIZE][BLOCK_SIZE+1];
template< typename T >
using data_ptr = T * __restrict__;
template< typename T >
using warp_shared_type = volatile T[WARP_SIZE];

template< typename T, typename IndexFunc >
__device__ void load_warp(data_ptr< T > in, warp_shared_type< T > sdata, IndexFunc f) {
    const T z = sqrt(3.0) - 2.0;
    const T z_0 = (1.0 + z) / z;

    int tid = threadIdx.x;

    if (tid == 0) {
        sdata[tid] = in[f(tid)] * z_0;
    } else {
        sdata[tid] = in[f(tid)] * power(z, tid);
    }
}

template< typename T >
__device__ void sum_warp(data_ptr< T > in, warp_shared_type< T > sdata, int offset) {
    const int tid = threadIdx.x;

    if (tid < 16) sdata[tid] += sdata[tid + 16];
    if (tid < 8)  sdata[tid] += sdata[tid + 8];
    if (tid < 4)  sdata[tid] += sdata[tid + 4];
    if (tid < 2)  sdata[tid] += sdata[tid + 2];

    if (tid == 0) {
        in[offset] = sdata[0] + sdata[1];
    }
}

/*
    For each row computes sum of the row weighted by some numbers,
    because of limited precision of float/double, it is done only for
    first 32 columns
    Saves sum to first column in the row
*/
template< typename T >
__global__ void sum_rows(data_ptr< T > in, int x) {
    int row = blockIdx.x * x;
    __shared__ warp_shared_type< T > sdata;

    load_warp(in, sdata, [row](int tid) { return row + tid; });
    sum_warp(in, sdata, row);
}

/*
    Alternative of sum_rows
    Saves sum to first row in the column
*/
template< typename T >
__global__ void sum_columns(data_ptr< T > in, int x) {
    const int col = blockIdx.x;
    __shared__ warp_shared_type< T > sdata;

    load_warp(in, sdata, [col, x](int tid) { return col + tid * x; });
    sum_warp(in, sdata, col);
}

template< typename T, int shift, bool is_last >
__device__ void load_to_shared(data_ptr< T > in, shared_type< T > sdata, int x, int block_col) {
    const T gain = shift ? 6.0 : 1.0;
    const int tid_x = threadIdx.x;
    const int block_row = blockIdx.x * blockDim. x * x;

    for (int j = 0; j < BLOCK_SIZE; ++j) {
        if (is_last && (block_col + tid_x >= x)) {
            break;
        }
        sdata[j][tid_x + shift] = in[block_row + block_col + x * j + tid_x] * gain;
    }
    __syncthreads();
}

template< typename T, int shift, bool is_last >
__device__ void store_to_global(data_ptr< T > in, shared_type< T > sdata, int x, int block_col) {
    const int tid_x = threadIdx.x;
    const int block_row = blockIdx.x * blockDim. x * x;

    for (int j = 0; j < BLOCK_SIZE; ++j) {
        if (is_last && (block_col + tid_x >= x)) {
            break;
        }
        in[block_row + block_col + x * j + tid_x] = sdata[j][tid_x + shift];
    }
    __syncthreads();
}

template< typename T >
__device__ void forward_pass(data_ptr< T > in, shared_type< T > sdata, int x) {
    const T z = sqrt(3.0) - 2.0;
    const int tid_x = threadIdx.x;

    // i = 0
    {
        load_to_shared<T, 1, false>(in, sdata, x, 0);

        sdata[tid_x][1] *= z;
        __syncthreads();

        // index 0 is not used, no previous block
        for (int j = 2; j <= BLOCK_SIZE; ++j) {
            sdata[tid_x][j] += z * sdata[tid_x][j - 1];
        }
        __syncthreads();

        store_to_global<T, 1, false>(in, sdata, x, 0);
    }

    // i > 0
    int i = 1;
    for (; i < x / BLOCK_SIZE; ++i) {
        const int block_col = i * BLOCK_SIZE;
        // copy last column from previous iteration
        sdata[tid_x][0] = sdata[tid_x][BLOCK_SIZE];
        __syncthreads();

        load_to_shared<T, 1, false>(in, sdata, x, block_col);

        for (int j = 1; j <= BLOCK_SIZE; ++j) {
            sdata[tid_x][j] += z * sdata[tid_x][j - 1];
        }
        __syncthreads();

        store_to_global<T, 1, false>(in, sdata, x, block_col);
    }

    if (i * BLOCK_SIZE == x) {
        return;
    }

    sdata[tid_x][0] = sdata[tid_x][BLOCK_SIZE];
    __syncthreads();

    const int block_col = i * BLOCK_SIZE;

    load_to_shared<T, 1, true>(in, sdata, x, block_col);

    for (int j = 1; j <= BLOCK_SIZE; ++j) {
        sdata[tid_x][j] += z * sdata[tid_x][j - 1];
    }
    __syncthreads();

    store_to_global<T, 1, true>(in, sdata, x, block_col);
}

template< typename T >
__device__ void backward_pass(data_ptr< T > in, shared_type< T > sdata, int x) {
    const T z = sqrt(3.0) - 2.0;
    const T k = z / (z - 1.0);
    const int tid_x = threadIdx.x;
    const int block_row = blockIdx.x * blockDim. x * x;

    in[block_row + x * tid_x + x - 1] *= k;
    __syncthreads();

    int i = x / BLOCK_SIZE;
    bool is_first_block = true;

    if ( i * BLOCK_SIZE != x ) {
        is_first_block = false;
        const int block_col = i * BLOCK_SIZE;

        load_to_shared<T, 0, true>(in, sdata, x, block_col);

        for (int j = BLOCK_SIZE - 1; j >= 0; --j) {
            if (block_col + j < x - 1) {
                sdata[tid_x][j] = z * (sdata[tid_x][j + 1] - sdata[tid_x][j]);
            }
        }
        __syncthreads();

        store_to_global<T, 0, true>(in, sdata, x, block_col);
    }


    for (int i = x / BLOCK_SIZE - 1; i >= 0; --i) {
            const int block_col = i * BLOCK_SIZE;

            if (!is_first_block) {
                sdata[tid_x][BLOCK_SIZE] = sdata[tid_x][0];
                __syncthreads();
            }

            load_to_shared<T, 0, false>(in, sdata, x, block_col);

            for (int j = BLOCK_SIZE - 1; j >= 0; --j) {
                if ( block_col + j < x - 1)
                sdata[tid_x][j] = z * (sdata[tid_x][j + 1] - sdata[tid_x][j]);
            }
            __syncthreads();

            store_to_global<T, 0, false>(in, sdata, x, block_col);

            is_first_block = false;
    }
}

template< typename T >
__global__ void rows_iterate(data_ptr< T > in, int x) {
    __shared__ shared_type< T > sdata;

    forward_pass(in, sdata, x);
    backward_pass(in, sdata, x);
}

template< typename T >
__global__ void cols_iterate(data_ptr< T > in, int x, int y) {
    const T z = sqrt(3.0) - 2.0;
    const T k = z / (z - 1.0);
    const T gain = 6.0;

    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    T *myCol = in + col;

    if (col < x) {

        myCol[0] *= z * gain;
        #pragma unroll UNROLL
        for (int j = 1; j < y; ++j) {
            myCol[j*x] = myCol[j*x] * gain + z * myCol[(j - 1)*x];
        }

        myCol[(y - 1)*x] *= k;
        #pragma unroll UNROLL
        for (int j = y - 2; 0 <= j; --j) {
            myCol[j*x] = z * (myCol[(j + 1)*x] - myCol[j*x]);
        }
    }
}

/*
 * Faster when compiled with fixed `x` and `y`
 * speed on GeForce GTX 1070 on size 8192x8192:
    a) fixed size: 6100 MPix/s
    b) variable size: 5100 MPix/s
*/
template< typename T >
void solveGPU(data_ptr< T > in, int x, int y) {

    const int x_padded = (x / COL_BLOCK_SIZE) * COL_BLOCK_SIZE + COL_BLOCK_SIZE * (x % COL_BLOCK_SIZE != 0);
    const int y_padded = (y / BLOCK_SIZE) * BLOCK_SIZE + BLOCK_SIZE * (y % BLOCK_SIZE != 0);

    sum_rows<<<y, WARP_SIZE>>>(in, x);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    rows_iterate<<<y_padded / BLOCK_SIZE, BLOCK_SIZE>>>(in, x);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    sum_columns<<<x, WARP_SIZE>>>(in, x);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    cols_iterate<<<x_padded / COL_BLOCK_SIZE, COL_BLOCK_SIZE>>>(in, x, y);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
}

} // namespace iirConvolve2D_Cardinal_BSpline_3_MirrorOffBoundKernels

void iirConvolve2D_Cardinal_Bspline_3_MirrorOffBoundInplace(float* input,
                int xDim, int yDim) {
    iirConvolve2D_Cardinal_BSpline_3_MirrorOffBoundKernels::solveGPU( input, xDim, yDim );
}

void iirConvolve2D_Cardinal_Bspline_3_MirrorOffBoundInplace(double* input,
                int xDim, int yDim) {
    iirConvolve2D_Cardinal_BSpline_3_MirrorOffBoundKernels::solveGPU( input, xDim, yDim );
}

