// inspired by https://devblogs.nvidia.com/efficient-matrix-transpose-cuda-cc/
// TILE_DIM=32, BLOCK_ROWS=8
// No bank-conflict transpose
// Same as transposeCoalesced except the first tile dimension is padded
// to avoid shared memory bank conflicts.
// can be used to transpose non-square 2D arrays
__global__
void transposeNoBankConflicts32x8(float *odata, const float *idata, int xdim, int ydim) {
    __shared__ float tile[32][32 + 1];
    int tilex = blockIdx.x * 32;
    int tiley = blockIdx.y * 32;
    int x = tilex + threadIdx.x;
    int y = tiley + threadIdx.y;

    for (int j = 0; j < 32; j += 8) {
        int index = (y + j) * xdim + x;
        if (index < (xdim*ydim)) {
            tile[threadIdx.y + j][threadIdx.x] = idata[index];
        }
    }

    __syncthreads();
    x = tiley + threadIdx.x; // transpose tiles
    y = tilex + threadIdx.y; // transpose tiles
    if (x >= ydim) return; // output matrix has y columns
    int maxJ = min(32, xdim - y); // output matrix has x rows
    for (int j = 0; j < maxJ; j += 8) {
        int index = (y+j) * ydim + x;
        odata[index] = tile[threadIdx.x][threadIdx.y + j];
    }
}

__global__
void iirConvolve2D_Cardinal_Bspline_3_MirrorOffBoundNew(float* input,
        float* output, int xDim, int yDim) {
    // assign column to thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // we will process data line by line, but data are stored in
    // columns!
    if (idx >= yDim) return; // only threads with data should continue
    float* line = output + idx; // FIXME rename to sth reasonable

    // adjust gain
    float z = sqrtf(3.f) - 2.f;
    float z1 = 1.0 - z;
    float gain = -(z1 * z1) / z;
    // copy original data
    for (int i = 0; i < xDim; i++) {
        line[i * yDim] = input[(i * yDim) + idx] * gain;
    }
    // prepare some values
    float sum = (line[0] + powf(z, xDim) * line[(xDim - 1) * yDim]) * (1.f + z) / z;
    z1 = z;
    float z2 = powf(z, 2 * xDim - 2);
    float iz = 1.f / z;
    for (int j = 1; j < (xDim - 1); ++j) {
        sum += (z2 + z1) * line[j * yDim];
        z1 *= z;
        z2 *= iz;
    }
    line[0] = sum * z / (1.f - powf(z, 2 * xDim));
    for (int j = 1; j < xDim; ++j) {
        line[j * yDim] += z * line[(j - 1) * yDim];
    }
    line[(xDim - 1) * yDim] *= z / (z - 1.f);
    for (int j = xDim - 2; 0 <= j; --j) {
        line[j * yDim] = z * (line[(j + 1) * yDim] - line[j * yDim]);
    }
}


__global__
void iirConvolve2D_Cardinal_Bspline_3_MirrorOffBound(float* input,
        float* output, size_t xDim, size_t yDim) {
    // assign line to thread
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idy >= yDim) return; // only threads with data should continue
    float* line = output + (idy * xDim);

    // adjust gain
    float z = sqrtf(3.f) - 2.f;
    float z1 = 1.0 - z;
    float gain = -(z1 * z1) / z;

    // copy original data
    for (int i = 0; i < xDim; i++) {
        line[i] = input[i + (idy * xDim)] * gain;
    }
    // prepare some values
    float sum = (line[0] + powf(z, xDim) * line[xDim - 1]) * (1.f + z) / z;
    z1 = z;
    float z2 = powf(z, 2 * xDim - 2);
    float iz = 1.f / z;
    for (int j = 1; j < (xDim - 1); ++j) {
        sum += (z2 + z1) * line[j];
        z1 *= z;
        z2 *= iz;
    }
    line[0] = sum * z / (1.f - powf(z, 2 * xDim));
    for (int j = 1; j < xDim; ++j) {
        line[j] += z * line[j - 1];
    }
    line[xDim - 1] *= z / (z - 1.f);
    for (int j = xDim - 2; 0 <= j; --j) {
        line[j] = z * (line[j + 1] - line[j]);
    }
}
