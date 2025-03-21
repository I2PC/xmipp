/*
 * Author: Martin Pernica, Masaryk University
 */

#define PI 3.14159265358979323846

__device__ double idx2freq(int idx, int size) {
    if (size <= 1) {
        return 0;
    }
    if (idx <= size / 2.0) {
        return (__int2double_rn(idx)) / (__int2double_rn(size));
    }
    return (__int2double_rn(idx - size)) / (__int2double_rn(size));
}

__global__ void projectKernel(double *cudaProjectionFourier,  float *cudaVfourierRealCoefs,
                               float *cudaVfourierImagCoefs,
                              double *cudaE, double *cudaCtf, bool useCtf,
                              int coefsSize, int volumeSize, int imageSizex, int imageSizey, int volumePaddedSize,
                              double maxFreq2,
                              int xinit, int yinit, int zinit, int work, double *cudaPhaseA, double *cudaPhaseB) {
    int i = blockIdx.y * blockDim.y * work + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    for (int w = 0; w < work; ++w) {
        if (i < imageSizey && j < imageSizex) {

            double freqy = idx2freq(i, volumeSize);
            double freqx = idx2freq(j, volumeSize);
            if ((freqy * freqy + freqx * freqx) <= maxFreq2) {
                double c = 0.0;
                double d = 0.0;

                // B-spline cubic interpolation
                double z = (cudaE[1 * 3 + 2] * freqy + cudaE[0 * 3 + 2] * freqx) * volumePaddedSize - zinit;
                double y = (cudaE[1 * 3 + 1] * freqy + cudaE[0 * 3 + 1] * freqx) * volumePaddedSize - yinit;
                double x = (cudaE[1 * 3 + 0] * freqy + cudaE[0 * 3 + 0] * freqx) * volumePaddedSize - xinit;

                int l1 = (int) ceil(x - 2);

                int m1 = (int) ceil(y - 2);

                int n1 = (int) ceil(z - 2);

                double aux;
                double yxsumRe;
                double yxsumIm;
                double xsumRe;
                double xsumIm;
                for (int nn = n1; nn <= n1 + 3; nn++) {
                    int equivalent_nn = nn;
                    if (nn < 0)
                        equivalent_nn = -nn - 1;
                    else if (nn >= coefsSize)
                        equivalent_nn = 2 * coefsSize - nn - 1;
                    yxsumRe = 0.0;
                    yxsumIm = 0.0;
                    for (int m = m1; m <= m1 + 3; m++) {
                        int equivalent_m = m;
                        if (m < 0)
                            equivalent_m = -m - 1;
                        else if (m >= coefsSize)
                            equivalent_m = 2 * coefsSize - m - 1;
                        xsumRe = 0.0;
                        xsumIm = 0.0;
                        for (int l = l1; l <= l1 + 3; l++) {
                            int equivalent_l = l;
                            if (l < 0) {
                                equivalent_l = -l - 1;
                            } else {
                                if (l >= coefsSize) {
                                    equivalent_l = 2 * coefsSize - l - 1;
                                }
                            }
                            if (coefsSize * coefsSize * equivalent_nn +
                                equivalent_m * coefsSize + equivalent_l >= coefsSize * coefsSize * coefsSize) {
                                return;
                            }
                            double CoeffRe = ((cudaVfourierRealCoefs)[
                                    (coefsSize * coefsSize * (equivalent_nn)) +
                                    ((equivalent_m) * coefsSize) +
                                    (equivalent_l)]);

                            double CoeffIm = ((cudaVfourierImagCoefs)[
                                    (coefsSize * coefsSize * (equivalent_nn)) +
                                    ((equivalent_m) * coefsSize) +
                                    (equivalent_l)]);

                            double xminusl = fabs(x - (double) l);
                            {
                                if (xminusl < 1.0) { aux = xminusl * xminusl * (xminusl - 2.0) * 0.5 + 2.0 / 3.0; }
                                else if (xminusl < 2.0) {
                                    xminusl -= 2.0;
                                    aux = xminusl * xminusl * xminusl * (-1.0 / 6.0);
                                } else { aux = 0.0; }
                            }
                            xsumRe += ((double) CoeffRe) * aux;
                            xsumIm += ((double) CoeffIm) * aux;
                        }

                        double yminusm = fabs(y - (double) m);
                        {
                            if (yminusm < 1.0) { aux = yminusm * yminusm * (yminusm - 2.0) * 0.5 + 2.0 / 3.0; }
                            else if (yminusm < 2.0) {
                                yminusm -= 2.0;
                                aux = yminusm * yminusm * yminusm * (-1.0 / 6.0);
                            } else { aux = 0.0; }
                        }
                        yxsumRe += xsumRe * aux;
                        yxsumIm += xsumIm * aux;
                    }

                    double zminusn = fabs(z - (double) nn);
                    {
                        if (zminusn < 1.0) { aux = zminusn * zminusn * (zminusn - 2.0) * 0.5 + 2.0 / 3.0; }
                        else if (zminusn < 2.0) {
                            zminusn -= 2.0;
                            aux = zminusn * zminusn * zminusn * (-1.0 / 6.0);
                        } else { aux = 0.0; }
                    }
                    c += yxsumRe * aux;
                    d += yxsumIm * aux;
                }
                double a = (i + j) % 2 == 0 ? 1.0 : -1.0;
                if (useCtf) {
                    a *= cudaCtf[i * imageSizex + j]; //WTF
                }
                cudaProjectionFourier[((i * imageSizex + j) * 2)] = a * c;
                cudaProjectionFourier[((i * imageSizex + j) * 2) + 1] = a * d;
            }
        }
        i += blockDim.y;
    }
}

__global__ void onlyCtf(double *cudaProjectionFourier, double *newCtf, double *oldCtf, int imagesizex, int imagesizey){
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i<imagesizey && j<imagesizex) {
        cudaProjectionFourier[(i * imagesizex + j)*2]=cudaProjectionFourier[(i * imagesizex + j)*2]/oldCtf[i * imagesizex + j]*newCtf[i * imagesizex + j];
        cudaProjectionFourier[(i * imagesizex + j)*2+1]=cudaProjectionFourier[(i * imagesizex + j)*2+1]/oldCtf[i * imagesizex + j]*newCtf[i * imagesizex + j];
    }
}