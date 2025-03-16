/***************************************************************************
 *
 * Authors:     Carlos Oscar S. Sorzano (coss@cnb.csic.es)
 *              Martin Pernica, Masaryk University
 *
 * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
 * 02111-1307  USA
 *
 *  All comments concerning this program package may be sent to the
 *  e-mail address 'xmipp@cnb.csic.es'
 ***************************************************************************/

#include <chrono>

#include "cuda_fourier_projection.h"
#include "core/bilib/kernel.h"

#include "core/transformations.h"
#include "core/xmipp_fftw.h"

#include "cuda_fourier_projection.cu"
//#include "core/geometry.h"

template<typename T>
void local_Euler_angles2matrix(T alpha, T beta, T gamma,
                               Matrix2D<T> &A, bool homogeneous = false) {

    static_assert(std::is_floating_point<T>::value, "Only double and double are allowed as template parameters");

    if (homogeneous) {
        A.initZeros(4, 4);
        MAT_ELEM(A, 3, 3) = 1;
    } else {
        if (MAT_XSIZE(A) != 3 || MAT_YSIZE(A) != 3) {
            A.resizeNoCopy(3, 3);
        }
    }

    T ca = std::cos(DEG2RAD(alpha));
    T sa = std::sin(DEG2RAD(alpha));
    T cb = std::cos(DEG2RAD(beta));
    T sb = std::sin(DEG2RAD(beta));
    T cg = std::cos(DEG2RAD(gamma));
    T sg = std::sin(DEG2RAD(gamma));

    T cc = cb * ca;
    T cs = cb * sa;
    T sc = sb * ca;
    T ss = sb * sa;

    MAT_ELEM(A, 0, 0) = cg * cc - sg * sa;
    MAT_ELEM(A, 0, 1) = cg * cs + sg * ca;
    MAT_ELEM(A, 0, 2) = -cg * sb;
    MAT_ELEM(A, 1, 0) = -sg * cc - cg * sa;
    MAT_ELEM(A, 1, 1) = -sg * cs + cg * ca;
    MAT_ELEM(A, 1, 2) = sg * sb;
    MAT_ELEM(A, 2, 0) = sc;
    MAT_ELEM(A, 2, 1) = ss;
    MAT_ELEM(A, 2, 2) = cb;
}


/* Reset =================================================================== */
void Projection::reset(int Ydim, int Xdim) {
    data.initZeros(Ydim, Xdim);
    data.setXmippOrigin();
}

/* Set angles ============================================================== */
void Projection::setAngles(double _rot, double _tilt, double _psi) {
    setEulerAngles(_rot, _tilt, _psi);
    local_Euler_angles2matrix(_rot, _tilt, _psi, euler);
    eulert = euler.transpose();
    euler.getRow(2, direction);
    direction.selfTranspose();
}

/* Read ==================================================================== */
void Projection::read(const FileName &fn, const bool only_apply_shifts,
                      DataMode datamode, MDRow *row) {
    Image<double>::read(fn, datamode);
    if (row != nullptr)
        applyGeo(*row, only_apply_shifts);
    local_Euler_angles2matrix(rot(), tilt(), psi(), euler);
    eulert = euler.transpose();
    euler.getRow(2, direction);
    direction.selfTranspose();
}

/* Another function for assignment ========================================= */
void Projection::assign(const Projection &P) {
    *this = P;
}

CudaFourierProjector::CudaFourierProjector(double paddFactor, double maxFreq, int degree, double threshold, int numT) {
    paddingFactor = paddFactor;
    maxFrequency = maxFreq;
    BSplineDeg = degree;
    volume = nullptr;
    skipThreshold=threshold;
    nThreads=numT;
}

CudaFourierProjector::CudaFourierProjector(MultidimArray<double> &V, double paddFactor, double maxFreq, int degree, double threshold, int numT) {
    paddingFactor = paddFactor;
    maxFrequency = maxFreq;
    BSplineDeg = degree;
    skipThreshold=threshold;
    nThreads=numT;
    updateVolume(V);
}


void CudaFourierProjector::initGPUs() {

    for (int i = 0; i < nThreads; ++i) {
        streams[i]=new cudaStream_t;
        cudaStreamCreate((cudaStream_t *)streams[i]);
    }

}

void CudaFourierProjector::freeObj(int rank) const {
    if (cudaAllocated) {
        if (cudaVfourierRealCoefs) {
            cudaFree(cudaVfourierRealCoefs);
        }
        if (cudaVfourierImagCoefs) {
            cudaFree(cudaVfourierImagCoefs);
        }

        cudaFree(cudaPhaseA);
        cudaFree(cudaPhaseB);

        for (int i = 0; i < nThreads; ++i) {

            if (cudaE[i]) {
                cudaFree(cudaE[i]);
            }
            if (cudaCtf[i]) {
                cudaFree(cudaCtf[i]);
            }
            if (cudaProjectionFourier[i]) {
                cudaFree(cudaProjectionFourier[i]);
            }

            if (cudaCtf[i]) {
                cudaFree(cudaCtf[i]);
            }
            cudaFree(copyProjectionFourier[i]);
            cudaFree(tmpE[i]);
        }
    }
}


void CudaFourierProjector::copyAllToGpu(int rank) {
    streams= new void*[nThreads];
    initGPUs();

    useCtf=new bool[nThreads];
    last_rot = new double[nThreads];
    last_tilt = new double[nThreads];
    last_psi = new double[nThreads];

    copyProjectionFourier = new double*[nThreads];
    cudaE= new double*[nThreads];
    cudaCtf = new double*[nThreads];
    cudaOldCtf = new double*[nThreads];
    cudaProjectionFourier = new double*[nThreads];
    tmpE=new double*[nThreads];

    cudaAllocated = true;
    cudaMalloc(&cudaVfourierRealCoefs, VfourierRealCoefs.zyxdim * sizeof(float));
    cudaMalloc(&cudaVfourierImagCoefs, VfourierImagCoefs.zyxdim * sizeof(float));

    float *tmp;

    tmp = (float *) malloc(VfourierRealCoefs.yxdim * sizeof(float));

    for (int i = 0; i < VfourierRealCoefs.zdim; ++i) {
        for (int j = 0; j < VfourierRealCoefs.yxdim; ++j) {
            tmp[j] = (float) VfourierRealCoefs.data[i * VfourierRealCoefs.yxdim + j];
        }
        cudaMemcpy(&cudaVfourierRealCoefs[i * VfourierRealCoefs.yxdim], tmp, VfourierRealCoefs.yxdim * sizeof(float),
                   cudaMemcpyHostToDevice);

        cudaDeviceSynchronize();

        for (int j = 0; j < VfourierRealCoefs.yxdim; ++j) {
            tmp[j] = (float) VfourierImagCoefs.data[i * VfourierRealCoefs.yxdim + j];
        }
        cudaMemcpy(cudaVfourierImagCoefs + i * VfourierRealCoefs.yxdim, tmp, VfourierRealCoefs.yxdim * sizeof(float),
                   cudaMemcpyHostToDevice);

        cudaDeviceSynchronize();
    }

    free(tmp);

    cudaDeviceSynchronize();

    fourierSize = projectionFourier[0].zyxdim * 2 * sizeof(double);
    realSize = projection[0].data.nzyxdim * sizeof(double);

    cudaMalloc(&cudaPhaseA,fourierSize/2);
    cudaMalloc(&cudaPhaseB,fourierSize/2);

    cudaMemcpy(cudaPhaseA,phaseShiftImgA.data,fourierSize/2,cudaMemcpyHostToDevice);
    cudaMemcpy(cudaPhaseB,phaseShiftImgB.data,fourierSize/2,cudaMemcpyHostToDevice);

    cudaError ee;
    for (int i=0;i<nThreads;++i){
        useCtf[i]=false;
        ee=cudaMalloc(&cudaProjectionFourier[i], fourierSize);
        //std::cout<< ", cudaProjectionFourier : " << cudaGetErrorName(ee) ;
        ee=cudaMallocHost(&copyProjectionFourier[i], fourierSize);
        //std::cout<< ", copyProjectionFourier : " << cudaGetErrorName(ee) ;
        ee=cudaMalloc(&cudaCtf[i], fourierSize/2);
        //std::cout<< ", cudaCtf : " << cudaGetErrorName(ee) ;
        ee=cudaMalloc(&cudaOldCtf[i], fourierSize/2);
        //std::cout<< ", cudaOldCtf : " << cudaGetErrorName(ee) ;
        ee=cudaMalloc(&cudaE[i], 9 * sizeof(double));
        //std::cout<< ", cudaE : " << cudaGetErrorName(ee) ;
        ee=cudaMallocHost(&tmpE[i], 9 * sizeof(double));
        //std::cout<< ", tmpE : " << cudaGetErrorName(ee) ;
        std::cout<< std::endl;

        projectionFourier[i].initZeros();
        cudaMemcpy(cudaProjectionFourier[i],projectionFourier[i].data,fourierSize,cudaMemcpyHostToDevice);
    }
    cudaDeviceSynchronize();
}


void
CudaFourierProjector::cudaProject(double rot, double tilt, double psi, const MultidimArray<double> *ctf, int degree,
                                  MultidimArray<double> *Ifilteredp,
                                  MultidimArray<double> *Ifiltered, const Matrix2D<double> *A, int thread, bool do_transform, bool updateCTF) {

    if (pow(rot - last_rot[thread], 2.0) + pow(tilt - last_tilt[thread], 2.0) + pow(psi - last_psi[thread], 2.0) < skipThreshold) {
        if (updateCTF){
            dim3 threads = {16, 16};
            dim3 blocks = {(unsigned int) ((projectionFourier[thread].xdim - 1) / (threads.x) + 1),
                           (unsigned int) ((projectionFourier[thread].ydim - 1) / (threads.y) + 1)};
            onlyCtf<<<blocks, threads,0 ,  *(cudaStream_t *)(streams[thread])>>>(cudaProjectionFourier[thread],cudaCtf[thread],cudaOldCtf[thread],projectionFourier[thread].xdim, projectionFourier[thread].ydim);
        }
        if (do_transform){
            applyGeometry(degree, *Ifilteredp, *Ifiltered, *A, xmipp_transformation::IS_NOT_INV,
                          xmipp_transformation::DONT_WRAP);
        }
        if (updateCTF){
            cudaMemcpyAsync(copyProjectionFourier[thread], cudaProjectionFourier[thread], fourierSize, cudaMemcpyDeviceToHost,*(cudaStream_t *)(streams[thread]));
            cudaStreamSynchronize(*(cudaStream_t *)(streams[thread]));
            memcpy(projectionFourier[thread].data,copyProjectionFourier[thread],fourierSize);
            transformer2D[thread].inverseFourierTransform();
        }
        return;
    } else {
        last_rot[thread] = rot;
        last_tilt[thread] = tilt;
        last_psi[thread] = psi;
    }

    local_Euler_angles2matrix(rot, tilt, psi, E[thread]);
    memcpy(tmpE[thread], E[thread].mdata, 9 * sizeof(double));

    cudaMemcpyAsync(cudaE[thread], tmpE[thread], 9 * sizeof(double), cudaMemcpyHostToDevice, *(cudaStream_t *)(streams[thread]));
    int work =4;

    dim3 threads = {8, 4};
    dim3 blocks = {(unsigned int) ((projectionFourier[thread].xdim - 1) / (threads.x) + 1),
                   (unsigned int) ((projectionFourier[thread].ydim - 1) / (threads.y*work) + 1)};

    auto start = std::chrono::high_resolution_clock::now();
    projectKernel<<<blocks, threads, 0, *(cudaStream_t *)(streams[thread])>>>(cudaProjectionFourier[thread], cudaVfourierRealCoefs, cudaVfourierImagCoefs, cudaE[thread], cudaCtf[thread], useCtf[thread], (int) VfourierRealCoefs.xdim, volumeSize, (int) projectionFourier[thread].xdim, (int) projectionFourier[thread].ydim, volumePaddedSize, (
            maxFrequency * maxFrequency), VfourierRealCoefs.xinit, VfourierRealCoefs.yinit, VfourierRealCoefs.zinit, work, cudaPhaseA, cudaPhaseB);

    if (do_transform){
        applyGeometry(degree, *Ifilteredp, *Ifiltered, *A, xmipp_transformation::IS_NOT_INV,
                      xmipp_transformation::DONT_WRAP);
    }

    cudaMemcpyAsync(copyProjectionFourier[thread], cudaProjectionFourier[thread], fourierSize, cudaMemcpyDeviceToHost,*(cudaStream_t *)(streams[thread]));

    cudaStreamSynchronize(*(cudaStream_t *)(streams[thread]));

    memcpy(projectionFourier[thread].data,copyProjectionFourier[thread],fourierSize);

    transformer2D[thread].inverseFourierTransform();

}

void CudaFourierProjector::updateVolume(MultidimArray<double> &V) {
    volume = &V;
    volumeSize = XSIZE(*volume);
    produceSideInfo();
}

void CudaFourierProjector::updateCtf(MultidimArray<double> *ctf, int thread) {

    std::swap(cudaCtf[thread],cudaOldCtf[thread]);
    memcpy(copyProjectionFourier[thread],ctf->data,fourierSize/2);
    cudaMemcpyAsync(cudaCtf[thread], copyProjectionFourier[thread], fourierSize/2, cudaMemcpyHostToDevice,
                    *(cudaStream_t *)(streams[thread]));
    cudaStreamSynchronize(*(cudaStream_t *)(streams[thread]));

    useCtf[thread] = true;
}


void CudaFourierProjector::orig_project(double rot, double tilt, double psi, int thread, const MultidimArray<double> *ctf) {
    double freqy;
    double freqx;
    //std::complex<double> f;
    local_Euler_angles2matrix(rot, tilt, psi, E[thread]);

    projectionFourier[thread].initZeros();
    double maxFreq2 = maxFrequency * maxFrequency;
    auto Xdim = (int) XSIZE(VfourierRealCoefs);
    auto Ydim = (int) YSIZE(VfourierRealCoefs);
    auto Zdim = (int) ZSIZE(VfourierRealCoefs);

    for (size_t i = 0; i < YSIZE(projectionFourier[thread]); ++i) {
        FFT_IDX2DIGFREQ(i, volumeSize, freqy);
        double freqy2 = freqy * freqy;

        double freqYvol_X = MAT_ELEM(E[thread], 1, 0) * freqy;
        double freqYvol_Y = MAT_ELEM(E[thread], 1, 1) * freqy;
        double freqYvol_Z = MAT_ELEM(E[thread], 1, 2) * freqy;
        for (size_t j = 0; j < XSIZE(projectionFourier[thread]); ++j) {
            // The frequency of pairs (i,j) in 2D
            FFT_IDX2DIGFREQ(j, volumeSize, freqx);

            // Do not consider pixels with high frequency
            if ((freqy2 + freqx * freqx) > maxFreq2)
                continue;

            // Compute corresponding frequency in the volume
            double freqvol_X = freqYvol_X + MAT_ELEM(E[thread], 0, 0) * freqx;
            double freqvol_Y = freqYvol_Y + MAT_ELEM(E[thread], 0, 1) * freqx;
            double freqvol_Z = freqYvol_Z + MAT_ELEM(E[thread], 0, 2) * freqx;

            double c;
            double d;
            if (BSplineDeg == xmipp_transformation::NEAREST) {
                // 0 order interpolation
                // Compute corresponding index in the volume
                auto kVolume = (int) round(freqvol_Z * volumePaddedSize);
                auto iVolume = (int) round(freqvol_Y * volumePaddedSize);
                auto jVolume = (int) round(freqvol_X * volumePaddedSize);
                c = A3D_ELEM(VfourierRealCoefs, kVolume, iVolume, jVolume);
                d = A3D_ELEM(VfourierImagCoefs, kVolume, iVolume, jVolume);
            } else if (BSplineDeg == xmipp_transformation::LINEAR) {
                // B-spline linear interpolation
                double kVolume = freqvol_Z * volumePaddedSize;
                double iVolume = freqvol_Y * volumePaddedSize;
                double jVolume = freqvol_X * volumePaddedSize;
                c = VfourierRealCoefs.interpolatedElement3D(jVolume, iVolume, kVolume);
                d = VfourierImagCoefs.interpolatedElement3D(jVolume, iVolume, kVolume);
            } else {
                // B-spline cubic interpolation
                double kVolume = freqvol_Z * volumePaddedSize;
                double iVolume = freqvol_Y * volumePaddedSize;
                double jVolume = freqvol_X * volumePaddedSize;

                // Commented for speed-up, the corresponding code is below
                // c=VfourierRealCoefs.interpolatedElementBSpline3D(jVolume,iVolume,kVolume);
                // d=VfourierImagCoefs.interpolatedElementBSpline3D(jVolume,iVolume,kVolume);

                // The code below is a replicate for speed reasons of interpolatedElementBSpline3D
                double z = kVolume;
                double y = iVolume;
                double x = jVolume;

                // Logical to physical
                z -= STARTINGZ(VfourierRealCoefs);
                y -= STARTINGY(VfourierRealCoefs);
                x -= STARTINGX(VfourierRealCoefs);

                auto l1 = (int) ceil(x - 2);
                int l2 = l1 + 3;

                auto m1 = (int) ceil(y - 2);
                int m2 = m1 + 3;

                auto n1 = (int) ceil(z - 2);
                int n2 = n1 + 3;

                c = d = 0.0;
                double aux;
                for (int nn = n1; nn <= n2; nn++) {
                    int equivalent_nn = nn;
                    if (nn < 0)
                        equivalent_nn = -nn - 1;
                    else if (nn >= Zdim)
                        equivalent_nn = 2 * Zdim - nn - 1;
                    double yxsumRe = 0.0;
                    double yxsumIm = 0.0;
                    for (int m = m1; m <= m2; m++) {
                        int equivalent_m = m;
                        if (m < 0)
                            equivalent_m = -m - 1;
                        else if (m >= Ydim)
                            equivalent_m = 2 * Ydim - m - 1;
                        double xsumRe = 0.0;
                        double xsumIm = 0.0;
                        for (int l = l1; l <= l2; l++) {
                            double xminusl = x - (double) l;
                            int equivalent_l = l;
                            if (l < 0)
                                equivalent_l = -l - 1;
                            else if (l >= Xdim)
                                equivalent_l = 2 * Xdim - l - 1;
                            auto CoeffRe = (double) DIRECT_A3D_ELEM(VfourierRealCoefs, equivalent_nn, equivalent_m,
                                                                    equivalent_l);
                            auto CoeffIm = (double) DIRECT_A3D_ELEM(VfourierImagCoefs, equivalent_nn, equivalent_m,
                                                                    equivalent_l);
                            BSPLINE03(aux, xminusl);
                            xsumRe += CoeffRe * aux;
                            xsumIm += CoeffIm * aux;
                        }

                        double yminusm = y - (double) m;
                        BSPLINE03(aux, yminusm);
                        yxsumRe += xsumRe * aux;
                        yxsumIm += xsumIm * aux;
                    }

                    double zminusn = z - (double) nn;
                    BSPLINE03(aux, zminusn);
                    c += yxsumRe * aux;
                    d += yxsumIm * aux;
                }
            }
            // Phase shift to move the origin of the image to the corner
            double a = (i + j) % 2 == 0 ? 1.0f : -1.0f;
            double b = 0.0f;


            if (useCtf[thread]) {
                double ctfij = DIRECT_A2D_ELEM(*ctf, i, j);
                a *= ctfij;
                b *= ctfij;
            }

            // Multiply Fourier coefficient in volume times phase shift
            double ac = a * c;
            double bd = b * d;
            double ab_cd = (a + b) * (c + d);

            // And store the multiplication
            auto *ptrI_ij = (double *) &DIRECT_A2D_ELEM(projectionFourier[thread], i, j);
            *ptrI_ij = ac - bd;
            *(ptrI_ij + 1) = ab_cd - ac - bd;
        }
    }

    transformer2D[thread].inverseFourierTransform();
}

void CudaFourierProjector::produceSideInfo() {

    // Zero padding
    MultidimArray<double> Vpadded;
    auto paddedDim = (int) (paddingFactor * volumeSize);
    volume->window(Vpadded, FIRST_XMIPP_INDEX(paddedDim), FIRST_XMIPP_INDEX(paddedDim), FIRST_XMIPP_INDEX(paddedDim),
                   LAST_XMIPP_INDEX(paddedDim), LAST_XMIPP_INDEX(paddedDim), LAST_XMIPP_INDEX(paddedDim));


    volume->clear();
    // Make Fourier transform, shift the volume origin to the volume center and center it
    MultidimArray<std::complex<double> > Vfourier;
    FourierTransformer transformer3D;
    transformer3D.completeFourierTransform(Vpadded, Vfourier);

    ShiftFFT(Vfourier, FIRST_XMIPP_INDEX(XSIZE(Vpadded)), FIRST_XMIPP_INDEX(YSIZE(Vpadded)),
             FIRST_XMIPP_INDEX(ZSIZE(Vpadded)));

    CenterFFT(Vfourier, true);
    Vfourier.setXmippOrigin();

    // Compensate for the Fourier normalization factor
    double K = (double) (XSIZE(Vpadded) * XSIZE(Vpadded) * XSIZE(Vpadded)) / (double) (volumeSize * volumeSize);
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Vfourier)DIRECT_MULTIDIM_ELEM(Vfourier, n) *= K;
    Vpadded.clear();
    // Compute Bspline coefficients

    if (BSplineDeg == xmipp_transformation::BSPLINE3) {
        MultidimArray<double> VfourierRealAux;
        MultidimArray<double> VfourierImagAux;
        Complex2RealImag(Vfourier, VfourierRealAux, VfourierImagAux);
        Vfourier.clear();
        produceSplineCoefficients(xmipp_transformation::BSPLINE3, VfourierRealCoefs, VfourierRealAux);

        // Release memory as soon as you can
        VfourierRealAux.clear();

        // Remove all those coefficients we are sure we will not use during the projections
        volumePaddedSize = XSIZE(VfourierRealCoefs);
        int idxMax = maxFrequency * XSIZE(VfourierRealCoefs) + 10; // +10 is a safety guard
        idxMax = std::min(FINISHINGX(VfourierRealCoefs), idxMax);
        int idxMin = std::max(-idxMax, STARTINGX(VfourierRealCoefs));
        VfourierRealCoefs.selfWindow(idxMin, idxMin, idxMin, idxMax, idxMax, idxMax);

        produceSplineCoefficients(xmipp_transformation::BSPLINE3, VfourierImagCoefs, VfourierImagAux);
        VfourierImagAux.clear();
        VfourierImagCoefs.selfWindow(idxMin, idxMin, idxMin, idxMax, idxMax, idxMax);

    } else {
        Complex2RealImag(Vfourier, VfourierRealCoefs, VfourierImagCoefs);
        volumePaddedSize = XSIZE(VfourierRealCoefs);
    }

    produceSideInfoProjection();
}

void CudaFourierProjector::produceSideInfoProjection() {

    transformer2D=new FourierTransformer[nThreads];
    projectionFourier=new MultidimArray< std::complex<double> >[nThreads];
    projection= new Image<double>[nThreads];
    E=new Matrix2D<double>[nThreads];
    // Allocate memory for the 2D Fourier transform
    for (int i=0; i< nThreads; ++i) {
        projection[i]().initZeros(volumeSize, volumeSize);
        projection[i]().setXmippOrigin();
        transformer2D[i].FourierTransform(projection[i](), projectionFourier[i], false);
    }

    // Calculate phase shift terms
    phaseShiftImgA.initZeros(projectionFourier[0]);
    phaseShiftImgB.initZeros(projectionFourier[0]);
    double shift=-FIRST_XMIPP_INDEX(volumeSize);
    double xxshift = -2 * PI * shift / volumeSize;
    for (size_t i=0; i<YSIZE(projectionFourier[0]); ++i)
    {
        double phasey=(double)(i) * xxshift;
        for (size_t j=0; j<XSIZE(projectionFourier[0]); ++j)
        {
            // Phase shift to move the origin of the image to the corner
            double dotp = (double)(j) * xxshift + phasey;
            //sincos(dotp,&DIRECT_A2D_ELEM(phaseShiftImgB,i,j),&DIRECT_A2D_ELEM(phaseShiftImgA,i,j));
            DIRECT_A2D_ELEM(phaseShiftImgB,i,j) = sin(dotp);
            DIRECT_A2D_ELEM(phaseShiftImgA,i,j) = cos(dotp);
        }
    }

    copyAllToGpu(0);
}

void
projectVolume(CudaFourierProjector &projector, MultidimArray<double> &P, int Ydim, int Xdim, double rot, double tilt,
              double psi, const MultidimArray<double> *ctf, int degree, MultidimArray<double> *Ifilteredp,
              MultidimArray<double> *Ifiltered, const Matrix2D<double> *A, int thread, bool transformuj, bool updateCTF) {

    projector.cudaProject(rot, tilt, psi, ctf, degree, Ifilteredp, Ifiltered, A, thread,transformuj, updateCTF);

    P = projector.projection[thread]();

}
