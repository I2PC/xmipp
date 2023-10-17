/***************************************************************************
 *
 * Authors:     Carlos Oscar S. Sorzano (coss@cnb.csic.es)
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

#include "fourier_projection.h"
#include "core/bilib/kernel.h"
#include "core/geometry.h"
#include "core/transformations.h"
#include "core/xmipp_fftw.h"

/* Reset =================================================================== */
void Projection::reset(int Ydim, int Xdim)
{
    data.initZeros(Ydim, Xdim);
    data.setXmippOrigin();
}

/* Set angles ============================================================== */
void Projection::setAngles(double _rot, double _tilt, double _psi)
{
    setEulerAngles(_rot, _tilt, _psi);
    Euler_angles2matrix(_rot, _tilt, _psi, euler);
    eulert = euler.transpose();
    euler.getRow(2, direction);
    direction.selfTranspose();
}

/* Read ==================================================================== */
void Projection::read(const FileName &fn, const bool only_apply_shifts,
                      DataMode datamode , MDRow * row)
{
    Image<double>::read(fn, datamode);
    if (row != nullptr)
        applyGeo(*row, only_apply_shifts);
    Euler_angles2matrix(rot(), tilt(), psi(), euler);
    eulert = euler.transpose();
    euler.getRow(2, direction);
    direction.selfTranspose();
}

/* Another function for assignment ========================================= */
void Projection::assign(const Projection &P)
{
    *this = P;
}

FourierProjector::FourierProjector(double paddFactor, double maxFreq, int degree)
{
    paddingFactor = paddFactor;
    maxFrequency = maxFreq;
    BSplineDeg = degree;
    volume = nullptr;
}

FourierProjector::FourierProjector(MultidimArray<double> &V, double paddFactor, double maxFreq, int degree)
{
    paddingFactor = paddFactor;
    maxFrequency = maxFreq;
    BSplineDeg = degree;
    updateVolume(V);
}

void FourierProjector::updateVolume(MultidimArray<double> &V)
{
    volume = &V;
    volumeSize=XSIZE(*volume);
    produceSideInfo();
}

void FourierProjector::projectToFourier(double rot, double tilt, double psi, const MultidimArray<double> *ctf)
{
    double freqy;
    double freqx;
    std::complex< double > f;
    Euler_angles2matrix(rot,tilt,psi,E);

    projectionFourier.initZeros();
    double maxFreq2=maxFrequency*maxFrequency;
    auto Xdim=(int)XSIZE(VfourierRealCoefs);
    auto Ydim=(int)YSIZE(VfourierRealCoefs);
    auto Zdim=(int)ZSIZE(VfourierRealCoefs);

    for (size_t i=0; i<YSIZE(projectionFourier); ++i)
    {
        FFT_IDX2DIGFREQ(i,volumeSize,freqy);
        double freqy2=freqy*freqy;

        double freqYvol_X=MAT_ELEM(E,1,0)*freqy;
        double freqYvol_Y=MAT_ELEM(E,1,1)*freqy;
        double freqYvol_Z=MAT_ELEM(E,1,2)*freqy;
        for (size_t j=0; j<XSIZE(projectionFourier); ++j)
        {
            // The frequency of pairs (i,j) in 2D
            FFT_IDX2DIGFREQ(j,volumeSize,freqx);

            // Do not consider pixels with high frequency
            if ((freqy2+freqx*freqx)>maxFreq2)
                continue;

            // Compute corresponding frequency in the volume
            double freqvol_X=freqYvol_X+MAT_ELEM(E,0,0)*freqx;
            double freqvol_Y=freqYvol_Y+MAT_ELEM(E,0,1)*freqx;
            double freqvol_Z=freqYvol_Z+MAT_ELEM(E,0,2)*freqx;

            double c;
            double d;
            if (BSplineDeg==xmipp_transformation::NEAREST)
            {
                // 0 order interpolation
                // Compute corresponding index in the volume
            	auto kVolume=(int)round(freqvol_Z*volumePaddedSize);
            	auto iVolume=(int)round(freqvol_Y*volumePaddedSize);
            	auto jVolume=(int)round(freqvol_X*volumePaddedSize);
                c = A3D_ELEM(VfourierRealCoefs,kVolume,iVolume,jVolume);
                d = A3D_ELEM(VfourierImagCoefs,kVolume,iVolume,jVolume);
            }
            else if (BSplineDeg==xmipp_transformation::LINEAR)
            {
                // B-spline linear interpolation
                double kVolume=freqvol_Z*volumePaddedSize;
                double iVolume=freqvol_Y*volumePaddedSize;
                double jVolume=freqvol_X*volumePaddedSize;
                c=VfourierRealCoefs.interpolatedElement3D(jVolume,iVolume,kVolume);
                d=VfourierImagCoefs.interpolatedElement3D(jVolume,iVolume,kVolume);
            }
            else
            {
                // B-spline cubic interpolation
                double kVolume=freqvol_Z*volumePaddedSize;
                double iVolume=freqvol_Y*volumePaddedSize;
                double jVolume=freqvol_X*volumePaddedSize;

                // Commented for speed-up, the corresponding code is below
                // c=VfourierRealCoefs.interpolatedElementBSpline3D(jVolume,iVolume,kVolume);
                // d=VfourierImagCoefs.interpolatedElementBSpline3D(jVolume,iVolume,kVolume);

                // The code below is a replicate for speed reasons of interpolatedElementBSpline3D
                double z=kVolume;
                double y=iVolume;
                double x=jVolume;

                // Logical to physical
                z -= STARTINGZ(VfourierRealCoefs);
                y -= STARTINGY(VfourierRealCoefs);
                x -= STARTINGX(VfourierRealCoefs);

                auto l1 = (int)ceil(x - 2);
                int l2 = l1 + 3;

                auto m1 = (int)ceil(y - 2);
                int m2 = m1 + 3;

                auto n1 = (int)ceil(z - 2);
                int n2 = n1 + 3;

                c = d = 0.0;
                double aux;
                for (int nn = n1; nn <= n2; nn++)
                {
                    int equivalent_nn=nn;
                    if      (nn<0)
                        equivalent_nn=-nn-1;
                    else if (nn>=Zdim)
                        equivalent_nn=2*Zdim-nn-1;
                    double yxsumRe = 0.0;
                    double yxsumIm = 0.0;
                    for (int m = m1; m <= m2; m++)
                    {
                        int equivalent_m=m;
                        if      (m<0)
                            equivalent_m=-m-1;
                        else if (m>=Ydim)
                            equivalent_m=2*Ydim-m-1;
                        double xsumRe = 0.0;
                        double xsumIm = 0.0;
                        for (int l = l1; l <= l2; l++)
                        {
                            double xminusl = x - (double) l;
                            int equivalent_l=l;
                            if      (l<0)
                                equivalent_l=-l-1;
                            else if (l>=Xdim)
                                equivalent_l=2*Xdim-l-1;
                            auto CoeffRe = (double) DIRECT_A3D_ELEM(VfourierRealCoefs,equivalent_nn,equivalent_m,equivalent_l);
                            auto CoeffIm = (double) DIRECT_A3D_ELEM(VfourierImagCoefs,equivalent_nn,equivalent_m,equivalent_l);
                            BSPLINE03(aux,xminusl);
                            xsumRe += CoeffRe * aux;
                            xsumIm += CoeffIm * aux;
                        }

                        double yminusm = y - (double) m;
                        BSPLINE03(aux,yminusm);
						yxsumRe += xsumRe * aux;
						yxsumIm += xsumIm * aux;
                    }

                    double zminusn = z - (double) nn;
                    BSPLINE03(aux,zminusn);
					c += yxsumRe * aux;
					d += yxsumIm * aux;
                }
            }

            // Phase shift to move the origin of the image to the corner
            double a=DIRECT_A2D_ELEM(phaseShiftImgA,i,j);
            double b=DIRECT_A2D_ELEM(phaseShiftImgB,i,j);
            if (ctf!=nullptr)
            {
            	double ctfij=DIRECT_A2D_ELEM(*ctf,i,j);
            	a*=ctfij;
            	b*=ctfij;
            }

            // Multiply Fourier coefficient in volume times phase shift
            double ac = a * c;
            double bd = b * d;
            double ab_cd = (a + b) * (c + d);

            // And store the multiplication
            auto *ptrI_ij=(double *)&DIRECT_A2D_ELEM(projectionFourier,i,j);
            *ptrI_ij = ac - bd;
            *(ptrI_ij+1) = ab_cd - ac - bd;
        }
    }
}

void FourierProjector::projectToFourier(double rot, double tilt, double psi, double shiftX, double shiftY, const MultidimArray<double> *ctf) {
    projectToFourier(rot, tilt, psi, ctf);
    shiftFourierProjection(shiftX, shiftY);
}

void FourierProjector::project(double rot, double tilt, double psi, const MultidimArray<double> *ctf) {
    projectToFourier(rot, tilt, psi, ctf);
    transformer2D.inverseFourierTransform();
}

void FourierProjector::project(double rot, double tilt, double psi, double shiftX, double shiftY, const MultidimArray<double> *ctf) {
    projectToFourier(rot, tilt, psi, shiftX, shiftY, ctf);
    transformer2D.inverseFourierTransform();
}

void FourierProjector::produceSideInfo()
{
    // Zero padding
    MultidimArray<double> Vpadded;
    auto paddedDim=(int)(paddingFactor*volumeSize);
    volume->window(Vpadded,FIRST_XMIPP_INDEX(paddedDim),FIRST_XMIPP_INDEX(paddedDim),FIRST_XMIPP_INDEX(paddedDim),
                   LAST_XMIPP_INDEX(paddedDim),LAST_XMIPP_INDEX(paddedDim),LAST_XMIPP_INDEX(paddedDim));
    volume->clear();
    // Make Fourier transform, shift the volume origin to the volume center and center it
    MultidimArray< std::complex<double> > Vfourier;
    FourierTransformer transformer3D;
    transformer3D.completeFourierTransform(Vpadded,Vfourier);
    ShiftFFT(Vfourier, FIRST_XMIPP_INDEX(XSIZE(Vpadded)), FIRST_XMIPP_INDEX(YSIZE(Vpadded)), FIRST_XMIPP_INDEX(ZSIZE(Vpadded)));
    CenterFFT(Vfourier,true);
    Vfourier.setXmippOrigin();

    // Compensate for the Fourier normalization factor
    double K=(double)(XSIZE(Vpadded)*XSIZE(Vpadded)*XSIZE(Vpadded))/(double)(volumeSize*volumeSize);
    FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Vfourier)
    DIRECT_MULTIDIM_ELEM(Vfourier,n)*=K;
    Vpadded.clear();
    // Compute Bspline coefficients
    if (BSplineDeg==xmipp_transformation::BSPLINE3)
    {
        MultidimArray< double > VfourierRealAux;
        MultidimArray< double > VfourierImagAux;
        Complex2RealImag(Vfourier, VfourierRealAux, VfourierImagAux);
        Vfourier.clear();
        produceSplineCoefficients(xmipp_transformation::BSPLINE3,VfourierRealCoefs,VfourierRealAux);

        // Release memory as soon as you can
        VfourierRealAux.clear();

        // Remove all those coefficients we are sure we will not use during the projections
        volumePaddedSize=XSIZE(VfourierRealCoefs);
        int idxMax=maxFrequency*XSIZE(VfourierRealCoefs)+10; // +10 is a safety guard
        idxMax=std::min(FINISHINGX(VfourierRealCoefs),idxMax);
        int idxMin=std::max(-idxMax,STARTINGX(VfourierRealCoefs));
        VfourierRealCoefs.selfWindow(idxMin,idxMin,idxMin,idxMax,idxMax,idxMax);

        produceSplineCoefficients(xmipp_transformation::BSPLINE3,VfourierImagCoefs,VfourierImagAux);
        VfourierImagAux.clear();
        VfourierImagCoefs.selfWindow(idxMin,idxMin,idxMin,idxMax,idxMax,idxMax);
    }
    else {
        Complex2RealImag(Vfourier, VfourierRealCoefs, VfourierImagCoefs);
        volumePaddedSize=XSIZE(VfourierRealCoefs);
    }

    produceSideInfoProjection();
}

void FourierProjector::produceSideInfoProjection()
{
    // Allocate memory for the 2D Fourier transform
    projection().initZeros(volumeSize,volumeSize);
    projection().setXmippOrigin();
    transformer2D.FourierTransform(projection(),projectionFourier,false);

    // Calculate phase shift terms
    phaseShiftImgA.initZeros(projectionFourier);
    phaseShiftImgB.initZeros(projectionFourier);
    double shift=-FIRST_XMIPP_INDEX(volumeSize);
    double xxshift = -2 * PI * shift / volumeSize;
    for (size_t i=0; i<YSIZE(projectionFourier); ++i)
    {
        double phasey=(double)(i) * xxshift;
        for (size_t j=0; j<XSIZE(projectionFourier); ++j)
        {
            // Phase shift to move the origin of the image to the corner
            double dotp = (double)(j) * xxshift + phasey;
            //sincos(dotp,&DIRECT_A2D_ELEM(phaseShiftImgB,i,j),&DIRECT_A2D_ELEM(phaseShiftImgA,i,j));
            DIRECT_A2D_ELEM(phaseShiftImgB,i,j) = sin(dotp);
			DIRECT_A2D_ELEM(phaseShiftImgA,i,j) = cos(dotp);
        }
    }
}

void projectVolume(FourierProjector &projector, Projection &P, int Ydim, int Xdim,
                   double rot, double tilt, double psi, const MultidimArray<double> *ctf)
{
	projector.project(rot,tilt,psi,ctf);
    P() = projector.projection();
}

void FourierProjector::shiftFourierProjection(double shiftX, double shiftY) {
    const int ny = YSIZE(projectionFourier);
    const int ny_2 = ny / 2;
    const auto ny_inv = 1.0 / ny;
    const int nx_2 = (XSIZE(projectionFourier) - 1);
    const int nx = nx_2 * 2;
    const auto nx_inv = 1.0 / nx;

    // Normalize the displacement
    const auto dy = (-2 * M_PI) * shiftY;
    const auto dx = (-2 * M_PI) * shiftX;

    // Compute the Fourier Transform of delta[i-y, j-x]
    double fy, fx;
    FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY2D(projectionFourier) {
        // Convert the indices to fourier coefficients
        FFT_IDX2DIGFREQ_FAST(static_cast<int>(i), ny, ny_2, ny_inv, fy);
        FFT_IDX2DIGFREQ_FAST(static_cast<int>(j), nx, nx_2, nx_inv, fx);

        const auto theta = fy*dy + fx*dx; // Dot product of (dx, dy) and (j, i)
        DIRECT_A2D_ELEM(projectionFourier, i, j) *= std::polar(1.0, theta); //e^(i*theta)
    }
}
