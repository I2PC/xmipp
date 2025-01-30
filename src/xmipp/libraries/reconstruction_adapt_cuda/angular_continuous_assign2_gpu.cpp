/***************************************************************************
 *
 * Authors:    Carlos Oscar Sanchez Sorzano coss@cnb.csic.es
 *             Martin Pernica, Masaryk University
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

#include "angular_continuous_assign2_gpu.h"
#include "core/transformations.h"
#include "core/xmipp_image_extension.h"
#include "core/xmipp_image_generic.h"
#include "data/mask.h"
#include <iostream>
#include <chrono>

// Empty constructor =======================================================
ProgCudaAngularContinuousAssign2::ProgCudaAngularContinuousAssign2() {
    produces_a_metadata = true;
    each_image_produces_an_output = true;
    projector = nullptr;

    rank = 0;
}

struct progIdx {
    ProgCudaAngularContinuousAssign2 *prog;
    int thread;
};

ProgCudaAngularContinuousAssign2::~ProgCudaAngularContinuousAssign2() {
    projector->freeObj(rank);
    delete projector;
    for (int i = 0; i < nThreads; ++i) {
        delete ctfImage[i];
    }
}

bool ProgCudaAngularContinuousAssign2::getImageToProcess(size_t &objId, size_t &objIndex) {

    thread_mutex.lock();
    bool result = false;
    try {
        result = XmippMetadataProgram::getImageToProcess(objId, objIndex);
        //std::cout<< "objId: " <<objId << ", objIndex: " << objIndex << std::endl;
    } catch (...) {
        thread_mutex.unlock();
        return false;
    }
    thread_mutex.unlock();
    return result;
}

void ProgCudaAngularContinuousAssign2::runThread(int threadIdx) {
    size_t objId;
    size_t objIndex;
    while (getImageToProcess(objId, objIndex)) {
        ++objIndex; //increment for composing starting at 1
        thread_mutex.lock();
        auto rowIn = getInputMd()->getRow(objId);
        rowIn->getValue(image_label, fnImg[threadIdx]);

        if (fnImg[threadIdx].empty())
            break;

        fnImgOut[threadIdx] = fnImg[threadIdx];

        MDRowVec rowOut;

        if (each_image_produces_an_output) {
            if (!oroot.empty()) // Compose out name to save as independent images
            {
                if (oext.empty()) // If oext is still empty, then use ext of indep input images
                {
                    if (input_is_stack)
                        oextBaseName = "spi";
                    else
                        oextBaseName = fnImg[threadIdx].getFileFormat();
                }

                if (!baseName.empty())
                    fnImgOut[threadIdx].compose(fullBaseName, objIndex, oextBaseName);
                else if (fnImg[threadIdx].isInStack())
                    fnImgOut[threadIdx].compose(
                            pathBaseName + (fnImg[threadIdx].withoutExtension()).getDecomposedFileName(), objIndex,
                            oextBaseName);
                else
                    fnImgOut[threadIdx] = pathBaseName + fnImg[threadIdx].withoutExtension() + "." + oextBaseName;
            } else if (!fn_out.empty()) {
                if (single_image)
                    fnImgOut[threadIdx] = fn_out;
                else
                    fnImgOut[threadIdx].compose(objIndex, fn_out); // Compose out name to save as stacks
            } else
                fnImgOut[threadIdx] = fnImg[threadIdx];
            setupRowOut(fnImg[threadIdx], *rowIn, fnImgOut[threadIdx], rowOut);
        } else if (produces_a_metadata)
            setupRowOut(fnImg[threadIdx], *rowIn, fnImgOut[threadIdx], rowOut);

        thread_mutex.unlock();
        processImage_t(fnImg[threadIdx], fnImgOut[threadIdx], *rowIn, rowOut, threadIdx);

        thread_mutex.lock();
        if (each_image_produces_an_output || produces_a_metadata)
            getOutputMd().addRow(rowOut);

        checkPoint();
        showProgress();
        thread_mutex.unlock();
    }

}


void ProgCudaAngularContinuousAssign2::run() {

    getOutputMd().clear(); //this allows multiple runs of the same Program object
    //Perform particular preprocessing
    preProcess();

    startProcessing();

    if (!oroot.empty()) {
        if (oext.empty())
            oext = oroot.getFileFormat();
        oextBaseName = oext;
        fullBaseName = oroot.removeFileFormat();
        baseName = fullBaseName.getBaseName();
        pathBaseName = fullBaseName.getDir();
    }


    std::thread threads[nThreads];
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < nThreads; ++i) {
        threads[i] = std::thread([this, i]() { this->runThread(i); });
    }

    for (int i = 0; i < nThreads; ++i) {
        threads[i].join();

    }

    //std::cout<< "total time: " << total_time<< std::endl;
    wait();

    /* Generate name to save mdOut when output are independent images. It uses as prefix
     * the dirBaseName in order not overwriting files when repeating same command on
     * different directories. If baseName is set it is used, otherwise, input name is used.
     * Then, the suffix _oext is added.*/
    if (fn_out.empty()) {
        if (!oroot.empty()) {
            if (!baseName.empty())
                fn_out = findAndReplace(pathBaseName, "/", "_") + baseName + "_" + oextBaseName + ".xmd";
            else
                fn_out = findAndReplace(pathBaseName, "/", "_") + fn_in.getBaseName() + "_" + oextBaseName + ".xmd";
        } else if (input_is_metadata) /// When nor -o neither --oroot is passed and want to overwrite input metadata
            fn_out = fn_in;
    }

    finishProcessing();

    postProcess();
    /* Reset the default values of the program in case
     * to be reused.*/
    init();
}


// Read arguments ==========================================================
void ProgCudaAngularContinuousAssign2::readParams() {
    XmippMetadataProgram::readParams();
    fnVol = getParam("--ref");
    maxShift = getDoubleParam("--max_shift");
    maxScale = getDoubleParam("--max_scale");
    maxDefocusChange = getDoubleParam("--max_defocus_change");
    maxAngularChange = getDoubleParam("--max_angular_change");
    maxResol = getDoubleParam("--max_resolution");
    maxA = getDoubleParam("--max_gray_scale");
    maxB = getDoubleParam("--max_gray_shift");
    Ts = getDoubleParam("--sampling");
    Rmax = getIntParam("--Rmax");
    pad = getIntParam("--padding");
    optimizeGrayValues = checkParam("--optimizeGray");
    optimizeShift = checkParam("--optimizeShift");
    optimizeScale = checkParam("--optimizeScale");
    optimizeAngles = checkParam("--optimizeAngles");
    optimizeDefocus = checkParam("--optimizeDefocus");
    ignoreCTF = checkParam("--ignoreCTF");
    originalImageLabel = getParam("--applyTo");
    phaseFlipped = checkParam("--phaseFlipped");
    // penalization = getDoubleParam("--penalization");
    fnResiduals = getParam("--oresiduals");
    fnProjections = getParam("--oprojections");
    sameDefocus = checkParam("--sameDefocus");
    keep_input_columns = true; // each output metadata row is a deep copy of the input metadata row
    skipThreshold = getDoubleParam("--skipThreshold");
    nThreads = getIntParam("--nThreads");
}


// Show ====================================================================
void ProgCudaAngularContinuousAssign2::show() {
    if (!verbose)
        return;
    XmippMetadataProgram::show();
    std::cout
            << "Reference volume:    " << fnVol << std::endl
            << "Max. Shift:          " << maxShift << std::endl
            << "Max. Scale:          " << maxScale << std::endl
            << "Max. Angular Change: " << maxAngularChange << std::endl
            << "Max. Resolution:     " << maxResol << std::endl
            << "Max. Defocus Change: " << maxDefocusChange << std::endl
            << "Max. Gray Scale:     " << maxA << std::endl
            << "Max. Gray Shift:     " << maxB << std::endl
            << "Sampling:            " << Ts << std::endl
            << "Max. Radius:         " << Rmax << std::endl
            << "Padding factor:      " << pad << std::endl
            << "Optimize gray:       " << optimizeGrayValues << std::endl
            << "Optimize shifts:     " << optimizeShift << std::endl
            << "Optimize scale:      " << optimizeScale << std::endl
            << "Optimize angles:     " << optimizeAngles << std::endl
            << "Optimize defocus:    " << optimizeDefocus << std::endl
            << "Ignore CTF:          " << ignoreCTF << std::endl
            << "Apply to:            " << originalImageLabel << std::endl
            << "Phase flipped:       " << phaseFlipped << std::endl
            // << "Penalization:        " << penalization       << std::endl
            << "Force same defocus:  " << sameDefocus << std::endl
            << "Output residuals:    " << fnResiduals << std::endl
            << "Output projections:  " << fnProjections << std::endl
            << "Skip projection threshold: " << skipThreshold << std::endl
            << "Number of threads:   " << nThreads << std::endl;
}

// usage ===================================================================
void ProgCudaAngularContinuousAssign2::defineParams() {
    addUsageLine("Make a continuous angular assignment");
    defaultComments["-i"].clear();
    defaultComments["-i"].addComment("Metadata with initial alignment");
    defaultComments["-o"].clear();
    defaultComments["-o"].addComment("Stack of images prepared for 3D reconstruction");
    XmippMetadataProgram::defineParams();
    addParamsLine("   --ref <volume>              : Reference volume");
    addParamsLine("  [--max_shift <s=-1>]         : Maximum shift allowed in pixels");
    addParamsLine("  [--max_scale <s=0.02>]       : Maximum scale change");
    addParamsLine("  [--max_angular_change <a=5>] : Maximum angular change allowed (in degrees)");
    addParamsLine("  [--max_defocus_change <d=500>] : Maximum defocus change allowed (in Angstroms)");
    addParamsLine("  [--max_resolution <f=4>]     : Maximum resolution (A)");
    addParamsLine("  [--max_gray_scale <a=0.05>]  : Maximum gray scale change");
    addParamsLine(
            "  [--max_gray_shift <b=0.05>]  : Maximum gray shift change as a factor of the image standard deviation");
    addParamsLine("  [--sampling <Ts=1>]          : Sampling rate (A/pixel)");
    addParamsLine("  [--Rmax <R=-1>]              : Maximum radius (px). -1=Half of volume size");
    addParamsLine("  [--padding <p=2>]            : Padding factor");
    addParamsLine("  [--optimizeGray]             : Optimize gray values");
    addParamsLine("  [--optimizeShift]            : Optimize shift");
    addParamsLine("  [--optimizeScale]            : Optimize scale");
    addParamsLine("  [--optimizeAngles]           : Optimize angles");
    addParamsLine("  [--optimizeDefocus]          : Optimize defocus");
    addParamsLine("  [--ignoreCTF]                : Ignore CTF");
    addParamsLine("  [--applyTo <label=image>]    : Which is the source of images to apply the final transformation");
    addParamsLine("  [--phaseFlipped]             : Input images have been phase flipped");
    // addParamsLine("  [--penalization <l=100>]     : Penalization for the average term");
    addParamsLine("  [--sameDefocus]              : Force defocusU = defocusV");
    addParamsLine("  [--oresiduals <stack=\"\">]  : Output stack for the residuals");
    addParamsLine("  [--oprojections <stack=\"\">] : Output stack for the projections");
    addParamsLine("  [--skipThreshold <t=0.000001>]: Condition for skipping too similar projections");
    addParamsLine("  [--nThreads <n=3>]:          : Number od threads to split the computation into");
    addExampleLine("A typical use is:", false);
    addExampleLine(
            "xmipp_angular_continuous_assign2 -i anglesFromDiscreteAssignment.xmd --ref reference.vol -o assigned_angles.stk");
}

void ProgCudaAngularContinuousAssign2::startProcessing() {
    XmippMetadataProgram::startProcessing();
    if (fnResiduals != "")
        createEmptyFile(fnResiduals, xdimOut, ydimOut, zdimOut, mdInSize, true, WRITE_OVERWRITE);
    if (fnProjections != "")
        createEmptyFile(fnProjections, xdimOut, ydimOut, zdimOut, mdInSize, true, WRITE_OVERWRITE);
}

// Produce side information ================================================
void ProgCudaAngularContinuousAssign2::preProcess() {
    if (verbose>=2) {
        std::cout<< "skipThreshold: " << skipThreshold << std::endl;
        std::cout<< "nThreads: " << nThreads << std::endl;
    }
    for (int i=0;i<5;++i){
        last_transform[i]=new double[nThreads];
    }
    fnImg = new FileName[nThreads];
    fnImgOut = new FileName[nThreads];
    I = new Image<double>[nThreads];
    Ip = new Image<double>[nThreads];
    E = new Image<double>[nThreads];
    Ifiltered = new Image<double>[nThreads];
    Ifilteredp = new Image<double>[nThreads];
    FourierIfiltered = new MultidimArray<std::complex<double>>[nThreads];
    FourierIfilteredp = new MultidimArray<std::complex<double>>[nThreads];
    FourierProjection = new MultidimArray<std::complex<double>>[nThreads];
    P = new Projection[nThreads];
    filter = new FourierFilter[nThreads];
    A = new Matrix2D<double>[nThreads];
    old_rot = new double[nThreads];
    old_tilt = new double[nThreads];
    old_psi = new double[nThreads];
    old_shiftX = new double[nThreads];
    old_shiftY = new double[nThreads];
    old_flip = new bool[nThreads];
    old_grayA = new double[nThreads];
    old_grayB = new double[nThreads];
    old_defocusU = new double[nThreads];
    old_defocusV = new double[nThreads];
    old_defocusAngle = new double[nThreads];
    ctf = new CTFDescription[nThreads];
    Istddev = new double[nThreads];
    currentDefocusU = new double[nThreads];
    currentDefocusV = new double[nThreads];
    currentAngle = new double[nThreads];
    ctfImage = new MultidimArray<double>*[nThreads];
    for (int i = 0; i < nThreads; ++i) {
        ctfImage[i] = nullptr;
    }
    ctfEnvelope = new std::unique_ptr<MultidimArray<double>>[nThreads];
    fftTransformer = new FourierTransformer[nThreads];
    fftE = new MultidimArray<std::complex<double>>[nThreads];

    // Read the reference volume

    Image<double> V;
    if (rank == 0) {
        V.read(fnVol);
        V().setXmippOrigin();
        Xdim = XSIZE(V());
    } else {
        size_t ydim, zdim, ndim;
        getImageSize(fnVol, Xdim, ydim, zdim, ndim);
    }

    // Construct mask
    if (Rmax < 0)
        Rmax = (int) Xdim / 2;
    Mask mask;
    mask.type = BINARY_CIRCULAR_MASK;
    mask.mode = INNER_MASK;
    mask.R1 = Rmax;

    mask.generate_mask((int) Xdim, (int) Xdim);
    mask2D = mask.get_binary_mask();
    iMask2Dsum = 1.0 / mask2D.sum();

    for (int i = 0; i < nThreads; ++i) {
        Ip[i]().initZeros(Xdim, Xdim);
        E[i]().initZeros(Xdim, Xdim);
        P[i]().initZeros(Xdim, Xdim);
        Ifilteredp[i]().initZeros(Xdim, Xdim);
        Ifilteredp[i]().setXmippOrigin();

        FourierIfiltered[i].initZeros(Xdim, Xdim / 2 + 1);
        FourierIfilteredp[i].initZeros(Xdim, Xdim / 2 + 1);
        FourierProjection[i].initZeros(Xdim, Xdim / 2 + 1);

        // Transformation matrix
        A[i].initIdentity(3);
        // Construct reference covariance
        FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(mask2D)if (DIRECT_MULTIDIM_ELEM(mask2D, n))
                DIRECT_MULTIDIM_ELEM(E[i](), n) = rnd_gaus(0, 1);
        /*
        covarianceMatrix(E(), C0);
        FOR_ALL_ELEMENTS_IN_MATRIX2D(C0) {
                double val = MAT_ELEM(C0, i, j);
                if (val < 0.5)
                    MAT_ELEM(C0, i, j) = 0;
                else
                    MAT_ELEM(C0, i, j) = 1;
            }
        */
        // Low pass filter
        filter[i].FilterBand = LOWPASS;
        filter[i].w1 = Ts / maxResol;
        filter[i].raised_w = 0.02;

    }

    // Construct projector
    if (rank == 0) {
        projector = new CudaFourierProjector(V(), pad, Ts / maxResol, xmipp_transformation::BSPLINE3, skipThreshold, nThreads);
        //projector = new FourierProjector(V(), pad, Ts / maxResol, xmipp_transformation::BSPLINE3);
    } else {
        projector = new CudaFourierProjector(pad, Ts / maxResol, xmipp_transformation::BSPLINE3, skipThreshold, nThreads);
        //projector = new FourierProjector(pad, Ts / maxResol, xmipp_transformation::BSPLINE3);
    }

    // Continuous cost
    if (optimizeGrayValues)
        contCost = CONTCOST_L1;
    else
        contCost = CONTCOST_CORR;

}

template<class T>
void local_generateCTF(int Ydim, int Xdim, MultidimArray<T> &ctf, CTFDescription *ctfdesc,
                 double Ts = -1, bool envelope=false) {
    double iTs = ctfdesc->initCTF(Ydim, Xdim, ctf, Ts);
    for (int i = 0; i < Ydim; ++i) {
        double wy;
        FFT_IDX2DIGFREQ(i, YSIZE(ctf), wy);
        double fy = wy * iTs;
        for (int j = 0; j < Xdim; ++j) {
            double wx;
            FFT_IDX2DIGFREQ(j, YSIZE(ctf), wx);
            double fx = wx * iTs;
            ctfdesc->precomputeValues(fx,fy);
            if (!envelope) {
                A2D_ELEM(ctf, i, j) = (T) ctfdesc->getValueAt();
            }else{
                A2D_ELEM(ctf, i, j) = (T) -ctfdesc->getValueDampingAt();
            }
        }
    }
}


//#define DEBUG
void ProgCudaAngularContinuousAssign2::updateCTFImage(double defocusU, double defocusV, double angle, int thread, bool gen_env=true) {
    ctf[thread].K = 1; // get pure CTF with no envelope
    currentDefocusU[thread] = ctf[thread].DeltafU = defocusU;
    currentDefocusV[thread] = ctf[thread].DeltafV = defocusV;
    currentAngle[thread] = ctf[thread].azimuthal_angle = angle;
    ctf[thread].produceSideInfo();
    if (ctfImage[thread] == nullptr) {
        ctfImage[thread] = new MultidimArray<double>();
        ctfImage[thread]->resizeNoCopy(FourierIfiltered[thread]);
        STARTINGY(*ctfImage[thread]) = STARTINGX(*ctfImage[thread]) = 0;
        if (gen_env) {
            ctfEnvelope[thread] = std::make_unique<MultidimArray<double>>();
            ctfEnvelope[thread]->resizeNoCopy(FourierIfiltered[thread]);
            STARTINGY(*ctfEnvelope[thread]) = STARTINGX(*ctfEnvelope[thread]) = 0;
        }
    }
    local_generateCTF((int) YSIZE(FourierIfiltered[thread]), (int) XSIZE(FourierIfiltered[thread]), *(ctfImage[thread]), &ctf[thread], Ts);
    if (gen_env) {
        local_generateCTF((int) YSIZE(FourierIfiltered[thread]), (int) XSIZE(FourierIfiltered[thread]),
                          *(ctfEnvelope[thread]), &ctf[thread], Ts, true);
    }
    if (phaseFlipped)
        FOR_ALL_ELEMENTS_IN_ARRAY2D(*ctfImage[thread])
                A2D_ELEM(*(ctfImage[thread]), i, j) = fabs(A2D_ELEM(*(ctfImage[thread]), i, j));

    projector->updateCtf(ctfImage[thread], thread);
}

//#define DEBUG
//#define DEBUG2
double cuda_tranformImage(ProgCudaAngularContinuousAssign2 *prm, double rot, double tilt, double psi,
                          double a, double b, const Matrix2D<double> &A, double deltaDefocusU, double deltaDefocusV,
                          double deltaDefocusAngle, int degree, int thread, bool transformuj) {

    bool updateCTF=false;
    if (prm->hasCTF) {
        double defocusU = prm->old_defocusU[thread] + deltaDefocusU;
        double defocusV;
        if (prm->sameDefocus) {
            defocusV = defocusU;
        } else {
            defocusV = prm->old_defocusV[thread] + deltaDefocusV;
        }
        double angle = prm->old_defocusAngle[thread] + deltaDefocusAngle;
        if (defocusU != prm->currentDefocusU[thread] || defocusV != prm->currentDefocusV[thread] ||
            angle != prm->currentAngle[thread]) {
            prm->updateCTFImage(defocusU, defocusV, angle, thread, false);
            updateCTF=true;
        }
    }

    if (prm->old_flip[thread]) {
        MAT_ELEM(A, 0, 0) *= -1;
        MAT_ELEM(A, 0, 1) *= -1;
        MAT_ELEM(A, 0, 2) *= -1;
    }

    projectVolume(*(prm->projector), prm->P[thread](), (int) XSIZE(prm->I[thread]()), (int) XSIZE(prm->I[thread]()),
                  rot, tilt, psi, prm->ctfImage[thread], degree, &(prm->Ifilteredp[thread]()),
                  &(prm->Ifiltered[thread]()), &A, thread, transformuj, updateCTF);

    const MultidimArray<double> &mP = prm->P[thread]();
    MultidimArray<double> &mIfilteredp = prm->Ifilteredp[thread]();

    double cost = 0.0;


    if (prm->contCost == CONTCOST_L1) {
        FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY2D(mP)cost += fabs((DIRECT_A2D_ELEM(mP, i, j)-DIRECT_A2D_ELEM(mIfilteredp, i, j))*DIRECT_A2D_ELEM(prm->mask2D, i, j));
        cost *= prm->iMask2Dsum;
    } else {
        cost = -correlationIndex(mIfilteredp, mP, &(prm->mask2D));
    }
    return cost;
}

double cuda_continuous2cost(double *x, void *_prm) {
    progIdx *pt = (progIdx *) _prm;
    ProgCudaAngularContinuousAssign2 *prm = pt->prog;
    int thread = pt->thread;

    double a = x[1];
    double b = x[2];
    double deltax = x[3];
    double deltay = x[4];
    double scalex = x[5];
    double scaley = x[6];
    double scaleAngle = x[7];
    double deltaRot = x[8];
    double deltaTilt = x[9];
    double deltaPsi = x[10];
    double deltaDefocusU = x[11];
    double deltaDefocusV = x[12];
    double deltaDefocusAngle = x[13];

    if (prm->maxShift > 0 && deltax * deltax + deltay * deltay > prm->maxShift * prm->maxShift)
        return 1e38;
    if (fabs(scalex) > prm->maxScale || fabs(scaley) > prm->maxScale)
        return 1e38;
    if (fabs(deltaRot) > prm->maxAngularChange || fabs(deltaTilt) > prm->maxAngularChange ||
        fabs(deltaPsi) > prm->maxAngularChange)
        return 1e38;
    if (fabs(a - prm->old_grayA[thread]) > prm->maxA)
        return 1e38;
    if (fabs(b) > prm->maxB * prm->Istddev[thread])
        return 1e38;
    if (fabs(deltaDefocusU) > prm->maxDefocusChange || fabs(deltaDefocusV) > prm->maxDefocusChange)
        return 1e38;
//	MAT_ELEM(prm->A,0,0)=1+scalex;
//	MAT_ELEM(prm->A,1,1)=1+scaley;
// In Matlab
//	syms sx sy t
//	R=[cos(t) -sin(t); sin(t) cos(t)]
//	S=[1+sx 0; 0 1+sy]
//	simple(transpose(R)*S*R)
//	[ sx - sx*sin(t)^2 + sy*sin(t)^2 + 1,            -sin(2*t)*(sx/2 - sy/2)]
//	[            -sin(2*t)*(sx/2 - sy/2), sy + sx*sin(t)^2 - sy*sin(t)^2 + 1]
    double sin2_t = sin(scaleAngle) * sin(scaleAngle);
    double sin_2t = sin(2 * scaleAngle);
    MAT_ELEM(prm->A[thread], 0, 0) = 1 + scalex + (scaley - scalex) * sin2_t;
    MAT_ELEM(prm->A[thread], 0, 1) = 0.5 * (scaley - scalex) * sin_2t;
    MAT_ELEM(prm->A[thread], 1, 0) = MAT_ELEM(prm->A[thread], 0, 1);
    MAT_ELEM(prm->A[thread], 1, 1) = 1 + scaley - (scaley - scalex) * sin2_t;
    MAT_ELEM(prm->A[thread], 0, 2) = prm->old_shiftX[thread] + deltax;
    MAT_ELEM(prm->A[thread], 1, 2) = prm->old_shiftY[thread] + deltay;

    bool do_transform = true;
    for (int i = 0; i < 5; ++i) {
        do_transform = do_transform && prm->last_transform[i][thread] == x[i + 3];
    }
    do_transform = !do_transform;
    if (do_transform) {
        for (int i = 0; i < 5; ++i) {
            prm->last_transform[i][thread] = x[i + 3];
        }
    }

    double result = cuda_tranformImage(prm, prm->old_rot[thread] + deltaRot, prm->old_tilt[thread] + deltaTilt,
                                       prm->old_psi[thread] + deltaPsi,
                                       a, b, prm->A[thread], deltaDefocusU, deltaDefocusV, deltaDefocusAngle,
                                       xmipp_transformation::LINEAR, thread, do_transform);
    return result;
}


void ProgCudaAngularContinuousAssign2::processImage(const FileName &fnImg, const FileName &fnImgOut, const MDRow &rowIn,
                                                    MDRow &rowOut) {
    return;
}

// Predict =================================================================
//#define DEBUG
void ProgCudaAngularContinuousAssign2::processImage_t(const FileName &locfnImg, const FileName &locfnImgOut,
                                                      const MDRow &rowIn,
                                                      MDRow &rowOut, int thread) {
    // Read input image and initial parameters
//  ApplyGeoParams geoParams;
//	geoParams.only_apply_shifts=false;
//	geoParams.wrap=DONT_WRAP;
    //AJ some definitions

    double corrIdx = 0.0;
    double corrMask = 0.0;
    double corrWeight = 0.0;
    double imedDist = 0.0;
    projector->last_rot[thread] = 1000.0;
    projector->last_tilt[thread] = 1000.0;
    projector->last_psi[thread] = 1000.0;
    for (int i = 0; i < 5; ++i) {
        last_transform[i][thread] = 1000.0;
    }

    if (verbose >= 2)
        std::cout << rank << ": Processing " << locfnImg << ", " << locfnImgOut << std::endl;
    I[thread].read(locfnImg);
    I[thread]().setXmippOrigin();

    Istddev[thread] = I[thread]().computeStddev();

    Ifiltered[thread]() = I[thread]();

    fftTransformer[thread].FourierTransform(Ifiltered[thread](), FourierIfiltered[thread], false);

    filter[thread].applyMaskFourierSpace(Ifiltered[thread](), FourierIfiltered[thread]);

    old_rot[thread] = rowIn.getValueOrDefault(MDL_ANGLE_ROT, 0.);
    old_tilt[thread] = rowIn.getValueOrDefault(MDL_ANGLE_TILT, 0.);
    old_psi[thread] = rowIn.getValueOrDefault(MDL_ANGLE_PSI, 0.);
    old_shiftX[thread] = rowIn.getValueOrDefault(MDL_SHIFT_X, 0.);
    old_shiftY[thread] = rowIn.getValueOrDefault(MDL_SHIFT_Y, 0.);
    old_flip[thread] = rowIn.getValueOrDefault(MDL_FLIP, false);
    double old_scaleX = 0, old_scaleY = 0, old_scaleAngle = 0;
    old_grayA[thread] = 1;
    old_grayB[thread] = 0;
    if (rowIn.containsLabel(MDL_CONTINUOUS_SCALE_X)) {
        old_scaleX = rowIn.getValue<double>(MDL_CONTINUOUS_SCALE_X);
        old_scaleY = rowIn.getValue<double>(MDL_CONTINUOUS_SCALE_Y);
        if (rowIn.containsLabel(MDL_CONTINUOUS_SCALE_ANGLE))
            old_scaleAngle = rowIn.getValue<double>(MDL_CONTINUOUS_SCALE_ANGLE);
        old_shiftX[thread] = rowIn.getValue<double>(MDL_CONTINUOUS_X);
        old_shiftY[thread] = rowIn.getValue<double>(MDL_CONTINUOUS_Y);
        old_flip[thread] = rowIn.getValue<bool>(MDL_CONTINUOUS_FLIP);
    }

    if (optimizeGrayValues && rowIn.containsLabel(MDL_CONTINUOUS_GRAY_A)) {
        old_grayA[thread] = rowIn.getValue<double>(MDL_CONTINUOUS_GRAY_A);
        old_grayB[thread] = rowIn.getValue<double>(MDL_CONTINUOUS_GRAY_B);
    }

    if ((rowIn.containsLabel(MDL_CTF_DEFOCUSU) || rowIn.containsLabel(MDL_CTF_MODEL)) && !ignoreCTF) {
        hasCTF = true;
        ctf[thread].readFromMdRow(rowIn);
        ctf[thread].produceSideInfo();
        old_defocusU[thread] = ctf[thread].DeltafU;
        old_defocusV[thread] = ctf[thread].DeltafV;
        old_defocusAngle[thread] = ctf[thread].azimuthal_angle;
        updateCTFImage(old_defocusU[thread], old_defocusV[thread], old_defocusAngle[thread], thread);
        //fftTransformer[thread].FourierTransform(Ifiltered[thread](), fftE[thread], false);
        FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY2D(FourierIfiltered[thread]) DIRECT_A2D_ELEM(FourierIfiltered[thread], i,
                                                                                     j) *= DIRECT_A2D_ELEM(
                        *(ctfEnvelope[thread]), i, j);

        //fftTransformer[thread].inverseFourierTransform();
    } else
        hasCTF = false;

    Matrix1D<double> p(13), steps(13);
    // COSS: Gray values are optimized in transform_image_adjust_gray_values
    if (optimizeGrayValues) {
        p(0) = old_grayA[thread]; // a in I'=a*I+b
        p(1) = old_grayB[thread]; // b in I'=a*I+b
    } else {
        p(0) = 1; // a in I'=a*I+b
        p(1) = 0; // b in I'=a*I+b
    }
    p(4) = old_scaleX;
    p(5) = old_scaleY;
    p(6) = old_scaleAngle;

    // default values
    if (fnResiduals != "") {
        rowOut.setValue<String>(MDL_IMAGE_RESIDUAL, "");
    }
    if (fnProjections != "") {
        rowOut.setValue<String>(MDL_IMAGE_REF, "");
    }

    // Optimize
    double cost = -1;

    fftTransformer[thread].inverseFourierTransform();

    if (fabs(old_scaleX) > maxScale || fabs(old_scaleY) > maxScale) {
        rowOut.setValue(MDL_ENABLED, -1);
    } else {
        try {
            cost = 1e38;
            int iter;
            steps.initZeros();
            if (optimizeGrayValues)
                steps(0) = steps(1) = 1.;
            if (optimizeShift)
                steps(2) = steps(3) = 1.;
            if (optimizeScale)
                steps(4) = steps(5) = steps(6) = 1.;
            if (optimizeAngles)
                steps(7) = steps(8) = steps(9) = 1.;
            if (optimizeDefocus) {
                if (sameDefocus)
                    steps(10) = steps(12) = 1.;
                else {
                    steps(10) = steps(11) = steps(12) = 1.;
                    if (hasCTF) {
                        currentDefocusU[thread] = old_defocusU[thread];
                        currentDefocusV[thread] = old_defocusV[thread];
                        currentAngle[thread] = old_defocusAngle[thread];
                    } else
                        currentDefocusU[thread] = currentDefocusV[thread] = currentAngle[thread] = 0;
                }
            }

            progIdx prog{};
            prog.prog = this;
            prog.thread = thread;

            powellOptimizer(p, 1, 13, &cuda_continuous2cost, &prog, 0.01, cost, iter, steps, verbose >= 2);

            E[thread]().initZeros();

            if (contCost == CONTCOST_L1) {
                FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(mask2D) {
                    if (DIRECT_MULTIDIM_ELEM(mask2D, n)) {
                        //DIRECT_MULTIDIM_ELEM(mIfilteredp,n)=DIRECT_MULTIDIM_ELEM(mIfilteredp,n);
                        double val = (DIRECT_MULTIDIM_ELEM(P[thread], n)) - DIRECT_MULTIDIM_ELEM(Ifilteredp[thread], n);
                        DIRECT_MULTIDIM_ELEM(E[thread], n) = val;
                    } else
                        DIRECT_MULTIDIM_ELEM(Ifilteredp[thread], n) = 0;
                }
            } else {
                FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(mask2D) {
                    if (DIRECT_MULTIDIM_ELEM(mask2D, n)) {
                        double val =
                                DIRECT_MULTIDIM_ELEM(P[thread](), n) - DIRECT_MULTIDIM_ELEM(Ifilteredp[thread](), n);
                        DIRECT_MULTIDIM_ELEM(E[thread](), n) = val;
                    } else {
                        DIRECT_MULTIDIM_ELEM(Ifilteredp[thread], n) = 0;
                    }
                }
            }

            if (cost > 1e30 || (cost > 0 && contCost == CONTCOST_CORR)) {
                rowOut.setValue(MDL_ENABLED, -1);
                p.initZeros();
                if (optimizeGrayValues) {
                    p(0) = old_grayA[thread]; // a in I'=a*I+b
                    p(1) = old_grayB[thread]; // b in I'=a*I+b
                } else {
                    p(0) = 1;
                    p(1) = 0;
                }
                p(4) = old_scaleX;
                p(5) = old_scaleY;
                p(6) = old_scaleAngle;

            } else {
                //Calculating several similarity measures between P and Ifilteredp (correlations and imed)
                corrIdx = correlationIndex(P[thread](), Ifilteredp[thread]());
                corrMask = correlationMasked(P[thread](), Ifilteredp[thread]());
                corrWeight = correlationWeighted(P[thread](), Ifilteredp[thread]());
                imedDist = imedDistance(P[thread](), Ifilteredp[thread]());

                if (fnResiduals != "") {
                    FileName fnResidual;
                    fnResidual.compose(locfnImgOut.getPrefixNumber(), fnResiduals);
                    E[thread].write(fnResidual);
                    rowOut.setValue(MDL_IMAGE_RESIDUAL, fnResidual);
                }
                if (fnProjections != "") {
                    FileName fnProjection;
                    fnProjection.compose(locfnImgOut.getPrefixNumber(), fnProjections);
                    P[thread].write(fnProjection);
                    rowOut.setValue(MDL_IMAGE_REF, fnProjection);
                }
            }

            if (contCost == CONTCOST_CORR)
                cost = -cost;
            if (verbose >= 2)
                std::cout << "I'=" << p(0) << "*I" << "+" << p(1) << " Dshift=(" << p(2) << "," << p(3) << ") "
                          << "scale=(" << 1 + p(4) << "," << 1 + p(5) << ", angle=" << p(6) << ") Drot=" << p(7)
                          << " Dtilt=" << p(8)
                          << " Dpsi=" << p(9) << " DU=" << p(10) << " DV=" << p(11) << " Dalpha=" << p(12) << std::endl;

            // Apply
            FileName fnOrig;
            rowIn.getValue(MDL::str2Label(originalImageLabel), fnOrig);
            I[thread].read(locfnImg);
            if (XSIZE(Ip[thread]()) != XSIZE(I[thread]())) {
                scaleToSize(xmipp_transformation::BSPLINE3, Ip[thread](), I[thread](), XSIZE(Ip[thread]()),
                            YSIZE(Ip[thread]()));
                I[thread]() = Ip[thread]();
            }

            A[thread](0, 2) = p(2) + old_shiftX[thread];
            A[thread](1, 2) = p(3) + old_shiftY[thread];
            double scalex = p(4);
            double scaley = p(5);
            double scaleAngle = p(6);
            double sin2_t = sin(scaleAngle) * sin(scaleAngle);
            double sin_2t = sin(2 * scaleAngle);
            A[thread](0, 0) = 1 + scalex + (scaley - scalex) * sin2_t;
            A[thread](0, 1) = 0.5 * (scaley - scalex) * sin_2t;
            A[thread](1, 0) = A[thread](0, 1);
            A[thread](1, 1) = 1 + scaley - (scaley - scalex) * sin2_t;
//			A(0,0)=1+p(4);
//			A(1,1)=1+p(5);

            if (old_flip[thread]) {
                MAT_ELEM(A[thread], 0, 0) *= -1;
                MAT_ELEM(A[thread], 0, 1) *= -1;
                MAT_ELEM(A[thread], 0, 2) *= -1;
            }
            applyGeometry(xmipp_transformation::BSPLINE3, Ip[thread](), I[thread](), A[thread],
                          xmipp_transformation::IS_NOT_INV,
                          xmipp_transformation::DONT_WRAP);

            if (optimizeGrayValues) {
                MultidimArray<double> &mIp = Ip[thread]();
                double ia = 1.0 / p(0);
                double b = p(1);
                FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(mIp) {
                    if (DIRECT_MULTIDIM_ELEM(mask2D, n))
                        DIRECT_MULTIDIM_ELEM(mIp, n) = ia * (DIRECT_MULTIDIM_ELEM(mIp, n) - b);
                    else
                        DIRECT_MULTIDIM_ELEM(mIp, n) = 0.0;
                }
            }
            Ip[thread].write(locfnImgOut);

        }
        catch (XmippError &XE) {
            std::cerr << XE.what() << std::endl;
            std::cerr << "Warning: Cannot refine " << locfnImg << std::endl;
            rowOut.setValue(MDL_ENABLED, -1);
        }
    }
    rowOut.setValue(MDL_IMAGE_ORIGINAL, locfnImg);
    rowOut.setValue(MDL_IMAGE, locfnImgOut);
    rowOut.setValue(MDL_ANGLE_ROT, old_rot[thread] + p(7));
    rowOut.setValue(MDL_ANGLE_TILT, old_tilt[thread] + p(8));
    rowOut.setValue(MDL_ANGLE_PSI, old_psi[thread] + p(9));
    rowOut.setValue(MDL_SHIFT_X, 0.);
    rowOut.setValue(MDL_SHIFT_Y, 0.);
    rowOut.setValue(MDL_FLIP, false);
    rowOut.setValue(MDL_COST, cost);
    if (optimizeGrayValues) {
        rowOut.setValue(MDL_CONTINUOUS_GRAY_A, p(0));
        rowOut.setValue(MDL_CONTINUOUS_GRAY_B, p(1));
    }
    rowOut.setValue(MDL_CONTINUOUS_SCALE_X, p(4));
    rowOut.setValue(MDL_CONTINUOUS_SCALE_Y, p(5));
    rowOut.setValue(MDL_CONTINUOUS_SCALE_ANGLE, p(6));
    rowOut.setValue(MDL_CONTINUOUS_X, p(2) + old_shiftX[thread]);
    rowOut.setValue(MDL_CONTINUOUS_Y, p(3) + old_shiftY[thread]);
    rowOut.setValue(MDL_CONTINUOUS_FLIP, old_flip[thread]);
    if (hasCTF) {
        rowOut.setValue(MDL_CTF_DEFOCUSU, old_defocusU[thread] + p(10));
        if (sameDefocus)
            rowOut.setValue(MDL_CTF_DEFOCUSV, old_defocusU[thread] + p(10));
        else
            rowOut.setValue(MDL_CTF_DEFOCUSV, old_defocusV[thread] + p(11));
        rowOut.setValue(MDL_CTF_DEFOCUS_ANGLE, old_defocusAngle[thread] + p(12));
        if (sameDefocus)
            rowOut.setValue(MDL_CTF_DEFOCUS_CHANGE, 0.5 * (p(10) + p(10)));
        else
            rowOut.setValue(MDL_CTF_DEFOCUS_CHANGE, 0.5 * (p(10) + p(11)));
        if (old_defocusU[thread] + p(10) < 0 || old_defocusU[thread] + p(11) < 0)
            rowOut.setValue(MDL_ENABLED, -1);
    }
    //Saving correlation and imed values in the metadata
    rowOut.setValue(MDL_CORRELATION_IDX, corrIdx);
    rowOut.setValue(MDL_CORRELATION_MASK, corrMask);
    rowOut.setValue(MDL_CORRELATION_WEIGHT, corrWeight);
    rowOut.setValue(MDL_IMED, imedDist);
}

#undef DEBUG

void ProgCudaAngularContinuousAssign2::postProcess() {

    MetaData &ptrMdOut = getOutputMd();
    ptrMdOut.removeDisabled();
    if (contCost == CONTCOST_L1) {
        double minCost = 1e38;
        for (size_t objId: ptrMdOut.ids()) {
            double cost;
            ptrMdOut.getValue(MDL_COST, cost, objId);
            if (cost < minCost)
                minCost = cost;
        }
        for (size_t objId: ptrMdOut.ids()) {
            double cost;
            ptrMdOut.getValue(MDL_COST, cost, objId);
            ptrMdOut.setValue(MDL_WEIGHT_CONTINUOUS2, minCost / cost, objId);
        }
    } else {
        double maxCost = -1e38;
        for (size_t objId: ptrMdOut.ids()) {
            double cost;
            ptrMdOut.getValue(MDL_COST, cost, objId);
            if (cost > maxCost)
                maxCost = cost;
        }
        for (size_t objId: ptrMdOut.ids()) {
            double cost;
            ptrMdOut.getValue(MDL_COST, cost, objId);
            ptrMdOut.setValue(MDL_WEIGHT_CONTINUOUS2, cost / maxCost, objId);
        }
    }
    ptrMdOut.write(fn_out.replaceExtension("xmd"));
}
