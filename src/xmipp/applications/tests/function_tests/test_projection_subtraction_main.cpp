 /* Authors:     Estrella Fernandez-Gimenez (me.fernandez@cnb.csic.es)*/

#include <core/multidim_array.h>
#include <core/xmipp_image.h>
#include <iostream>
#include <gtest/gtest.h>
#include "core/transformations.h"
#include <core/xmipp_fftw.h>
#include <reconstruction/subtract_projection.cpp>

class POCSTestProj : public ::testing::Test
{
public:
	POCSTestProj()
	{
		//get results to compare
		if (chdir(((String)(getXmippPath() + (String)"/resources/test/pocs")).c_str())==-1)
			REPORT_ERROR(ERR_UNCLASSIFIED,"Cannot change directory");
		pocsmask.read("pocsmask.mrc");
		pocsamplitude_radavg.read("pocsamplitude_radavg.mrc");
		pocsminmax.read("pocsminmax.mrc");
		pocsphase.read("pocsphase.mrc");
	}
protected:
    virtual void SetUp()
    {
        //get example volume
        if (chdir(((String)(getXmippPath() + (String)"/resources/test/pocs")).c_str())==-1)
            REPORT_ERROR(ERR_UNCLASSIFIED,"Cannot change directory");
        img.read("V1.mrc");
    }

    Image<double> img;
    Image<double> pocsmask;
    Image<double> pocsnonnegative;
    Image<double> pocsamplitude_radavg;
    Image<double> pocsminmax;
    Image<double> pocsphase;
    Image<double> result;
	double min;
	double max;
	FourierTransformer transformer;
	MultidimArray< std::complex<double> > IFourier;
	MultidimArray<double> IFourierMag;
	MultidimArray<double> radial_meanI;
	MultidimArray<double> radial_meanP;
	MultidimArray<double> radQuotient;
	MultidimArray<std::complex<double> > IFourierPhase;
	FourierTransformer transformer2;
};

/* Test for Projectors Onto Convex Sets (POCS) and subtraction used for projection subtraction.*/

TEST_F(POCSTestProj, pocsmask)
{
	Image<double> mask;
	mask().initZeros(XSIZE(img()),YSIZE(img()),YSIZE(img()));
	mask().initConstant(1);
	POCSmaskProj(mask(), img());
	ASSERT_EQ(img(), pocsmask());
}

TEST_F(POCSTestProj, pocsamplituderadAvg)
{
	radial_meanI = computeRadialAvg(img, radial_meanI);
	radial_meanP = computeRadialAvg(img, radial_meanP);
	radQuotient = computeRadQuotient(radQuotient, radial_meanI, radial_meanP);
	transformer.FourierTransform(img(), IFourier);
	FFT_magnitude(IFourier,IFourierMag);
	POCSFourierAmplitudeProj(IFourierMag, IFourier, 1, radQuotient, (int)XSIZE(img()));
	transformer.inverseFourierTransform();
	ASSERT_EQ(img(), pocsamplitude_radavg());
}

TEST_F(POCSTestProj, pocsminmax)
{
	img().computeDoubleMinMax(min, max);
	POCSMinMaxProj(img(), min, max);
	ASSERT_EQ(img(), pocsminmax());
}

TEST_F(POCSTestProj, pocsphase)
{
	transformer.FourierTransform(img(),IFourier,false);
	transformer2.FourierTransform(img(),IFourierPhase,true);
	POCSFourierPhaseProj(IFourierPhase,IFourier);
	transformer2.inverseFourierTransform();
	ASSERT_EQ(img(), pocsphase());
}

TEST_F(POCSTestProj, subtraction)
{
	// Substract two identical projections with a non-specific mask (all 1s)
	Image<double> mask;
	Image<double> invmask;
	mask().initZeros(XSIZE(img()),YSIZE(img()));
	mask().initConstant(1);
	invmask = invertMask(mask);
	img = subtraction(img, img, invmask(), mask());
	ASSERT_EQ(img(), result());
}

