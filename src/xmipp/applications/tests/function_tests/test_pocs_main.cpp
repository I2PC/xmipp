 /* Authors:     Estrella Fernandez-Gimenez (me.fernandez@cnb.csic.es)*/

#include <core/multidim_array.h>
#include <core/xmipp_image.h>
#include <iostream>
#include <gtest/gtest.h>
#include "core/transformations.h"
#include <core/xmipp_fftw.h>
#include <reconstruction/volume_subtraction.cpp>

class POCSTest : public ::testing::Test
{
protected:
    virtual void SetUp()
    {
        if (chdir(((String)(getXmippPath() + (String)"/resources/test")).c_str())==-1)
            REPORT_ERROR(ERR_UNCLASSIFIED,"Could not change directory");
        img().resize(3,3);
        img().initConstant(0.0);
        img(1,1,1) = 1;
        imgAux = img;
    }
    Image<double> img;
    Image<double> imgAux;
	FourierFilter filter;
	double min;
	double max;
	FourierTransformer transformer;
	MultidimArray< std::complex<double> > IFourier;
	MultidimArray<double> IFourierMag;
	MultidimArray<double> IFourierMag2;
	MultidimArray<double> radial_meanI;
	MultidimArray<double> radQuotient;
	MultidimArray<std::complex<double> > IFourierPhase;
	FourierTransformer transformer2;
};

/* Test for Projectors Onto Convex Sets (POCS) used for volume adjustment in program volume_subtraction.*/

TEST_F(POCSTest, pocsmask)
{
	Image<double> mask;
	mask().initZeros(XSIZE(img()),YSIZE(img()),YSIZE(img()));
	mask().initConstant(1);
	POCSmask(mask(), img());
	ASSERT_EQ(imgAux(), img());
}

TEST_F(POCSTest, pocsnonnegative)
{
	POCSnonnegative(img());
	ASSERT_EQ(imgAux(), img());
}

TEST_F(POCSTest, pocsamplitude)
{
	transformer.completeFourierTransform(img(), IFourier);
	FFT_magnitude(IFourier,IFourierMag);
	POCSFourierAmplitude(IFourierMag, IFourier, 1);
	transformer.inverseFourierTransform();
	ASSERT_EQ(imgAux(), img());
}

TEST_F(POCSTest, pocsamplituderadAvg)
{
	IFourierMag = computeMagnitude(img());
	IFourierMag2 = computeMagnitude(img());
	radQuotient = computeRadQuotient(IFourierMag, IFourierMag2, img(), img());
	transformer.FourierTransform(img(), IFourier);
	POCSFourierAmplitudeRadAvg(IFourier, 1, radQuotient, XSIZE(img()),  YSIZE(img()),  ZSIZE(img()));
	transformer.inverseFourierTransform();
	ASSERT_EQ(imgAux(), img());
}

TEST_F(POCSTest, pocsminmax)
{
	img().computeDoubleMinMax(min, max);
	POCSMinMax(img(), min, max);
	ASSERT_EQ(imgAux(), img());
}

TEST_F(POCSTest, pocsphase)
{
	transformer.FourierTransform(img(),IFourier,false);
	transformer2.FourierTransform(img(),IFourierPhase,true);
	POCSFourierPhase(IFourierPhase,IFourier);
	transformer2.inverseFourierTransform();
	ASSERT_EQ(imgAux(), img());
}
