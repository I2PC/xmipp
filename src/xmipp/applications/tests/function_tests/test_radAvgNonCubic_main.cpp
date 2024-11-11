 /* Authors:     Estrella Fernandez-Gimenez (me.fernandez@cnb.csic.es)*/

#include <core/multidim_array.h>
#include <core/xmipp_image.h>
#include <iostream>
#include <gtest/gtest.h>
#include "core/transformations.h"

class RadAvgNonCubicTest : public ::testing::Test
{
protected:
    virtual void SetUp()
    {
        //get example volume
        if (chdir(((String)(getXmippSrcPath() + (String)"/xmipp/resources/test/image")).c_str())==-1)
            REPORT_ERROR(ERR_UNCLASSIFIED,"Cannot change directory");
        img.read("smallVolume.vol");
    }

    Image<double> img;
	MultidimArray<double> radial_meanV;
	MultidimArray<int> radial_count;
};

TEST_F(RadAvgNonCubicTest, radavgNonCubic)
{
	img().setXmippOrigin();
	Matrix1D<int> center(2);
	center.initZeros();
    radialAverageNonCubic(img(), center, radial_meanV, radial_count);
    ASSERT_TRUE(XSIZE(radial_meanV)==46);
    ASSERT_TRUE(XSIZE(radial_count)==46);
    ASSERT_TRUE(radial_meanV(45)==0);
    ASSERT_TRUE(radial_count(0)==4);
}

TEST_F(RadAvgNonCubicTest, radavgNonCubicRounding)
{
	img().setXmippOrigin();
	Matrix1D<int> center(2);
	center.initZeros();
    radialAverageNonCubic(img(), center, radial_meanV, radial_count, true);
    ASSERT_TRUE(XSIZE(radial_meanV)==47);
    ASSERT_TRUE(XSIZE(radial_count)==47);
    ASSERT_TRUE(radial_meanV(46)==0);
    ASSERT_TRUE(radial_count(0)==4);
}
