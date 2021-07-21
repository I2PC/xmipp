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
#define len 128
        //get example volume
        if (chdir(((String)(getXmippPath() + (String)"/resources/test/image")).c_str())==-1)
            REPORT_ERROR(ERR_UNCLASSIFIED,"Cannot change directory");
        img.read("smallVolume.vol");
    }

    Image<double> img;

};

TEST_F(RadAvgNonCubicTest, radavgNonCubic)
{
	MultidimArray<double> radial_meanV;
	MultidimArray<int> radial_count;
	Matrix1D<int> center(2);
	center.initZeros();
	img().setXmippOrigin();

    radialAverageNonCubic(img(), center, radial_meanV, radial_count);

	ASSERT_TRUE(XSIZE(radial_meanV)==45);
}

