 /* Authors:     Estrella Fernandez-Gimenez (me.fernandez@cnb.csic.es)*/

#include <core/multidim_array.h>
#include <core/xmipp_image.h>
#include <iostream>
#include <gtest/gtest.h>
#include <reconstruction/volume_subtraction.cpp>

class VolSubtractionTest : public ::testing::Test
{
protected:
    virtual void SetUp()
    {
        //get example volume
        if (chdir(((String)(getXmippPath() + (String)"/resources/test/image")).c_str())==-1)
            REPORT_ERROR(ERR_UNCLASSIFIED,"Cannot change directory");
        img.read("smallVolume.vol");
    }

    Image<double> img;

};

TEST_F(VolSubtractionTest, subtraction)
{
	// Substract two identical images with a non-specific mask (all 1s)
	// expecting for the result to be an empty image (all 0s)
	Image<double> empty, mask;
	empty().initZeros(4, 64, 64);
	mask().initZeros(4, 64, 64);
	mask().initConstant(1);
	subtraction(img(), img(), img(), mask());
	ASSERT_EQ(img(), empty());
}

