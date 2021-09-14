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
        if (chdir(((String)(getXmippPath() + (String)"/resources/test/pocs")).c_str())==-1)
            REPORT_ERROR(ERR_UNCLASSIFIED,"Cannot change directory");
        img.read("V1.mrc");
        img2.read("V.mrc");
        //get results to compare
        result.read("subtraction.mrc");
    }

    Image<double> img;
    Image<double> img2;
    Image<double> result;
    FourierFilter filter;
};

TEST_F(VolSubtractionTest, subtraction)
{
	// Substract two identical volumes with a non-specific mask (all 1s)
	Image<double> mask;
	mask().initZeros(XSIZE(img()),YSIZE(img()),YSIZE(img()));
	mask().initConstant(1);
	createFilter(filter, 0);
	img = subtraction(img, img2, mask(), "", "", filter, 0);
	ASSERT_EQ(img(), result());
}

