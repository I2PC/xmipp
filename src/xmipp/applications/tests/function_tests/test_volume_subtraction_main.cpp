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
        A().resize(3,3);
		A().initConstant(0.0);
		A(1,1,1) = 1;
    }
    Image<double> A;
    Image<double> img;
	FourierFilter filter;
};

TEST_F(VolSubtractionTest, subtraction)
{
	// Substract two identical volumes with a non-specific mask (all 1s)
	Image<double> mask;
	mask().initZeros(XSIZE(A()),YSIZE(A()),YSIZE(A()));
	mask().initConstant(1);
	createFilter(filter, 0);
	img = subtraction(A, A, mask(), "", "", filter, 0);
	ASSERT_EQ(img().sum(), 0.0);
}

