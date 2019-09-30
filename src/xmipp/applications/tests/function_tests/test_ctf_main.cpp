
#include <iostream>

#include <core/xmipp_image.h>
#include <data/transform_downsample.h>
#include <gtest/gtest.h>
#include <data/ctf.h>

// MORE INFO HERE: http://code.google.com/p/googletest/wiki/AdvancedGuide
// This test is named "Size", and belongs to the "MetadataTest"
// test case.
class CtfTest : public ::testing::Test
{
protected:
    //init metadatas
    virtual void SetUp()
    {

        try
        {
            //get example images/staks
            if (chdir(((String)(getXmippPath() + (String)"/resources/test")).c_str())==-1)
            	REPORT_ERROR(ERR_UNCLASSIFIED,"Could not change directory");
            // testBaseName = xmippPath + "/resources/test";
            imageName = "image/singleImage.spi";

        }
        catch (XmippError &xe)
        {
            std::cerr << xe;
            exit(-1);
        }
    }

    // virtual void TearDown() {}//Destructor
    Image<double> myImageFloat;
    FileName imageName;


};

TEST_F( CtfTest, generateImageWithTwoCTFs)
{
    XMIPP_TRY
    MetaData metadata1;
    long objectId = metadata1.addObject();
    metadata1.setValue(MDL_CTF_SAMPLING_RATE, 1., objectId);
    metadata1.setValue(MDL_CTF_VOLTAGE, 300., objectId);
    metadata1.setValue(MDL_CTF_DEFOCUSU, 26000., objectId);
    metadata1.setValue(MDL_CTF_DEFOCUSV, 2000., objectId);
    metadata1.setValue(MDL_CTF_DEFOCUS_ANGLE, 45., objectId);
    metadata1.setValue(MDL_CTF_CS, 2., objectId);
    metadata1.setValue(MDL_CTF_Q0, 0.1, objectId);
    MultidimArray<double> in;
    generateCTFImageWith2CTFs(metadata1, metadata1, 256, in);
    Image<double> img;
    img().alias(in);
    //img.write("/tmp/kk.mrc");
    EXPECT_TRUE(true);
    XMIPP_CATCH
}

TEST_F( CtfTest, errorBetween2CTFs)
{
    XMIPP_TRY
    MetaData metadata1;
    long objectId = metadata1.addObject();
    metadata1.setValue(MDL_CTF_SAMPLING_RATE, 2.1, objectId);
    metadata1.setValue(MDL_CTF_VOLTAGE, 300., objectId);
    metadata1.setValue(MDL_CTF_DEFOCUSU, 5000., objectId);
    metadata1.setValue(MDL_CTF_DEFOCUSV, 10000., objectId);
    metadata1.setValue(MDL_CTF_DEFOCUS_ANGLE, -45., objectId);
    metadata1.setValue(MDL_CTF_CS, 2., objectId);
    metadata1.setValue(MDL_CTF_Q0, 0.1, objectId);

    MetaData metadata2;
    objectId = metadata2.addObject();
    metadata2.setValue(MDL_CTF_SAMPLING_RATE, 2.1, objectId);
    metadata2.setValue(MDL_CTF_VOLTAGE, 300., objectId);
    metadata2.setValue(MDL_CTF_DEFOCUSU, 10000., objectId);
    metadata2.setValue(MDL_CTF_DEFOCUSV, 10000., objectId);
    metadata2.setValue(MDL_CTF_DEFOCUS_ANGLE, 45., objectId);
    metadata2.setValue(MDL_CTF_CS, 2., objectId);
    metadata2.setValue(MDL_CTF_Q0, 0.1, objectId);

    double error = errorBetween2CTFs(metadata1,
                             metadata2,
                             256,
                             0.05,0.25);
    EXPECT_FLOAT_EQ(error,7121.4971);
    XMIPP_CATCH
}

TEST_F( CtfTest, errorMaxFreqCTFs)
{
    XMIPP_TRY
    MetaData metadata1;
    long objectId = metadata1.addObject();
    metadata1.setValue(MDL_CTF_SAMPLING_RATE, 2., objectId);
    metadata1.setValue(MDL_CTF_VOLTAGE, 300., objectId);
    metadata1.setValue(MDL_CTF_DEFOCUSU, 6000., objectId);
    metadata1.setValue(MDL_CTF_DEFOCUSV, 7500., objectId);
    metadata1.setValue(MDL_CTF_DEFOCUS_ANGLE, 45., objectId);
    metadata1.setValue(MDL_CTF_CS, 2., objectId);
    metadata1.setValue(MDL_CTF_Q0, 0.1, objectId);
    double resolution;
//    for (double f=0; f < 4000; f+=100.){
//    		metadata1.setValue(MDL_CTF_DEFOCUSV, f, objectId);
//            resolution = errorMaxFreqCTFs(metadata1,HALFPI);
//            std::cerr << "f=" << f << " " << resolution <<std::endl;
//    }
    resolution = errorMaxFreqCTFs(metadata1,HALFPI);
    EXPECT_FLOAT_EQ(resolution,7.6852355);
    XMIPP_CATCH
}

TEST_F( CtfTest, errorMaxFreqCTFs2D)
{
    XMIPP_TRY
    MetaData metadata1;
    long objectId = metadata1.addObject();
    metadata1.setValue(MDL_CTF_SAMPLING_RATE, 2., objectId);
    metadata1.setValue(MDL_CTF_VOLTAGE, 300., objectId);
    metadata1.setValue(MDL_CTF_DEFOCUSU, 10000., objectId);
    metadata1.setValue(MDL_CTF_DEFOCUSV, 5400., objectId);
    metadata1.setValue(MDL_CTF_DEFOCUS_ANGLE, 45., objectId);
    metadata1.setValue(MDL_CTF_CS, 2., objectId);
    metadata1.setValue(MDL_CTF_Q0, 0.1, objectId);

    MetaData metadata2;
    objectId = metadata2.addObject();
    metadata2.setValue(MDL_CTF_SAMPLING_RATE, 2., objectId);
    metadata2.setValue(MDL_CTF_VOLTAGE, 300., objectId);
    metadata2.setValue(MDL_CTF_DEFOCUSU, 5000., objectId);
    metadata2.setValue(MDL_CTF_DEFOCUSV, 5000., objectId);
    metadata2.setValue(MDL_CTF_DEFOCUS_ANGLE, 45., objectId);
    metadata2.setValue(MDL_CTF_CS, 2., objectId);
    metadata2.setValue(MDL_CTF_Q0, 0.1, objectId);

    double resolution = errorMaxFreqCTFs2D(metadata1,
                             metadata2,256,HALFPI);
    EXPECT_NEAR(resolution,13.921659080780355,0.00001);
    XMIPP_CATCH
}

GTEST_API_ int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

