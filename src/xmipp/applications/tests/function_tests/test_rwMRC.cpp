#include <iostream>
#include <gtest/gtest.h>
#include <core/xmipp_image_base.h>
#include <core/xmipp_image_extension.h>
#include <core/xmipp_image.h>
#include <filesystem>// MORE INFO HERE: http://code.google.com/p/googletest/wiki/AdvancedGuide

class rwMRCTest : public ::testing::Test {
protected:
    virtual void SetUp()
    {
        FileName path=getXmippSrcPath();
        path+="/xmipp/resources/test";
        if (chdir(path.c_str()) != 0 ) FAIL() << "Could not change path to: " << path;
        imageName = "image/singleImage.mrc";
    }
    //Image to be fitted:
    Image<double> im;
    FileName imageName;
};


TEST_F(rwMRCTest, readMRC) {
    //ASSERT_TRUE(im.read(imageName, 3, select_img, false, datamode) ==0);
    ASSERT_TRUE(im.read(imageName) == 0);
}


//TEST_F(rwMRCTest, writeMRC) {
//    im2w.setDataMode(_DATA_ALL);
//    ASSERT_TRUE(im2w.write(imageNameWrite) == 0);
//    im2w.clear();
//    ASSERT_TRUE(im2w.read(imageName) == 0);
//}