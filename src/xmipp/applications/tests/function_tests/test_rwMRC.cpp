#include <core/rwMRC.h>
#include <iostream>
#include <gtest/gtest.h>
#include <core/xmipp_image_base.h>
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
//        try
//        {
//            im.read(imageName);
//        }
//        catch (XmippError &xe)
//        {
//            std::cerr << xe;
//            exit(-1);
//        }
    }
    //Image to be fitted:
    Image<double> im;
    //File name of the image to process
    FileName imageName;
};


TEST_F(rwMRCTest, readMapped) {
//    int datamode = 2;
      size_t select_img = sizeof(imageName)
//    bool mapData = False;
//    int mode = 2;
    ASSERT_EQ(im.readMapped(imageName, select_img), 0);
    //ASSERT_EQ(im.read(imageName, datamode, select_img, mapData, mode), 0);
    //err = ImageBase::readMRC(imageName,false); is protected
    //ImageBase::dataMode
}

//TEST_F(rwMRCTest, writeMRC) {
//    while (file has still elements) {
//      wchar_t* expectedField = L"expectingThis";
//      wchar_t* readString = el->field;
//      EXPECT_EQ(expectedField,writeMRC);
//    }