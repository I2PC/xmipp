#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <core/metadata_extension.h>
#include <data/xmipp_image_convert.h>
#include <core/xmipp_funcs.h>
#include <iostream>
#include <gtest/gtest.h>
#include <string.h>
#include <fstream>
#include <sys/time.h>
#include "core/metadata_vec.h"

#define N_ROWS_TEST 2
#define N_ROWS_PERFORMANCE_TEST 8000


/*
 * Define a "Fixture so we may reuse the metadatas
 */
class MetadataTest : public ::testing::Test
{
protected:
    //init metadatas

    virtual void SetUp()
    {
        size_t id;

        if (chdir(((String)(getXmippPath() + (String)"/resources/test")).c_str())==-1)
            REPORT_ERROR(ERR_UNCLASSIFIED,"Could not change directory");
        // Md1
        id = mDsource.addObject();
        mDsourceIds.push_back(id);
        mDsource.setValue(MDL_X, 1., id);
        mDsource.setValue(MDL_Y, 2., id);

        id = mDsource.addObject();
        mDsourceIds.push_back(id);
        mDsource.setValue(MDL_X, 3., id);
        mDsource.setValue(MDL_Y, 4., id);

        // mDanotherSource
        id = mDanotherSource.addObject();
        mDanotherSource.setValue(MDL_X, 11., id);
        mDanotherSource.setValue(MDL_Y, 22., id);

        id = mDanotherSource.addObject();
        mDanotherSource.setValue(MDL_X, 33., id);
        mDanotherSource.setValue(MDL_Y, 44., id);

        // Md UnionAll
        mDunion = mDsource;
        id = mDunion.addObject();
        mDunion.setValue(MDL_X, 11., id);
        mDunion.setValue(MDL_Y, 22., id);

        id = mDunion.addObject();
        mDunion.setValue(MDL_X, 33., id);
        mDunion.setValue(MDL_Y, 44., id);
    }

    MetaDataVec mDsource, mDanotherSource;
    MetaDataVec mDunion;
    std::vector<int> mDsourceIds;
};

TEST_F(MetadataTest, IdIteration)
{
    auto it = mDsource.ids().begin();
    for (size_t i = 0; i < mDsourceIds.size(); i++, ++it);
    ASSERT_EQ(it, mDsource.ids().end());

    size_t i = 0;
    for (size_t objId : mDsource.ids()) {
        ASSERT_EQ(objId, mDsourceIds[i]);
        i++;
    }
    ASSERT_EQ(i, mDsourceIds.size());
}

TEST_F(MetadataTest, RowIteration)
{
    auto it = mDsource.begin();
    for (size_t i = 0; i < mDsourceIds.size(); i++, ++it);
    ASSERT_EQ(it, mDsource.end());

    size_t i = 0;
    for (const auto& row : mDsource) {
        ASSERT_EQ(row.id(), mDsourceIds[i]);
        i++;
    }
    ASSERT_EQ(i, mDsourceIds.size());
}

TEST_F(MetadataTest, SimilarToOperator)
{
    size_t id;
    MetaDataVec auxMetadata;

    ASSERT_EQ(mDsource, mDsource);
    ASSERT_FALSE(mDsource == mDanotherSource);

    // Attribute order should not be important
    id = auxMetadata.addObject();
    auxMetadata.setValue(MDL_Y, 2., id);
    auxMetadata.setValue(MDL_X, 1., id);
    id = auxMetadata.addObject();
    auxMetadata.setValue(MDL_Y, 4., id);
    auxMetadata.setValue(MDL_X, 3., id);
    ASSERT_EQ(auxMetadata,mDsource);

    // Test double with a given precission
    auxMetadata.clear();
    auxMetadata.setPrecission(2);
    id = auxMetadata.addObject();
    auxMetadata.setValue(MDL_Y, 2.001, id);
    auxMetadata.setValue(MDL_X, 1., id);
    id = auxMetadata.addObject();
    auxMetadata.setValue(MDL_Y, 4., id);
    auxMetadata.setValue(MDL_X, 3., id);
    ASSERT_TRUE(auxMetadata == mDsource);
    auxMetadata.setPrecission(4);
    ASSERT_FALSE(auxMetadata == mDsource);
}

TEST_F(MetadataTest, AddLabel)
{
    MetaDataVec auxMetadata = mDunion;
    auxMetadata.addLabel(MDL_Z);
    std::vector<MDLabel> v1 = {MDL_X, MDL_Y, MDL_Z};
    std::vector<MDLabel> v2 = auxMetadata.getActiveLabels();
    EXPECT_EQ(v2, v1);
}

TEST_F(MetadataTest, AddRow)
{
    MetaDataVec md, md2;

    MDRowVec row;
    row.setValue(MDL_X, 1.);
    row.setValue(MDL_Y, 2.);
    md.addRow(row);
    row.setValue(MDL_X, 3.);
    row.setValue(MDL_Y, 4.);
    md.addRow(row);

    row.setValue(MDL_X, 1.);
    row.setValue(MDL_Y, 2.);
    md2.addRow(row);
    row.setValue(MDL_X, 3.);
    row.setValue(MDL_Y, 4.);
    md2.addRow(row);

    EXPECT_EQ(md, mDsource);
    EXPECT_EQ(md2, mDsource);
}

TEST_F(MetadataTest, AddRowsPerformance)
{
    MetaDataVec md, md2, md3;
    MDRowVec row;  // Sample row

    printf("N_ROWS_PERFORMANCE_TEST = %d\n", N_ROWS_PERFORMANCE_TEST);

    // Initialize row.
    row.setValue(MDL_X,1.);
    row.setValue(MDL_Y,2.);
    row.setValue(MDL_ZSCORE,3.);
    row.setValue(MDL_ZSCORE_HISTOGRAM,4.);
    row.setValue(MDL_ZSCORE_RESMEAN,5.);
    row.setValue(MDL_ZSCORE_RESVAR,6.);
    row.setValue(MDL_ZSCORE_RESCOV,7.);
    row.setValue(MDL_ZSCORE_SHAPE1,8.);
    row.setValue(MDL_ZSCORE_SHAPE2,9.);
    row.setValue(MDL_ZSCORE_SNR1,10.);
    row.setValue(MDL_ZSCORE_SNR2,11.);
    row.setValue(MDL_IMAGE, String("particles.stk"));
    row.setValue(MDL_SHIFT_X_DIFF, 1.5);
    row.setValue(MDL_SHIFT_Y_DIFF, 2.5);
    row.setValue(MDL_CONTINUOUS_X, 1.5);
    row.setValue(MDL_CONTINUOUS_Y, 2.5);
    row.setValue(MDL_SHIFT_X, 1.5);
    row.setValue(MDL_SHIFT_Y, 2.5);
    row.setValue(MDL_SHIFT_Z, 3.5);

    Timer t;
    size_t s1, s2, s3;

    t.tic();

    for (int i=0; i<N_ROWS_PERFORMANCE_TEST; i++)
    {
        md.addRow(row);
    }
    s1 = t.toc("Time original:", false);

    t.tic();
    for (int i=0; i<N_ROWS_PERFORMANCE_TEST; i++)
    {
        md2.addRow(row);
    }

    s2 = t.toc("Time by row: ", false);
    printf("    Speed up from original: %f\n", ((float) s1 / (float) s2));

    // Initialize insertion.
    t.tic();
    // Add rows loop.
    int i=0;
    do
    {
        // Insert row and increase number of insertions.
        md3.addRow(row);
        i++;
    }
    while (i<N_ROWS_PERFORMANCE_TEST);
    s3 = t.toc("Time by set:", false);
    printf("    Speed up from original: %f\n", ((float) s1 / (float) s3));
    printf("    Speed up from row: %f\n", ((float) s2 / (float) s3));
    // Check result.
    EXPECT_EQ(md, md2);
    EXPECT_EQ(md2, md3);
}

TEST_F(MetadataTest, addLabelAlias)
{
    //metada with no xmipp labels    //metada with no xmipp labels
    FileName fnNonXmippSTAR = (String)"metadata/noXmipp.xmd";
    MDL::addLabelAlias(MDL_Y,(String)"noExixtingLabel");
    MetaDataVec md = MetaDataVec(fnNonXmippSTAR);
    EXPECT_EQ(mDsource, md);
}

TEST_F(MetadataTest, getNewAlias)
{
    //metada with no xmipp labels
    FileName fnNonXmippSTAR = (String)"metadata/noXmipp.xmd";
    String labelStr("noExixtingLabel");
    MDLabel newLabel = MDL::getNewAlias(labelStr);
    EXPECT_EQ(newLabel, BUFFER_01);
    EXPECT_EQ(MDL::label2Str(newLabel), labelStr);
    MetaDataVec md = MetaDataVec(fnNonXmippSTAR);

    std::vector<double> yValues;
    std::vector<std::string> y2Values;
    mDsource.getColumnValues(MDL_Y, yValues);
    md.getColumnValues(newLabel, y2Values);
    for (int i = 0; i < yValues.size(); ++i)
        EXPECT_FLOAT_EQ(yValues[i], textToFloat(y2Values[i]));
}

TEST_F(MetadataTest, Clear)
{
    MetaDataVec auxMetadata = mDsource;
    EXPECT_EQ((size_t)2,auxMetadata.size());
    auxMetadata.clear();
    EXPECT_EQ((size_t)0,auxMetadata.size());
}

TEST_F(MetadataTest, Copy)
{
    MetaDataVec auxMetadata = mDsource;
    EXPECT_EQ(mDsource,auxMetadata);
}

TEST_F(MetadataTest, MDInfo)
{
    FileName fn;
    fn.initUniqueName("MDInfo_XXXXXX");
    FileName fnSTAR;
    fnSTAR = fn + ".xmd";

    XMIPP_TRY

    mDsource.write(fnSTAR);

    MetaDataVec md;
    md.read(fnSTAR);

    MetaDataVec mdOnlyOne;
    mdOnlyOne.setMaxRows(1);
    mdOnlyOne.read(fnSTAR);
    EXPECT_EQ(md.size(), mdOnlyOne.getParsedLines());

    MDLabelVector labels = md.getActiveLabels();
    for (size_t i = 0; i < labels.size(); ++i)
        EXPECT_TRUE(mdOnlyOne.containsLabel(labels[i]));

    XMIPP_CATCH

    unlink(fn.c_str());
    unlink(fnSTAR.c_str());
}

TEST_F(MetadataTest, multiWrite)
{
    FileName fnSTAR;
    fnSTAR.initUniqueName("/tmp/testReadMultipleBlocks_XXXXXX");
    fnSTAR += ".xmd";

    FileName fnSTARref = (String)"metadata/mDsource.xmd";

    XMIPP_TRY
    mDsource.write((String)"myblock@"+fnSTAR);
    XMIPP_CATCH

    EXPECT_TRUE(compareTwoFiles(fnSTAR, fnSTARref));

    unlink(fnSTAR.c_str());
}

TEST_F(MetadataTest, ReadEmptyBlock)
{
    char sfn[64] = "";
    strncpy(sfn, "/tmp/testGetBlocks_XXXXXX", sizeof sfn);
    if (mkstemp(sfn)==-1)
        REPORT_ERROR(ERR_IO_NOTOPEN,"Cannot create temporary file");
    MetaDataVec md;
    FileName fn = (String)"block_Empty@"+sfn;
    md.write(fn, MD_OVERWRITE);
    md.clear();
    md.setValue(MDL_IMAGE,(String)"image_data_2_1.xmp",md.addObject());
    md.setValue(MDL_IMAGE,(String)"image_data_2_2.xmp",md.addObject());
    md.write((String)"block_B1@"+sfn,MD_APPEND);

    EXPECT_NO_THROW(MetaDataVec md2(fn););

    unlink(sfn);
}

TEST_F(MetadataTest, GetBlocksInMetadata)
{
    char sfn[64] = "";
    strncpy(sfn, "/tmp/testGetBlocks_XXXXXX", sizeof sfn);
    if (mkstemp(sfn)==-1)
        REPORT_ERROR(ERR_IO_NOTOPEN,"Cannot create temporary file");

    MetaDataVec auxMetadata;
    auxMetadata.setValue(MDL_IMAGE,(String)"image_1.xmp",auxMetadata.addObject());
    auxMetadata.setValue(MDL_IMAGE,(String)"image_2.xmp",auxMetadata.addObject());
    auxMetadata.write(sfn,MD_OVERWRITE);
    auxMetadata.clear();
    auxMetadata.setValue(MDL_IMAGE,(String)"image_data_1_1.xmp",auxMetadata.addObject());
    auxMetadata.setValue(MDL_IMAGE,(String)"image_data_1_2.xmp",auxMetadata.addObject());
    auxMetadata.write((String)"block_000001@"+sfn,MD_APPEND);
    auxMetadata.clear();
    auxMetadata.setValue(MDL_IMAGE,(String)"image_data_2_1.xmp",auxMetadata.addObject());
    auxMetadata.setValue(MDL_IMAGE,(String)"image_data_2_2.xmp",auxMetadata.addObject());
    auxMetadata.write((String)"block_000002@"+sfn,MD_APPEND);
    auxMetadata.clear();

    StringVector compBlockList;
    compBlockList.push_back(DEFAULT_BLOCK_NAME);
    compBlockList.push_back("block_000001");
    compBlockList.push_back("block_000002");

    StringVector readBlockList;
    getBlocksInMetaDataFile(sfn,readBlockList);

    EXPECT_EQ(compBlockList,readBlockList);
    unlink(sfn);
}

TEST_F(MetadataTest, CheckRegularExpression)
{
    char sfn[64] = "";
    strncpy(sfn, "/tmp/testGetBlocks_XXXXXX", sizeof sfn);
    if (mkstemp(sfn)==-1)
        REPORT_ERROR(ERR_IO_NOTOPEN,"Cannot create temporary file");

    MetaDataVec auxMd, auxMd2;
    auxMd.setValue(MDL_IMAGE,(String)"image_1.xmp",auxMd.addObject());
    auxMd.setValue(MDL_IMAGE,(String)"image_2.xmp",auxMd.addObject());
    auxMd.write(sfn,MD_OVERWRITE);
    auxMd.clear();
    auxMd.setValue(MDL_IMAGE,(String)"image_data_1_1.xmp",auxMd.addObject());
    auxMd.setValue(MDL_IMAGE,(String)"image_data_1_2.xmp",auxMd.addObject());
    auxMd.write((String)"block_000001@"+sfn,MD_APPEND);
    auxMd.clear();
    auxMd.setValue(MDL_IMAGE,(String)"image_data_2_1.xmp",auxMd.addObject());
    auxMd.setValue(MDL_IMAGE,(String)"image_data_2_2.xmp",auxMd.addObject());
    auxMd.write((String)"block_000002@"+sfn,MD_APPEND);
    auxMd.clear();
    auxMd.setValue(MDL_IMAGE,(String)"image_data_3_1.xmp",auxMd.addObject());
    auxMd.setValue(MDL_IMAGE,(String)"image_data_3_2.xmp",auxMd.addObject());
    auxMd.write((String)"block_000003@"+sfn,MD_APPEND);
    auxMd.clear();
    auxMd.setValue(MDL_IMAGE,(String)"image_data_A_1.xmp",auxMd.addObject());
    auxMd.setValue(MDL_IMAGE,(String)"image_data_A_2.xmp",auxMd.addObject());
    auxMd.write((String)"block_A@"+sfn,MD_APPEND);
    auxMd.clear();

    auxMd.clear();
    auxMd.setValue(MDL_IMAGE,(String)"image_data_1_1.xmp",auxMd.addObject());
    auxMd.setValue(MDL_IMAGE,(String)"image_data_1_2.xmp",auxMd.addObject());
    auxMd.setValue(MDL_IMAGE,(String)"image_data_2_1.xmp",auxMd.addObject());
    auxMd.setValue(MDL_IMAGE,(String)"image_data_2_2.xmp",auxMd.addObject());
    auxMd.setValue(MDL_IMAGE,(String)"image_data_3_1.xmp",auxMd.addObject());
    auxMd.setValue(MDL_IMAGE,(String)"image_data_3_2.xmp",auxMd.addObject());

    auxMd2.read((String)"block_000[0-9][0-9][123]@" + sfn);
    EXPECT_EQ(auxMd, auxMd2);

    unlink(sfn);
}

TEST_F(MetadataTest, CheckRegularExpression2)
{
    char sfn[64] = "";
    strncpy(sfn, "/tmp/testGetBlocks_XXXXXX", sizeof sfn);
    if (mkstemp(sfn)==-1)
        REPORT_ERROR(ERR_IO_NOTOPEN,"Cannot create temporary file");

    MetaDataVec auxMd, auxMd2;
    auxMd.setValue(MDL_IMAGE,(String)"image_1.xmp",auxMd.addObject());
    auxMd.setValue(MDL_IMAGE,(String)"image_2.xmp",auxMd.addObject());
    auxMd.write(sfn,MD_OVERWRITE);
    auxMd.clear();
    auxMd.setValue(MDL_IMAGE,(String)"image_data_1_1.xmp",auxMd.addObject());
    auxMd.setValue(MDL_IMAGE,(String)"image_data_1_2.xmp",auxMd.addObject());
    auxMd.write((String)"block_000001@"+sfn,MD_APPEND);
    auxMd.clear();
    auxMd.setValue(MDL_IMAGE,(String)"image_data_2_1.xmp",auxMd.addObject());
    auxMd.setValue(MDL_IMAGE,(String)"image_data_2_2.xmp",auxMd.addObject());
    auxMd.write((String)"block_000002@"+sfn,MD_APPEND);
    auxMd.clear();
    auxMd.setValue(MDL_IMAGE,(String)"image_data_A_1.xmp",auxMd.addObject());
    auxMd.setValue(MDL_IMAGE,(String)"image_data_A_2.xmp",auxMd.addObject());
    auxMd.write((String)"block_0000023@"+sfn,MD_APPEND);
    auxMd.clear();

    auxMd.clear();
    auxMd.setValue(MDL_IMAGE,(String)"image_data_1_1.xmp",auxMd.addObject());
    auxMd.setValue(MDL_IMAGE,(String)"image_data_1_2.xmp",auxMd.addObject());
    auxMd.setValue(MDL_IMAGE,(String)"image_data_2_1.xmp",auxMd.addObject());
    auxMd.setValue(MDL_IMAGE,(String)"image_data_2_2.xmp",auxMd.addObject());

    auxMd2.read((String)"block_000[0-9][0-9][0-9]$@" + sfn);
    EXPECT_EQ(auxMd, auxMd2);

    unlink(sfn);
}

TEST_F(MetadataTest, compareTwoMetadataFiles)
{
    XMIPP_TRY
    char sfn[64] = "";
    strncpy(sfn, "/tmp/testGetBlocks_XXXXXX", sizeof sfn);
    if (mkstemp(sfn)==-1)
        REPORT_ERROR(ERR_IO_NOTOPEN,"Cannot create temporary file");
    char sfn2[64] = "";
    strncpy(sfn2, "/tmp/testGetBlocks_XXXXXX", sizeof sfn2);
    if (mkstemp(sfn2)==-1)
        REPORT_ERROR(ERR_IO_NOTOPEN,"Cannot create temporary file");
    char sfn3[64] = "";
    strncpy(sfn3, "/tmp/testGetBlocks_XXXXXX", sizeof sfn3);
    if (mkstemp(sfn3)==-1)
        REPORT_ERROR(ERR_IO_NOTOPEN,"Cannot create temporary file");

    MetaDataVec auxMd, auxMd2;
    auxMd.setValue(MDL_IMAGE,(String)"image_1.xmp",auxMd.addObject());
    auxMd.setValue(MDL_IMAGE,(String)"image_2.xmp",auxMd.addObject());
    auxMd.write(sfn,MD_OVERWRITE);
    auxMd.clear();
    auxMd.setValue(MDL_IMAGE,(String)"image_data_1_1.xmp",auxMd.addObject());
    auxMd.setValue(MDL_IMAGE,(String)"image_data_1_2.xmp",auxMd.addObject());
    auxMd.write((String)"block_000001@"+sfn,MD_APPEND);
    auxMd.clear();
    auxMd.setValue(MDL_IMAGE,(String)"image_data_2_1.xmp",auxMd.addObject());
    auxMd.setValue(MDL_IMAGE,(String)"image_data_2_2.xmp",auxMd.addObject());
    auxMd.write(sfn2, MD_OVERWRITE);
    auxMd.clear();
    auxMd.setValue(MDL_IMAGE,(String)"image_data_A_1.xmp",auxMd.addObject());
    auxMd.setValue(MDL_IMAGE,(String)"image_data_A_2.xmp",auxMd.addObject());
    auxMd.write((String)"block_000001@"+sfn2,MD_APPEND);
    auxMd.clear();

    EXPECT_FALSE(compareTwoMetadataFiles(sfn, sfn2));
    EXPECT_TRUE(compareTwoMetadataFiles(sfn, sfn));

    auxMd.setValue(MDL_IMAGE,(String)"image_1.xmpSPACE",auxMd.addObject());//extra space
    auxMd.setValue(MDL_IMAGE,(String)"image_2.xmp",auxMd.addObject());
    auxMd.write(sfn2,MD_OVERWRITE);
    auxMd.clear();
    auxMd.setValue(MDL_IMAGE,(String)"image_data_1_1.xmp",auxMd.addObject());
    auxMd.setValue(MDL_IMAGE,(String)"image_data_1_2.xmp",auxMd.addObject());
    auxMd.write((String)"block_000001@"+sfn2,MD_APPEND);

    String command=(String)"sed 's/SPACE/ /g' " + sfn2 + (String) ">" + sfn3;
    if (system (command.c_str())==-1)
        REPORT_ERROR(ERR_UNCLASSIFIED,"Could not open shell");

    EXPECT_TRUE(compareTwoMetadataFiles(sfn, sfn3));

    unlink(sfn);
    unlink(sfn2);
    unlink(sfn3);
    XMIPP_CATCH
}

TEST_F(MetadataTest, ImportObject)
{
    MetaDataVec auxMetadata = mDsource;
    for (size_t objId : mDanotherSource.ids())
        auxMetadata.importObject(mDanotherSource, objId, false);
    EXPECT_EQ(auxMetadata, mDunion);
}

TEST_F(MetadataTest, MultiQuery)
{
    MetaDataVec auxMetadata;
    MetaDataVec auxMetadata3;
    size_t id = auxMetadata3.addObject();
    auxMetadata3.setValue(MDL_X,1.,id);
    auxMetadata3.setValue(MDL_Y,2.,id);
    auxMetadata3.setValue(MDL_Z,222.,id);
    id = auxMetadata3.addObject();
    auxMetadata3.setValue(MDL_X,3.,id);
    auxMetadata3.setValue(MDL_Y,4.,id);
    auxMetadata3.setValue(MDL_Z,333.,id);
    id = auxMetadata3.addObject();
    auxMetadata3.setValue(MDL_X,3.,id);
    auxMetadata3.setValue(MDL_Y,4.,id);
    auxMetadata3.setValue(MDL_Z,444.,id);

    MDValueEQ eq1(MDL_X, 3.);
    MDValueEQ eq2(MDL_Y, 4.);
    MDMultiQuery multi;

    //Test empty query
    auxMetadata.importObjects(auxMetadata3, multi);
    EXPECT_EQ(auxMetadata3,auxMetadata);

    multi.addAndQuery(eq1);
    multi.addAndQuery(eq2);

    auxMetadata.importObjects(auxMetadata3, multi);

    MetaDataVec outMetadata;
    id = outMetadata.addObject();
    outMetadata.setValue(MDL_X,3.,id);
    outMetadata.setValue(MDL_Y,4.,id);
    outMetadata.setValue(MDL_Z,333.,id);
    id = outMetadata.addObject();
    outMetadata.setValue(MDL_X,3.,id);
    outMetadata.setValue(MDL_Y,4.,id);
    outMetadata.setValue(MDL_Z,444.,id);

    EXPECT_EQ(outMetadata,auxMetadata);
}

TEST_F(MetadataTest, MDValueEQ)
{
    try
    {
        MetaDataVec md;
        md.setValue(MDL_IMAGE, (String)"a", md.addObject());
        md.setValue(MDL_IMAGE, (String)"b", md.addObject());
        md.setValue(MDL_IMAGE, (String)"c", md.addObject());
        md.setValue(MDL_IMAGE, (String)"a", md.addObject());

        MetaDataVec md2;
        md2.setValue(MDL_IMAGE, (String)"a", md2.addObject());
        md2.setValue(MDL_IMAGE, (String)"a", md2.addObject());

        MDValueEQ eq(MDL_IMAGE,(String)"a");

        MetaDataVec md3;
        md3.importObjects(md, eq);

        // FIXME: consult with David
        // This test fails because ids are compared too
        // Possible solutions:
        //  a) do not compare ids
        //  b) reid in importObjects
        // std::cout << md << std::endl << md2 << std::endl << md3;

        EXPECT_EQ(md2, md3);
    }
    catch (XmippError &xe)
    {
        std::cerr << "DEBUG_JM: xe: " << xe << std::endl;
    }
}

TEST_F(MetadataTest, RegularExp)
{
    XMIPP_TRY

    //create temporal file with three metadas
    //char sfnStar[64] = "";
    //char sfnSqlite[64] = "";
    //strncpy(sfnStar, "/tmp/testReadMultipleBlocks_XXXXXX.xmd", sizeof sfnStar);
    //if (mkstemps(sfnStar,4)==-1)
    //  REPORT_ERROR(ERR_IO_NOTOPEN,"Cannot create temporary STAR file");
//    strncpy(sfnSqlite, "/tmp/testReadMultipleBlocks_XXXXXX.sqlite", sizeof sfnSqlite);
//    if (mkstemps(sfnSqlite,7)==-1)
//      REPORT_ERROR(ERR_IO_NOTOPEN,"Cannot create temporary SQLITE file");

    FileName fn;
    fn.initUniqueName("/tmp/testReadMultipleBlocks_XXXXXX");
    FileName sfnStar;
    sfnStar = fn + ".xmd";
    MetaDataVec auxMetadata;
    auxMetadata.setValue(MDL_IMAGE,(String)"image_1.xmp",auxMetadata.addObject());
    auxMetadata.setValue(MDL_IMAGE,(String)"image_2.xmp",auxMetadata.addObject());
    auxMetadata.write(sfnStar,MD_OVERWRITE);
//    auxMetadata.write(sfnSqlite,MD_OVERWRITE);
    auxMetadata.clear();
    auxMetadata.setValue(MDL_IMAGE,(String)"image_data_1_1.xmp",auxMetadata.addObject());
    auxMetadata.setValue(MDL_IMAGE,(String)"image_data_1_2.xmp",auxMetadata.addObject());
    auxMetadata.write((String)"block_000001@"+sfnStar,MD_APPEND);
//    auxMetadata.write((String)"block_000001@"+sfnSqlite,MD_APPEND);
    auxMetadata.clear();
    auxMetadata.setValue(MDL_IMAGE,(String)"image_data_2_1.xmp",auxMetadata.addObject());
    auxMetadata.setValue(MDL_IMAGE,(String)"image_data_2_2.xmp",auxMetadata.addObject());
    auxMetadata.write((String)"block_000002@"+sfnStar,MD_APPEND);
//    auxMetadata.write((String)"block_000002@"+sfnSqlite,MD_APPEND);
    auxMetadata.clear();
    auxMetadata.setValue(MDL_IMAGE,(String)"image_data_no_1.xmp",auxMetadata.addObject());
    auxMetadata.setValue(MDL_IMAGE,(String)"image_data_no_2.xmp",auxMetadata.addObject());
    auxMetadata.write((String)"noblock@"+sfnStar,MD_APPEND);
//    auxMetadata.write((String)"noblock@"+sfnSqlite,MD_APPEND);
    auxMetadata.clear();
    auxMetadata.setValue(MDL_IMAGE,(String)"image_data_3_1.xmp",auxMetadata.addObject());
    auxMetadata.setValue(MDL_IMAGE,(String)"image_data_3_2.xmp",auxMetadata.addObject());
    auxMetadata.write((String)"block_000003@"+sfnStar,MD_APPEND);
//    auxMetadata.write((String)"block_000003@"+sfnSqlite,MD_APPEND);
    auxMetadata.clear();
    MetaDataVec auxMetadata2;
    auxMetadata2.setValue(MDL_IMAGE,(String)"image_data_1_1.xmp",auxMetadata2.addObject());
    auxMetadata2.setValue(MDL_IMAGE,(String)"image_data_1_2.xmp",auxMetadata2.addObject());
    auxMetadata2.setValue(MDL_IMAGE,(String)"image_data_2_1.xmp",auxMetadata2.addObject());
    auxMetadata2.setValue(MDL_IMAGE,(String)"image_data_2_2.xmp",auxMetadata2.addObject());
    MDSql::activateRegExtensions();
    //query file
    MetaDataVec md;
    FileName blockFileName;
    md.read((String)"block_000001@"+sfnStar);
    //compare with reference metada
//    EXPECT_EQ(md,auxMetadata2);
//    md.read((String)"block_000001@"+sfnSqlite);
    //compare with reference metada
//    EXPECT_EQ(md,auxMetadata2);
    unlink(fn.c_str());
    unlink(sfnStar.c_str());
    //unlink(sfnSqlite);

    XMIPP_CATCH
}

TEST_F(MetadataTest, Query)
{
    MetaDataVec auxMetadata;
    MetaDataVec auxMetadata3;
    size_t id = auxMetadata3.addObject();
    auxMetadata3.setValue(MDL_X,1.,id);
    auxMetadata3.setValue(MDL_Y,2.,id);
    auxMetadata3.setValue(MDL_Z,222.,id);
    id = auxMetadata3.addObject();
    auxMetadata3.setValue(MDL_X,3.,id);
    auxMetadata3.setValue(MDL_Y,4.,id);
    auxMetadata3.setValue(MDL_Z,333.,id);
    id = auxMetadata3.addObject();
    auxMetadata3.setValue(MDL_X,3.,id);
    auxMetadata3.setValue(MDL_Y,4.,id);
    auxMetadata3.setValue(MDL_Z,444.,id);

    auxMetadata.importObjects(auxMetadata3, MDValueEQ(MDL_X, 3.));

    MetaDataVec outMetadata;
    id = outMetadata.addObject();
    outMetadata.setValue(MDL_X,3.,id);
    outMetadata.setValue(MDL_Y,4.,id);
    outMetadata.setValue(MDL_Z,333.,id);
    id = outMetadata.addObject();
    outMetadata.setValue(MDL_X,3.,id);
    outMetadata.setValue(MDL_Y,4.,id);
    outMetadata.setValue(MDL_Z,444.,id);

    EXPECT_EQ(outMetadata,auxMetadata);
}


TEST_F(MetadataTest, Randomize)
{
    MetaDataVec auxMetadata;
    const int tries = 50;
    for (int var = 0; var < tries; var++)
    {
        // randomize the content of the metadata
        auxMetadata.randomize(mDsource);
        if (mDsource == auxMetadata) {
            continue; // try again
        } else {
            // if they don't equal, the randomization probably works correctly
            SUCCEED();
            return;
        }
    }
    // if we got here, non of the previous try detected a change
    FAIL() << "Randomization did not change the content of the metadata even after " << tries << " times.";
}

TEST_F(MetadataTest, ReadMultipleBlocks)
{
    char sfn[64] = "";
    strncpy(sfn, "/tmp/testReadMultipleBlocks_XXXXXX", sizeof sfn);
    if (mkstemp(sfn)==-1)
        REPORT_ERROR(ERR_IO_NOTOPEN,"Cannot create temporary file");

    MetaDataVec auxMetadata;
    auxMetadata.setValue(MDL_IMAGE,(String)"image_1.xmp",auxMetadata.addObject());
    auxMetadata.setValue(MDL_IMAGE,(String)"image_2.xmp",auxMetadata.addObject());
    auxMetadata.write(sfn,MD_OVERWRITE);
    auxMetadata.clear();
    auxMetadata.setValue(MDL_IMAGE,(String)"image_data_1_1.xmp",auxMetadata.addObject());
    auxMetadata.setValue(MDL_IMAGE,(String)"image_data_1_2.xmp",auxMetadata.addObject());
    auxMetadata.write((String)"block_000001@"+sfn,MD_APPEND);
    auxMetadata.clear();
    auxMetadata.setValue(MDL_IMAGE,(String)"image_data_2_1.xmp",auxMetadata.addObject());
    auxMetadata.setValue(MDL_IMAGE,(String)"image_data_2_2.xmp",auxMetadata.addObject());
    auxMetadata.write((String)"block_000002@"+sfn,MD_APPEND);
    auxMetadata.clear();
    auxMetadata.setValue(MDL_IMAGE,(String)"image_data_no_1.xmp",auxMetadata.addObject());
    auxMetadata.setValue(MDL_IMAGE,(String)"image_data_no_2.xmp",auxMetadata.addObject());
    auxMetadata.write((String)"noblock@"+sfn,MD_APPEND);
    auxMetadata.clear();
    auxMetadata.setValue(MDL_IMAGE,(String)"image_data_3_1.xmp",auxMetadata.addObject());
    auxMetadata.setValue(MDL_IMAGE,(String)"image_data_3_2.xmp",auxMetadata.addObject());
    auxMetadata.write((String)"block_000003@"+sfn,MD_APPEND);
    auxMetadata.clear();

    MetaDataVec compMetadata;
    compMetadata.setValue(MDL_IMAGE,(String)"image_data_1_1.xmp",compMetadata.addObject());
    compMetadata.setValue(MDL_IMAGE,(String)"image_data_1_2.xmp",compMetadata.addObject());
    compMetadata.setValue(MDL_IMAGE,(String)"image_data_2_1.xmp",compMetadata.addObject());
    compMetadata.setValue(MDL_IMAGE,(String)"image_data_2_2.xmp",compMetadata.addObject());
    auxMetadata.read((String)"block_00000[12]@"+sfn);
    EXPECT_EQ(compMetadata,auxMetadata);
    compMetadata.clear();

    compMetadata.setValue(MDL_IMAGE,(String)"image_data_3_1.xmp",compMetadata.addObject());
    compMetadata.setValue(MDL_IMAGE,(String)"image_data_3_2.xmp",compMetadata.addObject());
    auxMetadata.read((String)"block_000003@"+sfn);
    EXPECT_EQ(compMetadata,auxMetadata);
    compMetadata.clear();

    compMetadata.setValue(MDL_IMAGE,(String)"image_1.xmp",compMetadata.addObject());
    compMetadata.setValue(MDL_IMAGE,(String)"image_2.xmp",compMetadata.addObject());
    auxMetadata.read(sfn);
    EXPECT_EQ(compMetadata,auxMetadata);
    compMetadata.clear();
    unlink(sfn);
}

TEST_F(MetadataTest, ReadEmptyBlocks)
{
#define sizesfn 64
    char sfn[sizesfn] = "";
    strncpy(sfn, "/tmp/testReadMultipleBlocks_XXXXXX", sizesfn);
    if (mkstemp(sfn)==-1)
        REPORT_ERROR(ERR_IO_NOTOPEN,"Cannot create temporary file");

    MetaDataVec auxMetadata;
    size_t id = auxMetadata.addObject();
    auxMetadata.setValue(MDL_X,1.,id);
    auxMetadata.setValue(MDL_Y,2.,id);
    auxMetadata.setValue(MDL_Z,222.,id);
    auxMetadata.write((String)"block_000001@"+sfn,MD_APPEND);

    auxMetadata.clear();
    auxMetadata.addLabel(MDL_X);
    auxMetadata.addLabel(MDL_Y);
    auxMetadata.addLabel(MDL_Z);
    auxMetadata.write((String)"block_000002@"+sfn,MD_APPEND);

    auxMetadata.clear();
    id=auxMetadata.addObject();
    auxMetadata.setValue(MDL_X,1.,id);
    auxMetadata.setValue(MDL_Y,2.,id);
    auxMetadata.setValue(MDL_Z,222.,id);
    auxMetadata.write((String)"block_000003@"+sfn,MD_APPEND);

    auxMetadata.clear();
    auxMetadata.addLabel(MDL_X);
    auxMetadata.addLabel(MDL_Y);
    auxMetadata.addLabel(MDL_Z);
    auxMetadata.write((String)"block_000004@"+sfn,MD_APPEND);

    auxMetadata.read((String)"block_000002@"+sfn);
    EXPECT_EQ(auxMetadata.size(),(size_t)0);

    auxMetadata.read((String)"block_000004@"+sfn);
    EXPECT_EQ(auxMetadata.size(),(size_t)0);

    unlink(sfn);
}

TEST_F(MetadataTest, ReadEmptyBlocksII)
{
    char sfn[64] = "";
    strncpy(sfn, "/tmp/testReadMultipleBlocks_XXXXXX", sizeof sfn);
    if (mkstemp(sfn)==-1)
        REPORT_ERROR(ERR_IO_NOTOPEN,"Cannot create temporary file");

    MetaDataVec auxMetadata;

    auxMetadata.addLabel(MDL_X);
    auxMetadata.addLabel(MDL_Y);
    auxMetadata.addLabel(MDL_Z);
    auxMetadata.write((String)"block_000002@"+sfn,MD_APPEND);

    auxMetadata.read((String)"block_000002@"+sfn);
    EXPECT_EQ(auxMetadata.size(),(size_t)0);
    unlink(sfn);
}

TEST_F(MetadataTest, ReadWrite)
{
    //temp file name
    char sfn[32] = "";
    strncpy(sfn, "/tmp/testWrite_XXXXXX", sizeof sfn);
    if (mkstemp(sfn)==-1)
        REPORT_ERROR(ERR_IO_NOTOPEN,"Cannot create temporary file");
    mDsource.write(sfn);
    MetaDataVec auxMetadata;
    auxMetadata.read(sfn);

    EXPECT_EQ(mDsource,auxMetadata);
    unlink(sfn);
}

TEST_F(MetadataTest, WriteIntermediateBlock)
{
    //read metadata block between another two
    FileName filename("metadata/WriteIntermediateBlock.xmd");
    FileName blockFileName;
    blockFileName.compose("two", filename);
    MetaDataVec auxMetadata(blockFileName);
    MDRowVec row;
    row.setValue(MDL_X, 11.);
    row.setValue(MDL_Y, 22.);
    auxMetadata.addRow(row);
    row.setValue(MDL_X, 33.);
    row.setValue(MDL_Y, 44.);
    auxMetadata.addRow(row);
    auxMetadata.setValue(MDL_X,111.,auxMetadata.firstRowId());

    //temporal file for modified metadata
    char sfn2[32] = "";
    strncpy(sfn2, "/tmp/testWrite_XXXXXX", sizeof sfn2);
    if (mkstemp(sfn2)==-1)
        REPORT_ERROR(ERR_IO_NOTOPEN,"Cannot create temporary file");

    //copy input metadata file
    std::ifstream src; // the source file
    std::ofstream dest; // the destination file
    src.open (filename.c_str(), std::ios::binary); // open in binary to prevent jargon at the end of the buffer
    dest.open (sfn2, std::ios::binary); // same again, binary
    if (!src.is_open())
        std::cerr << "Can not open file: " << filename.c_str() <<std::endl; // could not be copied
    if (!dest.is_open())
        std::cerr << "Can not open file: " <<sfn2 <<std::endl; // could not be copied
    dest << src.rdbuf (); // copy the content
    dest.close (); // close destination file
    src.close (); // close source file

    blockFileName.compose("two", sfn2);
    auxMetadata.write(blockFileName,MD_APPEND);
    //file with correct values
    FileName fn2("metadata/ReadWriteAppendBlock2.xmd");
    EXPECT_TRUE(compareTwoFiles("metadata/WriteIntermediateBlock2.xmd",sfn2,0));
    unlink(sfn2);
}

TEST_F(MetadataTest, ReadWriteAppendBlock)
{
    XMIPP_TRY
    //temp file name
    char sfn[32] = "";
    strncpy(sfn, "/tmp/testWrite_XXXXXX", sizeof sfn);
    if (mkstemp(sfn)==-1)
        REPORT_ERROR(ERR_IO_NOTOPEN,"Cannot create temporary file");
    mDsource.write((String)"one@"+sfn);
    mDsource.write((String)"two@"+sfn,MD_APPEND);
    mDsource.write((String)"three@"+sfn,MD_APPEND);
    MetaDataVec auxMetadata;
    FileName sfn2 = "metadata/ReadWriteAppendBlock.xmd";
    EXPECT_TRUE(compareTwoFiles(sfn,sfn2,0));
    unlink(sfn);
    XMIPP_CATCH
}

TEST_F(MetadataTest, RemoveDuplicates)
{
    MetaDataVec auxMetadata1,auxMetadata3;
    size_t id = auxMetadata3.addObject();
    auxMetadata3.setValue(MDL_X,1.,id);
    auxMetadata3.setValue(MDL_Y,2.,id);
    id = auxMetadata3.addObject();
    auxMetadata3.setValue(MDL_X,3.,id);
    auxMetadata3.setValue(MDL_Y,4.,id);
    id = auxMetadata3.addObject();
    auxMetadata3.setValue(MDL_X,1.,id);
    auxMetadata3.setValue(MDL_Y,2.,id);
    auxMetadata1.removeDuplicates(auxMetadata3);
    EXPECT_EQ(auxMetadata1,mDsource);//print mDjoin if error
}

TEST_F(MetadataTest, Removelabel)
{
    MetaDataVec auxMetadata = mDunion;
    auxMetadata.removeLabel(MDL_X);
    std::vector<MDLabel> v1,v2;
    v1.push_back(MDL_Y);
    v2 = auxMetadata.getActiveLabels();
    EXPECT_EQ(v2,v1);
}

TEST_F(MetadataTest, Select)
{
    MetaDataVec auxMetadata;
    MetaDataVec auxMetadata2;
    size_t id = auxMetadata2.addObject();
    auxMetadata2.setValue(MDL_X,3.,id);
    auxMetadata2.setValue(MDL_Y,4.,id);

    auxMetadata.importObjects(mDsource,MDExpression((String)"x>2"));
    EXPECT_EQ(auxMetadata,auxMetadata2);
}


TEST_F(MetadataTest, Size)
{
    EXPECT_EQ((size_t)2, mDsource.size());
}

TEST_F(MetadataTest, Sort)
{
    MetaDataVec auxMetadata,auxMetadata2,auxMetadata3,outMetadata;
    size_t id = auxMetadata.addObject();
    auxMetadata.setValue(MDL_X,3.,id);
    auxMetadata.setValue(MDL_Y,4.,id);
    id = auxMetadata.addObject();
    auxMetadata.setValue(MDL_X,1.,id);
    auxMetadata.setValue(MDL_Y,2.,id);
    auxMetadata2.sort(auxMetadata,MDL_X);
    EXPECT_EQ(auxMetadata2,mDsource);

    id = auxMetadata.addObject();
    auxMetadata.setValue(MDL_X,5.,id);
    auxMetadata.setValue(MDL_Y,6.,id);

    auxMetadata2.clear();
    auxMetadata2.sort(auxMetadata,MDL_X,true,1,0);
    id = outMetadata.addObject();
    outMetadata.setValue(MDL_X,1.,id);
    outMetadata.setValue(MDL_Y,2.,id);
    EXPECT_EQ(auxMetadata2,outMetadata);

    auxMetadata2.clear();
    auxMetadata2.sort(auxMetadata,MDL_X,true,2,1);
    outMetadata.clear();
    id = outMetadata.addObject();
    outMetadata.setValue(MDL_X,3.,id);
    outMetadata.setValue(MDL_Y,4.,id);
    id = outMetadata.addObject();
    outMetadata.setValue(MDL_X,5.,id);
    outMetadata.setValue(MDL_Y,6.,id);
    EXPECT_EQ(auxMetadata2,outMetadata);
}

//check if mdl label match its type and
//check if int is different from size_t
TEST_F(MetadataTest, setGetValue)
{
    size_t t;
    int i;
    EXPECT_EQ(MDL::labelType(MDL_ORDER),LABEL_SIZET);
    MetaDataVec auxMetadata;
    size_t id = auxMetadata.addObject();
    auxMetadata.setValue(MDL_ORDER,(size_t)1, id);
    auxMetadata.getValue(MDL_ORDER,t, id);
    EXPECT_EQ((size_t)1,t);
    //We expect that MetaDataVec will throw an exception
    //if you use getValue with a variable of type that
    // doesn't match the label type
    std::cerr << "TEST COMMENT: you should get the ERROR: Mismatch Label (order_) and value type(INT)" <<std::endl;
    EXPECT_THROW(auxMetadata.getValue(MDL_ORDER, i, id), XmippError);
}
TEST_F(MetadataTest, Comment)
{
    XMIPP_TRY
    char sfn[64] = "";
    MetaDataVec md1(mDsource);
    strncpy(sfn, "/tmp/testComment_XXXXXX", sizeof sfn);
    if (mkstemp(sfn)==-1)
        REPORT_ERROR(ERR_IO_NOTOPEN,"Cannot create temporary file");
    String s1((String)"This is a very long comment that has more than 80 characters"+
              " Therefore should be split in several lines"+
              " Let us see what happened");
    md1.setComment(s1);
    md1.write(sfn, MD_OVERWRITE);
    MetaDataVec md2;
    md2.read(sfn);
    String s2;
    s2=md2.getComment();
    EXPECT_EQ(s1, s2);
    unlink(sfn);

    XMIPP_CATCH
}
//read file with vector
TEST_F(MetadataTest, getValue)
{
    XMIPP_TRY
    std::vector<double> v1(3);
    std::vector<double> v2(3);
    MetaDataVec auxMD1;
    size_t id = auxMD1.addObject();
    v1[0]=1.;
    v1[1]=2.;
    v1[2]=3.;
    auxMD1.setValue(MDL_CLASSIFICATION_DATA,v1,id);
    id = auxMD1.firstRowId();
    auxMD1.getValue(MDL_CLASSIFICATION_DATA,v2, id);

    EXPECT_EQ(v1[0],v2[0]);
    EXPECT_EQ(v1[1],v2[1]);
    EXPECT_EQ(v1[2],v2[2]);
    XMIPP_CATCH
}

TEST_F(MetadataTest, getValueDefault)
{
    XMIPP_TRY
    MetaDataVec auxMD1;
    MetaDataVec auxMD2;
    double rot=1., tilt=2., psi=3.;
    double rot2=0., tilt2=0., psi2=0.;
    size_t id = auxMD1.addObject();
    auxMD1.setValue(MDL_ANGLE_ROT,rot,id);
    auxMD1.setValue(MDL_ANGLE_TILT,tilt,id);
    //psi assigned by defaults
    id = auxMD1.firstRowId();
    auxMD1.getValueOrDefault(MDL_ANGLE_ROT,rot2, id, 0.);
    auxMD1.getValueOrDefault(MDL_ANGLE_TILT,tilt2, id, 0.);
    auxMD1.getValueOrDefault(MDL_ANGLE_PSI,psi2, id, 3.);

    EXPECT_EQ(rot,rot2);
    EXPECT_EQ(tilt,tilt2);
    EXPECT_EQ(psi,psi2);

    MDRowVec  rowIn;
    psi2=0;
    auxMD1.getRow(rowIn, id);
    rowIn.getValueOrDefault(MDL_ANGLE_PSI,psi2,3.);
    EXPECT_EQ(psi,psi2);

    auxMD1.getRow(rowIn, id);
    rowIn.getValueOrDefault(MDL_ANGLE_PSI,psi2,3.);
    EXPECT_EQ(psi,psi2);

    XMIPP_CATCH
}

TEST_F(MetadataTest, getValueAbort)
{
    XMIPP_TRY
    size_t id;
    MetaDataVec auxMD1;
    double rot=1.;
    id = auxMD1.addObject();
    auxMD1.setValue(MDL_ANGLE_ROT,rot,id);
    //psi assigned by defaults
    id = auxMD1.firstRowId();
    std::cerr << "TEST COMMENT: You should get the error  Cannot find label: order_" <<std::endl;
    EXPECT_THROW(auxMD1.getValueOrAbort(MDL_ORDER, rot, id), XmippError);
    MDRowVec rowIn;
    auxMD1.getRow(rowIn, id);
    std::cerr << "TEST COMMENT: You should get the error  Cannot find label: anglePsi" <<std::endl;
    EXPECT_THROW(rowGetValueOrAbort(rowIn,MDL_ANGLE_PSI,rot), XmippError);
    XMIPP_CATCH
}

TEST_F(MetadataTest, CopyColumn)
{
    XMIPP_TRY
    MetaDataVec md1(mDsource), md2(mDsource);
    double value;

    for (size_t objId : md1.ids())
    {
        md1.getValue(MDL_Y, value, objId);
        md1.setValue(MDL_Z, value, objId);
    }

    md2.copyColumn(MDL_Z, MDL_Y);

    EXPECT_EQ(md1, md2);
    XMIPP_CATCH
}

TEST_F(MetadataTest, RenameColumn)
{
    size_t id;
    XMIPP_TRY
    MetaDataVec md1(mDsource);
    MetaDataVec md2;
    md1.renameColumn(MDL_Y,MDL_Z);
    id = md2.addObject();
    md2.setValue(MDL_X,1.,id);
    md2.setValue(MDL_Z,2.,id);
    id = md2.addObject();
    md2.setValue(MDL_X,3.,id);
    md2.setValue(MDL_Z,4.,id);


    EXPECT_EQ(md1, md2);
    XMIPP_CATCH
}

//Copy images on metadata using ImageConvert logic
TEST_F(MetadataTest, copyImages)
{
    XMIPP_TRY
    FileName fn = "metadata/smallStack.stk";
    FileName out;
    FileName oroot;
    oroot.initUniqueName("/tmp/smallImg_XXXXXX");
    oroot.deleteFile();
    out = oroot.addExtension("xmd");
    oroot = oroot + ":mrc";

    FileName fn1, fn2;
    MetaDataVec md(fn);
    ProgConvImg conv;
    conv.verbose = 0;
    conv.setup(&md, out, oroot);
    conv.tryRun();
    MetaDataVec& mdOut = dynamic_cast<MetaDataVec&>(conv.getOutputMd());

    auto itermd = md.ids().begin();
    auto itermdout = mdOut.ids().begin();

    for (; itermd != md.ids().end(); ++itermd, ++itermdout)
    {
        md.getValue(MDL_IMAGE, fn1, *itermd);
        mdOut.getValue(MDL_IMAGE, fn2, *itermdout);
        EXPECT_TRUE(compareImage(fn1, fn2));
        fn2.deleteFile();
    }

    out.deleteFile();
    out.initUniqueName("/tmp/smallStack_XXXXXX");
    out = out + ":mrcs";
    conv.setup(&md, out);
    conv.tryRun();

    itermd = md.ids().begin();
    itermdout = mdOut.ids().begin();

    for (; itermd != md.ids().end(); ++itermd, ++itermdout)
    {
        md.getValue(MDL_IMAGE, fn1, *itermd);
        mdOut.getValue(MDL_IMAGE, fn2, *itermdout);
        EXPECT_TRUE(compareImage(fn1, fn2));
    }
    out.deleteFile();

    out.initUniqueName("/tmp/smallStackVol_XXXXXX");
    out = out + ":mrc";
    conv.setup(&md, out);
    conv.setType("vol");
    conv.tryRun();
    Image<float> imgStk, imgVol;
    imgStk.read(fn);
    imgVol.read(out);

    int n = imgStk.getDimensions().ndim;
    for (int i = FIRST_IMAGE; i <= n; ++i)
    {
        imgStk.movePointerTo(1, i);
        imgVol.movePointerTo(i);
        EXPECT_TRUE(imgStk == imgVol);
    }
    out.deleteFile();
    XMIPP_CATCH
}

TEST_F(MetadataTest, updateRow)
{
    size_t id1, id2;

    ASSERT_EQ(mDsource,mDsource);
    ASSERT_FALSE(mDsource==mDanotherSource);

    //attribute order should not be important
    MetaDataVec auxMetadata;
    id1 = auxMetadata.addObject();
    auxMetadata.setValue(MDL_Y,0.,id1);
    auxMetadata.setValue(MDL_X,0.,id1);
    id2 = auxMetadata.addObject();
    auxMetadata.setValue(MDL_Y,0.,id2);
    auxMetadata.setValue(MDL_X,0.,id2);
    ASSERT_FALSE(auxMetadata==mDsource);

    MDRowVec row;
    row.setValue(MDL_X, 1.);
    row.setValue(MDL_Y, 2.);
    auxMetadata.setRow(row, id1);
    row.setValue(MDL_X, 3.);
    row.setValue(MDL_Y, 4.);
    auxMetadata.setRow(row, id2);
    ASSERT_EQ(auxMetadata,mDsource);

    row.setValue(MDL_X, 1.);
    row.setValue(MDL_Y, 2.);
    auxMetadata.setRow(row, id1);
    row.setValue(MDL_X, 3.);
    row.setValue(MDL_Y, 4.);
    auxMetadata.setRow(row, id2);

    ASSERT_EQ(auxMetadata,mDsource);
}
