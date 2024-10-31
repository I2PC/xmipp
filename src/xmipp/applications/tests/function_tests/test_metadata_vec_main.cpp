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
#include "core/metadata_db.h"

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

        if (chdir(((String)(getXmippSrcPath() + (String)"/resources/test")).c_str())==-1)
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
    for (size_t i = 0; i < mDsourceIds.size(); i++, ++it); // reach end of MetaData
    ASSERT_EQ(it, mDsource.ids().end());

    size_t i = 0;
    for (size_t objId : mDsource.ids()) {
        ASSERT_EQ(objId, mDsourceIds[i]);
        ASSERT_NE(objId, BAD_OBJID);
        i++;
    }
    ASSERT_EQ(i, mDsourceIds.size());
}

TEST_F(MetadataTest, GetValue)
{
    MetaDataVec md;
    MDRowVec row;
    row.setValue(MDL_X, 10.);
    size_t id = md.addRow(row);

    EXPECT_EQ(md.getValue<double>(MDL_X, id), 10.);
    EXPECT_EQ(md.getValue<double>(MDL_X, id), 10.);

    {
        double value;
        md.getValue<double>(MDL_X, value, id);
        EXPECT_EQ(value, 10.);
    }

    MDObject obj(MDL_X);
    md.getValue(obj, id);
    EXPECT_EQ(obj.getValue2(1.), 10.);

    {
        const double value = md.getValueOrDefault<double>(MDL_X, id, 0.);
        EXPECT_EQ(value, 10.);
    }
    {
        const double value = md.getValueOrDefault<double>(MDL_Y, id, 0.);
        EXPECT_EQ(value, 0.);
    }
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

TEST_F(MetadataTest, AssignmentFromVecOperator)
{
    MetaDataVec orig, assigned;
    MDRowVec row;
    row.setValue(MDL_X, 10.);
    orig.addRow(row);
    row.setValue(MDL_Y, 100.);
    assigned.addRow(row);
    assigned.addRow(row);

    assigned = orig;

    ASSERT_EQ(orig.getColumnValues<double>(MDL_X), (std::vector<double>{10.}));
    EXPECT_EQ(orig.size(), assigned.size());
    EXPECT_EQ(assigned.getColumnValues<double>(MDL_X), (std::vector<double>{10.}));
    EXPECT_FALSE(assigned.containsLabel(MDL_Y));
    EXPECT_EQ(orig, assigned);
}

TEST_F(MetadataTest, AssignmentFromDbOperator)
{
    MetaDataDb orig;
    MetaDataVec assigned;

    {
        MDRowSql row;
        row.setValue(MDL_X, 10.);
        orig.addRow(row);
    }

    {
        MDRowVec row;
        row.setValue(MDL_X, 10.);
        row.setValue(MDL_Y, 100.);
        assigned.addRow(row);
        assigned.addRow(row);
    }

    assigned = orig;

    ASSERT_EQ(orig.getColumnValues<double>(MDL_X), (std::vector<double>{10.}));
    EXPECT_EQ(orig.size(), assigned.size());
    EXPECT_EQ(assigned.getColumnValues<double>(MDL_X), (std::vector<double>{10.}));
    EXPECT_FALSE(assigned.containsLabel(MDL_Y));
    EXPECT_EQ(orig, assigned);
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
    MetaDataVec md;
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
    size_t s1;
    t.tic();

    for (int i=0; i<N_ROWS_PERFORMANCE_TEST; i++)
        md.addRow(row);
    s1 = t.toc("Time:", false);
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

    unlink(fn.c_str());
    unlink(fnSTAR.c_str());
}

TEST_F(MetadataTest, multiWrite)
{
    FileName fnSTAR;
    fnSTAR.initUniqueName("/tmp/testReadMultipleBlocks_XXXXXX");
    fnSTAR += ".xmd";

    FileName fnSTARref = (String)"metadata/mDsource.xmd";

    mDsource.write((String)"myblock@"+fnSTAR);

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

    EXPECT_EQ(md2, md3);
}

TEST_F(MetadataTest, RegularExp)
{
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
    MetaDataVec aux = mDunion;
    ASSERT_TRUE(aux.containsLabel(MDL_X));
    ASSERT_EQ(aux.getColumnValues<double>(MDL_X), (std::vector<double>{1., 3., 11., 33.}));
    ASSERT_EQ(aux.getColumnValues<double>(MDL_Y), (std::vector<double>{2., 4., 22., 44.}));

    ASSERT_TRUE(aux.removeLabel(MDL_X));
    EXPECT_FALSE(aux.containsLabel(MDL_X));
    EXPECT_EQ(aux.getColumnValues<double>(MDL_Y), (std::vector<double>{2., 4., 22., 44.}));
    for (const auto& row : aux) {
        EXPECT_FALSE(row.containsLabel(MDL_X));
        EXPECT_TRUE(row.containsLabel(MDL_Y));
        EXPECT_EQ(row.getValueOrDefault<double>(MDL_X, 42.), 42.);
        EXPECT_NE(row.getValueOrDefault<double>(MDL_Y, 42.), 42.);
    }

    EXPECT_FALSE(aux.removeLabel(MDL_X));
    aux.setColumnValues(MDL_Z, std::vector<double>{0., 1., 2., 3.});
    EXPECT_EQ(aux.getColumnValues<double>(MDL_Z), (std::vector<double>{0., 1., 2., 3.}));
    for (size_t i = 0; i < aux.size(); i++)
        EXPECT_EQ(aux.getValue<double>(MDL_Z, aux.getRowId(i)), i);

    ASSERT_TRUE(aux.addLabel(MDL_X));
    EXPECT_TRUE(aux.containsLabel(MDL_X));
    ASSERT_TRUE(aux.removeLabel(MDL_X));
    EXPECT_FALSE(aux.containsLabel(MDL_X));

    aux.setColumnValues(MDL_X, std::vector<double>{1., 2., 3., 4.});
    EXPECT_EQ(aux.getColumnValues<double>(MDL_X), (std::vector<double>{1., 2., 3., 4.}));
    EXPECT_EQ(aux.getColumnValues<double>(MDL_Z), (std::vector<double>{0., 1., 2., 3.}));
    for (size_t i = 0; i < aux.size(); i++) {
        EXPECT_EQ(aux.getValue<double>(MDL_X, aux.getRowId(i)), i+1);
        EXPECT_EQ(aux.getValue<double>(MDL_Z, aux.getRowId(i)), i);
    }
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
}
//read file with vector
TEST_F(MetadataTest, getValue)
{
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
}

TEST_F(MetadataTest, getValueDefault)
{
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
}

TEST_F(MetadataTest, getValueAbort)
{
    size_t id;
    MetaDataVec auxMD1;
    double rot=1.;
    id = auxMD1.addObject();
    auxMD1.setValue(MDL_ANGLE_ROT,rot,id);

    id = auxMD1.firstRowId();
    std::cerr << "TEST COMMENT: You should get the error  Cannot find label: order_" << std::endl;
    EXPECT_THROW(auxMD1.getValueOrAbort(MDL_ORDER, rot, id), XmippError);

    MDRowVec rowIn;
    auxMD1.getRow(rowIn, id);
    std::cerr << "TEST COMMENT: You should get the error  Cannot find label: anglePsi" << std::endl;
    EXPECT_THROW(rowGetValueOrAbort(rowIn, MDL_ANGLE_PSI, rot), XmippError);
}

TEST_F(MetadataTest, CopyColumn)
{
    MetaDataVec md1(mDsource), md2(mDsource);
    double value;

    for (size_t objId : md1.ids())
    {
        md1.getValue(MDL_Y, value, objId);
        md1.setValue(MDL_Z, value, objId);
    }

    md2.copyColumn(MDL_Z, MDL_Y);

    EXPECT_EQ(md1, md2);
}

TEST_F(MetadataTest, RenameColumn)
{
    size_t id;
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
}

//Copy images on metadata using ImageConvert logic
TEST_F(MetadataTest, copyImages)
{
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

TEST_F(MetadataTest, VecToDbAndBack)
{
    MetaDataDb mdDb(mDsource);
    ASSERT_EQ(mdDb.size(), mDsource.size());
    MetaDataVec mdVec(mdDb);
    ASSERT_EQ(mDsource, mdVec);
}

TEST_F(MetadataTest, split)
{
    MetaDataVec original;
    for (int value = 3; value >= 0; value--) {
        MDRowVec row;
        row.setValue(MDL_X, static_cast<double>(value));
        original.addRow(row);
    }
    ASSERT_EQ(original.size(), 4);
    ASSERT_EQ(original.getColumnValues<double>(MDL_X), (std::vector<double>{3., 2., 1., 0.}));

    std::vector<MetaDataVec> splitted;

    original.split(1, splitted, MDL_X);
    ASSERT_EQ(splitted.size(), 1);
    ASSERT_EQ(splitted[0].size(), 4);
    ASSERT_EQ(splitted[0].getColumnValues<double>(MDL_X), (std::vector<double>{0., 1., 2., 3.}));
    ASSERT_EQ(original.getColumnValues<double>(MDL_X), (std::vector<double>{3., 2., 1., 0.}));

    original.split(2, splitted, MDL_X);
    ASSERT_EQ(splitted.size(), 2);
    ASSERT_EQ(splitted[0].size(), 2);
    ASSERT_EQ(splitted[1].size(), 2);
    ASSERT_EQ(splitted[0].getColumnValues<double>(MDL_X), (std::vector<double>{0., 1.}));
    ASSERT_EQ(splitted[1].getColumnValues<double>(MDL_X), (std::vector<double>{2., 3.}));
    ASSERT_EQ(original.getColumnValues<double>(MDL_X), (std::vector<double>{3., 2., 1., 0.}));

    original.split(3, splitted, MDL_X);
    ASSERT_EQ(splitted.size(), 3);
    size_t total_size = 0;
    for (const auto& split : splitted) {
        ASSERT_TRUE(split.size() >= 1);
        ASSERT_TRUE(split.size() <= 2);
        total_size += split.size();
    }
    ASSERT_EQ(total_size, original.size());
    ASSERT_EQ(original.getColumnValues<double>(MDL_X), (std::vector<double>{3., 2., 1., 0.}));
}

TEST_F(MetadataTest, rowDetach)
{
    MetaDataVec orig;
    MDRowVec row;
    row.setValue(MDL_X, 10.);
    orig.addRow(row);

    // Change in iteration changes original value
    ASSERT_EQ(orig.getValue<double>(MDL_X, orig.firstRowId()), 10.);
    for (auto& row : orig)
        row.setValue(MDL_X, 5.);
    ASSERT_EQ(orig.getValue<double>(MDL_X, orig.firstRowId()), 5.);

    // Original value is not changed after detaching
    for (auto& row : orig) {
        row.detach();
        row.setValue(MDL_X, 10.);
    }
    ASSERT_EQ(orig.getValue<double>(MDL_X, orig.firstRowId()), 5.);

    // MetaDataRowVec::deepCopy should work too
    for (auto& row : orig) {
        MDRowVec detached = MDRowVec::deepCopy(dynamic_cast<MDRowVec&>(row));
        detached.setValue(MDL_X, 10.);
    }
    ASSERT_EQ(orig.getValue<double>(MDL_X, orig.firstRowId()), 5.);
}

TEST_F(MetadataTest, selectPart)
{
    MetaDataVec orig;
    for (size_t i = 0; i < 2; i++) {
        MDRowVec row;
        row.setValue(MDL_X, static_cast<double>(i));
        orig.addRow(row);
    }

    for (size_t i = 0; i < 2; i++) {
        MetaDataVec part;
        part.selectPart(orig, i, 1, MDL_OBJID);
        EXPECT_EQ(part.size(), 1);
        EXPECT_EQ(part.getValue<double>(MDL_X, part.firstRowId()), static_cast<double>(i));
    }

    {
        MetaDataVec part;
        part.selectPart(orig, 0, 2, MDL_OBJID);
        EXPECT_EQ(part.size(), 2);
        EXPECT_EQ(part.getColumnValues<double>(MDL_X), (std::vector<double>{0., 1.}));
    }
}
