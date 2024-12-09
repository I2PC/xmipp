#include <string>
#include "data/pdb.h"
#include "core/xmipp_filename.h"
#include <gtest/gtest.h>
#include <iostream>
#include <cstdio>

// MORE INFO HERE: http://code.google.com/p/googletest/wiki/AdvancedGuide
// This test is named "Size", and belongs to the "MetadataTest"
// test case.
class CIFTest : public ::testing::Test
{
protected:

    // virtual void TearDown() {}//Destructor
    FileName source1;
    FileName source2;
    //init metadatas
    virtual void SetUp()
    {
    }
};

std::string readPath = std::string(getXmippSrcPath()) + "/libcifpp/examples/1cbs.cif.gz";
std::string writePath = std::string(getXmippSrcPath()) + "/libcifpp/examples/1cbs_test.cif";

/**
 * @brief Checks the values for a given RichAtom
 * 
 * This function compares the given first atom of an AtomList
 * with some hard-coded known values.
 * 
 * @param firstAtom Atom whose values will be checked.
*/
void compareFirstAtom(RichAtom &firstAtom) {
    // Declaring auxiliary variables
    bool condition;
    float threshold = 0.0001;

    // Comparing values
    ASSERT_EQ(firstAtom.serial, 1);
    ASSERT_EQ(firstAtom.name[0], 'N');
    ASSERT_EQ(firstAtom.name, "N");
    ASSERT_EQ(firstAtom.name, "N");
    ASSERT_EQ(firstAtom.altId, "");
    ASSERT_EQ(firstAtom.resname, "PRO");
    ASSERT_EQ(firstAtom.altloc, "A");
    ASSERT_EQ(firstAtom.resseq, 1);
    ASSERT_EQ(firstAtom.seqId, 1);
    ASSERT_EQ(firstAtom.icode, "");
    // X position (threshold used because of floating point rounding errors)
    if (abs(firstAtom.x - 16.979) < 0.0001) condition = true; else condition = false;
    ASSERT_EQ(condition, true);
    // Y position (threshold used because of floating point rounding errors)
    if (abs(firstAtom.y - 13.301) < 0.0001) condition = true; else condition = false;
    ASSERT_EQ(condition, true);
    // Z position (threshold used because of floating point rounding errors)
    if (abs(firstAtom.z - 44.555) < 0.0001) condition = true; else condition = false;
    ASSERT_EQ(condition, true);
    // Occupancy (threshold used because of floating point rounding errors)
    if (abs(firstAtom.occupancy - 1.00) < 0.0001) condition = true; else condition = false;
    ASSERT_EQ(condition, true);
    // B factor (threshold used because of floating point rounding errors)
    if (abs(firstAtom.bfactor - 30.05) < 0.0001) condition = true; else condition = false;
    ASSERT_EQ(condition, true);
    ASSERT_EQ(firstAtom.charge, "");
    ASSERT_EQ(firstAtom.authSeqId, 1);
    ASSERT_EQ(firstAtom.authCompId, "PRO");
    ASSERT_EQ(firstAtom.authAsymId, "A");
    ASSERT_EQ(firstAtom.authAtomId, "N");
    ASSERT_EQ(firstAtom.pdbNum, 1);
}


TEST_F(CIFTest, readFile)
{
    // Declaring pdb object
    PDBRichPhantom pdb;

    // Reading cif file from relative path
    FileName fileName(readPath); // Sample file from libcifpp examples
    pdb.read(fileName);

    // Comparing values for the first atom in list
    // ATOM   1    N N   . PRO A 1 1   ? 16.979 13.301 44.555 1.00 30.05 ? 1   PRO A N   1 
    compareFirstAtom(pdb.atomList[0]);
}

TEST_F(CIFTest, writeFile)
{
    // Declaring pdb object
    PDBRichPhantom pdb;

    // Reading cif file from relative path
    FileName fileName(readPath); // Sample file from libcifpp examples
    pdb.read(fileName);

    // Writing file
    pdb.write(writePath);

    // Reading written file
    pdb.read(writePath);

    // Comparing values for the first atom in list
    // ATOM   1    N N   . PRO A 1 1   ? 16.979 13.301 44.555 1.00 30.05 ? 1   PRO A N   1 
    compareFirstAtom(pdb.atomList[0]);

    // Attempting to remove produced file
    remove(writePath.c_str());
}