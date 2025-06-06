# ***************************************************************************
# * Authors:     J.M. De la Rosa Trevin (jmdelarosa@cnb.csic.es)
# *
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# * GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public License
# * along with this program; if not, write to the Free Software
# * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# * 02111-1307  USA
# *
# *  All comments concerning this program package may be sent to the
# *  e-mail address 'scipion@cnb.csic.es'
# ***************************************************************************/
VAHID = "vahid"
RM = 'rmarabini'
COSS = 'coss'
JMRT = 'delarosatrevin'
JOTON = 'joton'
DISCONTINUED = 'nobody'
JMOTA = 'javimota'
EFG = 'estrellafg'
JMK = 'jamesmkrieger'

# import math
import os

# import pyworkflow.utils as pwutils
# import xmipp3
# from pyworkflow.tests import DataSet
from tests.test import ProgramTest

def yellow(text):
    return f"\033[93m{text}\033[0m"

class XmippProgramTest(ProgramTest):
    
    @classmethod
    def setUpClass(cls):
        #cls.setTestDir(os.path.join(os.environ['SCIPION_TESTS', 'testXmipp']))
        cls.program = cls.getProgram()
        # cls.env = xmipp3.getEnviron()
        # cls.dataset = DataSet.getDataSet('xmipp_programs') # ----------------------------------------------
        print(' ')
        print(">>>>> OWNER: %s" % cls._owner)
        #cls._counter = 0 # count number of cases per test
        
    def runCase(self, *args, **kwargs):
        if 'mpi' not in kwargs:
            kwargs['mpi'] = 2 if self.program.startswith('xmipp_mpi') else 0
        ProgramTest.runCase(self, *args, **kwargs)


class AngularDiscreteAssign(XmippProgramTest):
    _owner = COSS
    @classmethod
    def getProgram(cls):
        return 'xmipp_angular_discrete_assign'

    def test_case1(self):
        self.runCase("-i input/aFewProjections.sel -o %o/assigned_angles.xmd --ref %o/reference.doc",
                preruns=["xmipp_angular_project_library -i input/phantomBacteriorhodopsin.vol -o %o/reference.stk --sampling_rate 10" ],
                outputs=["assigned_angles.xmd"])


class AngularDistance(XmippProgramTest):
    _owner = COSS
    @classmethod
    def getProgram(cls):
        return 'xmipp_angular_distance'

    def test_case1(self):
        self.runCase("--ang1 input/discreteAssignment.xmd --ang2 input/aFewProjections.sel --oroot %o/angleComparison --sym c3 -v 2",
                outputs=["angleComparison.xmd"])


class AngularDistributionShow(XmippProgramTest):
    _owner = RM
    @classmethod
    def getProgram(cls):
        return 'xmipp_angular_distribution_show'

    def test_case1(self):
        self.runCase("-i input/randomAngularDistribution.sel -o %o/distribution.xmd histogram",
                outputs=["distribution.xmd"])
    def test_case2(self):
        self.runCase("-i input/randomAngularDistribution.sel -o %o/distribution.bild chimera",
                outputs=["distribution.bild"])


class AngularNeighbourhood(XmippProgramTest):
    _owner = RM
    @classmethod
    def getProgram(cls):
        return 'xmipp_angular_neighbourhood'

    def test_case1(self):
        self.runCase("--i1 input/randomAngularDistribution.sel --i2 input/aFewProjections.sel -o %o/neighborhood.sel",
                outputs=["neighborhood.sel"])


class AngularProjectLibrary(XmippProgramTest):
    _owner = RM
    @classmethod
    def getProgram(cls):
        return 'xmipp_angular_project_library'

    def test_case1(self):
        self.runCase("-i input/phantomBacteriorhodopsin.vol -o %o/output_projections.stk --sym c6 --sampling_rate 5",
                outputs=["output_projections.doc","output_projections.stk"])
    def test_case2(self):
        self.runCase("-i input/phantomBacteriorhodopsin.vol -o %o/output_projections.stk --sym c6 --sampling_rate 5 --compute_neighbors --experimental_images input/aFewProjections.sel  --angular_distance -1",
                outputs=["output_projections.doc","output_projections.stk","output_projections_sampling.xmd"])
    def test_case3(self):
        self.runCase("-i input/phantomBacteriorhodopsin.vol -o %o/output_projections.stk --sym c6 --sampling_rate 5 --compute_neighbors --experimental_images input/aFewProjections.sel  --angular_distance 10",
                outputs=["output_projections.doc","output_projections.stk","output_projections_sampling.xmd"])
    def test_case4(self):
        self.runCase("-i input/phantomBacteriorhodopsin.vol -o %o/output_projections.stk --sym c6 --sampling_rate 5 --compute_neighbors --experimental_images input/aFewProjections.sel  --angular_distance 10 --near_exp_data",
                outputs=["output_projections.doc","output_projections.stk","output_projections_sampling.xmd"])
    def test_case5(self):
        self.runCase("-i input/phantomBacteriorhodopsin.vol -o %o/output_projections.stk --sym c6 --sampling_rate 5 --method real_space",
                outputs=["output_projections.doc","output_projections.stk"])


class AngularProjectLibraryMpi(AngularProjectLibrary):
    _owner = RM
    @classmethod
    def getProgram(cls):
        return 'xmipp_mpi_angular_project_library'

    def test_case6(self):
        self.runCase("-i input/phantomBacteriorhodopsin.vol -o %o/output_projections.stk --sym c6 --sampling_rate 5",
                outputs=["output_projections.doc","output_projections.stk"])
    def test_case7(self):
        self.runCase("-i input/phantomBacteriorhodopsin.vol -o %o/output_projections.stk --sym c6 --sampling_rate 5 --compute_neighbors --experimental_images input/aFewProjections.sel  --angular_distance -1",
                outputs=["output_projections.doc","output_projections.stk","output_projections_sampling.xmd"])
    def test_case8(self):
        self.runCase("-i input/phantomBacteriorhodopsin.vol -o %o/output_projections.stk --sym c6 --sampling_rate 5 --compute_neighbors --experimental_images input/aFewProjections.sel  --angular_distance 10",
                outputs=["output_projections.doc","output_projections.stk","output_projections_sampling.xmd"])


class AngularProjectionMatching(XmippProgramTest):
    _owner = RM
    @classmethod
    def getProgram(cls):
        return 'xmipp_angular_projection_matching'

    def test_case1(self):
        self.runCase("-i input/aFewProjections.sel -o %o/assigned_angles.xmd --ref %o/reference.stk --thr 3",
                preruns=["xmipp_angular_project_library -i input/phantomBacteriorhodopsin.vol --experimental_images input/aFewProjections.sel -o %o/reference.stk --sampling_rate 10 --compute_neighbors --angular_distance -1" ],
                outputs=["reference.doc","reference_sampling.xmd","reference.stk","assigned_angles.xmd"])


class AngularProjectionMatchingMpi(AngularProjectionMatching):
    _owner = RM
    @classmethod
    def getProgram(cls):
        return 'xmipp_mpi_angular_projection_matching'

    def test_case2(self):
        self.runCase("-i input/aFewProjections.sel -o %o/assigned_angles.xmd --ref %o/reference.stk",
                preruns=["xmipp_angular_project_library -i input/phantomBacteriorhodopsin.vol --experimental_images input/aFewProjections.sel -o %o/reference.stk --sampling_rate 10 --compute_neighbors --angular_distance -1" ],
                outputs=["reference.doc","reference_sampling.xmd","reference.stk","assigned_angles.xmd"])


class AngularRotate(XmippProgramTest):
    _owner = COSS
    @classmethod
    def getProgram(cls):
        return 'xmipp_angular_rotate'

    def test_case1(self):
        self.runCase("-i input/aFewProjections.sel --axis 10 1 2 3 -o %o/newAnglesFewProjections.sel",
                outputs=["newAnglesFewProjections.sel"])


class ClassifyAnalyzeCluster(XmippProgramTest):
    _owner = COSS
    @classmethod
    def getProgram(cls):
        return 'xmipp_classify_analyze_cluster'

    def test_case1(self):
        self.runCase("-i input/smallStack.stk --ref 1@input/smallStack.stk -o %o/pca.xmd",
                outputs=["pca.xmd"])


class ClassifyCompareClasses(XmippProgramTest):
    _owner = COSS
    @classmethod
    def getProgram(cls):
        return 'xmipp_classify_compare_classes'

    def test_case1(self):
        self.runCase("--i1 input/CL2DBacteriorhodopsin/level_00/class_classes.xmd --i2 input/CL2DBacteriorhodopsin/level_01/class_classes.xmd -o %o/classes_hierarchy.txt",
                outputs=["classes_hierarchy.txt"])


class ClassifyEvaluateClasses(XmippProgramTest):
    _owner = COSS
    @classmethod
    def getProgram(cls):
        return 'xmipp_classify_evaluate_classes'

    def test_case1(self):
        self.runCase("-i input/CL2DBacteriorhodopsin/level_00/class_classes.xmd -o %o/evaluated_classes.xmd",
                outputs=["evaluated_classes.xmd"])


class ClassifyKerdensom(XmippProgramTest):
    _owner = COSS
    @classmethod
    def getProgram(cls):
        return 'xmipp_classify_kerdensom'

    def test_case1(self):
        self.runCase("-i input/clusterVectors.xmd --oroot %o/kerdensom --deterministic_annealing 1",random=True,
                     postruns=["xmipp_image_vectorize -i %o/kerdensom_vectors.xmd -o %o/kerdensom_vectors.stk"],
                     outputs=["kerdensom_vectors.stk", "kerdensom_vectors.xmd"])


class CtfCorrectWiener3d(XmippProgramTest):
    _owner = RM
    @classmethod
    def getProgram(cls):
        return 'xmipp_ctf_correct_wiener3d'

    def test_case1(self):
        # FIX ME: We change 'wiener' to 'wiener2' to avoid delta error in devel and master devel.
        #         This is because some calculations have changed giving ALMOST the same result.
        self.runCase("-i input/ctf_correct3d.xmd --oroot %o/wiener2",
                outputs=["wiener2_deconvolved.vol","wiener2_ctffiltered_group000001.vol"])

    def test_case2(self):
        self.runCase("-i input/ctf_correct3d.xmd --oroot %o/wiener",
                     outputs=["wiener_deconvolved.vol",
                              "wiener_ctffiltered_group000001.vol"])


class CtfEnhancePsd(XmippProgramTest):
    _owner = COSS
    @classmethod
    def getProgram(cls):
        return 'xmipp_ctf_enhance_psd'

    def test_case1(self):
        self.runCase("-i input/down1_01nov26b.001.001.001.002_Periodogramavg.psd -o %o/enhanced_psd.xmp",
                outputs=["enhanced_psd.xmp"])


class CtfEstimateFromMicrograph(XmippProgramTest):
    _owner = COSS
    @classmethod
    def getProgram(cls):
        return 'xmipp_ctf_estimate_from_micrograph'

    def test_case1(self):
        self.runCase("--micrograph input/Protocol_Preprocess_Micrographs/Micrographs/01nov26b.001.001.001.002.mrc --oroot %o/micrograph --dont_estimate_ctf",
                outputs=["micrograph.psd"])
    
    def test_case2(self):
        cause = 'outputs of xmipp_ctf_estimate_from_micrograph are highly unstable'
        print(yellow('test_case2 is skipped as ' + cause))
        self.skipTest(cause)
        self.setTimeOut(400)
        self.runCase("--micrograph input/Protocol_Preprocess_Micrographs/Micrographs/01nov26b.001.001.001.002.mrc --oroot %o/micrograph --sampling_rate 1.4 --voltage 200 --spherical_aberration 2.5 --pieceDim 256 --downSamplingPerformed 2.5 --ctfmodelSize 256  --defocusU 14900 --defocusV 14900 --min_freq 0.01 --max_freq 0.3 --defocus_range 1000",
                postruns=["xmipp_metadata_utilities -i %o/micrograph.ctfparam --operate keep_column 'ctfDefocusU ctfDefocusV' -o %o/Defocus.xmd" ,
                          'xmipp_metadata_utilities -i %o/Defocus.xmd --operate  modify_values "ctfDefocusU = round(ctfDefocusU/100.0)" ',
                          'xmipp_metadata_utilities -i %o/Defocus.xmd --operate  modify_values "ctfDefocusV = round(ctfDefocusV/100.0)" '],
                outputs=["micrograph.psd","micrograph_enhanced_psd.xmp","micrograph.ctfparam","Defocus.xmd"])
    
    def test_case3(self):
        cause = 'outputs of xmipp_ctf_estimate_from_micrograph are highly unstable'
        print(yellow('test_case3 is skipped as ' + cause))
        self.skipTest(cause)
        self.runCase("--micrograph input/Protocol_Preprocess_Micrographs/Micrographs/01nov26b.001.001.001.002.mrc --oroot %o/micrograph --sampling_rate 1.4 --voltage 200 --spherical_aberration 2.5 --pieceDim 256 --downSamplingPerformed 2.5 --ctfmodelSize 256  --defocusU 14900 --defocusV 14900 --min_freq 0.01 --max_freq 0.3 --defocus_range 1000 --acceleration1D",
        postruns=["xmipp_metadata_utilities -i %o/micrograph.ctfparam --operate keep_column 'ctfDefocusU ctfDefocusV' -o %o/Defocus.xmd",
            'xmipp_metadata_utilities -i %o/Defocus.xmd --operate  modify_values "ctfDefocusU = round(ctfDefocusU/100.0)" ',
            'xmipp_metadata_utilities -i %o/Defocus.xmd --operate  modify_values "ctfDefocusV = round(ctfDefocusV/100.0)" '],
        outputs=["micrograph.psd", "micrograph_enhanced_psd.xmp",
                 "micrograph.ctfparam", "Defocus.xmd"])


class CtfEstimateFromPsd(XmippProgramTest):
    _owner = RM
    @classmethod
    def getProgram(cls):
        return 'xmipp_ctf_estimate_from_psd'

    def test_case1(self):
        self.runCase("--psd %o/down1_01nov26b.001.001.001.002_Periodogramavg2.psd --sampling_rate 1.4 --voltage 200 --spherical_aberration 2.5 --defocusU 15000 --defocus_range 200 --ctfmodelSize 170",
                preruns=["cp input/down1_01nov26b.001.001.001.002_Periodogramavg.psd %o" ," xmipp_transform_downsample -i %o/down1_01nov26b.001.001.001.002_Periodogramavg.psd -o %o/down1_01nov26b.001.001.001.002_Periodogramavg2.psd --step 3" ],
                postruns=["xmipp_metadata_utilities -i %o/down1_01nov26b.001.001.001.002_Periodogramavg2.ctfparam --operate keep_column 'ctfDefocusU ctfDefocusV' -o %o/Defocus.xmd" ,'xmipp_metadata_utilities -i %o/Defocus.xmd --operate  modify_values "ctfDefocusU = (round(abs(15132.6-ctfDefocusU)/15132.6)*100)" ','xmipp_metadata_utilities -i %o/Defocus.xmd --operate  modify_values "ctfDefocusV = (round(abs(14800-ctfDefocusV)/14800)*100)" '],
                outputs=["Defocus.xmd"])

class CtfEstimateFromPsdFast(XmippProgramTest):
    _owner = RM
    @classmethod
    def getProgram(cls):
        return 'xmipp_ctf_estimate_from_psd_fast'

    def test_case1(self):
        self.runCase("--psd %o/down1_01nov26b.001.001.001.002_Periodogramavg2.psd --sampling_rate 1.4 --voltage 200 --spherical_aberration 2.5 --defocusU 15000 --defocus_range 200 --ctfmodelSize 170",
                preruns=["cp input/down1_01nov26b.001.001.001.002_Periodogramavg.psd %o" ," xmipp_transform_downsample -i %o/down1_01nov26b.001.001.001.002_Periodogramavg.psd -o %o/down1_01nov26b.001.001.001.002_Periodogramavg2.psd --step 3" ],
                postruns=["xmipp_metadata_utilities -i %o/down1_01nov26b.001.001.001.002_Periodogramavg2.ctfparam --operate keep_column 'ctfDefocusU ctfDefocusV' -o %o/Defocus.xmd" ,'xmipp_metadata_utilities -i %o/Defocus.xmd --operate  modify_values "ctfDefocusU = (round(abs(15132.6-ctfDefocusU)/15132.6)*100)" ','xmipp_metadata_utilities -i %o/Defocus.xmd --operate  modify_values "ctfDefocusV = (round(abs(14800-ctfDefocusV)/14800)*100)" '],
                outputs=["Defocus.xmd"])

class CtfGroup(XmippProgramTest):
    _owner = RM
    @classmethod
    def getProgram(cls):
        return 'xmipp_ctf_group'

    #  FIXME: replace ctf2_* to ctf_* from -o arg and outputs
    def test_case1(self):
        self.runCase("--ctfdat input/ctf_group/all_images_new.ctfdat -o %o/ctf2 --wiener --wc -1 --pad 2 --phase_flipped --error 0.5 --resol 5.6",
                outputs=["ctf2_ctf.xmp","ctf2Info.xmd","ctf2_wien.xmp"])
    def test_case2(self):
        self.runCase("--ctfdat input/ctf_group/all_images_new.ctfdat -o %o/ctf2 --wiener --wc -1 --pad 2 --phase_flipped --split input/ctf_group/ctf_split.doc",
                outputs=["ctf2_ctf.xmp","ctf2Info.xmd","ctf2_wien.xmp"])


class CtfPhaseFlip(XmippProgramTest):
    _owner = COSS
    @classmethod
    def getProgram(cls):
        return 'xmipp_ctf_phase_flip'

    def test_case1(self):
        self.runCase("-i input/Protocol_Preprocess_Micrographs/Micrographs/01nov26b.001.001.001.002.mrc -o %o/micrographFlipped.mrc --ctf input/Protocol_Preprocess_Micrographs/Micrographs/01nov26b.001.001.001.002.mrc.ctfparam",
                outputs=["micrographFlipped.mrc"])


class CtfSortPsds(XmippProgramTest):
    _owner = COSS
    @classmethod
    def getProgram(cls):
        return 'xmipp_ctf_sort_psds'

    def test_case1(self):
        self.runCase("-i all_micrographs.sel",
                preruns=["cp -r input/Protocol_Preprocess_Micrographs/Preprocessing/* %o" ],
                outputs=["all_micrographs.sel"])


class ImageAlign(XmippProgramTest):
    _owner = COSS
    @classmethod
    def getProgram(cls):
        return 'xmipp_image_align'

    def test_case1(self):
        self.runCase("-i input/smallStack.stk --oroot %o/aligned",
                outputs=["aligned_alignment.xmd"])


class ImageConvert(XmippProgramTest):
    _owner = RM
    @classmethod
    def getProgram(cls):
        return 'xmipp_image_convert'

    def test_case1(self):
        self.runCase("-i input/smallStack.stk -o %o/smallStack.mrcs -t stk",
                outputs=["smallStack.mrcs"])
    def test_case2(self):
        self.runCase("-i input/singleImage.spi -o %o/singleImage.mrc",
                outputs=["singleImage.mrc"])
    def test_case3(self):
        self.runCase("-i input/singleImage.spi -o %o/singleImage.raw",
                outputs=["singleImage.raw"])
    def test_case4(self):
        self.runCase("-i input/singleImage.spi -o %o/singleImage.img",
                outputs=["singleImage.img"])


class ImageFindCenter(XmippProgramTest):
    _owner = COSS
    @classmethod
    def getProgram(cls):
        return 'xmipp_image_find_center'

    def test_case1(self):
        self.runCase("-i input/projectionsBacteriorhodopsin.stk --oroot %o/info",
                outputs=["info_center.xmd"])


class ImageHeader(XmippProgramTest):
    _owner = RM
    @classmethod
    def getProgram(cls):
        return 'xmipp_image_header'

    def test_case1(self):
        self.runCase("-i input/smallStack.stk --extract -o %o/header.xmd",
                outputs=["header.xmd"])
    def test_case2(self):
        self.runCase("-i %o/newHeaderStack.xmd --assign",
                preruns=["cp input/smallStack.stk %o" ," xmipp_metadata_selfile_create -p %o/smallStack.stk -o %o/smallStack.xmd -s" ," xmipp_metadata_utilities -i %o/smallStack.xmd -o %o/newHeaderStack.xmd --fill shiftY constant 2" ],
                postruns=["xmipp_image_header -i %o/smallStack.xmd --extract -o %o/outputHeader.xmd" ],
                outputs=["outputHeader.xmd"])
    def test_case3(self):
        self.runCase("-i %o/newHeader.xmd --assign",
                preruns=[" xmipp_image_convert input/smallStack.stk --oroot %o/smallImages:spi -o %o/images.xmd" ," xmipp_metadata_utilities -i %o/images.xmd -o %o/newHeader.xmd --fill shiftY constant 2" ],
                postruns=["xmipp_image_header %o/images.xmd --extract -o %o/outputHeader.xmd" ],
                outputs=["outputHeader.xmd"])
    def test_case4(self):
        self.runCase("-i %o/smallStack.xmd --reset",
                preruns=["cp input/smallStack.stk %o" ," xmipp_metadata_selfile_create -p %o/smallStack.stk -o %o/smallStack.xmd -s" ," xmipp_metadata_utilities -i %o/smallStack.xmd -o %o/newHeaderStack.xmd --fill shiftY constant 2" ," xmipp_image_header -i %o/newHeaderStack.xmd --assign" ,"xmipp_image_header -i %o/smallStack.xmd --extract -o %o/preOutputHeader.xmd" ],
                postruns=["xmipp_image_header -i %o/smallStack.xmd --extract -o %o/outputHeader.xmd" ],
                outputs=["outputHeader.xmd"])
    def test_case5(self):
        self.runCase("-i %o/newHeader.xmd --assign",
                preruns=[" xmipp_image_convert input/smallStack.stk --oroot %o/smallImages:spi -o %o/images.xmd" ," xmipp_metadata_utilities -i %o/images.xmd -o %o/newHeader.xmd --fill shiftY constant 2" ],
                postruns=["xmipp_image_header %o/images.xmd --extract -o %o/outputHeader.xmd" ],
                outputs=["outputHeader.xmd"])
    def test_case6(self):
        self.runCase("-i %o/images.xmd --reset",
                preruns=[" xmipp_image_convert input/smallStack.stk --oroot %o/smallImages:spi -o %o/images.xmd" ,"xmipp_metadata_utilities -i %o/images.xmd -o %o/newHeader.xmd --fill shiftY constant 2" ,"xmipp_image_header -i %o/newHeader.xmd --assign" ,"xmipp_image_header %o/images.xmd --extract -o %o/preOutputHeader.xmd" ],
                postruns=["xmipp_image_header %o/images.xmd --extract -o %o/outputHeader.xmd" ],
                outputs=["outputHeader.xmd"])


class ImageHistogram(XmippProgramTest):
    _owner = COSS
    @classmethod
    def getProgram(cls):
        return 'xmipp_image_histogram'

    def test_case1(self):
        self.runCase("-i input/smallStack.stk -o %o/hist.doc --steps 256 --range -0.5 0.7",
                outputs=["hist.doc"])


class ImageOperate(XmippProgramTest):
    _owner = RM
    @classmethod
    def getProgram(cls):
        return 'xmipp_image_operate'

    def test_case1(self):
        self.runCase("-i input/singleImage.spi --mult 2.5 -o %o/mult.spi",
                outputs=["mult.spi"])
    def test_case2(self):
        self.runCase("-i input/singleImage.spi --pow  -o %o/power.spi",
                outputs=["power.spi"])
    def test_case3(self):
        self.runCase("-i input/singleImage.spi --radial_avg -o %o/rad_avg.spi",
                outputs=["rad_avg.spi"])
    def test_case4(self):
        self.runCase("-i input/phantomCandida.vol --slice 32 -o %o/slice.spi",
                outputs=["slice.spi"])
    def test_case5(self):
        self.runCase("-i input/phantomCandida.vol --column 32 -o %o/column.spi",
                outputs=["column.spi"])
    def test_case6(self):
        self.runCase("-i input/projectionsBacteriorhodopsin.stk --minus input/projectionsBacteriorhodopsinWithCTF.stk -o %o/diff.stk",
                outputs=["diff.stk"])


class ImageOperateMpi(ImageOperate):
    _owner = RM
    @classmethod
    def getProgram(cls):
        return 'xmipp_mpi_image_operate'

    def test_case7(self):
        self.runCase("-i input/singleImage.spi --mult 2.5 -o %o/mult.spi",
                outputs=["mult.spi"])
    def test_case8(self):
        self.runCase("-i input/singleImage.spi --pow  -o %o/power.spi",
                outputs=["power.spi"])
    def test_case9(self):
        self.runCase("-i input/singleImage.spi --radial_avg -o %o/rad_avg.spi",
                outputs=["rad_avg.spi"])
    def test_case10(self):
        self.runCase("-i input/phantomCandida.vol --slice 32 -o %o/slice.spi",
                outputs=["slice.spi"])
    def test_case11(self):
        self.runCase("-i input/phantomCandida.vol --column 32 -o %o/column.spi",
                outputs=["column.spi"])


class ImageResize(XmippProgramTest):
    _owner = RM
    @classmethod
    def getProgram(cls):
        return 'xmipp_image_resize'

    def test_case1(self):
        self.runCase("-i input/phantomBacteriorhodopsin.vol --pyramid -1.5 -o %o/bdr.vol",
                outputs=["bdr.vol"])
    def test_case2(self):
        self.runCase("-i input/phantomBacteriorhodopsin.vol --factor 2 -o %o/volume_64.vol",
                outputs=["volume_64.vol"])
    def test_case3(self):
        self.runCase("-i input/smallStack.stk --dim 32 -o %o/stack_32.stk",
                outputs=["stack_32.stk"])
    def test_case4(self):
        self.runCase("-i input/header.doc --fourier 32 --dont_apply_geo --oroot %o/halvedFourierDim:xmp -o %o/halvedFourierDim.xmd",
                outputs=["halvedFourierDim000001.xmp","halvedFourierDim000002.xmp","halvedFourierDim000003.xmp","halvedFourierDim000004.xmp"])
    def test_case5(self):
        self.runCase("-i input/header.doc --fourier 16 16 2 --oroot %o/halvedFourierDim:xmp -o %o/halvedFourierDim.xmd",
                outputs=["halvedFourierDim000001.xmp","halvedFourierDim000002.xmp","halvedFourierDim000003.xmp","halvedFourierDim000004.xmp"])
    def test_case6(self):
        self.runCase("-i input/phantomBacteriorhodopsin.vol --fourier 32 32 32 -o %o/volume_32.vol",
                outputs=["volume_32.vol"])




class ImageRotationalPcaMpi(XmippProgramTest):
    _owner = VAHID
    @classmethod
    def getProgram(cls):
        return 'xmipp_mpi_image_rotational_pca'

    def test_case1(self):
        self.runCase("-i input/smallStack.stk --oroot %o/invariant",
                postruns=['xmipp_metadata_utilities -i %o/invariant.xmd -o %o/rename.xmd    --operate rename_column "weight wRobust"  --mode overwrite','xmipp_metadata_utilities -i %o/rename.xmd    -o %o/merge.xmd     --set merge  %o/invariant.xmd image  --mode overwrite','xmipp_metadata_utilities -i %o/merge.xmd     -o %o/substract.xmd  --operate modify_values "weight=round(abs(weight-wRobust)*10)"  --mode overwrite','xmipp_metadata_utilities -i %o/substract.xmd -o %o/reference.xmd --operate drop_column "wRobust"  --mode overwrite'],
                outputs=["reference.xmd"])



class ImageSortByStatistics(XmippProgramTest):
    _owner = COSS
    @classmethod
    def getProgram(cls):
        return 'xmipp_image_sort_by_statistics'

    def test_case1(self):
        self.runCase("-i input/smallStack.stk -o %o/smallStackSorted.xmd",
                outputs=["smallStackSorted.xmd"])


class ImageStatistics(XmippProgramTest):
    _owner = COSS
    @classmethod
    def getProgram(cls):
        return 'xmipp_image_statistics'

    def test_case1(self):
        self.runCase("-i input/smallStack.stk -o %o/stats.xmd",
                outputs=["stats.xmd"])


class ImageVectorize(XmippProgramTest):
    _owner = COSS
    @classmethod
    def getProgram(cls):
        return 'xmipp_image_vectorize'

    def test_case1(self):
        self.runCase("-i input/aFewProjections.sel -o %o/vectors.xmd",
                outputs=["vectors.xmd"])
    def test_case2(self):
        self.runCase("-i input/vectors.xmd -o %o/images.stk",
                outputs=["images.stk"])

#use scipion
#class MetadataConvertToSpider(XmippProgramTest):

class MetadataSplit(XmippProgramTest):
    _owner = RM
    @classmethod
    def getProgram(cls):
        return 'xmipp_metadata_split'

    def test_case1(self):
        self.runCase("-i input/projectionsBacteriorhodopsin.xmd -n 3 --oroot %o/splitted --dont_randomize",
                outputs=["splitted000001.xmd","splitted000002.xmd","splitted000003.xmd"])
    def test_case2(self):
        self.runCase("-i input/projectionsBacteriorhodopsin.xmd -n 3 --oroot %o/splitted",random=True,
                outputs=["splitted000001.xmd","splitted000002.xmd","splitted000003.xmd"])


class MetadataUtilities(XmippProgramTest):
    _owner = RM
    @classmethod
    def getProgram(cls):
        return 'xmipp_metadata_utilities'

    def test_case1(self):
        self.runCase("-i input/mD1.doc --set union input/mD2.doc  -o %o/out1.doc",
                outputs=["out1.doc"])
    def test_case2(self):
        self.runCase("-i input/mD1.doc --operate sort -o %o/out2.doc",
                outputs=["out2.doc"])
    def test_case3(self):
        self.runCase("-i input/mD1.doc --fill shiftX rand_uniform 0 10 -o %o/out4.doc",
                outputs=["out4.doc"],random=True)
    def test_case4(self):
        self.runCase("-i input/mD1.doc -l \"shiftX shiftY\" constant 5 -o %o/out5.doc",
                outputs=["out5.doc"])
    def test_case5(self):
        self.runCase("-i input/mD1.doc --query select \"angleRot > 10 AND anglePsi < 0.5\" -o %o/out6.doc",
                outputs=["out6.doc"])
    def test_case6(self):
        self.runCase("-i input/mD1.doc --operate modify_values \"angleRot=(angleRot*3.1416/180.)\" -o %o/out7.doc",
                outputs=["out7.doc"])
    def test_case7(self):
        self.runCase("-i input/mD1.doc --operate modify_values \"image=replace(image, 'xmp','spi')\" -o %o/out8.doc",
                outputs=["out8.doc"])


class MicrographScissor(XmippProgramTest):
    _owner = RM
    @classmethod
    def getProgram(cls):
        return 'xmipp_micrograph_scissor'

    def test_case1(self):
        self.runCase("-i input/micrograph8bits.tif --pos input/micrograph8bits.tif.Common.pos -o %o/images --Xdim 256",
                outputs=["images.stk","images.xmd"])


class AngularClassAverageMpi(XmippProgramTest):
    _owner = RM
    @classmethod
    def getProgram(cls):
        return 'xmipp_mpi_angular_class_average'

    def test_case1(self):
        self.runCase("-i input/angular_class_average/current_angles.doc --lib input/angular_class_average/gallery_Ref3D_003.doc -o %o//projmatch --wien input/angular_class_average/ctf_wien.stk --pad 2 --split",
                outputs=["projmatch_discarded.xmd","projmatch_Ref3D_001.stk","projmatch_Ref3D_001.xmd"])


class AngularContinuousAssignMpi(XmippProgramTest):
    _owner = RM
    @classmethod
    def getProgram(cls):
        return 'xmipp_mpi_angular_continuous_assign'

    def test_case1(self):
        self.runCase("-i input/aFewProjections.sel --ref input/phantomBacteriorhodopsin.vol -o %o/assigned_angles.xmd",
                postruns=["xmipp_metadata_utilities -i %o/assigned_angles.xmd -e sort -o %o/assigned_angles.xmd" ],
                outputs=["assigned_angles.xmd"])


class ClassifyCL2DMpi(XmippProgramTest):
    _owner = COSS
    @classmethod
    def getProgram(cls):
        return 'xmipp_mpi_classify_CL2D'

    def test_case1(self):
        self.runCase("-i input/projectionsBacteriorhodopsin.stk --nref 4 --oroot class --odir %o --iter 4 --classicalMultiref",
                outputs=["images.xmd","level_00/class_classes.stk","level_00/class_classes.xmd","level_01/class_classes.stk","level_01/class_classes.xmd"])


class ClassifyCL2DCoreAnalysisMpi(XmippProgramTest):
    _owner = COSS
    @classmethod
    def getProgram(cls):
        return 'xmipp_mpi_classify_CL2D_core_analysis'

    def test_case1(self):
        self.runCase("--dir %o/input/CL2DBacteriorhodopsin --root class --computeCore 3.000000 3.000000",
                preruns=["mkdir %o/input ; cp -r input/CL2DBacteriorhodopsin %o/input ; rm -rf %o/input/CL2DBacteriorhodopsin/.svn ; cp input/projectionsBacteriorhodopsin.stk %o/input" ],
                outputs=['input/CL2DBacteriorhodopsin/level_00/class_classes_core.xmd'])




class ImageSortMpi(XmippProgramTest):
    _owner = COSS
    @classmethod
    def getProgram(cls):
        return 'xmipp_mpi_image_sort'

    def test_case1(self):
        self.runCase("-i input/aFewProjections.sel --oroot %o/sorted",
                outputs=["sorted.stk","sorted.xmd"])


class NmaAlignment(XmippProgramTest):
    _owner = COSS
    @classmethod
    def getProgram(cls):
        return 'xmipp_nma_alignment'

    def test_case1(self):
        self.runCase("-i 2tbv_prj00001.xmp  --pdb 2tbv.pdb --modes modelist.xmd --sampling_rate 6.4 -o output.xmd --resume",
                preruns=["cp input/2tbv* %o ; cp input/modelist.xmd %o ; cp input/mode0.mod0028 %o" ],
                postruns=["xmipp_metadata_utilities -i %o/output.xmd --operate keep_column 'cost' -o %o/cost.xmd" ,'xmipp_metadata_utilities -i %o/cost.xmd --operate  modify_values "cost = round(cost*100.0)" '],
                outputs=["cost.xmd"],
		changeDir=True)


class NmaAlignmentMpi(NmaAlignment):
    _owner = COSS
    @classmethod
    def getProgram(cls):
        return 'xmipp_mpi_nma_alignment'

    def test_case2(self):
        self.runCase("-i 2tbv_prj00001.xmp  --pdb 2tbv.pdb --modes modelist.xmd --sampling_rate 6.4 -o output.xmd --resume",
                preruns=["cp input/2tbv* %o ; cp input/modelist.xmd %o ; cp input/mode0.mod0028 %o" ],
                postruns=["xmipp_metadata_utilities -i %o/output.xmd --operate keep_column 'cost' -o %o/cost.xmd" ,'xmipp_metadata_utilities -i %o/cost.xmd --operate  modify_values "cost = round(cost*100.0)" '],
                outputs=["cost.xmd"],
		changeDir=True)


class RunMpi(XmippProgramTest):
    _owner = RM
    @classmethod
    def getProgram(cls):
        return 'xmipp_mpi_run'

    def test_case1(self):
        self.runCase("-i commands.cmd", changeDir=True,
                preruns=["cp input/commands.cmd %o" ],
                outputs=["file1.txt","file2.txt","file3.txt","file4.txt"])


class PdbReducePseudoatoms(XmippProgramTest):
    _owner = JMOTA
    @classmethod
    def getProgram(cls):
        return 'xmipp_pdb_reduce_pseudoatoms' 

    def test_case1(self):
        self.runCase("-i pseudoatoms.pdb -o pdbreduced.pdb --number 500",
                     preruns=["cp input/pseudoatoms.pdb %o"],
                     outputs=["pdbreduced.pdb"],
                     changeDir=True)


class PdbNmaDeform(XmippProgramTest):
    _owner = COSS
    @classmethod
    def getProgram(cls):
        return 'xmipp_pdb_nma_deform'

    def test_case1(self):
        self.runCase("--pdb 2tbv.pdb -o deformed2.pdb --nma modelist.xmd --deformations 1000",
                preruns=["cp input/2tbv* %o ; cp input/modelist.xmd %o ; cp input/mode0.mod0028 %o" ],
                outputs=["deformed2.pdb"], random=True, validate=self.validate,
		        changeDir=True)

    def validate(self):
        fileGoldStd = os.path.join(self.goldDir, "deformed2.pdb")
        outFile = os.path.join(self._testDir, self.outputDir, "deformed2.pdb")
        print("Checking ",fileGoldStd,outFile)
        with open(fileGoldStd,"r") as fh:
            linesGold = [line.rstrip() for line in fh.readlines()]
        with open(outFile,"r") as fh:
            lines = [line.rstrip() for line in fh.readlines()]

        def splitPDBLine(line):
            # ATOM      1  CA  GLY A 102     -22.617  31.293 119.792  1.00 20.00       CA
            token0 = line[0:6]
            token1 = line[6:11]
            token2 = line[12:16]
            token3 = line[16:17]
            token4 = line[17:20]
            token5 = line[21:22]
            token6 = line[22:26]
            token7 = line[26:27]
            token8 = line[30:38] # x
            token9 = line[38:46] # y
            token10 = line[46:54] # z
            token11 = line[54:60] # occupancy
            token12 = line[60:66] # temperature
            token13 = line[76:78]
            return [token0, token1, token2, token3, token4, token5, token6, token7, token8, token9, token10, token11,
                    token12, token13]

        ok = True
        for lineG, line in zip(linesGold[1:],lines[1:]):
            try:
                tokensG = splitPDBLine(lineG)
                tokens = splitPDBLine(line)
            except:
                ok = False
                break
            if len(tokensG)!=14 or len(tokens)!=14:
                print("The following two lines are not well formed")
                print("Gold: %s"%lineG)
                print("Test: %s"%line)
                ok = False
                break
            for i in [0,1,2,3,4,5,6,7,13]:
                if not tokensG[i]==tokens[i]:
                    print("The following two lines are not equal")
                    print("Gold: %s"%lineG)
                    print("Test: %s"%line)
                    ok = False
                    break
            for i in [8,9,10,11,12]:
                if abs(float(tokensG[i])-float(tokens[i]))>1e-2:
                    print("The following two lines are not equal")
                    print("Gold: %s"%lineG)
                    print("Test: %s"%line)
                    ok = False
                    break
        self.assertTrue(ok)

class PhantomCreate(XmippProgramTest):
    _owner = VAHID
    @classmethod
    def getProgram(cls):
        return 'xmipp_phantom_create'

    def test_case1(self):
        self.runCase("-i input/rings.descr -o %o/rings.vol",
                outputs=["rings.vol"])


class PhantomSimulateMicroscope(XmippProgramTest):
    _owner = COSS
    @classmethod
    def getProgram(cls):
        return 'xmipp_phantom_simulate_microscope'

    def test_case1(self):
        self.runCase("-i input/projectionsBacteriorhodopsin.stk -o %o/smallStackPlusCtf.stk --ctf input/input.ctfparam --noNoise",
                outputs=["smallStackPlusCtf.stk"],errorthreshold=0.05)
    def test_case2(self):
        self.runCase("-i input/projectionsBacteriorhodopsin.stk -o %o/smallStackPlusCtf.stk --ctf input/input.ctfparam --targetSNR 0.3",
                outputs=["smallStackPlusCtf.stk"],random=True)


class PhantomTransform(XmippProgramTest):
    _owner = COSS
    @classmethod
    def getProgram(cls):
        return 'xmipp_phantom_transform'

    def test_case1(self):
        self.runCase("-i input/1o7d.pdb -o %o/shifted.pdb --operation shift 1 2 3",
                outputs=["shifted.pdb"])


class ReconstructArt(XmippProgramTest):
    _owner = RM
    @classmethod
    def getProgram(cls):
        return 'xmipp_reconstruct_art'

    def test_case1(self):
        self.runCase("-i input/projectionsBacteriorhodopsin.xmd -n 2 --thr 2 -o  %o/rec_art.vol --noisy_reconstruction",
                postruns=["xmipp_image_statistics -i %o/rec_art.vol -o %o/stats.xmd" ,"xmipp_metadata_utilities -i %o/stats.xmd --operate keep_column 'avg' -o %o/average.xmd" ,'xmipp_metadata_utilities -i %o/average.xmd --operate  modify_values "avg = round(avg*10000.0)" '],
                outputs=["average.xmd"])
    def test_case2(self):
        self.runCase("-i input/art.xmd -o  %o/rec_art.vol  --sym c1 --thr 1 --WLS  -l 0.2 -k 0.5 -n 1",
                postruns=["xmipp_image_statistics -i %o/rec_art.vol -o %o/stats.xmd" ,"xmipp_metadata_utilities -i %o/stats.xmd --operate keep_column 'avg' -o %o/average.xmd" ,'xmipp_metadata_utilities -i %o/average.xmd --operate  modify_values "avg = round(avg*100000.0)" '],
                outputs=["average.xmd"])

#mpi version cannot use threads.
# That is the reason why this class does not inherit from  ReconstructArt
class ReconstructArtMpi(XmippProgramTest):
    _owner = RM
    @classmethod
    def getProgram(cls):
        return 'xmipp_mpi_reconstruct_art'

    def test_case1(self):
        self.runCase("-i input/projectionsBacteriorhodopsin.xmd -n 2  -o  %o/rec_art.vol --noisy_reconstruction",
                postruns=["xmipp_image_statistics -i %o/rec_art.vol -o %o/stats.xmd" ,"xmipp_metadata_utilities -i %o/stats.xmd --operate keep_column 'avg' -o %o/average.xmd" ,'xmipp_metadata_utilities -i %o/average.xmd --operate  modify_values "avg = round(avg*10000.0)" '],
                outputs=["average.xmd"])
#WLS and mpi is not compatible, thr may be used
#    def test_case2(self):
#        self.runCase("-i input/art.xmd -o  %o/rec_art.vol  --sym c1 --WLS  -l 0.2 -k 0.5 -n 1",
#                postruns=["xmipp_image_statistics -i %o/rec_art.vol -o %o/stats.xmd" ,"xmipp_metadata_utilities -i %o/stats.xmd --operate keep_column 'avg' -o %o/average.xmd" ,'xmipp_metadata_utilities -i %o/average.xmd --operate  modify_values "avg = round(avg*100000.0)" '],
#                outputs=["average.xmd"])
    def test_case2(self):
        self.runCase("-i input/projectionsBacteriorhodopsin.xmd -n 1 -o  %o/rec_art --noisy_reconstruction",
                postruns=["xmipp_image_statistics -i %o/rec_art.vol -o %o/stats.xmd" ,"xmipp_metadata_utilities -i %o/stats.xmd --operate keep_column 'avg' -o %o/average.xmd" ,'xmipp_metadata_utilities -i %o/average.xmd --operate  modify_values "avg = round(avg*10000.0)" '],
                outputs=["average.xmd"])


class ReconstructFourier(XmippProgramTest):
    _owner = VAHID
    @classmethod
    def getProgram(cls):
        return 'xmipp_reconstruct_fourier'

    def test_case1(self):
        self.runCase("-i input/aFewProjections.sel -o  %o/recon.vol ",
                outputs=["recon.vol"])


class ResolutionFsc(XmippProgramTest):
    _owner = RM
    @classmethod
    def getProgram(cls):
        return 'xmipp_resolution_fsc'

    def test_case1(self):
        self.runCase("--ref input/phantomBacteriorhodopsin.vol -i input/phantomCandida.vol -s 5.6 --do_dpr --oroot %o/phantomBacteriorhodopsin ",
                outputs=["phantomBacteriorhodopsin.frc"])


class TomoDetectMissingWedge(XmippProgramTest):
    _owner = COSS
    @classmethod
    def getProgram(cls):
        return 'xmipp_tomo_detect_missing_wedge'

    def test_case1(self):
        self.runCase("-i input/smallTomogram.vol",
                validate=self.validate_case1, random=True,
                outputs=["stdout.txt"])
    
    def validate_case1(self):
        stdout = os.path.join(self.outputDir, "stdout.txt")
        f = open(stdout).readlines()
        plane1 = 0.0
        plane2 = 0.0
        for line in f:
            if 'Plane1: ' in line:
                plane1 = list(map(float, line.split()[1:]))
            if 'Plane2: ' in line:
                plane2 = list(map(float, line.split()[1:]))
        if plane1[1] is None:
            self.assertIsNotNone(abs(plane1[1]))    # Assure test fails if var is None
        else:
            self.assertAlmostEqual(abs(plane1[1]), 67.0539, delta=1)
        
        if plane2[1] is None:     
            self.assertIsNotNone(abs(plane2[1]))    # Assure test fails if var is None
        else:
            self.assertAlmostEqual(abs(plane2[1]), 56.7034, delta=1)


class TomoProject(XmippProgramTest):
    _owner = RM
    @classmethod
    def getProgram(cls):
        return 'xmipp_tomo_project'

    def test_case1(self):
        self.runCase("-i input/phantomBacteriorhodopsin.vol -o %o/image.xmp --angles 0 90 90",
                outputs=["image.xmp"])
    def test_case2(self):
        self.runCase("-i input/phantomBacteriorhodopsin.vol --oroot %o/projections --params input/tomoProjection.param",
                outputs=["projections.stk","projections.sel"])


class TransformAddNoise(XmippProgramTest):
    _owner = COSS
    @classmethod
    def getProgram(cls):
        return 'xmipp_transform_add_noise'

    def test_case1(self):
        ''' Test to check if noise is properly simulated '''
        self.runCase("-i input/cleanImage.spi --type gaussian 10 5 -o %o/noisyGaussian.spi",
                outputs=["noisyGaussian.spi"], random=True)

    def test_case2(self):
        ''' Test to check if particle alignment is not applied '''
        self.runCase("-i input/projectionsBacteriorhodopsin.xmd --type gaussian 0 0 -o %o/notNoisyGaussian.stk",
                outputs=["notNoisyGaussian.stk"], random=True, validate=self.validate_case2)
    
    def validate_case2(self):
        import filecmp
        output = os.path.join(self.outputDir, "notNoisyGaussian.stk")
        self.assertTrue(filecmp.cmp(output, "input/projectionsBacteriorhodopsin.stk"))


class TransformCenterImage(XmippProgramTest):
    _owner = COSS
    @classmethod
    def getProgram(cls):
        return 'xmipp_transform_center_image'

    def test_case1(self):
        self.runCase("-i input/smallStack.stk -o %o/smallStackCentered.stk",
                outputs=["smallStackCentered.stk"])


class TransformDownsample(XmippProgramTest):
    _owner = COSS
    @classmethod
    def getProgram(cls):
        return 'xmipp_transform_downsample'

    def test_case1(self):
        self.runCase("-i input/micrograph8bits.tif -o %o/downsampledFourier.xmp --step 2",
                outputs=["downsampledFourier.xmp"])
    def test_case2(self):
        self.runCase("-i input/micrograph8bits.tif -o %o/downsampledSmooth.mrc --step 2 --method smooth",
                outputs=["downsampledSmooth.mrc"])


class TransformGeometry(XmippProgramTest):
    _owner = RM
    @classmethod
    def getProgram(cls):
        return 'xmipp_transform_geometry'

    def test_case1(self):
        self.runCase("-i input/header.doc --apply_transform -o %o/images.stk",
                outputs=["images.stk","images.xmd"])
    def test_case2(self):
        self.runCase("-i %o/header.xmd --apply_transform",
                preruns=["xmipp_image_convert input/header.doc -o %o/header.stk --save_metadata_stack" ],
                outputs=["header.stk","header.xmd"])
    def test_case3(self):
        self.runCase("-i input/header.doc --scale 0.5 --shift 5 10 -5 --rotate -45 -o %o/newGeo.xmd",
                outputs=["newGeo.xmd"])
    def test_case4(self):
        self.runCase("-i input/header.doc --scale 0.5 --shift 5 10 -5 --rotate -45 -o %o/newGeo.stk",
                outputs=["newGeo.xmd","newGeo.stk"])
    def test_case5(self):
        self.runCase("-i input/header.doc --scale 0.5 --shift 5 10 -5 --rotate -45 -o %o/newGeo.stk --apply_transform",
                outputs=["newGeo.xmd","newGeo.stk"])
    def test_case6(self):
        self.runCase("-i input/phantomBacteriorhodopsin.vol --shift 10 5 -10 --scale 0.8 -o %o/volume.vol --dont_wrap",
                outputs=["volume.vol"])
    def test_case7(self):
        self.runCase("-i input/phantomBacteriorhodopsin.vol --rotate_volume euler 10 5 -10 -o %o/volume.vol --dont_wrap",
                outputs=["volume.vol"])
    def test_case8(self):
        self.runCase("-i input/phantomBacteriorhodopsin.vol --rotate_volume alignZ 1 1 1 -o %o/volume.vol --dont_wrap",
                outputs=["volume.vol"])
    def test_case9(self):
        self.runCase("-i input/phantomBacteriorhodopsin.vol --rotate_volume axis 90 1 1 1 -o %o/volume.vol --dont_wrap",
                outputs=["volume.vol"])


class TransformMask(XmippProgramTest):
    _owner = COSS
    @classmethod
    def getProgram(cls):
        return 'xmipp_transform_mask'

    def test_case1(self):
        self.runCase("-i input/singleImage.spi -o %o/singleImage_mask.xmp --mask circular -15",
                outputs=["singleImage_mask.xmp"])
    def test_case2(self):
        self.runCase("-i input/phantomBacteriorhodopsin.vol -o %o/outputVol_mask.vol --mask rectangular -20 -20 -20",
                outputs=["outputVol_mask.vol"])
    def test_case3(self):
        self.runCase("-i input/smallStack.stk -o %o/outputStack_mask.stk --mask circular -20",
                outputs=["outputStack_mask.stk"])


class TransformMirror(XmippProgramTest):
    _owner = COSS
    @classmethod
    def getProgram(cls):
        return 'xmipp_transform_mirror'

    def test_case1(self):
        self.runCase("-i input/singleImage.spi -o %o/singleImage_X.xmp --flipX",
                outputs=["singleImage_X.xmp"])


class TransformMorphology(XmippProgramTest):
    _owner = COSS
    @classmethod
    def getProgram(cls):
        return 'xmipp_transform_morphology'

    def test_case1(self):
        self.runCase("-i input/maskBacteriorhodopsin.vol -o %o/dilated.spi --binaryOperation dilation",
                outputs=["dilated.spi"])


class TransformNormalize(XmippProgramTest):
    _owner = COSS
    @classmethod
    def getProgram(cls):
        return 'xmipp_transform_normalize'

    def test_case1(self):
        self.runCase("-i input/smallStack.stk -o %o/smallStackNormalized.stk --method NewXmipp --background circle 32",
                outputs=["smallStackNormalized.stk"])


class TransformSymmetrize(XmippProgramTest):
    _owner = RM
    @classmethod
    def getProgram(cls):
        return 'xmipp_transform_symmetrize'

    def test_case1(self):
        self.runCase("-i input/smallStack.stk -o %o/smallStackSymmetrized.stk --sym 5",
                outputs=["smallStackSymmetrized.stk"])
    def test_case2(self):
        self.runCase("-i input/phantomBacteriorhodopsin.vol -o %o/symmetrizedVolume.vol --sym C5",
                outputs=["symmetrizedVolume.vol"])


class TransformThreshold(XmippProgramTest):
    _owner = COSS
    @classmethod
    def getProgram(cls):
        return 'xmipp_transform_threshold'

    def test_case1(self):
        self.runCase("-i input/phantomBacteriorhodopsin.vol -o %o/mask.vol --select below 0.01 --substitute binarize",
                outputs=["mask.vol"])


class TransformWindow(XmippProgramTest):
    _owner = COSS
    @classmethod
    def getProgram(cls):
        return 'xmipp_transform_window'

    def test_case1(self):
        self.runCase("-i input/singleImage.spi -o %o/image.xmp --size 32",
                outputs=["image.xmp"])
    def test_case2(self):
        self.runCase("-i input/singleImage.spi -o %o/image.xmp --corners -16 -16 15 15",
                outputs=["image.xmp"])
    def test_case3(self):
        self.runCase("-i input/singleImage.spi -o %o/image.xmp --corners 0 0 31 31 --physical",
                outputs=["image.xmp"])
    def test_case4(self):
        self.runCase("-i input/singleImage.spi -o %o/image.xmp --crop -10",
                outputs=["image.xmp"])
    def test_case5(self):
        self.runCase("-i input/xray_import/Images/img48949.spe -o %o/image.xmp --size 512",outputs=["image.xmp"])

class VolumeAlign(XmippProgramTest):
    _owner = COSS
    @classmethod
    def getProgram(cls):
        return 'xmipp_volume_align'

    def test_case1(self):
        self.runCase("--i1 input/phantomBacteriorhodopsin.vol --i2 input/phantomBacteriorhodopsinRotated30.vol --rot 0 120 5 --apply %o/i2aligned.vol",
                outputs=["i2aligned.vol"])
    def test_case2(self):
        self.runCase("--i1 input/phantomBacteriorhodopsin.vol --i2 input/phantomBacteriorhodopsinRotated30.vol --rot 90 --local --apply %o/i2aligned.vol",
                outputs=["i2aligned.vol"])


class VolumeCenter(XmippProgramTest):
    _owner = COSS
    @classmethod
    def getProgram(cls):
        return 'xmipp_volume_center'

    def test_case1(self):
        self.runCase("-i input/1FFKfull_reconstructed.vol -o %o/centered.vol",
                outputs=["centered.vol"])


class VolumeCorrectBfactor(XmippProgramTest):
    _owner = COSS
    @classmethod
    def getProgram(cls):
        return 'xmipp_volume_correct_bfactor'

    def test_case1(self):
        self.runCase("-i input/1FFKfull_reconstructed.vol -o %o/corrected.vol --auto --sampling 1 --maxres 10",
                outputs=["corrected.vol"])




class VolumeFindSymmetry(XmippProgramTest):
    _owner = RM
    @classmethod
    def getProgram(cls):
        return 'xmipp_volume_find_symmetry'

    def test_case1(self):
        self.runCase("-i input/phantomBacteriorhodopsin.vol --sym rot 3 -o %o/symmetry.xmd",
                outputs=["symmetry.xmd"])


class VolumeFromPdb(XmippProgramTest):
    _owner = COSS
    @classmethod
    def getProgram(cls):
        return 'xmipp_volume_from_pdb'

    def test_case1(self):
        self.runCase("-i input/1o7d.pdb -o %o/1o7d --centerPDB",
                outputs=["1o7d.vol"])



class VolumeSegment(XmippProgramTest):
    _owner = RM
    @classmethod
    def getProgram(cls):
        return 'xmipp_volume_segment'

    def test_case1(self):
        self.runCase("-i input/phantomBacteriorhodopsin.vol -o %o/maskOtsu.vol --method otsu",
                outputs=["maskOtsu.vol"])
    def test_case2(self):
        self.runCase("-i input/phantomBacteriorhodopsin.vol -o %o/maskVoxelMass.vol --method voxel_mass 48000",
                outputs=["maskVoxelMass.vol"])
    def test_case3(self):
        self.runCase("-i input/phantomBacteriorhodopsin.vol -o %o/maskProb.vol --method prob 1",
                outputs=["maskProb.vol"])


class VolumeToPseudoatoms(XmippProgramTest):
    _owner = COSS
    @classmethod
    def getProgram(cls):
        return 'xmipp_volume_to_pseudoatoms'

    def test_case1(self):
        self.runCase("-i input/phantomBacteriorhodopsin.vol -o %o/pseudo --sigma 10",
                outputs=["pseudo.pdb"])


class VolumeToWeb(XmippProgramTest):
    _owner = COSS
    @classmethod
    def getProgram(cls):
        return 'xmipp_volume_to_web'

    def test_case1(self):
        self.runCase("-i input/phantomBacteriorhodopsin.vol --central_slices %o/imgOut.jpg",
                outputs=["imgOut.jpg"])


#xmipp no longer handle emx, import/export using scipion
#class ExportEmx(XmippProgramTest):
#class ImportEmx(XmippProgramTest):


class ReconstructWbp(XmippProgramTest):
    _owner = RM
    @classmethod
    def getProgram(cls):
        return 'xmipp_reconstruct_wbp'

    def test_case1(self):
        self.runCase("-i input/projectionsBacteriorhodopsin.xmd --use_each_image -o  %o/recon.vol ",
                outputs=["recon.vol"])


class ReconstructWbpMpi(ReconstructWbp):
    _owner = RM
    @classmethod
    def getProgram(cls):
        return 'xmipp_mpi_reconstruct_wbp'

    def test_case2(self):
        self.runCase("-i input/projectionsBacteriorhodopsin.xmd --use_each_image -o  %o/recon.vol ",
                outputs=["recon.vol"])


class MicrographAutomaticPicking(XmippProgramTest):
    _owner = VAHID
    @classmethod
    def getProgram(cls):
        return 'xmipp_micrograph_automatic_picking'

    def test_case1(self):
        self.runCase("input/ParticlePicking/BPV_1386.mrc --particleSize 100 "
                     "--model input/ParticlePicking/model "
                     "--outputRoot tmpLink/xmipp_micrograph_automatic_picking_01/automatically_selected "
                     "--mode autoselect --fast",
                outputs=["automatically_selected.pos"])


class ReconstructFourierMpi(ReconstructFourier):
    _owner = VAHID
    @classmethod
    def getProgram(cls):
        return 'xmipp_mpi_reconstruct_fourier'

    def test_case2(self):
        self.runCase("-i input/aFewProjections.sel -o  %o/recon.vol ",
                outputs=["recon.vol"])


class PhantomProject(XmippProgramTest):
    _owner = VAHID
    @classmethod
    def getProgram(cls):
        return 'xmipp_phantom_project'

    def test_case1(self):
        self.runCase("-i input/phantomBacteriorhodopsin.vol -o %o/image.xmp --angles 0 0 0",
                outputs=["image.xmp"])
    def test_case2(self):
        self.runCase("-i input/phantomBacteriorhodopsin.vol     -o %o/projections --params input/clusterProjection_xmd.param",
                outputs=["projections.stk","projections.xmd"])
    def test_case3(self):
        self.runCase("-i input/phantomBacteriorhodopsin.vol     -o %o/projections --params input/uniformProjection_xmd.param",
                outputs=["projections.stk","projections.xmd"])
    def test_case4(self):
        self.runCase("-i input/phantomBacteriorhodopsin.vol     -o %o/projections --params input/uniformProjection_xmd.param --method shears",
                outputs=["projections.stk","projections.xmd"])
    def test_case5(self):
        self.runCase("-i input/phantomBacteriorhodopsin.vol     -o %o/projections --params input/uniformProjection_xmd.param --method fourier 2 0.5 bspline",
                outputs=["projections.stk","projections.xmd"], errorthreshold=0.0021)


class MlAlign2d(XmippProgramTest):
    _owner = COSS
    _outputs = ["ml2d_extra/iter001/iter_classes.stk", "ml2d_extra/iter001/iter_images.xmd"]

    @classmethod
    def getProgram(cls):
        return 'xmipp_ml_align2d'

    # Redefine runCase to add default outputs
    def runCase(self, *args, **kwargs):
        kwargs['outputs'] = self._outputs
        kwargs['random'] = True

        XmippProgramTest.runCase(self, *args, **kwargs)

    def test_case1(self):
        self.runCase("-i input/images_some.stk --ref input/seeds2.stk --iter 2 --oroot %o/ml2d_ --fast --mirror --random_seed 100")

    def test_case2(self):
        self.runCase("-i input/mlData/phantom_images.xmd --ref input/mlData/refs.xmd --iter 2 --oroot %o/ml2d_ --random_seed 100")

    def test_case3(self):
        self.runCase("-i input/mlData/phantom_images.xmd --ref input/mlData/refs.xmd --iter 2 --oroot %o/ml2d_ --thr 2 --random_seed 100")


class MlAlign2dMpi(MlAlign2d):
   _owner = DISCONTINUED
   #_outputs = ["ml2d_images.xmd","ml2d_classes.stk", "ml2d_classes.xmd"]

   @classmethod
   def getProgram(cls):
       return 'xmipp_mpi_ml_align2d'

   def test_case4(self):
       self.runCase("-i input/images_some.stk --ref input/seeds2.stk --iter 2 --oroot %o/ml2d_ --fast --mirror")


class MlfAlign2d(XmippProgramTest):
    _owner = DISCONTINUED
    @classmethod
    def getProgram(cls):
        return 'xmipp_mlf_align2d'

    def test_case1(self):
        self.runCase("-i input/mlData/phantom_images.xmd --ref input/mlData/refs.xmd --iter 2 --oroot %o/mlf2d_",random=True,
                outputs=["mlf2d_images.xmd",
                         "mlf2d_classes.stk",
                         "mlf2d_classes.xmd"])


class MlfAlign2dMpi(MlfAlign2d):
    _owner = DISCONTINUED
    @classmethod
    def getProgram(cls):
        return 'xmipp_mpi_mlf_align2d'

    def test_case2(self):
        self.runCase("-i input/mlData/phantom_images.xmd --ref input/mlData/refs.xmd --iter 2 --oroot %o/mlf2d_",random=True,
                outputs=["mlf2d_images.xmd","mlf2d_classes.stk","mlf2d_classes.xmd"])


class VolSubtraction(XmippProgramTest):
    _owner = EFG
    @classmethod
    def getProgram(cls):
        return 'xmipp_volume_subtraction'

    def test_case1(self):
        """Test subtraction with radial average"""
        str = "--i1 input/phantomVolSubtraction/V1.vol " \
              "--i2 input/phantomVolSubtraction/V.vol " +\
              "-o %o/subtraction.mrc " \
              "--mask1 input/phantomVolSubtraction/V1_mask.mrc " +\
              "--mask2 input/phantomVolSubtraction/V_mask.mrc" \
              " --iter 5 --lambda 1.0 --sub --cutFreq 1.333333 --sigma 3 --computeEnergy"
        self.runCase(str, outputs=["subtraction.mrc"])

    def test_case2(self):
        """Test subtraction without radial average"""
        self.runCase("--i1 input/phantomVolSubtraction/V1.vol --i2 input/phantomVolSubtraction/V.vol "
                     "-o %o/subtraction_radAvg.mrc --mask1 input/phantomVolSubtraction/V1_mask.mrc "
                     "--mask2 input/phantomVolSubtraction/V_mask.mrc --iter 5 --radavg --lambda 1.0 --sub "
                     "--cutFreq 1.333333 --sigma 3 --computeEnergy",
                     outputs=["subtraction_radAvg.mrc"])

    def test_case3(self):
        """Test adjustment without radial average"""
        self.runCase("--i1 input/phantomVolSubtraction/V1.vol --i2 input/phantomVolSubtraction/V.vol  "
                     "-o %o/Vadjust.mrc --mask1 input/phantomVolSubtraction/V1_mask.mrc "
                     "--mask2 input/phantomVolSubtraction/V_mask.mrc --iter 5 --lambda 1.0 "
                     "--cutFreq 1.333333 --sigma 3 --computeEnergy",
                     outputs=["Vadjust.mrc"])

    def test_case4(self):
        """Test adjustment with radial average"""
        self.runCase("--i1 input/phantomVolSubtraction/V1.vol --i2 input/phantomVolSubtraction/V.vol "
                     "-o %o/Vadjust_radAvg.mrc --mask1 input/phantomVolSubtraction/V1_mask.mrc "
                     "--mask2 input/phantomVolSubtraction/V_mask.mrc --iter 5 --lambda 1.0 --radavg "
                     "--cutFreq 1.333333 --sigma 3 --computeEnergy",
                     outputs=["Vadjust_radAvg.mrc"])


class ProjSubtraction(XmippProgramTest):
    _owner = EFG
    @classmethod
    def getProgram(cls):
        return 'xmipp_subtract_projection'

    def test_case1(self):
        """Test projection subtraction"""
        self.runCase("-i input/projectionSubtraction/images.xmd  --ref input/projectionSubtraction/phantom.vol "
                     "-o %o/output_particles.xmd --sampling 1.0 --max_resolution 3.0 "
                     "--padding 2.0 --sigma 3 --limit_freq 0 --cirmaskrad -1 "
                     "--save input/projectionSubtraction --oroot %o/subtracted_part",
                     outputs=["output_particles.xmd"], errorthreshold=1)

class ContinuousCreateResiduals(XmippProgramTest):
    _owner = JMK
    @classmethod
    def getProgram(cls):
        return 'xmipp_continuous_create_residuals'

    def test_case1(self):
        "Test all 3 outputs"
        self.runCase("-i input/aFewProjections.sel -o %o/output.xmd --ref input/phantomBacteriorhodopsin.vol --oresiduals %o/residuals.stk --oprojections %o/projections.stk",
                preruns=["xmipp_angular_project_library -i input/phantomBacteriorhodopsin.vol -o %o/reference.stk --sampling_rate 10" ],
                outputs=["output.xmd", "residuals.stk", "projections.stk"])

    def test_case2(self):
        "Test output xmd and residuals only"
        self.runCase("-i input/aFewProjections.sel -o %o/output.xmd --ref input/phantomBacteriorhodopsin.vol --oresiduals %o/residuals.stk",
                preruns=["xmipp_angular_project_library -i input/phantomBacteriorhodopsin.vol -o %o/reference.stk --sampling_rate 10" ],
                outputs=["output.xmd", "residuals.stk"])

    def test_case3(self):
        "Test output xmd and projections only"
        self.runCase("-i input/aFewProjections.sel -o %o/output.xmd --ref input/phantomBacteriorhodopsin.vol --oprojections %o/projections.stk",
                preruns=["xmipp_angular_project_library -i input/phantomBacteriorhodopsin.vol -o %o/reference.stk --sampling_rate 10" ],
                outputs=["output.xmd", "projections.stk"])

    def test_case4(self):
        "Test output xmd only"
        self.runCase("-i input/aFewProjections.sel -o %o/output.xmd --ref input/phantomBacteriorhodopsin.vol",
                preruns=["xmipp_angular_project_library -i input/phantomBacteriorhodopsin.vol -o %o/reference.stk --sampling_rate 10" ],
                outputs=["output.xmd"])
