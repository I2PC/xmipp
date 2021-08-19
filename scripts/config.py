#!/usr/bin/env python3
# ***************************************************************************
# * Authors:     Carlos Oscar S. Sorzano (coss@cnb.csic.es)
# *              David Maluenda (dmaluenda@cnb.csic.es)
# *              David Strelak (dstrelak@cnb.csic.es)
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


import os
import sys
from .utils import *


class Config:
    FILE_NAME = 'xmipp.conf'
    KEY_BUILD_TESTS = 'BUILD_TESTS'

    def __init__(self):
        self._create_empty()

    def get(self):
        return self.configDict

    def is_true(self, key):
        return self.configDict and (key in self.configDict) and (self.configDict[key].lower() == 'true')

    def read(self, fnConfig=FILE_NAME):
        try:
            from ConfigParser import ConfigParser, ParsingError
        except ImportError:
            from configparser import ConfigParser, ParsingError  # Python 3
        cf = ConfigParser()
        cf.optionxform = str  # keep case (stackoverflow.com/questions/1611799)
        try:
            if os.path.isdir(fnConfig):
                if os.path.exists(os.path.join(fnConfig, Config.FILE_NAME)):
                    fnConfig = os.path.join(fnConfig, Config.FILE_NAME)
                else:
                    fnConfig = os.path.join(fnConfig, "xmipp.template")
            if os.path.exists(fnConfig):
                cf.read(fnConfig)
                if not 'BUILD' in cf.sections():
                    print(red("Cannot find section BUILD in %s" % fnConfig))
                    self._create_empty()
                self.configDict = dict(cf.items('BUILD'))
        except:
            sys.exit("%s\nPlease fix the configuration file %s." %
                     (sys.exc_info()[1], fnConfig))

    def _create_empty(self):
        labels = [Config.KEY_BUILD_TESTS, 'CC', 'CXX', 'LINKERFORPROGRAMS', 'INCDIRFLAGS', 'LIBDIRFLAGS', 'CCFLAGS', 'CXXFLAGS',
                  'LINKFLAGS', 'PYTHONINCFLAGS', 'MPI_CC', 'MPI_CXX', 'MPI_RUN', 'MPI_LINKERFORPROGRAMS', 'MPI_CXXFLAGS',
                  'MPI_LINKFLAGS', 'NVCC', 'CXX_CUDA', 'NVCC_CXXFLAGS', 'NVCC_LINKFLAGS',
                  'MATLAB_DIR', 'CUDA', 'DEBUG', 'MATLAB', 'OPENCV', 'OPENCVSUPPORTSCUDA', 'OPENCV3',
                  'JAVA_HOME', 'JAVA_BINDIR', 'JAVAC', 'JAR', 'JNI_CPPPATH',
                  'STARPU', 'STARPU_HOME', 'STARPU_INCLUDE', 'STARPU_LIB', 'STARPU_LIBRARY',
                  'USE_DL', 'VERIFIED', 'CONFIG_VERSION', 'PYTHON_LIB']
        self.configDict = {}
        for label in labels:
            # We let to set up the xmipp configuration via environ.
            self.configDict[label] = os.environ.get(label, "")

    def _config_OpenCV(self):
        cppProg = "#include <opencv2/core/core.hpp>\n"
        cppProg += "int main(){}\n"
        with open("xmipp_test_opencv.cpp", "w") as cppFile:
            cppFile.write(cppProg)

        if not runJob("%s -c -w %s xmipp_test_opencv.cpp -o xmipp_test_opencv.o %s"
                      % (self.configDict["CXX"], self.configDict["CXXFLAGS"],
                         self.configDict["INCDIRFLAGS"]), show_output=False):
            print(yellow("OpenCV not found"))
            self.configDict["OPENCV"] = False
            self.configDict["OPENCVSUPPORTSCUDA"] = False
            self.configDict["OPENCV3"] = False
        else:
            self.configDict["OPENCV"] = True

            # Check version
            with open("xmipp_test_opencv.cpp", "w") as cppFile:
                cppFile.write('#include <opencv2/core/version.hpp>\n')
                cppFile.write('#include <fstream>\n')
                cppFile.write('int main()'
                              '{std::ofstream fh;'
                              ' fh.open("xmipp_test_opencv.txt");'
                              ' fh << CV_MAJOR_VERSION << std::endl;'
                              ' fh.close();'
                              '}\n')
            if not runJob("%s -w %s xmipp_test_opencv.cpp -o xmipp_test_opencv %s "
                          % (self.configDict["CXX"], self.configDict["CXXFLAGS"],
                             self.configDict["INCDIRFLAGS"]), show_output=False):
                self.configDict["OPENCV3"] = False
                version = 2  # Just in case
            else:
                runJob("./xmipp_test_opencv")
                f = open("xmipp_test_opencv.txt")
                versionStr = f.readline()
                f.close()
                version = int(versionStr.split('.', 1)[0])
                self.configDict["OPENCV3"] = version >= 3

            # Check CUDA Support
            cppProg = "#include <opencv2/core/version.hpp>\n"
            cppProg += "#include <opencv2/cudaoptflow.hpp>\n" if self.configDict[
                "OPENCV3"] else "#include <opencv2/core/cuda.hpp>\n"
            cppProg += "int main(){}\n"
            with open("xmipp_test_opencv.cpp", "w") as cppFile:
                cppFile.write(cppProg)
            self.configDict["OPENCVSUPPORTSCUDA"] = runJob("%s -c -w %s xmipp_test_opencv.cpp -o xmipp_test_opencv.o %s" %
                                                           (self.configDict["CXX"], self.configDict["CXXFLAGS"], self.configDict["INCDIRFLAGS"]), show_output=False)

            print(green("OPENCV-%s detected %s CUDA support"
                        % (version, 'with' if self.configDict["OPENCVSUPPORTSCUDA"] else 'without')))
        runJob("rm -v xmipp_test_opencv*", show_output=False)
