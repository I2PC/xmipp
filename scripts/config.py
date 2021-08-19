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
from.environment import Environment


class Config:
    FILE_NAME = 'xmipp.conf'
    KEY_BUILD_TESTS = 'BUILD_TESTS'

    def __init__(self, askUser=False):
        self.ask = askUser
        self._create_empty()

    def create(self):
        print("Configuring -----------------------------------------")
        self._create_empty()

        if self.configDict['VERIFIED'] == '':
            self.configDict['VERIFIED'] = 'False'

        self._config_compiler()

    def get(self):
        return self.configDict

    def writeEnviron(self): # FIXME remove
        self.environment.write()

    def updateXmippEnv(self, pos='begin', realPath=True, **kwargs): # FIXME remove
        self.environment.update(pos, realPath, **kwargs)

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
        self.environment = Environment()
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

    def _config_compiler(self):
        if self.configDict["DEBUG"] == "":
            self.configDict["DEBUG"] = "False"

        if self.configDict["CC"] == "" and checkProgram("gcc"):
            self.configDict["CC"] = "gcc"
            print(green('gcc detected'))
        if self.configDict["CXX"] == "":
            if isCIBuild():
                # we can use cache to speed up the build
                self.configDict["CXX"] = "ccache g++" if checkProgram(
                    "g++") else ""
            else:
                self.configDict["CXX"] = "g++" if checkProgram("g++") else ""
        if self.configDict["LINKERFORPROGRAMS"] == "":
            if isCIBuild():
                # we can use cache to speed up the build
                self.configDict["LINKERFORPROGRAMS"] = "ccache g++" if checkProgram(
                    "g++") else ""
            else:
                self.configDict["LINKERFORPROGRAMS"] = "g++" if checkProgram(
                    "g++") else ""

        if self.configDict["CC"] == "gcc":
            if not "-std=c99" in self.configDict["CCFLAGS"]:
                self.configDict["CCFLAGS"] += " -std=c99"
        if 'g++' in self.configDict["CXX"]:
            # optimize for current machine
            self.configDict["CXXFLAGS"] += " -mtune=native -march=native"
            if "-std=c99" not in self.configDict["CXXFLAGS"]:
                self.configDict["CXXFLAGS"] += " -std=c++11"
            if isCIBuild():
                # don't tolerate any warnings on build machine
                self.configDict["CXXFLAGS"] += " -Werror"
                # don't optimize, as it slows down the build
                self.configDict["CXXFLAGS"] += " -O0"
            else:
                self.configDict["CXXFLAGS"] += " -O3"
            if self.is_true("DEBUG"):
                self.configDict["CXXFLAGS"] += " -g"
        # Nothing special to add to LINKFLAGS
        from sysconfig import get_paths
        info = get_paths()

        if self.configDict["LIBDIRFLAGS"] == "":
            # /usr/local/lib or /path/to/virtEnv/lib
            localLib = "%s/lib" % info['data']
            self.configDict["LIBDIRFLAGS"] = "-L%s" % localLib
            self.environment.update(LD_LIBRARY_PATH=localLib)

            # extra libs
            hdf5InLocalLib = findFileInDirList("libhdf5*", localLib)
            isHdf5CppLinking = checkLib(self.configDict['CXX'], '-lhdf5_cpp')
            isHdf5Linking = checkLib(self.configDict['CXX'], '-lhdf5')
            if not (hdf5InLocalLib or (isHdf5CppLinking and isHdf5Linking)):
                print(yellow("\n'libhdf5' not found at '%s'." % localLib))
                hdf5Lib = findFileInDirList("libhdf5*", ["/usr/lib",
                                                         "/usr/lib/x86_64-linux-gnu"])
                hdf5Lib = askPath(hdf5Lib, self.ask)
                if hdf5Lib:
                    self.configDict["LIBDIRFLAGS"] += " -L%s" % hdf5Lib
                    self.environment.update(LD_LIBRARY_PATH=hdf5Lib)
                else:
                    installDepConda('hdf5', self.ask)

        if not checkLib(self.configDict['CXX'], '-lfftw3'):
            print(red("'libfftw3' not found in the system"))
            installDepConda('fftw', self.ask)
        if not checkLib(self.configDict['CXX'], '-ltiff'):
            print(red("'libtiff' not found in the system"))
            installDepConda('libtiff', self.ask)

        if self.configDict["INCDIRFLAGS"] == "":
            # /usr/local/include or /path/to/virtEnv/include
            localInc = "%s/include" % info['data']
            self.configDict["INCDIRFLAGS"] += ' '.join(
                map(lambda x: '-I' + str(x), getDependenciesInclude()))
            self.configDict["INCDIRFLAGS"] += " -I%s" % localInc

            # extra includes
            if not findFileInDirList("hdf5.h", [localInc, "/usr/include"]):
                print(yellow("\nHeaders for 'libhdf5' not found at '%s'." % localInc))
                # Add more candidates if needed
                hdf5Inc = findFileInDirList(
                    "hdf5.h", "/usr/include/hdf5/serial")
                hdf5Inc = askPath(hdf5Inc, self.ask)
                if hdf5Inc:
                    self.configDict["INCDIRFLAGS"] += " -I%s" % hdf5Inc

        if self.configDict["PYTHON_LIB"] == "":
            # malloc flavour is not needed from 3.8
            malloc = "m" if sys.version_info.minor < 8 else ""
            self.configDict["PYTHON_LIB"] = "python%s.%s%s" % (sys.version_info.major,
                                                               sys.version_info.minor,
                                                               malloc)

        if self.configDict["PYTHONINCFLAGS"] == "":
            import numpy
            incDirs = [info['include'], numpy.get_include()]

            self.configDict["PYTHONINCFLAGS"] = ' '.join(
                ["-I%s" % iDir for iDir in incDirs])

        self.configDict["OPENCV"] = os.environ.get("OPENCV", "")
        if self.configDict["OPENCV"] == "" or self.configDict["OPENCVSUPPORTSCUDA"] or self.configDict["OPENCV3"]:
            self._config_OpenCV()
