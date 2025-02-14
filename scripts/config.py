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
    KEY_USE_DL = 'USE_DL'
    KEY_VERSION = 'CONFIG_VERSION'
    KEY_LINKERFORPROGRAMS = 'LINKERFORPROGRAMS'
    KEY_CXX = 'CXX'
    KEY_LINKFLAGS = 'LINKFLAGS'
    OPT_CUDA = 'CUDA'
    OPT_CXX_CUDA = 'CXX_CUDA'
    OPT_NVCC = 'NVCC'
    OPT_NVCC_LINKFLAGS = 'NVCC_LINKFLAGS'
    OPT_NVCC_CXXFLAGS = 'NVCC_CXXFLAGS'
    MINIMUM_GCC_VERSION = '8.4.0'
    MINIMUM_CUDA_VERSION = 10.1
    vGCC = ['12.3', '12.2', '12.1',
            '11.3', '11.2', '11.1', '11',
            '10.4', '10.3', '10.2', '10.1', '10',
            '9.4', '9.3', '9.2', '9.1', '9',
            '8.5', '8.4', '8.3', '8.2', '8.1', '8']
    CUDA_GCC_COMPATIBILITY = {
        '10.1-10.2': vGCC[vGCC.index('8.5'):],
        '11.0-11.0': vGCC[vGCC.index('9.4'):],
        '11.1-11.4': vGCC[vGCC.index('10.4'):],
        '11.5-11.8': vGCC[vGCC.index('11.2'):],
        '12.0-12.3': vGCC[vGCC.index('12.3'):],
    }
    CMAKE_VERSION_REQUIRED = '3.16'


    def __init__(self, askUser=False):
        self.ask = askUser
        self._create_empty()

    def create(self):
        print("Configuring -----------------------------------------")
        self._create_empty()

        if self.configDict['VERIFIED'] == '':
            self.configDict['VERIFIED'] = 'False'

        self._config_compiler()
        self._config_cmake()
        self._set_CUDA()
        self._config_MPI()
        self._config_Java()

        self._config_Matlab()
        self._config_StarPU()
        self._config_DL()

        self.configDict[Config.KEY_VERSION] = self._get_version()

        self.write()
        self.environment.write()
        print(green("\nConfiguration completed and written on xmipp.conf\n"))


    def check(self):
        print("\nChecking configuration -----------------------------------")
        if self.configDict['VERIFIED'] != 'True':
            status = self._check_compiler()
            if status[0] == False:
                return status #status = [False, index, error msg, suport Msg]
            status = self._check_hdf5()
            if not status:
                return status
            status = self._check_MPI()
            if not status[0]:
                runJob("rm xmipp_mpi_test_main*", show_command=False)
                return status
            status = self._check_cmake()
            if status[0] == False:
                return status
            status = self._check_Java()
            if not status[0]:
                runJob("rm Xmipp.java Xmipp.class xmipp_jni_test*", show_command=False)
                return status
            if not self._check_CUDA():
                print(red("Cannot compile with NVCC, continuing without CUDA"))
                # if fails, the test files remains
                runJob("rm xmipp_cuda_test*", show_command=False)
                self.configDict["CUDA"] = "False"
            if not self._check_Matlab():
                print(red("Cannot compile with Matlab, continuing without Matlab"))
                self.configDict["MATLAB"] = "False"
                runJob("rm xmipp_mex*", show_command=False)
            if not self._check_StarPU():
                print(red("Cannot compile with StarPU, continuing without StarPU"))
                self.configDict["STARPU"] = "False"
            self.configDict['VERIFIED'] = "True"
            self.write()  # store result
        else:
            print(green("'%s' is already checked. Set VERIFIED=False to have it re-checked"
                       % Config.FILE_NAME))
        return True, 0

    def get(self, option=None):
        if option:
            return self.configDict[option]
        return self.configDict

    def _set(self, option, value):
        self.get()[option] = str(value)

    def is_true(self, key):
        return self.configDict and (key in self.configDict) and (self.configDict[key].lower() == 'true')

    def is_empty(self, option):
        return self.has(option) and self.get(option) == ''

    def has(self, option, value=None):
        if value is None:
            return option in self.get()
        return option in self.get() and value in self.get()[option]

    def _set_if_empty(self, option, value):
        if self.is_empty(option):
            self._set(option, value)

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
            sys.exit("%s\nPlease fix the configuration file %s.\n"
                     "Visit https://github.com/I2PC/xmipp/wiki/Xmipp-configuration" %
                     (sys.exc_info()[1], fnConfig))

    def write(self):
        with open(Config.FILE_NAME, "w") as configFile:
            configFile.write("[BUILD]\n")
            for label in sorted(self.configDict.keys()):
                configFile.write("%s=%s\n" % (label, self.configDict[label]))

    def _create_empty(self):
        labels = [ 'CC', 'CXX', 'LINKERFORPROGRAMS', 'INCDIRFLAGS', 'LIBDIRFLAGS', 'CCFLAGS', 'CXXFLAGS',
                  'LINKFLAGS', 'PYTHONINCFLAGS', 'MPI_CC', 'MPI_CXX', 'MPI_RUN', 'MPI_LINKERFORPROGRAMS', 'MPI_CXXFLAGS',
                  'MPI_LINKFLAGS', 'NVCC', 'CXX_CUDA', 'NVCC_CXXFLAGS', 'NVCC_LINKFLAGS',
                  'MATLAB_DIR', 'CUDA', 'DEBUG', 'MATLAB', 'OPENCV', 'OPENCVSUPPORTSCUDA', 'OPENCV_VERSION',
                  'JAVA_HOME', 'JAVA_BINDIR', 'JAVAC', 'JAR', 'JNI_CPPPATH',
                  'STARPU', 'STARPU_HOME', 'STARPU_INCLUDE', 'STARPU_LIB', 'STARPU_LIBRARY',
                  'USE_DL', 'VERIFIED', 'CONFIG_VERSION', 'PYTHON_LIB']
        self.configDict = {}
        self.environment = Environment()
        for label in labels:
            # We let to set up the xmipp configuration via environ.
            value = os.environ.get(label, "")

            if value !="":
                print("%s variable found in the environment with this value: %s." % (label, value))
            self.configDict[label] = os.environ.get(label, "")

    def _config_OpenCV(self):
        cppProg = "#include <opencv2/core/core.hpp>\n"
        cppProg += "int main(){}\n"
        with open("xmipp_test_opencv.cpp", "w") as cppFile:
            cppFile.write(cppProg)

        if not runJob("%s -c -w %s xmipp_test_opencv.cpp -o xmipp_test_opencv.o %s"
                      % (self.get(Config.KEY_CXX), self.configDict["CXXFLAGS"],
                         self.configDict["INCDIRFLAGS"]), show_command=False,
                      show_output=False, showWithReturn=True):
            print(yellow("OpenCV not found"))
            self.configDict["OPENCV"] = False
            self.configDict["OPENCVSUPPORTSCUDA"] = False
            self.configDict["OPENCV_VERSION"] = ''
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
                          % (self.get(Config.KEY_CXX), self.configDict["CXXFLAGS"],
                             self.configDict["INCDIRFLAGS"]),
                          show_command=False, show_output=False):
                self.configDict["OPENCV_VERSION"] = 2
            else:
                runJob("./xmipp_test_opencv", show_output=False, show_command=False)
                f = open("xmipp_test_opencv.txt")
                versionStr = f.readline()
                f.close()
                version = int(versionStr.split('.', 1)[0])
                self.configDict["OPENCV_VERSION"] = version


            # Check CUDA Support
            cppProg = "#include <opencv2/core/version.hpp>\n"
            if self.configDict["OPENCV_VERSION"] < 3:
                cppProg += "#include <opencv2/core/cuda.hpp>\n"
            else:
                cppProg += "#include <opencv2/cudaoptflow.hpp>\n"
            cppProg += "int main(){}\n"
            with open("xmipp_test_opencv.cpp", "w") as cppFile:
                cppFile.write(cppProg)
            if runJob("%s -c -w %s xmipp_test_opencv.cpp -o xmipp_test_opencv.o %s" %
                 (self.get(Config.KEY_CXX), self.configDict["CXXFLAGS"],
                  self.configDict["INCDIRFLAGS"]),
                      show_output=False, log=[], show_command=False):
                self.configDict["OPENCVSUPPORTSCUDA"] = True
            else:
                self.configDict["OPENCVSUPPORTSCUDA"] = False
            print(green("OPENCV-%s detected %s CUDA support"
                        % (version, 'with' if self.configDict["OPENCVSUPPORTSCUDA"] else 'without')))
        runJob("rm -v xmipp_test_opencv*", show_output=False, show_command=False)

    def _get_help_msg(self):
        msg_missing = 'The config file %s is missing.\n' % Config.FILE_NAME
        msg_obsolete = ('The config file %s is obsolete and has to be regenerated.\n'
                        'We recommend you do create a manual backup of it first\n' % Config.FILE_NAME)
        msg_common = 'Please, run ./xmipp --help or check the online documention for further details'
        if not self.config_file_exists():
            return msg_missing + msg_common
        if not self._is_up_to_date():
            return msg_obsolete + msg_common
        return msg_common

    def config_file_exists(self):
        return os.path.isfile(Config.FILE_NAME)

    def _is_up_to_date(self):
        return self.get(Config.KEY_VERSION) == self._get_version()

    def get_supported_GCC(self):
        # we need GCC with C++17 support
        # https://gcc.gnu.org/projects/cxx-status.html
        return ['',
                11.4, 11.3, 11.2, 11.1, 11,
                10.4, 10.3, 10.2, 10.1, 10,
                9.5, 9.4, 9.3, 9.2, 9.1, 9,
                8.5, 8.4, 8.3, 8.2, 8.1, 8]

    def _set_compiler_linker_helper(self, opt, prg, versions, show=False):
        if not self.is_empty(opt):
            return
        prg = find_newest(prg, versions, show=show)
        if isCIBuild() and prg:
            prg = 'ccache ' + prg
        self._set(opt, prg)

    def _set_cxx(self):
        self._set_compiler_linker_helper(
            Config.KEY_CXX, 'g++', self.get_supported_GCC(), show=True)

    def _set_linker(self):
        self._set_compiler_linker_helper(
            Config.KEY_LINKERFORPROGRAMS, 'g++', self.get_supported_GCC())

    def _config_compiler(self):
        print('Configuring compiler')
        if self.configDict["DEBUG"] == "":
            self.configDict["DEBUG"] = "False"
        if self.configDict["CC"] == "" and checkProgram("gcc")[0]:
            self.configDict["CC"] = "gcc"
            if versionToNumber(get_GCC_version("gcc")[1]) < versionToNumber(Config.MINIMUM_GCC_VERSION):
                print(red("gcc version required >= {}, detected gcc {}".format(
                    Config.MINIMUM_GCC_VERSION, get_GCC_version("gcc")[0])))
            else:
                print(green('gcc {} detected'.format(get_GCC_version("gcc")[0])))
        self._set_cxx()
        self._set_linker()
        if self.configDict["CC"] == "gcc":
            if not "-std=c99" in self.configDict["CCFLAGS"]:
                self.configDict["CCFLAGS"] += " -std=c99"
        if 'g++' in self.get(Config.KEY_CXX):
            # optimize for current machine
            self.configDict["CXXFLAGS"] += " -mtune=native -march=native -flto"
            if "-std=c99" not in self.configDict["CXXFLAGS"]:
                self.configDict["CXXFLAGS"] += " -std=c++17"
            if isCIBuild():
                # don't tolerate any warnings on build machine
                self.configDict["CXXFLAGS"] += " -Werror"
                # don't optimize, as it slows down the build
                self.configDict["CXXFLAGS"] += " -O0"
            else:
                self.configDict["CXXFLAGS"] += " -O3"
            if self.is_true("DEBUG"):
                self.configDict["CXXFLAGS"] += " -g"
        
        if self.is_empty(Config.KEY_LINKFLAGS):
            self._set(Config.KEY_LINKFLAGS, '-flto')
        
        from sysconfig import get_paths
        info = get_paths()

        if self.configDict["LIBDIRFLAGS"] == "":
            print('Configuring HDF5')
            hdf5Found = 'hdf5 library found at: '
            # /usr/local/lib or /path/to/virtEnv/lib
            localLib = "%s/lib" % info['data']
            self.configDict["LIBDIRFLAGS"] = "-L%s" % localLib
            self.environment.update(LD_LIBRARY_PATH=localLib)
            # extra libs
            path2FindHDF5 = ["/usr/lib",
                             "/usr/lib/x86_64-linux-gnu/hdf5/serial",
                             "/usr/lib/x86_64-linux-gnu"]
            hdf5InLocalLib = findFileInDirList("libhdf5*", localLib)
            isHdf5CppLinking = checkLib(self.get(Config.KEY_CXX), '-lhdf5_cpp')
            isHdf5Linking = checkLib(self.get(Config.KEY_CXX), '-lhdf5')
            if not (hdf5InLocalLib or (isHdf5CppLinking and isHdf5Linking)):
                hdf5Lib = findFileInDirList("libhdf5*", path2FindHDF5)
                if hdf5Lib == '':
                    print(yellow('HDF5 not found at {}'.format(path2FindHDF5)))
                    hdf5Lib = askPath('', self.ask)
                    if hdf5Lib == '':
                        installDepConda('hdf5', self.ask)
                else:
                    self.configDict["LIBDIRFLAGS"] += " -L%s" % hdf5Lib
                    self.environment.update(LD_LIBRARY_PATH=hdf5Lib)

            if hdf5InLocalLib != '':
                print(green(str(hdf5Found + hdf5InLocalLib)))
            elif isHdf5CppLinking != False:
                print(green('{} found on the system'.format(
                    self.get(Config.KEY_CXX), '-lhdf5_cpp')))
            elif isHdf5Linking !=False:
                print(green('{} found on the system'.format(
                    self.get(Config.KEY_CXX), '-lhdf5')))
            elif hdf5Lib != '':
                print(green(str(hdf5Found + hdf5Lib)))

        if not checkLib(self.get(Config.KEY_CXX), '-lfftw3'):
            print(red("'libfftw3' not found in the system"))
            installDepConda('fftw', self.ask)
        if not checkLib(self.get(Config.KEY_CXX), '-ltiff'):
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
                # Add more candidates if needed
                hdf5Inc = findFileInDirList(
                    "hdf5.h", "/usr/include/hdf5/serial")
                if hdf5Inc == '':
                    print(yellow(
                        "Headers for 'libhdf5' not found at '%s'." % "/usr/include/hdf5/serial"))
                    hdf5Inc = askPath('', self.ask)
                    if hdf5Inc == '':
                        print(red("Headers for 'libhdf5' not found"))
                if hdf5Inc:
                    print(green(
                        "Headers for 'libhdf5' found at '%s'." % hdf5Inc))
                    self.configDict["INCDIRFLAGS"] += " -I%s" % hdf5Inc

            if findFileInDirList("opencv4/opencv2/core/core.hpp", ["/usr/include"]):
                self.configDict["INCDIRFLAGS"] += " -I%s" % "/usr/include/opencv4"


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

        print('Configuring OpenCV')
        self.configDict["OPENCV"] = os.environ.get("OPENCV", "")
        if self.configDict["OPENCV"] == "" or self.configDict["OPENCVSUPPORTSCUDA"]:
            self._config_OpenCV()

    def _config_cmake(self):
        error = checkCMakeVersion(Config.CMAKE_VERSION_REQUIRED)
        if error[0] == False:
            print(red(error[1][2]))
            print(red(error[1][3]))



    def _ensure_GCC_GPP_version(self, compiler):
        status = checkProgram(compiler)
        if status[0] == False:
            return status #status = [False, index, error msg, suport Msg]
        gccVersion, fullVersion = get_GCC_version(compiler)
        if gccVersion == '':
            return False, 3
        print(green('Detected ' + compiler + " in version " + fullVersion + '.'))
        if versionToNumber(fullVersion) < versionToNumber(Config.MINIMUM_GCC_VERSION):
            return False, 4
        return True, 0

    def _ensure_compiler_version(self, compiler):
        if 'g++' in compiler or 'gcc' in compiler\
		        or 'cc' in compiler or 'c++' in compiler:
            status = self._ensure_GCC_GPP_version(compiler)
            if status[0] == False:
                return status #status = [False, index, error msg, suport Msg]
        else:
            return False, 2
        return True, 0


    def _get_Hdf5_name(self, libdirflags):
        libdirs = libdirflags.split("-L")
        for dir in libdirs:
            if os.path.exists(os.path.join(dir.strip(), "libhdf5.so")):
                return "hdf5"
            elif os.path.exists(os.path.join(dir.strip(), "libhdf5_serial.so")):
                return "hdf5_serial"
        return "hdf5"

    def _check_compiler(self):
        print("Checking compiler configuration")
        # in case user specified some wrapper of the compiler
        # get rid of it: 'ccache g++' -> 'g++'
        currentCxx = self.get(Config.KEY_CXX).split()[-1]
        status = self._ensure_compiler_version(currentCxx)
        if status[0] == False:
            return status


        cppProg = """
    #include <fftw3.h>
    #include <hdf5.h>
    #include <tiffio.h>
    #include <jpeglib.h>
    #include <sqlite3.h>
    #include <pthread.h>
    #include <Python.h>
    #include <numpy/ndarraytypes.h>
        """
        if self.configDict["OPENCV"] == "True":
            cppProg += "#include <opencv2/core/core.hpp>\n"
            if self.configDict["OPENCVSUPPORTSCUDA"] == "True":
                if self.configDict["OPENCV_VERSION"] == 3:
                    cppProg += "#include <opencv2/cudaoptflow.hpp>\n"
                else:
                    cppProg += "#include <opencv2/core/cuda.hpp>\n"
        cppProg += "\n int main(){}\n"
        with open("xmipp_test_main.cpp", "w") as cppFile:
            cppFile.write(cppProg)

        if not runJob("%s -c -w %s xmipp_test_main.cpp -o xmipp_test_main.o %s %s" %
                      (self.get(Config.KEY_CXX), self.configDict["CXXFLAGS"],
                       self.configDict["INCDIRFLAGS"], self.configDict["PYTHONINCFLAGS"]),
                      show_command=False, show_output=False):
            return False, 5

        return True, 0

    def _check_hdf5(self):
        print("Checking hdf5 configuration")
        libhdf5 = self._get_Hdf5_name(self.configDict["LIBDIRFLAGS"])
        if not runJob("%s %s %s xmipp_test_main.o -o xmipp_test_main -lfftw3 -lfftw3_threads -l%s  -lhdf5_cpp -ltiff -ljpeg -lsqlite3 -lpthread" %
                      (self.get(Config.KEY_LINKERFORPROGRAMS), self.configDict["LINKFLAGS"],
                       self.configDict["LIBDIRFLAGS"], libhdf5),
                      show_command=False, show_output=False):
            return False, 6
        runJob("rm xmipp_test_main*", show_command=False, show_output=False)
        print(green('Done ' + (' ' * 70)))
        return True, 0

    def _get_CUDA_version(self, nvcc):
        log = []
        runJob(nvcc + " --version", show_output=False,
               show_command=False, log=log)
        # find 'Cuda compilation tools' line (last for older versions, one before last otherwise)
        # expected format: 'Cuda compilation tools, release 8.0, V8.0.61'
        full_version_line = next(l for l in log if 'compilation tools' in l)
        full_version = full_version_line.strip().split(', ')[-1].lstrip('V')
        tokens = full_version.split('.')
        if len(tokens) < 2:
            tokens.append('0')  # just in case when only one digit is returned
        nvccVersion = float(str(tokens[0] + '.' + tokens[1]))
        return nvccVersion, full_version

    def _get_compatible_GCC(self, nvcc_version):
        # https://gist.github.com/ax3l/9489132
        for key, value in Config.CUDA_GCC_COMPATIBILITY.items():
            list = key.split('-')
            if float(nvcc_version) >= float(list[0]) and\
                    float(nvcc_version) <= float(list[1]):
                return value, True
        return Config.vGCC, False

    def _join_with_prefix(self, collection, prefix):
        return ' '.join([prefix + i for i in collection if i])

    def _set_nvcc(self):
        if not self.is_empty(Config.OPT_NVCC):
            return True

        nvcc_loc_candidates = {os.environ.get('XMIPP_CUDA_BIN', ''),
                               os.environ.get('CUDA_BIN', ''),
                               '/usr/local/cuda/bin',
                               '/usr/local/cuda*/bin'} #if there is no -ln on /usr/local/cuda ?
        nvcc_loc = find('nvcc', nvcc_loc_candidates)
        if not nvcc_loc:
            print(yellow('nvcc not found (searched in %s). If you want to '
                         'enable CUDA, add \'nvcc\' to PATH. %s' % (
                nvcc_loc_candidates, self._get_help_msg())))
            return False
        print(green('nvcc found in ' + nvcc_loc))
        self._set(Config.OPT_NVCC, nvcc_loc)
        return True

    def _set_nvcc_cxx(self, nvcc_version):
        if not self.is_empty(Config.OPT_CXX_CUDA):
            return
        candidates, resultBool = self._get_compatible_GCC(nvcc_version)
        if not resultBool:
            print(yellow('CUDA version {} not compatible with Xmipp. Please '
                         'install CUDA>={} and not higher than the maximum version required by the compiler'.format(nvcc_version,
                                                   Config.MINIMUM_CUDA_VERSION)), *candidates, sep=", ")
            print('gcc candidates based on nvcc version:', *candidates, sep=", ")
            return

        prg = find_GCC(candidates, minimumGCC=Config.MINIMUM_GCC_VERSION, show=True)
        if not prg:# searching a g++ for devToolSet on CentOS
            print('gcc candidates based on nvcc version:', *candidates,
                  sep=", ")
            print(yellow('No valid compiler found for CUDA host code. ' +
                'nvcc_version : ' + str(nvcc_version) +' ' + self._get_help_msg()))
            return

        self._set(Config.OPT_CXX_CUDA, prg)

    def _set_nvcc_lib_dir(self):
        opt = Config.OPT_NVCC_LINKFLAGS
        if not self.is_empty(opt):
            return True

        dirs = ['lib', 'lib64', 'targets/x86_64-linux/lib',
                'lib/x86_64-linux-gnu']
        # assume cuda_dir/bin/nvcc
        locs = [os.path.dirname(os.path.dirname(self.get(Config.OPT_NVCC))),
                os.environ.get('XMIPP_CUDA_LIB', ''),
                os.environ.get('CUDA_LIB', ''),
                os.environ.get('LD_LIBRARY_PATH', ''),
                '/usr'
                ]

        def search():
            for l in locs:
                for d in dirs:
                    tmp = os.path.join(l, d, 'libcudart.so')
                    if os.path.isfile(tmp):
                        return os.path.dirname(tmp)
            return None

        path = search()
        if path is None:
            print(yellow(
                'WARNING: CUDA libraries (libcudart.so) not found. ' + self._get_help_msg()))
            return False
        # nvidia-ml is in stubs folder
        self._set(opt, self._join_with_prefix(
            [path, os.path.join(path, 'stubs')], '-L'))
        return True

    def _set_nvcc_flags(self, nvcc_version):
        flags = ("--x cu -D_FORCE_INLINES -Xcompiler -fPIC "
                 "-ccbin %(CXX_CUDA)s -std=c++11 --expt-extended-lambda "
                 # generate PTX only, and SASS at the runtime (by setting code=virtual_arch)
                 "-gencode=arch=compute_35,code=compute_35 "
                 "-gencode=arch=compute_50,code=compute_50 "
                 "-gencode=arch=compute_60,code=compute_60 "
                 "-gencode=arch=compute_61,code=compute_61")
        if nvcc_version >= 11:
            flags = ("--x cu -D_FORCE_INLINES -Xcompiler -fPIC "
                     "-ccbin %(CXX_CUDA)s -std=c++14 --expt-extended-lambda "
                     # generate PTX only, and SASS at the runtime (by setting code=virtual_arch)
                     "-gencode=arch=compute_60,code=compute_60 "
                     "-gencode=arch=compute_61,code=compute_61 "
                     "-gencode=arch=compute_75,code=compute_75 "
                     "-gencode=arch=compute_86,code=compute_86")
        self._set_if_empty(Config.OPT_NVCC_CXXFLAGS, flags)

    def _set_CUDA(self):
        print('Configuring CUDA')
        def no_CUDA():
            print(red("No valid compiler found. "
                  "Skipping CUDA compilation.\n"))
            self._set(Config.OPT_CUDA, False)
            self.environment.update(CUDA=False)

        if not self._set_nvcc():
            no_CUDA()
            return
        nvcc_version, nvcc_full_version = self._get_CUDA_version(
            self.get(Config.OPT_NVCC))
        print(green('CUDA-' + nvcc_full_version + ' found.'))
        self._set_nvcc_cxx(nvcc_version)
        if not self._set_nvcc_lib_dir():
            no_CUDA()
            return

        self._set_nvcc_flags(nvcc_version)

        # update config and environment
        self._set(Config.OPT_CUDA, True)
        self.environment.update(CUDA=True)
        LD = ':'.join(self.get(Config.OPT_NVCC_LINKFLAGS).split('-L'))
        self.environment.update(LD_LIBRARY_PATH=LD)

    def _check_CUDA(self):
        if self.configDict["CUDA"] == "True":
            print("Checking CUDA configuration")
            print(yellow('Working...'), end='\r')
            if not checkProgram(self.configDict["NVCC"][0]):
                return False
            cppProg = """
        #include <cuda_runtime.h>
        #include <cufft.h>
        int main(){}
        """
            with open("xmipp_cuda_test.cpp", "w") as cppFile:
                cppFile.write(cppProg)

            if not runJob("%s -c -w %s %s xmipp_cuda_test.cpp -o xmipp_cuda_test.o" %
                          (self.configDict["NVCC"], self.configDict["NVCC_CXXFLAGS"],
                           self.configDict["INCDIRFLAGS"]), show_command=False,show_output=False):
                print(red("Check the NVCC, NVCC_CXXFLAGS and INCDIRFLAGS"))
                return False
            if not runJob("%s %s xmipp_cuda_test.o -o xmipp_cuda_test -lcudart -lcufft" %
                          (self.configDict["NVCC"], self.configDict["NVCC_LINKFLAGS"])
                          , show_command=False,show_output=False):
                print(red("Check the NVCC and NVCC_LINKFLAGS"))
                return False
            if not runJob("%s %s xmipp_cuda_test.o -o xmipp_cuda_test -lcudart -lcufft" %
                          (self.get(Config.KEY_CXX), self.configDict["NVCC_LINKFLAGS"])
                          , show_command=False,show_output=False):
                print(red("Check the CXX and NVCC_LINKFLAGS"))
                return False
            runJob("rm xmipp_cuda_test*", show_command=False,show_output=False)
        print(green('Done ' + (' ' * 70)))
        return True

    def _config_MPI(self):
        print('Configuring MPI')
        mpiBinCandidates = [os.environ.get('MPI_BINDIR', 'None'),
                            '/usr/lib/openmpi/bin',
                            '/usr/lib64/openmpi/bin']
        if self.configDict["MPI_RUN"] == "":
            if checkProgram("mpirun")[0]:
                self.configDict["MPI_RUN"] = "mpirun"
                print(green("'mpirun' detected."))
            elif checkProgram("mpiexec")[0]:
                self.configDict["MPI_RUN"] = "mpiexec"
                print(green("'mpiexec' detected."))
            else:
                print(yellow("\n'mpirun' and 'mpiexec' not found in the PATH"))
                mpiDir = findFileInDirList('mpirun', mpiBinCandidates)
                if mpiDir == '':
                    mpiDir = askPath(mpiDir, self.ask)
                if mpiDir:
                    self.configDict["MPI_RUN"] = os.path.join(mpiDir, "mpirun")
                    checkProgram(self.configDict["MPI_RUN"])
                    self.environment.update(PATH=mpiDir)
        if self.configDict["MPI_CC"] == "":
            if checkProgram("mpicc")[0]:
                self.configDict["MPI_CC"] = "mpicc"
                print(green("'mpicc' detected."))
            else:
                mpiDir = findFileInDirList('mpicc', mpiBinCandidates)
                if mpiDir == '':
                    print(yellow("\n'mpicc' not found in the PATH"))
                    mpiDir = askPath(mpiDir, self.ask)
                if mpiDir:
                    self.configDict["MPI_CC"] = os.path.join(mpiDir, "mpicc")
                    checkProgram(self.configDict["MPI_CC"])
        if self.configDict["MPI_CXX"] == "":
            if checkProgram("mpicxx")[0]:
                self.configDict["MPI_CXX"] = "mpicxx"
                print(green("'mpicxx' detected."))
            else:
                mpiDir = findFileInDirList('mpicxx', mpiBinCandidates)
                if mpiDir == '':
                    print(yellow("\n'mpicxx' not found in the PATH"))
                    mpiDir = askPath(mpiDir, self.ask)
                if mpiDir:
                    self.configDict["MPI_CXX"] = os.path.join(mpiDir, "mpicxx")
                    checkProgram(self.configDict["MPI_CXX"])

        mpiLib_env = os.environ.get('MPI_LIBDIR', '')
        if mpiLib_env:
            self.configDict['MPI_CXXFLAGS'] += ' -L'+mpiLib_env

        mpiInc_env = os.environ.get('MPI_INCLUDE', '')
        if mpiInc_env:
            self.configDict['MPI_CXXFLAGS'] += ' -I'+mpiInc_env

        if self.configDict["MPI_LINKERFORPROGRAMS"] == "":
            self.configDict["MPI_LINKERFORPROGRAMS"] = self.configDict["MPI_CXX"]

    def _check_cmake(self):
        print("\nChecking cmake configuration")
        status = checkCMakeVersion(Config.CMAKE_VERSION_REQUIRED)
        if status[0] == False:
            return status[1]
        return True, []


    def _check_MPI(self):
        print("\nChecking MPI configuration")
        cppProg = """
    #include <mpi.h>
    int main(){}
    """
        with open("xmipp_mpi_test_main.cpp", "w") as cppFile:
            cppFile.write(cppProg)

        if not runJob("%s -c -w %s %s %s xmipp_mpi_test_main.cpp -o xmipp_mpi_test_main.o"
                      % (self.configDict["MPI_CXX"], self.configDict["INCDIRFLAGS"],
                         self.configDict["CXXFLAGS"], self.configDict["MPI_CXXFLAGS"]),
                      show_output=False,show_command=False):
            return False, 8

        libhdf5 = self._get_Hdf5_name(self.configDict["LIBDIRFLAGS"])
        if not runJob("%s %s %s %s xmipp_mpi_test_main.o -o xmipp_mpi_test_main "
                      "-lfftw3 -lfftw3_threads -l%s  -lhdf5_cpp -ltiff -ljpeg -lsqlite3 -lpthread"
                      % (self.configDict["MPI_LINKERFORPROGRAMS"], self.configDict["LINKFLAGS"],
                         self.configDict["MPI_LINKFLAGS"], self.configDict["LIBDIRFLAGS"], libhdf5),
                      show_output=False, show_command=False):
            return False, 9
        runJob("rm xmipp_mpi_test_main*", show_output=False,show_command=False)

        echoString = blue(
            "   > This sentence should be printed 2 times if mpi runs fine")
        if not (runJob("%s -np 2 echo '%s.'" % (self.configDict['MPI_RUN'], echoString),
                       show_command=False, showWithReturn=False) or
                runJob("%s -np 2 --allow-run-as-root echo '%s.'" % (self.configDict['MPI_RUN'], echoString))):
            print(red("mpirun or mpiexec have failed."))
            return False, 10
        print(green('Done ' + (' ' * 70)))
        return True, 0

    def _config_Java(self):
        print('Configuring JAVA')
        if self.configDict["JAVA_HOME"] == "":
            javaProgramPath = whereis('javac', findReal=True)
            if not javaProgramPath:
                javaProgramPath = findFileInDirList(
                    'javac', ['/usr/lib/jvm/java-*/bin'])  # put candidates here
                if javaProgramPath == '':
                    print(yellow("\n'javac' not found in the PATH"))
                    javaProgramPath = askPath(javaProgramPath, self.ask)
            if not os.path.isdir(javaProgramPath):
                installDepConda('openjdk', self.ask)
                javaProgramPath = whereis('javac', findReal=True)

            if javaProgramPath:
                self.environment.update(PATH=javaProgramPath)
                javaHomeDir = javaProgramPath.replace("/jre/bin", "")
                javaHomeDir = javaHomeDir.replace("/bin", "")
                self.configDict["JAVA_HOME"] = javaHomeDir
            print(green("JAVA_HOME (%s) guessed from javac or installed in conda." % self.configDict["JAVA_HOME"]))
        else:
            print(green("JAVA_HOME (%s) already available. Either coming from the environment or in a previous config file." %
                        self.configDict["JAVA_HOME"]))


        def addSecondaryJavaVariable(varName, defaultValue, checkMethod=os.path.isfile, multiplePaths=False):
            if self.configDict[varName] == "" and self.configDict["JAVA_HOME"]:
                self.configDict[varName] = defaultValue

            resolvedValue = (self.configDict[varName] % self.configDict) % self.configDict

            if multiplePaths:
                resolvedValue = resolvedValue.split(":")
            else:
                resolvedValue = [resolvedValue]

            for path in resolvedValue:

                if checkMethod(path):
                    print(green("%s detected at: %s" % (varName, path)))
                else:
                    print(red("%s NOT detected at: %s" % (varName, path)))


        addSecondaryJavaVariable("JAVA_BINDIR", "%(JAVA_HOME)s/bin", checkMethod=os.path.isdir)
        addSecondaryJavaVariable("JAVAC", "%(JAVA_BINDIR)s/javac")
        addSecondaryJavaVariable("JAR", "%(JAVA_BINDIR)s/jar")
        addSecondaryJavaVariable("JNI_CPPPATH", "%(JAVA_HOME)s/include:%(JAVA_HOME)s/include/linux",
                                 checkMethod=os.path.isdir, multiplePaths=True)


    def _check_Java(self):
        print("Checking Java configuration")
        print(yellow('Working ...'), end='\r')
        if not checkProgram(self.configDict['JAVAC'][0]):
            return False, 11
        javaProg = """
        public class Xmipp {
        public static void main(String[] args) {}
        }
    """
        with open("Xmipp.java", "w") as javaFile:
            javaFile.write(javaProg)
        if not runJob("%s Xmipp.java" % self.configDict["JAVAC"],
                      show_command=False,show_output=False):
            return False, 12
        runJob("rm Xmipp.java Xmipp.class",show_command=False,show_output=False)

        cppProg = """
    #include <jni.h>
    int dummy(){return 0;}
    """
        with open("xmipp_jni_test.cpp", "w") as cppFile:
            cppFile.write(cppProg)

        incs = ""
        for x in self.configDict['JNI_CPPPATH'].split(':'):
            incs += " -I"+x
        if not runJob("%s -c -w %s %s xmipp_jni_test.cpp -o xmipp_jni_test.o" %
                      (self.get(Config.KEY_CXX), incs, self.configDict["INCDIRFLAGS"]),
                      show_command=False,show_output=False):
            return False, 13
        runJob("rm xmipp_jni_test*", show_command=False,show_output=False)
        print(green('Done ' + (' ' * 150)))
        return True, 0

    def _config_Matlab(self):
        if self.configDict["MATLAB"] == "":
            if checkProgram("matlab")[0]:
                self.configDict["MATLAB"] = "True"
            else:
                self.configDict["MATLAB"] = "False"
        if self.configDict["MATLAB"] == "True":
            if self.configDict["MATLAB_DIR"] == "":
                if checkProgram("matlab")[0]:
                    matlabBinDir = whereis("matlab", findReal=True)
                    self.environment.update(MATLAB_BIN_DIR=matlabBinDir)
                    self.configDict["MATLAB_DIR"] = matlabBinDir.replace(
                        "/bin", "")
                    print(green("Matlab detected at " + matlabBinDir))

    def _check_Matlab(self):
        ans = True
        if self.configDict["MATLAB"] == "True":
            print("Checking Matlab configuration")
            if not checkProgram("matlab")[0]:
                return False
            print("Checking Matlab configuration ...")
            cppProg = """
        #include <mex.h>
        int dummy(){return 0;}
        """
            with open("xmipp_mex.cpp", "w") as cppFile:
                cppFile.write(cppProg)

            if not runJob("%s/bin/mex -silent xmipp_mex.cpp" % self.configDict["MATLAB_DIR"]):
                print(red("Check the MATLAB_DIR"))
                ans = False
            runJob("rm xmipp_mex*")
        return ans

    def _config_StarPU(self):
        # TODO(Jan Polak): This check would be probably better done with pkg-config
        if self.configDict["STARPU"] == "":
            # Heuristic only, StarPU has no main executable
            if checkProgram("starpu_sched_display")[0]:
                self.configDict["STARPU"] = "True"
            else:
                self.configDict["STARPU"] = "False"
        if self.configDict["STARPU"] == "True":
            if self.configDict["STARPU_HOME"] == "" and checkProgram("starpu_sched_display")[0]:
                starpuBinDir = os.path.dirname(os.path.realpath(
                    distutils.spawn.find_executable("starpu_sched_display")))
                self.configDict["STARPU_HOME"] = starpuBinDir.replace(
                    "/bin", "")
        if self.configDict["STARPU_INCLUDE"] == "":
            self.configDict["STARPU_INCLUDE"] = "%(STARPU_HOME)s/include/starpu/1.3"
        if self.configDict["STARPU_LIB"] == "":
            self.configDict["STARPU_LIB"] = "%(STARPU_HOME)s/lib"
        if self.configDict["STARPU_LIBRARY"] == "":
            self.configDict["STARPU_LIBRARY"] = "libstarpu-1.3"

    def _check_StarPU(self):
        ans = True
        if self.configDict["STARPU"] == "True":
            print("Checking StarPU configuration")
            if self.configDict["CUDA"] != "True":
                ans = False
                print(red("CUDA must be enabled together with STARPU"))
            if self.configDict["STARPU_INCLUDE"] == "" or not os.path.isdir(self.configDict["STARPU_INCLUDE"]):
                ans = False
                print(red("Check the STARPU_INCLUDE directory: " +
                      self.configDict["STARPU_INCLUDE"]))
            if self.configDict["STARPU_LIB"] == "" or not os.path.isdir(self.configDict["STARPU_LIB"]):
                ans = False
                print(red("Check the STARPU_LIB directory: " +
                      self.configDict["STARPU_LIB"]))
            if self.configDict["STARPU_LIBRARY"] == "":
                ans = False
                print(red("STARPU_LIBRARY must be specified (link library name)"))

            if ans:
                with open("xmipp_starpu_config_test.cpp", "w") as cppFile:
                    cppFile.write("""
                    #include <starpu.h>
                    int dummy(){return 0;}
                    """)

                if not runJob("%s -c -w %s %s -I%s -L%s -l%s xmipp_starpu_config_test.cpp -o xmipp_starpu_config_test.o" %
                              (self.configDict["NVCC"], self.configDict["NVCC_CXXFLAGS"], self.configDict["INCDIRFLAGS"],
                               self.configDict["STARPU_INCLUDE"], self.configDict["STARPU_LIB"], self.configDict["STARPU_LIBRARY"])):
                    print(red("Check STARPU_* settings"))
                    ans = False
                runJob("rm xmipp_starpu_config_test*")
        return ans

    def _config_DL(self):
        if (Config.KEY_USE_DL in self.configDict) and (self.configDict[Config.KEY_USE_DL] != 'True'):
            self.configDict[Config.KEY_USE_DL] = 'False'

    def check_version(self, XMIPP_VERNAME):
        if Config.KEY_VERSION not in self.configDict or self.configDict[Config.KEY_VERSION] != self._get_version():
            print(yellow("There are some changes in repository which may not be compatible\n"
                         " with your config file. Run './xmipp config' to generate a new config file."))

    def _get_version(self):
        """ If git not present means it is in production mode
            and version can be retrieved from the commit.info file
        """
        commitFn = os.path.join(
            'src', 'xmipp', 'commit.info')  # FIXME check if this is still true
        notFound = "(no git repo detected)"
        if ensureGit(False)[0] and isGitRepo():
            scriptName = []
            runJob('git ls-files --full-name ' +
                   __file__, show_command=False, log=scriptName, show_output=False)
            lastCommit = []
            # get hash of the last commit changing this script
            if runJob('git log -n 1 --pretty=format:%H -- ' + scriptName[0].strip(), '.', False, lastCommit, False):
                return lastCommit[0].strip()
        elif os.path.isfile(commitFn):
            with open(commitFn, 'r') as file:
                commitInfo = file.readline()
            return commitInfo
        else:
            return notFound

    # def _config_tests(self):
    #     if self.configDict[Config.KEY_BUILD_TESTS] == "":
    #         self.configDict[Config.KEY_BUILD_TESTS] = askYesNo(yellow(
    #             '\nDo you want to build tests [YES/no]'), default=True, actually_ask=self.ask)
