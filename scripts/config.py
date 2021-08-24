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


from sysconfig import get_paths
from configparser import ConfigParser
import os
import sys
from .utils import *
from.environment import Environment


class Config:
    '''
    This class is responsible for generation and validation of the Xmipp config file.
    The main idea is that the config file is a dictionary with multiple values.
    Newly generated config follows these rules:
    1. By default, all values are undefined
    2. Existing values found in the Environment take precedence
    3. We do our best to look for the rest
    '''
    FILE_NAME = 'xmipp.conf'

    SECTION_BUILD = "BUILD"

    OPT_BUILD_TESTS = 'BUILD_TESTS'
    OPT_CC = 'CC'
    OPT_CCFLAGS = 'CCFLAGS'
    OPT_CONFIG_VERSION = 'CONFIG_VERSION'
    OPT_CXX = 'CXX'
    OPT_CXXFLAGS = 'CXXFLAGS'
    OPT_INCDIRFLAGS = 'INCDIRFLAGS'
    OPT_JAVA_HOME = 'JAVA_HOME'
    OPT_JAVA_BINDIR = 'JAVA_BINDIR'
    OPT_JAVAC = 'JAVAC'
    OPT_JAR = 'JAR'
    OPT_JNI_CPPPATH = 'JNI_CPPPATH'
    OPT_LIBDIRFLAGS = 'LIBDIRFLAGS'
    OPT_LINKERFORPROGRAMS = 'LINKERFORPROGRAMS'
    OPT_LINKFLAGS = 'LINKFLAGS'
    OPT_PYTHONINCFLAGS = 'PYTHONINCFLAGS'
    OPT_MPI_CC = 'MPI_CC'
    OPT_MPI_CXX = 'MPI_CXX'
    OPT_MPI_RUN = 'MPI_RUN'
    OPT_MPI_LINKERFORPROGRAMS = 'MPI_LINKERFORPROGRAMS'
    OPT_MPI_CXXFLAGS = 'MPI_CXXFLAGS'
    OPT_MATLAB_DIR = 'MATLAB_DIR'
    OPT_CUDA = 'CUDA'
    OPT_DEBUG = 'DEBUG'
    OPT_MATLAB = 'MATLAB'
    OPT_OPENCV = 'OPENCV'
    OPT_OPENCVSUPPORTSCUDA = 'OPENCVSUPPORTSCUDA'
    OPT_OPENCV3 = 'OPENCV3'
    OPT_MPI_LINKFLAGS = 'MPI_LINKFLAGS'
    OPT_NVCC = 'NVCC'
    OPT_CXX_CUDA = 'CXX_CUDA'
    OPT_NVCC_CXXFLAGS = 'NVCC_CXXFLAGS'
    OPT_NVCC_LINKFLAGS = 'NVCC_LINKFLAGS'
    OPT_STARPU = 'STARPU'
    OPT_STARPU_HOME = 'STARPU_HOME'
    OPT_STARPU_INCLUDE = 'STARPU_INCLUDE'
    OPT_STARPU_LIB = 'STARPU_LIB'
    OPT_STARPU_LIBRARY = 'STARPU_LIBRARY'
    OPT_USE_DL = 'USE_DL'
    OPT_USE_DL = 'USE_DL'
    OPT_VERIFIED = 'VERIFIED'
    OPT_CONFIG_VERSION = 'CONFIG_VERSION'
    OPT_PYTHON_LIB = 'PYTHON_LIB'

    def __init__(self, askUser=False):
        self.ask = askUser
        self.config = ConfigParser()
        # keep case (stackoverflow.com/questions/1611799)
        self.environment = Environment()
        self.configDict = {}  # FIXME remove (use _set() etc.)

    def create(self):
        print("Configuring -----------------------------------------")
        self._create_default()

        self._set_compilers()
        self._set_OpenCV()
        self._set_CUDA()
        # FIXME refactor methods below, rename them to _set_XXX()
        self.configDict = self.get()  # FIXME remove (use _set() etc.)
        self._config_MPI()
        self._config_Java()

        self._config_Matlab()
        self._config_StarPU()
        self._config_DL()
        self._config_tests()

        self.write()
        self.environment.write()
        print(blue("Configuration completed....."))

    def check(self):
        print("Checking configuration ------------------------------")
        if self._is_verified():
            print(blue("'%s' is already checked. Set VERIFIED=False to re-checked"
                       % Config.FILE_NAME))
            return True
        if not self._is_up_to_date():
            print(blue(self._get_help_msg()))
            return False
        if not self._check_compiler:
            print(red("Cannot compile"))
            print("Possible solutions")  # FIXME: check libraries
            print("In Ubuntu: sudo apt-get -y install libsqlite3-dev libfftw3-dev libhdf5-dev libopencv-dev python3-dev "
                  "python3-numpy python3-scipy python3-mpi4py")
            print(
                "In Manjaro: sudo pacman -Syu install hdf5 python3-numpy python3-scipy --noconfirm")
            print("Please, see 'https://scipion-em.github.io/docs/docs/scipion-modes/"
                  "install-from-sources.html#step-2-dependencies' for more information about libraries dependencies.")
            print("\nRemember to re-run './xmipp config' after installing libraries in order to "
                  "take into account the new system configuration.")
            runJob("rm xmipp_test_main*")
            return False
        if not self._check_MPI():
            print(red("Cannot compile with MPI or use it"))
            runJob("rm xmipp_mpi_test_main*")
            return False
        if not self._check_Java():
            print(red("Cannot compile with Java"))
            runJob("rm Xmipp.java Xmipp.class xmipp_jni_test*")
            return False
        if not self._check_CUDA():
            print(red("Cannot compile with NVCC, continuing without CUDA"))
            # if fails, the test files remains
            runJob("rm xmipp_cuda_test*")
            self.configDict["CUDA"] = "False"
        if not self._check_Matlab():
            print(red("Cannot compile with Matlab, continuing without Matlab"))
            self.configDict["MATLAB"] = "False"
            runJob("rm xmipp_mex*")
        if not self._check_StarPU():
            print(red("Cannot compile with StarPU, continuing without StarPU"))
            self.configDict["STARPU"] = "False"
        self.configDict['VERIFIED'] = "True"
        self.write()  # store result

        return True

    def get(self, option=None):
        tmp = self.config['BUILD']
        if option:
            return tmp[option]
        return tmp

    def _set(self, option, value):
        self.get()[option] = str(value)

    def _set_version(self):
        self._set(Config.OPT_CONFIG_VERSION, Config.get_version())

    def _set_if_empty(self, option, value):
        if self.is_empty(option):
            self._set(option, value)

    def _append(self, option, value):
        vals = self.get()[option]
        self._set(option, vals + ' ' + value)

    def is_true(self, option):
        if self.has(option):
            return self.get()[option].lower() == 'true'
        return False

    def is_empty(self, option):
        return self.has(option) and self.get(option) == ''

    def is_valid(self):
        return self._is_verified() and self._is_up_to_date()

    def ensure_is_valid(self):
        if not (Config.config_file_exists() and self.is_valid()):
            sys.exit(red(self._get_help_msg()))

    def config_file_exists():
        return os.path.isfile(Config.FILE_NAME)

    def _is_verified(self):
        return self.is_true(Config.OPT_VERIFIED)

    def _is_up_to_date(self):
        return self.get(Config.OPT_CONFIG_VERSION) == Config.get_version()

    def has(self, option, value=None):
        if value is None:
            return option in self.get()
        return option in self.get() and value in self.get()[option]

    def read(self, file=FILE_NAME):
        self.config.read(file)

    def get_opts():
        import inspect
        attrs = inspect.getmembers(Config, lambda a: not(inspect.isroutine(a)))
        return [a[1] for a in attrs if a[0].startswith('OPT')]

    def write(self):
        with open(Config.FILE_NAME, "w") as f:
            self.config.write(f)

    def _get_help_msg(self):
        msg_missing = 'The config file %s is missing.\n' % Config.FILE_NAME
        msg_obsolete = ('The config file %s is obsolete and has to be regenerated.\n'
                        'We recommend you do create a manual backup of it first\n' % Config.FILE_NAME)
        msg_common = 'Please, run ./xmipp --help or check the online documention for further details'
        if not Config.config_file_exists():
            return msg_missing + msg_common
        if not self._is_up_to_date():
            return msg_obsolete + msg_common
        return msg_common

    def _create_default(self):
        options = Config.get_opts()
        self.config.optionxform = str
        # remove existing section, if any (e.g. regenerating existing config)
        self.config.remove_section(Config.SECTION_BUILD)
        self.config.add_section(Config.SECTION_BUILD)
        for o in options:
            self.config[Config.SECTION_BUILD][o] = environ.get(o, '')
        self._set_version()
        self._set(Config.OPT_VERIFIED, False)

    def _set_OpenCV(self):
        if not self.is_empty(Config.OPT_OPENCV) or not self.is_empty(Config.OPT_OPENCV3):
            return

        # try to compile C++ file which reports the version
        curr_dir = get_script_dir(__file__)
        cpp = os.path.join(curr_dir, 'templates', 'opencv_version.cpp')
        bin = os.path.splitext(cpp)[0]
        cmd_version = '%s %s -o %s %s %s' % (self.get(
            Config.OPT_CXX), cpp, bin, self.get(Config.OPT_CXXFLAGS), self.get(Config.OPT_INCDIRFLAGS))
        log = []
        if not runJob(cmd_version, show_output=False, show_command=False, log=log):
            if any('No such file or directory' in l for l in log):
                print(yellow('OpenCV not found'))
            else:
                print(red(
                    'Cannot compile OpenCV test file. The CXX related configuration is probably invalid'))
            return
        # execute the binary to get the version
        log = []
        runJob(bin, show_output=False, log=log, show_command=False)
        os.remove(bin)
        version = int(log[0])
        if 2 == version:
            self._set(Config.OPT_OPENCV, True)
        elif 3 == version:
            self._set(Config.OPT_OPENCV3, True)
        else:
            print(yellow('Unsupported version of OpenCV (%s) found.' % version))
            return
        print(green('OpenCV %s found' % version))

        # check if CUDA is supported
        cpp = os.path.join(curr_dir, 'templates',
                           'opencv' + str(version) + '_cuda.cpp')
        bin = os.path.splitext(cpp)[0]
        cmd_cuda = '%s %s -o %s %s %s' % (self.get(
            Config.OPT_CXX), cpp, bin, self.get(Config.OPT_CXXFLAGS), self.get(Config.OPT_INCDIRFLAGS))
        cuda_supported = runJob(
            cmd_cuda, show_output=False, show_command=False)
        os.remove(bin)
        self._set(Config.OPT_OPENCVSUPPORTSCUDA, cuda_supported)
        if cuda_supported:
            print(green('CUDA support for OpenCV found'))
        else:
            print(yellow('CUDA support for OpenCV NOT found'))

    def get_supported_GCC():
        # we need GCC with C++14 support
        # https://gcc.gnu.org/projects/cxx-status.html
        return [11.2, 11.1, 11, 10.3, 10.2, 10.1, 10,
                9.3, 9.2, 9.1, 9, 8.5, 8.4, 8.3, 8.2, 8.1, 8,
                7.5, 7.4, 7.3, 7.2, 7.1, 7, 6.5, 6.4, 6.3, 6.2, 6.1, 6,
                5.5, 5.4, 5.3, 5.2, 5.1, 5, '']

    def _set_compiler_linker_helper(self, opt, prg, versions):
        if not self.is_empty(opt):
            return
        prg = find_newest(prg, versions, True)
        if is_CI_build() and prg:
            prg = 'ccache ' + prg
        self._set(opt, prg)

    def _set_cc(self):
        self._set_compiler_linker_helper(
            Config.OPT_CC, 'gcc', Config.get_supported_GCC())

    def _set_cxx(self):
        self._set_compiler_linker_helper(
            Config.OPT_CXX, 'g++', Config.get_supported_GCC())

    def _set_linker(self):
        opt = Config.OPT_LINKERFORPROGRAMS
        self._set_compiler_linker_helper(
            opt, 'g++', Config.get_supported_GCC())

    def _set_cc_flags(self):
        std = '-std=c99'
        opt = Config.OPT_CCFLAGS
        if self.has(opt, std):
            return
        self._append(opt, std)

    def _set_cxx_flags(self):
        opt = Config.OPT_CXXFLAGS
        if not self.is_empty(opt):
            return
        # optimize for current machine
        self._append(opt, '-mtune=native -march=native')
        # c++ standard
        self._append(opt, '-std=c++14')
        if is_CI_build():
            # don't tolerate any warnings on build machine
            self._append(opt, '-Werror')
        # set optimization level
        level = '-O0' if is_CI_build() else '-O3'
        level = '-O1' if self.is_true(Config.OPT_DEBUG) else level
        self._append(opt, level)
        # optimize at the link time
        self._append(opt, '-flto')
        if self.is_true(Config.OPT_DEBUG):
            self._append(opt, '-g')
            self._append(opt, '-fno-omit-frame-pointer')

    def _set_compilers(self):
        self._set_if_empty(Config.OPT_DEBUG, False)
        self._set_cc()
        self._set_cc_flags()
        self._set_cxx()
        self._set_cxx_flags()
        self._set_linker()
        self._set_libs_includes()

    def _get_conda_include_dir(self):
        tmp = os.environ.get('CONDA_PREFIX', '')
        return os.path.join(tmp, 'include') if tmp else ''

    def _get_conda_lib_dir(self):
        tmp = os.environ.get('CONDA_PREFIX', '')
        return os.path.join(tmp, 'lib') if tmp else ''

    def _get_python_include_dir(self):
        return os.path.join(get_paths()['data'], 'include')

    def _get_python_lib_dir(self):
        return os.path.join(get_paths()['data'], 'lib')

    def _get_include_dirs(self):
        dirs = {self._get_conda_include_dir(),
                self._get_python_include_dir()}
        return self._join_with_prefix(dirs, '-I')

    def _get_lib_dirs(self):
        dirs = {self._get_conda_lib_dir(), self._get_python_lib_dir()}
        return self._join_with_prefix(dirs, '-L')

    def _exists(self, lib):
        # see https://serverfault.com/a/679586
        log = []
        runJob('%s %s' % (self.get(Config.OPT_CXX), lib),
               show_output=False, show_command=False, log=log)
        return any('undefined reference' in l for l in log)

    def _set_libs_includes(self):
        self._set_if_empty(Config.OPT_INCDIRFLAGS,
                           self._get_include_dirs())
        self._set_if_empty(Config.OPT_LIBDIRFLAGS, self._get_lib_dirs())

        if not self._exists('-lhdf5_cpp'):
            print(red("'lhdf5_cpp' not found in the system"))
            installDepConda('hdf5', self.ask)
        if not self._exists('-lfftw3'):
            print(red("'libfftw3' not found in the system"))
            installDepConda('fftw', self.ask)
        if not self._exists('-ltiff'):
            print(red("'libtiff' not found in the system"))
            installDepConda('libtiff', self.ask)

        # optimize at the link time
        f = '-flto'
        if self.has(Config.OPT_LIBDIRFLAGS, f):
            return
        self._append(Config.OPT_LIBDIRFLAGS, f)

        # set python, assume we have python 3
        # pymalloc flag is not needed from 3.8, see https://docs.python.org/3/whatsnew/3.8.html
        pymalloc = 'm' if sys.version_info.minor < 8 else ''
        lib = 'python%s.%s%s' % (
            sys.version_info.major, sys.version_info.minor, pymalloc)
        self._set_if_empty(Config.OPT_PYTHON_LIB, lib)
        import numpy
        self._set_if_empty(
            Config.OPT_PYTHONINCFLAGS, self._get_include_dirs() + '-I' + numpy.get_include())

    def _get_GCC_version(self, compiler):
        log = []
        runJob(compiler + " -dumpversion", show_output=False,
               show_command=False, log=log)
        full_version = log[0].strip()
        tokens = full_version.split('.')
        if len(tokens) < 2:
            tokens.append('0')  # for version 5.0, only '5' is returned
        gccVersion = float(str(tokens[0] + '.' + tokens[1]))
        return gccVersion, full_version

    def _ensure_GCC_GPP_version(self, compiler):
        if not checkProgram(compiler, True):
            sys.exit(-7)
        gccVersion, fullVersion = self._get_GCC_version(compiler)
        if gccVersion < 5.0:
            print(red('found ' + compiler + " in version " +
                  fullVersion + '. Version 5.0 or higher is required.'))
            sys.exit(-8)
        elif gccVersion < 7.0:
            print(yellow(
                'Consider updating your compiler. Xmipp will require GCC 7 or newer soon'))
        else:
            print(green(compiler + ' ' + fullVersion + ' found'))

    def _ensure_compiler_version(self, compiler):
        if 'g++' in compiler or 'gcc' in compiler:
            self._ensure_GCC_GPP_version(compiler)
        else:
            print(red('Version detection for \'' +
                  compiler + '\' is not implemented.'))

    def _get_Hdf5_name(self, libdirflags):
        libdirs = libdirflags.split("-L")
        for dir in libdirs:
            if os.path.exists(os.path.join(dir.strip(), "libhdf5.so")):
                return "hdf5"
            elif os.path.exists(os.path.join(dir.strip(), "libhdf5_serial.so")):
                return "hdf5_serial"
        return "hdf5"

    def _check_compiler(self):
        print("Checking compiler configuration ...")
        # in case user specified some wrapper of the compiler
        # get rid of it: 'ccache g++' -> 'g++'
        currentCxx = self.configDict["CXX"].split()[-1]
        self._ensure_compiler_version(currentCxx)

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
                if self.configDict["OPENCV3"] == "True":
                    cppProg += "#include <opencv2/cudaoptflow.hpp>\n"
                else:
                    cppProg += "#include <opencv2/core/cuda.hpp>\n"
        cppProg += "\n int main(){}\n"
        with open("xmipp_test_main.cpp", "w") as cppFile:
            cppFile.write(cppProg)

        if not runJob("%s -c -w %s xmipp_test_main.cpp -o xmipp_test_main.o %s %s" %
                      (self.configDict["CXX"], self.configDict["CXXFLAGS"], self.configDict["INCDIRFLAGS"], self.configDict["PYTHONINCFLAGS"])):
            print(
                red("Check the INCDIRFLAGS, CXX, CXXFLAGS and PYTHONINCFLAGS in xmipp.conf"))
            # FIXME: Check the dependencies list
            print(red("If some of the libraries headers fail, try installing fftw3_dev, tiff_dev, jpeg_dev, sqlite_dev, hdf5, pthread"))
            return False
        libhdf5 = self._get_Hdf5_name(self.configDict["LIBDIRFLAGS"])
        if not runJob("%s %s %s xmipp_test_main.o -o xmipp_test_main -lfftw3 -lfftw3_threads -l%s  -lhdf5_cpp -ltiff -ljpeg -lsqlite3 -lpthread" %
                      (self.configDict["LINKERFORPROGRAMS"], self.configDict["LINKFLAGS"], self.configDict["LIBDIRFLAGS"], libhdf5)):
            print(red("Check the LINKERFORPROGRAMS, LINKFLAGS and LIBDIRFLAGS"))
            return False
        runJob("rm xmipp_test_main*")
        return True

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
        v = ['10.2', '10.1', '10',
             '9.3', '9.2', '9.1', '9',
             '8.4', '8.3', '8.2', '8.1', '8',
             '7.5', '7.4', '7.3', '7.2', '7.1', '7',
             '6.5', '6.4', '6.3', '6.2', '6.1', '6',
             '5.5', '5.4', '5.3', '5.2', '5.1', '5',
             '4.9', '4.8']
        if 8.0 <= nvcc_version < 9.0:
            return v[v.index('5.3'):]
        elif 9.0 <= nvcc_version < 9.2:
            return v[v.index('5.5'):]
        elif 9.2 <= nvcc_version < 10.1:
            return v[v.index('7.3'):]
        elif 10.1 <= nvcc_version <= 10.2:
            return v[v.index('8.4'):]
        elif 11.0 <= nvcc_version <= 11.2:
            return v[v.index('9.3'):]
        return []  # not supported

    def _join_with_prefix(self, collection, prefix):
        return ' '.join([prefix + i for i in collection if i])

    def _set_nvcc(self):
        if not self.is_empty(Config.OPT_NVCC):
            return True

        nvcc_loc_candidates = {os.environ.get('XMIPP_CUDA_BIN', ''),
                               os.environ.get('CUDA_BIN', ''),
                               '/usr/local/cuda/bin'}
        nvcc_loc = find('nvcc', nvcc_loc_candidates)
        if not nvcc_loc:
            print(yellow('nvcc not found (searched in %s). If you want to enable CUDA, add \'nvcc\' to PATH. %s' % (
                nvcc_loc_candidates, self._get_help_msg())))
            return False
        print(green('nvcc found in ' + nvcc_loc))
        self._set(Config.OPT_NVCC, nvcc_loc)
        return True

    def _set_nvcc_cxx(self, nvcc_version):
        opt = Config.OPT_CXX_CUDA
        if not self.is_empty(opt):
            return True
        prg = find_newest('g++', self._get_compatible_GCC(nvcc_version), True)
        if not prg:
            print(
                yellow('No valid compiler found for CUDA host code. ' + self._get_help_msg()))
            return False
        self._set(opt, prg)
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
        self._set(opt, self._join_with_prefix([path, os.path.join(path, 'stubs')], '-L'))
        return True

    def _set_CUDA(self):
        if not self.is_empty(Config.OPT_CUDA):
            return

        if not self._set_nvcc():
            return
        nvcc_version, nvcc_full_version = self._get_CUDA_version(
            self.get(Config.OPT_NVCC))
        print(green('CUDA-' + nvcc_full_version + ' found.'))
        if nvcc_version != 10.2:
            print(yellow('CUDA-10.2 is recommended.'))

        if not self._set_nvcc_cxx(nvcc_version) or not self._set_nvcc_lib_dir():
            return
        self._set_nvcc_flags(nvcc_version)

        # update config and environment
        self._set(Config.OPT_CUDA, True)
        self.environment.update(CUDA=True)
        LD = ':'.join(self.get(Config.OPT_NVCC_LINKFLAGS).split('-L'))
        self.environment.update(LD_LIBRARY_PATH=LD)

    def _check_CUDA(self):
        if self.configDict["CUDA"] == "True":
            if not checkProgram(self.configDict["NVCC"]):
                return False
            print("Checking CUDA configuration ...")
            cppProg = """
        #include <cuda_runtime.h>
        #include <cufft.h>
        int main(){}
        """
            with open("xmipp_cuda_test.cpp", "w") as cppFile:
                cppFile.write(cppProg)

            if not runJob("%s -c -w %s %s xmipp_cuda_test.cpp -o xmipp_cuda_test.o" %
                          (self.configDict["NVCC"], self.configDict["NVCC_CXXFLAGS"], self.configDict["INCDIRFLAGS"])):
                print(red("Check the NVCC, NVCC_CXXFLAGS and INCDIRFLAGS"))
                return False
            if not runJob("%s %s xmipp_cuda_test.o -o xmipp_cuda_test -lcudart -lcufft" %
                          (self.configDict["NVCC"], self.configDict["NVCC_LINKFLAGS"])):
                print(red("Check the NVCC and NVCC_LINKFLAGS"))
                return False
            if not runJob("%s %s xmipp_cuda_test.o -o xmipp_cuda_test -lcudart -lcufft" %
                          (self.configDict["CXX"], self.configDict["NVCC_LINKFLAGS"])):
                print(red("Check the CXX and NVCC_LINKFLAGS"))
                return False
            runJob("rm xmipp_cuda_test*")
        return True

    def _config_MPI(self):
        mpiBinCandidates = [os.environ.get('MPI_BINDIR', 'None'),
                            '/usr/lib/openmpi/bin',
                            '/usr/lib64/openmpi/bin']
        if self.configDict["MPI_RUN"] == "":
            if checkProgram("mpirun", False):
                self.configDict["MPI_RUN"] = "mpirun"
                print(green("'mpirun' found."))
            elif checkProgram("mpiexec", False):
                self.configDict["MPI_RUN"] = "mpiexec"
                print(green("'mpiexec' found."))
            else:
                print(yellow("\n'mpirun' and 'mpiexec' not found in the PATH"))
                mpiDir = findFileInDirList('mpirun', mpiBinCandidates)
                mpiDir = askPath(mpiDir, self.ask)
                if mpiDir:
                    self.configDict["MPI_RUN"] = os.path.join(mpiDir, "mpirun")
                    checkProgram(self.configDict["MPI_RUN"])
                    self.environment.update(PATH=mpiDir)
        if self.configDict["MPI_CC"] == "":
            if checkProgram("mpicc", False):
                self.configDict["MPI_CC"] = "mpicc"
                print(green("'mpicc' found."))
            else:
                print(yellow("\n'mpicc' not found in the PATH"))
                mpiDir = findFileInDirList('mpicc', mpiBinCandidates)
                mpiDir = askPath(mpiDir, self.ask)
                if mpiDir:
                    self.configDict["MPI_CC"] = os.path.join(mpiDir, "mpicc")
                    checkProgram(self.configDict["MPI_CC"])
        if self.configDict["MPI_CXX"] == "":
            if checkProgram("mpicxx", False):
                self.configDict["MPI_CXX"] = "mpicxx"
                print(green("'mpicxx' found."))
            else:
                print(yellow("\n'mpicxx' not found in the PATH"))
                mpiDir = findFileInDirList('mpicxx', mpiBinCandidates)
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

    def _check_MPI(self):
        print("Checking MPI configuration ...")
        cppProg = """
    #include <mpi.h>
    int main(){}
    """
        with open("xmipp_mpi_test_main.cpp", "w") as cppFile:
            cppFile.write(cppProg)

        if not runJob("%s -c -w %s %s %s xmipp_mpi_test_main.cpp -o xmipp_mpi_test_main.o"
                      % (self.configDict["MPI_CXX"], self.configDict["INCDIRFLAGS"],
                         self.configDict["CXXFLAGS"], self.configDict["MPI_CXXFLAGS"])):
            print(red(
                "MPI compilation failed. Check the INCDIRFLAGS, MPI_CXX and CXXFLAGS in 'xmipp.conf'"))
            print(red("In addition, MPI_CXXFLAGS can also be used to add flags to MPI compilations."
                      "'%s --showme:compile' might help" % self.configDict['MPI_CXX']))
            return False

        libhdf5 = self._get_Hdf5_name(self.configDict["LIBDIRFLAGS"])
        if not runJob("%s %s %s %s xmipp_mpi_test_main.o -o xmipp_mpi_test_main "
                      "-lfftw3 -lfftw3_threads -l%s  -lhdf5_cpp -ltiff -ljpeg -lsqlite3 -lpthread"
                      % (self.configDict["MPI_LINKERFORPROGRAMS"], self.configDict["LINKFLAGS"],
                         self.configDict["MPI_LINKFLAGS"], self.configDict["LIBDIRFLAGS"], libhdf5)):
            print(red("Check the LINKERFORPROGRAMS, LINKFLAGS and LIBDIRFLAGS"))
            print(red("In addition, MPI_LINKFLAGS can also be used to add flags to MPI links. "
                      "'%s --showme:compile' might help" % self.configDict['MPI_CXX']))
            return False
        runJob("rm xmipp_mpi_test_main*")

        echoString = blue(
            "   > This sentence should be printed 2 times if mpi runs fine")
        if not (runJob("%s -np 2 echo '%s.'" % (self.configDict['MPI_RUN'], echoString)) or
                runJob("%s -np 2 --allow-run-as-root echo '%s.'" % (self.configDict['MPI_RUN'], echoString))):
            print(red("mpirun or mpiexec have failed."))
            return False
        return True

    def _config_Java(self):
        if self.configDict["JAVA_HOME"] == "":
            javaProgramPath = whereis('javac', findReal=True)
            if not javaProgramPath:
                print(yellow("\n'javac' not found in the PATH"))
                javaProgramPath = findFileInDirList(
                    'javac', ['/usr/lib/jvm/java-*/bin'])  # put candidates here
                javaProgramPath = askPath(javaProgramPath, self.ask)
            if not os.path.isdir(javaProgramPath):
                installDepConda('openjdk')
                javaProgramPath = whereis('javac', findReal=True)

            if javaProgramPath:
                self.environment.update(PATH=javaProgramPath)
                javaHomeDir = javaProgramPath.replace("/jre/bin", "")
                javaHomeDir = javaHomeDir.replace("/bin", "")
                self.configDict["JAVA_HOME"] = javaHomeDir

        if self.configDict["JAVA_BINDIR"] == "" and self.configDict["JAVA_HOME"]:
            self.configDict["JAVA_BINDIR"] = "%(JAVA_HOME)s/bin"
        if self.configDict["JAVAC"] == "" and self.configDict["JAVA_HOME"]:
            self.configDict["JAVAC"] = "%(JAVA_BINDIR)s/javac"
        if self.configDict["JAR"] == "" and self.configDict["JAVA_HOME"]:
            self.configDict["JAR"] = "%(JAVA_BINDIR)s/jar"
        if self.configDict["JNI_CPPPATH"] == "" and self.configDict["JAVA_HOME"]:
            self.configDict["JNI_CPPPATH"] = "%(JAVA_HOME)s/include:%(JAVA_HOME)s/include/linux"

        if (os.path.isfile((self.configDict["JAVAC"] % self.configDict) % self.configDict) and
                os.path.isfile((self.configDict["JAR"] % self.configDict) % self.configDict) and
                os.path.isdir("%(JAVA_HOME)s/include" % self.configDict)):
            print(green("Java found at: %s" % self.configDict["JAVA_HOME"]))
        else:
            print(red("No development environ for 'java' found. "
                      "Please, check JAVA_HOME, JAVAC, JAR and JNI_CPPPATH variables."))

    def _check_Java(self):
        if not checkProgram(self.configDict['JAVAC']):
            return False
        print("Checking Java configuration...")
        javaProg = """
        public class Xmipp {
        public static void main(String[] args) {}
        }
    """
        with open("Xmipp.java", "w") as javaFile:
            javaFile.write(javaProg)
        if not runJob("%s Xmipp.java" % self.configDict["JAVAC"]):
            print(red("Check the JAVAC"))
            return False
        runJob("rm Xmipp.java Xmipp.class")

        cppProg = """
    #include <jni.h>
    int dummy(){}
    """
        with open("xmipp_jni_test.cpp", "w") as cppFile:
            cppFile.write(cppProg)

        incs = ""
        for x in self.configDict['JNI_CPPPATH'].split(':'):
            incs += " -I"+x
        if not runJob("%s -c -w %s %s xmipp_jni_test.cpp -o xmipp_jni_test.o" %
                      (self.configDict["CXX"], incs, self.configDict["INCDIRFLAGS"])):
            print(red("Check the JNI_CPPPATH, CXX and INCDIRFLAGS"))
            return False
        runJob("rm xmipp_jni_test*")
        return True

    def _config_Matlab(self):
        if self.configDict["MATLAB"] == "":
            if checkProgram("matlab", False):
                self.configDict["MATLAB"] = "True"
            else:
                self.configDict["MATLAB"] = "False"
        if self.configDict["MATLAB"] == "True":
            if self.configDict["MATLAB_DIR"] == "":
                if checkProgram("matlab"):
                    matlabBinDir = whereis("matlab", findReal=True)
                    self.environment.update(MATLAB_BIN_DIR=matlabBinDir)
                    self.configDict["MATLAB_DIR"] = matlabBinDir.replace(
                        "/bin", "")
                    print(green("Matlab found at " + matlabBinDir))

    def _check_Matlab(self):
        ans = True
        if self.configDict["MATLAB"] == "True":
            if not checkProgram("matlab"):
                return False
            print("Checking Matlab configuration ...")
            cppProg = """
        #include <mex.h>
        int dummy(){}
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
            if checkProgram("starpu_sched_display", show=False):
                self.configDict["STARPU"] = "True"
            else:
                self.configDict["STARPU"] = "False"
        if self.configDict["STARPU"] == "True":
            if self.configDict["STARPU_HOME"] == "" and checkProgram("starpu_sched_display"):
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
                    int dummy(){}
                    """)

                if not runJob("%s -c -w %s %s -I%s -L%s -l%s xmipp_starpu_config_test.cpp -o xmipp_starpu_config_test.o" %
                              (self.configDict["NVCC"], self.configDict["NVCC_CXXFLAGS"], self.configDict["INCDIRFLAGS"],
                               self.configDict["STARPU_INCLUDE"], self.configDict["STARPU_LIB"], self.configDict["STARPU_LIBRARY"])):
                    print(red("Check STARPU_* settings"))
                    ans = False
                runJob("rm xmipp_starpu_config_test*")
        return ans

    def _config_DL(self):
        if (Config.OPT_USE_DL in self.configDict) and (self.configDict[Config.OPT_USE_DL] != 'True'):
            self.configDict[Config.OPT_USE_DL] = 'False'

    def get_version():
        """ If git not present means it is in production mode
            and version can be retrieved from the commit.info file
        """
        commitFn = os.path.join(
            'src', 'xmipp', 'commit.info')  # FIXME check if this is still true
        notFound = "(no git repo found)"
        if ensureGit(False):
            scriptName = ''
            runJob('git ls-files --full-name ' +
                   os.path.basename(__file__), '.', False, scriptName, False)
            lastCommit = []
            # get hash of the last commit changing this script
            if runJob('git log -n 1 --pretty=format:%H -- ' + scriptName, '.', False, lastCommit, False):
                return lastCommit[0].strip()
            elif os.path.isfile(commitFn):
                with open(commitFn, 'r') as file:
                    commitInfo = file.readline()
                return commitInfo
            else:
                return notFound
        else:
            return notFound

    def _config_tests(self):
        if self.configDict[Config.OPT_BUILD_TESTS] == "":
            self._set(Config.OPT_BUILD_TESTS, askYesNo(yellow(
                '\nDo you want to build tests [yes/NO]'), default=False, actually_ask=self.ask))
