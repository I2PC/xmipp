#!/usr/bin/env python3
# ***************************************************************************
# * Authors:     Carlos Oscar S. Sorzano (coss@cnb.csic.es)
# *              David Maluenda (dmaluenda@cnb.csic.es)
# *              David Strelak (dstrelak@cnb.csic.es)
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
import os.path
import subprocess
import glob
import distutils.spawn
from os import environ, path, remove
from shutil import which
from os.path import realpath
import sys

def green(text):
    return "\033[92m "+text+"\033[0m"

def yellow(text):
    return "\033[93m " + text + "\033[0m"

def red(text):
    return "\033[91m "+text+"\033[0m"

def blue(text):
    return "\033[34m "+text+"\033[0m"

def bold(text):
    return "\033[1m "+text+"\033[0m"


def get_GCC_version(compiler):
    def get_version_tokens(v):
        log = []
        runJob(compiler + v, show_output=False,
               show_command=False, log=log)
        if log[0].find('command not found') != -1:
            return '', ''
        else:
            return log[0].strip(), log[0].strip().split('.')

    full_version, tokens = get_version_tokens(" -dumpversion")
    if full_version == '':
        return
    elif len(tokens) < 2:
        full_version, tokens = get_version_tokens(" -dumpfullversion")
    gccVersion = float(str(tokens[0] + '.' + tokens[1]))
    return gccVersion, full_version

def find_GCC(candidates, minimumGCC='', show=False):
    gccVersion, full_version = get_GCC_version('gcc')
    if gccVersion == '':
        print(red('Not compiler found, please install it. We require gcc/g++ >={}'.
                  format(minimumGCC)))
        return ''
    if str(gccVersion) in candidates:
        log=[]
        runJob('type gcc', log=log, show_output=False, show_command=False)
        if log[0].find('not found') == -1:
            loc = log[0].split(' ')[2]
            if show:
                print(green('gcc {} found for CUDA: {}'.format(full_version, loc)))
            return loc
    else:
        return find_newest('g++', candidates,  False)


def find_newest(program, versions, show=True):
    for v in versions:
        p = program + '-' + str(v) if v else program
        loc = find(p)
        if loc:
            if show:
                print(green(p + ' found in ' + loc))
            return loc
    if show:
        print(yellow(program + ' not found'))
    return ''


def endMessage(XMIPP_VERNAME):
    strXmipp = 'Xmipp {} has been successfully installed, enjoy it!'.format(
        XMIPP_VERNAME)
    lenStr = len(strXmipp)
    border = '*' * (lenStr + 5)
    spaceStr = ' ' * (lenStr + 3)
    print(border)
    print('*' + spaceStr + '*')
    print('* ', end='')
    print(green(strXmipp), end='')
    print(' *')
    print('*' + spaceStr + '*')
    print(border)
    print('\n')


def errorEndMessage(XMIPP_VERNAME, errorNum, status=''):
    if status:
        ErrMsg1 = status[2]
        ErrMsg2 = status[3]
    else:
        ErrMsg1 = errorList(errorNum)[1]
        ErrMsg2 = errorList(errorNum)[2]

    print(('\n---------------------------------------------------------------'))

    print( " Unable to install Xmipp.\n")

    if XMIPP_VERNAME == 'devel':
        print(' Devel version of Xmipp is constantly beeing improved, some errors might appear temporary.' )
        print(red('{}. '.format(ErrMsg1) + '\n {} '.format(ErrMsg2)))
        print(' For more information about the error check compileLOG.txt file if exist.\n'
              ' Developers note: If you have modified code inside Xmipp please check it.')

    else:#release
        print(red('{}. '.format(ErrMsg1) + '\n {} '.format(ErrMsg2)))
        print(' For more information about the error check compileLOG.txt file if exist.\n'
              ' You can visit our guide of installation https://github.com/I2PC/xmipp,\n'
              ' visit the wiki page with some details https://github.com/I2PC/xmipp/wiki \n'
              ' and you can contact us xmipp@cnb.csic.es or Discord/Scipion/xmipp\n'
              'To report the details of the error please share the reportInstallation.tar.gz file')

    print(('---------------------------------------------------------------'))

def find(program, path=[]):
    location = which(program)
    if location:
        return location
    else:
        for p in path:
            location = which(program, path=p)
            if location:
                return realpath(location)
        return None


def binariesPrecompiled(log):
    n = 0
    for l in log:
        if 'is up to date' in str(l):
            n += 1
    if n>20:
        return True
    else:
        return False




def printProgressBar(value, sizeBar=40):#value 0 - 100
    sizeValue = int((value * sizeBar) / 100)
    templateStr = green(str(value) + '%') + green('[') + green('#' * sizeValue) \
                  + yellow('-' * (sizeBar - sizeValue)) + green(']') + (' ' * 50)
    return templateStr

def runJob(cmd, cwd='./', show_output=True, log=None, show_command=True,
           showWithReturn=True, in_parallel=False, sconsProgress=False,
           progresLines=False, progresLinesPrecompiled=False, printProgress=False,
           COMPILE_LOG=''):
    str_out = []
    precompiled = False
    if show_command:
        if showWithReturn == True:
            print(yellow(completeSplitLine(80, cmd)), end='\r')
        else:
            print(green(cmd))
    p = subprocess.Popen(cmd, cwd=cwd, env=environ, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, shell=True)
    n = 0
    while sconsProgress:
        UP = "\x1B[1A\r" #Move the coursor one line up and move to the begin in
        if precompiled == True:
            progresL = progresLinesPrecompiled
        else:
            progresL = progresLines
        line = p.stdout.readline().decode("utf-8")
        if line:
            log.append(line)
            write_compileLog(line, COMPILE_LOG)

            if printProgress == True:
                if n == 30 and binariesPrecompiled(log):
                        precompiled=True
                prg = round((n*100)/progresL)
                if prg > 100: prg = 100
                line = line.replace('\n', "")
                line = line.replace("\r", "")
                line = line.replace("\b", "")
                line = completeSplitLine(70, line)
                str2Print = line + "\n" + printProgressBar(prg)
                print(f"{yellow(str2Print)}", end=UP)
                n += 1
        else:
            if p.wait() == 0:
                if printProgress:
                    print(printProgressBar(100))
                return True
            else:
                print('\n')
                return False


    while not in_parallel:
        output = p.stdout.readline().decode("utf-8")
        if output == '' and p.poll() is not None:
            break
        if output:
            l = output.rstrip()
            if show_output:
                print(l)
            elif log is None:
                str_out.append(l)
            if log is not None:
                log.append(l)
    if in_parallel:
        return p
    elif 0 == p.wait():
        return True
    else:
        if show_output is False and log is None:
            print(yellow(''.join(str_out)))
        return False


def completeSplitLine(sizeLine, line):
    emptyLine = sizeLine - len(line)
    if emptyLine > 0:
        line = line + (" " * emptyLine)
    if emptyLine < 0:
        line = line[:sizeLine - 3] + '...'
    return line

def write_compileLog(log, COMPILE_LOG='', append=True):
    if append ==True:
        HTW = 'a'
    else:
        HTW = 'w'
    with open(COMPILE_LOG, HTW) as logFile:#no imprime con salto de linea ni imprime tod
        logFile.write(log)

def whereis(program, findReal=False, env=None):
    programPath = distutils.spawn.find_executable(program, path=env)
    if programPath:
        if findReal:
            programPath = path.realpath(programPath)
        return path.dirname(programPath)
    else:
        return None


def checkProgram(programName):
    systems = ["Ubuntu/Debian", "ManjaroLinux"]
    try:
        osInfo = subprocess.Popen(["lsb_release", "--id"],
                                  stdout=subprocess.PIPE, env=environ).stdout.read().decode("utf-8")
        osName = osInfo.split('\t')[1].strip('\n')
        osId = -1  # no default OS
        for idx, system in enumerate(systems):
            if osName in system:
                osId = idx
    except:
        osId = -1

    systemInstructions = {}  # Ubuntu/Debian          ;      ManjaroLinux
    systemInstructions["git"] = [
        "sudo apt-get -y install git", "sudo pacman -Syu --noconfirm git"]
    systemInstructions["gcc"] = [
        "sudo apt-get -y install gcc", "sudo pacman -Syu --noconfirm gcc"]
    systemInstructions["g++"] = ["sudo apt-get -y install g++",
                                 "sudo pacman -Syu --noconfirm g++"]
    systemInstructions["mpicc"] = [
        "sudo apt-get -y install libopenmpi-dev", "sudo pacman -Syu --noconfirm openmpi"]
    systemInstructions["mpicxx"] = [
        "sudo apt-get -y install libopenmpi-dev", "sudo pacman -Syu --noconfirm openmpi"]
    systemInstructions["scons"] = [
        'sudo apt-get -y install scons or make sure that Scipion Scons is in the path', "sudo pacman -Syu --noconfirm scons"]
    systemInstructions["javac"] = [
        'sudo apt-get -y install default-jdk default-jre', "sudo pacman -Syu --noconfirm jre"]
    systemInstructions["rsync"] = [
        "sudo apt-get -y install rsync", "sudo pacman -Syu --noconfirm rsync"]
    systemInstructions["pip"] = [
        "sudo apt-get -y install python3-pip", "sudo pacman -Syu --noconfirm pip"]
    systemInstructions["make"] = [
        "sudo apt-get -y install make", "sudo pacman -Syu --noconfirm make"]
    if not whereis(programName):
        strError = "Cannot find '%s'." % path.basename(programName)
        idx = 0
        if programName in systemInstructions:
            if osId >= 0:
                strSupport = " - %s OS detected, please try: %s"% (systems[osId],
                             systemInstructions[programName][osId])
            else:
                strSupport = "   Do:"
                for instructions in systemInstructions[programName]:
                    strSupport += "    - In %s: %s" %(systems[idx], instructions)
                    idx += 1
            strSupport +="\nRemember to re-run './xmipp config' after install " \
                         "new software in order to take into account the new system configuration."
            status = [False, 1, strError, strSupport]
        else:
            status = [False, 0]
        return status
    return True, []


def isCIBuild():
    return 'CIBuild' in environ


def findFileInDirList(fnH, dirlist):
    """ :returns the dir where found or an empty string if not found.
        dirs can contain *, then first found is returned.
    """
    if isinstance(dirlist, str):
        dirlist = [dirlist]

    for dir in dirlist:
        validDirs = glob.glob(path.join(dir, fnH))
        if len(validDirs) > 0:
            return path.dirname(validDirs[0])
    return ''


def checkLib(gxx, libFlag):
    """ Returns True if lib is found. """
    log = []
    result = runJob('echo "int main(){}" > xmipp_check_lib.cpp ; ' +
                    gxx + ' ' + libFlag + ' xmipp_check_lib.cpp',
                    show_output=False, show_command=False, log=log)
    remove('xmipp_check_lib.cpp')
    remove('a.out') if path.isfile('a.out') else None
    return result


def askPath(default='', ask=True):
    question = "type a path where to locate it"
    if ask:
        if default:
            print(yellow("Alternative found at '%s'." % default))
            question = "press [return] to use it or " + question
        else:
            question = question+" or press [return] to continue"
        result = input(yellow("Please, "+question+": "))
        if not result and default:
            print(green(" -> "+default))
        print()
        return result if result else default
    else:
        if default:
            print(yellow("Using '%s'." % default))
        else:
            print(red("No alternative found in the system."))
        return default


def askYesNo(msg='', default=True, actually_ask=True):
    if not actually_ask:
        print(msg, default)
        return default
    r = input(msg)
    return (r.lower() not in ['n', 'no', '0'] if default else
            r.lower() in ['y', 'yes', '1'])


def getDependenciesInclude():
    return ['../']


def installDepConda(dep, askUser):
    condaEnv = environ.get('CONDA_DEFAULT_ENV', 'base')
    if condaEnv != 'base':
        if not askUser or askYesNo(yellow("'%s' dependency not found. Do you want "
                                          "to install it using conda? [YES/no] "
                                          % dep)):
            print(yellow("Trying to install %s with conda" % dep))
            if runJob("conda activate %s ; conda install %s -y -c defaults" % (condaEnv, dep)):
                print(green("'%s' installed in conda environ '%s'.\n" %
                      (dep, condaEnv)))
                return True
    return False


def ensureGit(critical=False):
    status = checkProgram('git')
    if status[0] == False:
        return status
        # if critical or path.isdir('.git'):
        #     # .git dir found means devel mode, which needs git
        #     return False, 7
        # else:
    return True, 0


def isGitRepo(path='./'):
    return runJob('git rev-parse --git-dir > /dev/null 2>&1', cwd=path,
                  show_command=False, show_output=False)

def version_tuple(versionStr):
    """
    This function returns the given version sting ('1.0.7' for example) into a tuple, so it can be compared.
    It also accepts other version schemes, like 1.0.9-rc, but only the numeric part is taken into account.
    """
    # Split the version string by dots
    version_parts = versionStr.split('.')
    # Initialize an empty list to store the numerical parts of the version string
    numerical_parts = []
    # Iterate over each part of the version string
    for part in version_parts:
        # Split the part by hyphens
        subparts = part.split('-')
        # The first subpart is always numerical, so we append it to our list
        numerical_parts.append(int(subparts[0]))
    # Convert the list of numerical parts to a tuple and return it
    return tuple(numerical_parts)


def versionToNumber(strVersion):
    listVersion = strVersion.split('.')
    numberVersion = int(listVersion[0]) * 100 + int(listVersion[1]) * 10
    try:
        numberVersion = numberVersion + int(listVersion[2]) * 1
    except Exception:
        pass
    return numberVersion
def checkCMakeVersion(minimumRequired=None):
    """
    ### This function checks if the current installed version, if installed, is above the minimum required version.
    ### If no version is provided it just checks if CMake is installed.

    #### Params:
    minimumRequired (str): Optional. Minimum required CMake version.

    #### Returns:
    An error message in color red in a string if there is a problem with CMake, None otherwise.
    """
    # Defining link for cmake installation & update guide
    cmakeInstallURL = 'https://github.com/I2PC/xmipp/wiki/Cmake-update-and-install'

    try:
        # Getting CMake version
        outputLog = []
        runJob('cmake --version', show_output=False, log=outputLog, show_command=False)
        result = '\n'.join(outputLog)
        cmakVersion = result.split('\n')[0].split()[-1]

        # Checking if installed version is below minimum required
        if minimumRequired and (version_tuple(cmakVersion) < version_tuple(minimumRequired)):
            return False, [False, 1, f"\033[91mYour CMake version ({cmakVersion}) is below {minimumRequired}. ",
                       f"Please update your CMake version by following the instructions at {cmakeInstallURL}\033[0m"]
        return True, []
    except FileNotFoundError:
        return False, [False, 1, f"\033[91mCMake is not installed",
                   f" Please install your CMake version by following the instructions at {cmakeInstallURL}\033[0m"]
    except Exception:
        return False, [False, 1, "Can not get the cmake version",
        "Please visit https://github.com/I2PC/xmipp/wiki/Cmake-troubleshoting"]

def removeCompileAndReportFile(COMPRESED_REPORT, COMPILE_LOG):
    if os.path.isfile(COMPRESED_REPORT):
        runJob('rm {}'.format(COMPRESED_REPORT), show_command=False)
    if os.path.isfile(COMPILE_LOG):
        runJob('rm {}'.format(COMPILE_LOG), show_command=False)

def errorList(errorNum):
    errorList = [
        #[index, error description, possible solutions]
        [0, 'No error', ''],
        [1, 'Specific error. Sentence overwritten', 'Check the xmipp.conf flags'],
        [2, 'Version of compiler is not implemented', 'Review the CXX flag in the xmipp.conf'],
        [3, 'Compiler read from xmipp.conf CXX not found', 'Review the CXX flag in the xmipp.conf'],
        [4, 'Version 8.0 or higher of the compiler is required', 'Please go to https://github.com/I2PC/xmipp#compiler to solve it.'],
        [5, 'Some library has fail beeing included', 'Check the INCDIRFLAGS, CXX, CXXFLAGS and PYTHONINCFLAGS in xmipp.conf\n'
                'If some of the libraries headers fail, try installing fftw3_dev, tiff_dev, jpeg_dev, sqlite_dev, hdf5, pthread'],
        [6, 'hdf5 can not be included', 'Check the LINKERFORPROGRAMS, LINKFLAGS and LIBDIRFLAGS in xmipp.conf and visit https://github.com/I2PC/xmipp/wiki/HDF5-Troubleshooting'],
        [7, 'Git not found on the repository', 'Please install git'],
        [8, 'MPI compilation failed', 'Check the INCDIRFLAGS, MPI_CXX and CXXFLAGS in xmipp.conf'
         'In addition, MPI_CXXFLAGS can also be used to add flags to MPI compilations'],
        [9, 'MPI compilation failed', 'Check the LINKERFORPROGRAMS, LINKFLAGS and LIBDIRFLAGS'],
        [10, 'mpi test (mpirun or mpiexec) has faild.', ''],
        [11, 'Java not found in system', ''],
        [12, 'JAVAC does not work', 'Check the JAVAC flag on xmipp.conf'],
        [13, 'JAVA fails. jni include fails', 'Check the JNI_CPPPATH, CXX and INCDIRFLAGS'],
        [14, 'Error git', 'Please review the conexion to the repository'],
        [15, 'Scons package not found', 'Install it in the enviroment you install Xmipp (pip install SCons) or in the system'],
        [16, 'Cannot build cuFFTAdvisor dependence', 'Review the documentation about CUDA and if the error persist \n'
            'you could disable CUDA functionalities with CUDA=False on the xmipp.conf file'],
        [17, 'Cannot build googletest dependence', 'Review the repository has been downloaded correctly. Run /xmipp cleanAll \nto remove all repositories (local changes will be removed) and compile Xmipp from scratch \n'
                                                   'Review the cmake version of your system and review compileLOG.txt file\n for more details about the error '],
        [18, 'Cannot build libsvm dependence', 'Review the repository has been downloaded correctly run ./xmipp cleanAll \nto remove all repositories (local changes would be removed'],
        [19, 'Cannot build libcifpp dependences', 'Review the repository has been downloaded correctly. Review the cmake version of your system. Visit https://github.com/I2PC/xmipp/wiki/Cmake-update-and-install \n'
                                                  'and https://github.com/I2PC/xmipp/wiki/Cmake-troubleshoting ']

    ]
    return errorList[errorNum]
