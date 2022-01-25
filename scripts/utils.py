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

import subprocess
from os import environ, path, remove
import distutils.spawn
import glob
from shutil import which
from os.path import realpath


def green(text):
    return "\033[92m "+text+"\033[0m"


def yellow(text):
    return "\033[93m " + text + "\033[0m"


def red(text):
    return "\033[91m "+text+"\033[0m"


def blue(text):
    return "\033[34m "+text+"\033[0m"


def printXmippLogo(indentation):
    print(indentation, r'''    ____________ ''')
    print(indentation, r'''  / \\         // \ ''')
    print(indentation, r''' /   \\       //   \ ''')
    print(indentation, r'''|     \\     //     | ''')
    print(indentation, r'''|      ''', end='')
    print(yellow('xmipp'), end='')
    print('''       | ''')
    print(indentation, r'''|     //     \\     | ''')
    print(indentation, r''' \   //       \\   / ''')
    print(indentation, r'''  \ //_________\\ / ''')



def find_newest(program, versions, show):
    for v in versions:
        p = program + '-' + str(v) if v else program
        loc = find(p)
        if loc:
            if show:
                print(green(p + ' found in ' + loc))
            return loc
    if show:
        print(red(program + ' not found'))
    return ''


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


def runJob(cmd, cwd='./', show_output=True, log=None, show_command=True,
           in_parallel=False):
    str_out = []
    if show_command:
        print(green(cmd))
    p = subprocess.Popen(cmd, cwd=cwd, env=environ, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, shell=True)
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
    elif 0 == p.poll():
        return True
    else:
        if show_output is False and log is None:
            print(yellow(''.join(str_out)))
            print('\n')
        return False


def whereis(program, findReal=False, env=None):
    programPath = distutils.spawn.find_executable(program, path=env)
    if programPath:
        if findReal:
            programPath = path.realpath(programPath)
        return path.dirname(programPath)
    else:
        return None


def checkProgram(programName, show=True):
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
    ok = True
    cont = True
    if not whereis(programName):
        if cont:
            if show:
                print(red("Cannot find '%s'." % path.basename(programName)))
                idx = 0
                if programName in systemInstructions:
                    if osId >= 0:
                        print(red(" - %s OS detected, please try: %s"
                                  % (systems[osId],
                                     systemInstructions[programName][osId])))
                    else:
                        print(red("   Do:"))
                        for instructions in systemInstructions[programName]:
                            print(red("    - In %s: %s" %
                                  (systems[idx], instructions)))
                            idx += 1
                    print("\nRemember to re-run './xmipp config' after install new software in order to "
                          "take into account the new system configuration.")
            ok = False
        else:
            ok = False
    return ok


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
    result = runJob('echo "int main(){}" > xmipp_check_lib.cpp ; ' +
                    gxx + ' ' + libFlag + ' xmipp_check_lib.cpp',
                    show_output=False, show_command=False)
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
    if not checkProgram('git', critical):
        if critical or path.isdir('.git'):
            # .git dir found means devel mode, which needs git
            print(red("Git not found."))
            exit(-1)
        else:
            return False
    return True


def isGitRepo(path='./'):
    return runJob('git rev-parse --git-dir > /dev/null 2>&1', cwd=path,
                  show_command=False, show_output=False)

