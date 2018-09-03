#!/usr/bin/env python


import sys
import os
import shutil
from os.path import dirname, realpath, join, isfile, exists


def usage(error):
    print ("\n"
           "    ERROR: %s\n"
           "\n"
           "    Usage: python tar.py <mode> <version>\n"
           "\n"
           "             mode: Binaries: Just the binaries \n"
           "                   Sources: Just the source code.\n"
           "\n"
           "             version: X.YY.MM  (version, year and month)\n"
           "    ") % error
    sys.exit(1)


def run(label, version):

    XMIPP_PATH = realpath(dirname(dirname(dirname(realpath(__file__)))))
    MODES = {'Binaries': 'build', 'Sources': 'src'}

    def makeTarget(target, label):
        if exists(target):
            print("%s already exists. Removing it...")
            os.system("rm -rf %s" % target)
        print("...preparing the bundle...")
        shutil.copytree(MODES[label], target, symlinks=True)

    excludeTgz = ''
    tgzPath = "xmipp%s-%s"
    if label == 'Binaries':
        print("Recompiling to make sure that last version is there...")
        try:
            os.system("./xmipp compile 4")
            os.system("./xmipp install")
        except:
            print("  ...some error occurred during the compilation.\nFollowing with the bundle creation.")
        target = tgzPath % ('', version)
        if not isfile(join(XMIPP_PATH, 'build', 'bin', 'xmipp_reconstruct_significant')):
            print("\n"
                  "     ERROR: %s not found. \n"
                  "            Xmipp needs to be compiled to make the binaries.tgz."
                  % target)
            sys.exit(1)
        excludeTgz = "--exclude='*.tgz' --exclude='*.h' --exclude='*.cpp' --exclude='*.java'"
        makeTarget(target, label)
    elif label == 'Sources':
        target = tgzPath % ('Src', version)
        os.mkdir(target)
        makeTarget(join(target, 'src'), label)
    else:
        usage("Incorrect <mode>")


    args = {'excludeTgz': excludeTgz,
            'target': target}

    cmdStr = "tar czf %(target)s.tgz --exclude=.git --exclude='software/tmp/*' " \
             "--exclude='*.o' --exclude='*.os' --exclude='*pyc' " \
             "--exclude='*.mrc' --exclude='*.stk' --exclude='*.gz' %(excludeTgz)s" \
             "--exclude='*.scons*' --exclude='config/*.conf' %(target)s"

    cmd = cmdStr % args

    if exists(target+'.tgz'):
        print("%s.tgz already exists. Removing it...")
        os.system("rm -rf %s.tgz" % target)

    print(cmd)
    os.system(cmd)
    os.system("rm -rf %s" % target)


if __name__  == '__main__':

    if len(sys.argv) != 3:
        usage("Incorrect number of input parameters")

    label = sys.argv[1]
    version = sys.argv[2]

    run(label, version)
