#!/usr/bin/env python
# **************************************************************************
# *
# * Authors:      David Maluenda (dmaluenda@cnb.csic.es)  
# *      based on J.M. De la Rosa Trevin (jmdelarosa@cnb.csic.es)
# *
# * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
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
# *
# **************************************************************************

import os

import time
import unittest
import threading
import subprocess
import sys
import shutil
from traceback import format_exception

VAHID = "vahid"
RM = 'rmarabini'
COSS = 'coss'
JMRT = 'delarosatrevin'
JOTON = 'joton'
DISCONTINUED = 'nobody'
JMOTA = 'javimota'


class Command(object):
    def __init__(self, cmd, env=None):
        self.cmd = cmd
        self.process = None
        self.env = env

    def run(self, timeout):
        # type: (object) -> object
        def target():
            self.process = subprocess.Popen(self.cmd, shell=True, env=self.env)
            self.process.communicate()

        thread = threading.Thread(target=target)
        thread.start()

        thread.join(timeout)
        if thread.is_alive():
            print(red('ERROR: timeout reached for this process'))
            self.process.terminate()
            thread.join()
        self.process = None
    
    def terminate(self):
        if self.process != None:
            self.process.terminate()
            print('Ctrl-c pressed, aborting this test')
            
            
class ProgramTest(unittest.TestCase):
    _testDir = None
    _environ = None
    _timeout = 300

    # _labels = [WEEKLY]
               
    @classmethod
    def setTestDir(cls, newTestDir):
        cls._testDir = newTestDir
    
    @classmethod
    def setEnviron(cls, newEnviron):
        cls._environ = newEnviron
        
    @classmethod
    def setTimeOut(cls, newTimeOut):
        cls._timeout = newTimeOut
    
    def _parseArgs(self, args):
        ''' Expand our tags %o, %p and %d with corresponding values '''
        args = args.replace("%o", self.outputDir)
        args = args.replace("%p", self.program)
        #args = args.replace("%d", self.fnDir)
        return args     
    
    def _runCommands(self, cmdList, cmdType):
        """ Run several commands.
        Params:
            cmdList: the list of commands to execute. 
            cmdType: either 'preruns' or 'postruns'
        """
        pipe = '>'
        outDir = self.outputDir
        for cmd in cmdList:
            if cmd:
                cmd = self._parseArgs(cmd)
                cmd = " %(cmd)s %(pipe)s %(outDir)s/%(cmdType)s_stdout.txt 2%(pipe)s %(outDir)s/%(cmdType)s_stderr.txt" % locals()
                print("    Running %s: %s" % (cmdType, blue(cmd)))
                command = Command(cmd, env=os.environ)
                command.run(timeout=self._timeout)
                pipe = ">>"
                sys.stdout.flush()
                
    def runCase(self, args, mpi=0, changeDir=False, 
                preruns=None, postruns=None, validate=None,
                outputs=None, random=False, errorthreshold=0.001):
        # Retrieve the correct case number from the test name id
        # We asumme here that 'test_caseXXX' should be in the name
        caseId = unittest.TestCase.id(self)
        if not 'test_case' in caseId:
            raise Exception("'test_case' string should be in the test function name followed by a number")
        _counter = int(caseId.split('test_case')[1])

        self._testDir = os.environ.get("XMIPP_TEST_DATA")
        self.outputDir = os.path.join('tmpLink', '%s_%02d' % (self.program, _counter))
        self.outputDirAbs = os.path.join(self._testDir, self.outputDir)
        self.goldDir = os.path.join(self._testDir, 'gold', '%s_%02d' % (self.program, _counter))
        
        # Change to tests root folder (self._testDir)
        cwd = os.getcwd()
        os.chdir(self._testDir)
        # Clean and create the program output folder if not exists
        createDir(self.outputDir, clean=True)

        if preruns:
            self._runCommands(preruns, 'preruns')
            
        if mpi:
            cmd = "mpirun -np %d `which %s`" % (mpi, self.program)
        else:
            cmd = self.program
        
        args = self._parseArgs(args)
        
        if changeDir:
            cmd = "cd %s ; %s %s > stdout.txt 2> stderr.txt" % (self.outputDir, cmd, args)
        else:
            
            cmd = "%s %s > %s/stdout.txt 2> %s/stderr.txt" % (cmd, args, self.outputDir, self.outputDir)
        print("    Command: ")
        print("       ", blue(cmd))
        sys.stdout.flush()
        #run the test itself
        command = Command(cmd, env=os.environ)
        self._command = command
        try:
            command.run(timeout=self._timeout)
        except KeyboardInterrupt:
            command.terminate()

        stderrFn = "%s/stderr.txt" % self.outputDir
        if os.path.exists(stderrFn):
            errFile = open(stderrFn, 'r')
            errStr = errFile.read()
            errFile.close()
            if 'XMIPP_ERROR' in errStr:
                print(errStr)

        if postruns:
            self._runCommands(postruns, 'postruns')
            
        if outputs:
            self._checkOutputs(outputs, random, errorthreshold=errorthreshold)
            
        if validate:
            validate()
            
        os.chdir(cwd)
        
    def _checkOutputs(self, outputs, random=False, errorthreshold=0.001):
        """ Check that all output files are produced
        and are equivalent to the ones in goldStandard folder.
        """
        import xmippLib
        for out in outputs:
            outFile = os.path.join(self._testDir, self.outputDir, out)
            fileGoldStd = os.path.join(self.goldDir, out)
            
            # Check the expect output file was produced
            msg = "Missing expected output file:\n  output: %s" % outFile
            self.assertTrue(os.path.exists(outFile), red(msg))
            
            if random:
                print(yellow("WARNING: %s was created using a random seed, check skipped..." % outFile))
            else:
                fnGoldStd = xmippLib.FileName(fileGoldStd)
                if fnGoldStd.isImage():
                    im1 = xmippLib.Image(fileGoldStd)
                    im2 = xmippLib.Image(outFile)
                    msg = "Images are not equal (+-%f):\n  output: %s\n  gold: %s" % \
                          (errorthreshold, outFile, fileGoldStd)
                    self.assertTrue(im1.equal(im2, errorthreshold), red(msg))
                elif fnGoldStd.isMetaData():
                    msg = "MetaDatas are not equal:\n  output: %s\n  gold: %s" % (outFile, fileGoldStd)
                    self.assertTrue(xmippLib.compareTwoMetadataFiles(outFile, fileGoldStd), red(msg))
                else:
                    msg = "Files are not equal:\n  output: %s\n  gold: %s" % (outFile, fileGoldStd)
                    self.assertTrue(xmippLib.compareTwoFiles(outFile, fileGoldStd, 0), red(msg))


class GTestResult(unittest.TestResult):
    """ Subclass TestResult to output tests results with colors (green for success and red for failure)
    and write a report on an .xml file.
    """
    xml = None
    testFailed = 0
    numberTests = 0

    def __init__(self):
        unittest.TestResult.__init__(self)
        self.startTimeAll = time.time()

    def openXmlReport(self, classname, filename):
        # self.xml = open(filename, 'w')
        # self.xml.write('<testsuite name="%s">\n' % classname)
        pass

    def doReport(self):
        secs = time.time() - self.startTimeAll
        sys.stderr.write("%s run %d tests (%0.3f secs)\n" %
                         (green("[==========]"), self.numberTests, secs))
        if self.testFailed:
            sys.stderr.write(red("[  FAILED  ]") + " %d tests" % self.testFailed)
        sys.stderr.write(green("[  PASSED  ]") + " %d tests" % (self.numberTests - self.testFailed))
        sys.stdout.flush()
        # self.xml.write('</testsuite>\n')
        # self.xml.close()

    def tic(self):
        self.startTime = time.time()

    def toc(self):
        return time.time() - self.startTime

    def startTest(self, test):
        self.tic()
        self.numberTests += 1

    def getTestName(self, test):
        parts = str(test).split()
        name = parts[0]
        parts = parts[1].split('.')
        classname = parts[-1].replace(")", "")
        return "%s.%s" % (classname, name)

    def addSuccess(self, test):
        secs = self.toc()
        sys.stderr.write("%s %s (%0.3f secs)\n\n" % (green('[ RUN   OK ]'), self.getTestName(test), secs))

    def reportError(self, test, err):
        sys.stderr.write("\n%s" % ("".join(format_exception(*err))))
        sys.stderr.write("%s %s\n\n" % (red('[  FAILED  ]'),
                                      self.getTestName(test)))
        self.testFailed += 1

    def addError(self, test, err):
        self.reportError(test, err)

    def addFailure(self, test, err):
        self.reportError(test, err)

def green(text):
    return "\033[92m "+text+"\033[0m"

def red(text):
    return "\033[91m "+text+"\033[0m"

def blue(text):
    return "\033[34m "+text+"\033[0m"

def yellow(text):
    return "\033[93m "+text+"\033[0m"

def createDir(dirname, clean=False):
    if clean and os.path.exists(dirname):
        shutil.rmtree(dirname)
    os.makedirs(dirname)


def visitTests(tests, grepStr=''):
    """ Show the list of tests available """

    # First flatten the list of tests.
    testsFlat = []
    toCheck = [t for t in tests]
    while toCheck:
        test = toCheck.pop()
        if isinstance(test, unittest.TestSuite):
            toCheck += [t for t in test]
        else:
            if grepStr in str(type(test)):
                testsFlat.append(test)
    # testsFlat.sort()

    # Follow the flattened list of tests and show the module, class
    # and name, in a nice way.
    lastClass = None
    lastModule = None
    
    grepPrint = '' if grepStr is '' else red(' (grep: %s)'%grepStr)

    for t in testsFlat:
        moduleName, className, testName = t.id().rsplit('.', 2)
        
        # If there is a failure loading the test, show it
        if moduleName.startswith('unittest.loader.ModuleImportFailure'):
            print(red(moduleName), "  test:", t.id())
            continue

        if moduleName != lastModule:
            lastModule = moduleName
            print(" - From  %s.py (to run all use --allPrograms)"
                  % '/'.join(moduleName.split('.')) + grepPrint)


        if className != lastClass:
            lastClass = className
            print("  ./xmipp test %s" % className)


if __name__ == "__main__":

    cudaTests = True
    for i, arg in enumerate(sys.argv):
        if arg == '--noCuda':
            cudaTests = False
            sys.argv.pop(i)

    testNames = sys.argv[1:]

    cudaExcludeStr = '| grep -v xmipp_test_cuda_' if not cudaTests else ''
    cTests = subprocess.check_output('compgen -ac | grep xmipp_test_ %s' % cudaExcludeStr,
                                     shell=True, executable='/bin/bash').decode('utf-8').splitlines()

    tests = unittest.TestSuite()
    if '--show' in testNames or '--allPrograms' in testNames:
        # tests.addTests(unittest.defaultTestLoader.discover(os.environ.get("XMIPP_TEST_DATA")+'/..',
        #                pattern='test*.py'))#,top_level_dir=os.environ.get("XMIPP_TEST_DATA")+'/..'))
        listDir = os.listdir(os.environ.get("XMIPP_TEST_DATA")+'/..')
        # print listDir
        for path in listDir:
            if path.startswith('test_') and path.endswith('.py'):
                tests.addTests(unittest.defaultTestLoader.loadTestsFromName('tests.' + path[:-3]))

        if '--show' in testNames:
            print(blue("\n    > >  You can run any of the following tests by:\n"))
            grepStr = '' if len(testNames)<2 else testNames[1]
            visitTests(tests, grepStr)
            print("\n - From applications/function_tests (to run all use --allFuncs):")
            for test in cTests:
                print("  %s" % test)
        elif '--allPrograms' in testNames:
            result = GTestResult()
            tests.run(result)
            result.doReport()
    elif '--allFuncs' in testNames:
        xmippBinDir = os.path.join(os.environ.get("XMIPP_SRC"), 'xmipp', 'bin')
        errors = []
        startTimeAll = time.time()
        for test in cTests:
            sys.stdout.write(blue("\n\n>> Running %s:\n" % test))
            sys.stdout.flush()
            result = os.system(test)
            sys.stdout.flush()
            if result != 0:
                errors.append(test)

        secs = time.time() - startTimeAll
        sys.stdout.write(blue("\n\n -- End of all function tests -- \n\n"))
        sys.stdout.write("%s run %d tests (%0.3f secs)\n" %
                         (green("[==========]"), len(cTests), secs))
        sys.stdout.write(green("[  PASSED  ]") + " %d tests \n" % (len(cTests) - len(errors)))
        sys.stdout.flush()
        if errors:
            sys.stdout.write(red("[  FAILED  ]") + " %d tests:\n" % len(errors))
        for fail in errors:
            sys.stdout.write(red("\t* %s\n" % fail))
        sys.stdout.flush()
    else:
        for test in testNames:
            test = 'tests.test_programs_xmipp.' + test
            try:
                tests.addTests(unittest.defaultTestLoader.loadTestsFromName(test))
            except Exception as e:
                print(red('Cannot find test %s -- skipping') % test)
                print('error: %s' % e)

            result = GTestResult()
            tests.run(result)
            result.doReport()