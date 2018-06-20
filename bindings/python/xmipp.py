from xmippCore import *
from xmippLib import *

class XmippScript():
    ''' This class will serve as wrapper around the XmippProgram class
    to have same facilities from Python scripts'''
    def __init__(self, runWithoutArgs=False):
        self._prog = Program(runWithoutArgs)
        
    def defineParams(self):
        ''' This function should be overwrited by subclasses for 
        define its own parameters'''
        pass
    
    def readParams(self):
        ''' This function should be overwrited by subclasses for 
        and take desired params from command line'''
        pass
    
    def checkParam(self, param):
        return self._prog.checkParam(param)
    
    def getParam(self, param, index=0):
        return self._prog.getParam(param, index)
    
    def getIntParam(self, param, index=0):
        return int(self._prog.getParam(param, index))
    
    def getDoubleParam(self, param, index=0):
        return float(self._prog.getParam(param, index))
    
    def getListParam(self, param):
        return self._prog.getListParam(param)
    
    def addUsageLine(self, line, verbatim=False):
        self._prog.addUsageLine(line, verbatim)

    def addExampleLine(self, line, verbatim=True):
        self._prog.addExampleLine(line, verbatim)
        
    def addParamsLine(self, line):
        self._prog.addParamsLine(line)
    
    def run(self):
        ''' This function should be overwrited by subclasses and
        it the main body of the script'''   
        pass
     
    def tryRun(self):
        ''' This function should be overwrited by subclasses and
        it the main body of the script'''
        try:
            self.defineParams()
            doRun = self._prog.read(sys.argv)
            if doRun:
                self.readParams()
                self.run()
        except Exception:
            import traceback
            traceback.print_exc(file=sys.stderr)

def createMetaDataFromPattern(pattern, isStack=False, label="image"):
    ''' Create a metadata from files matching pattern'''
    import glob
    files = glob.glob(pattern)
    files.sort()

    label = xmipp.str2Label(label) #Check for label value
    
    mD = xmipp.MetaData()
    inFile = xmipp.FileName()
    
    nSize = 1
    for file in files:
        fileAux=file
        if isStack:
            if file.endswith(".mrc"):
                fileAux=file+":mrcs"
            x, x, x, nSize = xmipp.getImageSize(fileAux)
        if nSize != 1:
            counter = 1
            for jj in range(nSize):
                inFile.compose(counter, fileAux)
                objId = mD.addObject()
                mD.setValue(label, inFile, objId)
                mD.setValue(xmipp.MDL_ENABLED, 1, objId)
                counter += 1
        else:
            objId = mD.addObject()
            mD.setValue(label, fileAux, objId)
            mD.setValue(xmipp.MDL_ENABLED, 1, objId)
    return mD            
