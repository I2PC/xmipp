import sys, os
import xmippLib
from xmipp3.convert import readPosCoordinates
from pyworkflow.em.data import Coordinate, Micrograph


def writeCoordsListToPosFname(mic_fname, list_x_y, outputRoot):

  s = """# XMIPP_STAR_1 *
#
data_header
loop_
 _pickingMicrographState
Auto
data_particles
loop_
 _xcoor
 _ycoor
"""
  baseName= os.path.basename(mic_fname).split(".")[0]
  print("%d %s %s"%(len(list_x_y), mic_fname, os.path.join(outputRoot, baseName+".pos")))

  if len(list_x_y)>0:
    with open(os.path.join(outputRoot, baseName+".pos"), "w") as f:
        f.write(s)
        for x, y in list_x_y:
          f.write(" %d %d\n"%(x,y) )


        
def readPosCoordsFromFName(fname):
  mData= readPosCoordinates(fname)
  coords=[]
  for mdId in mData:
    x=  int( mData.getValue( xmippLib.MDL_XCOOR, mdId) )
    y=  int( mData.getValue( xmippLib.MDL_XCOOR, mdId) )
    coords.append((x,y) )
  print("N coords: %d"%(len(coords) ))
  return coords 
