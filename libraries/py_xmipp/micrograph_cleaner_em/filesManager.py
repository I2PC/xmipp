import sys, os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore","No training configuration found")
warnings.filterwarnings("ignore","Cannot provide views on a non-contiguous")

try:
  #Import xmipp libraries to read/write files
  import xmippLib
  def loadMic(fname):
    I= xmippLib.Image()      
    I.read( fname )
    return  I.getData()
    
  def writeMic(fname, data):
    I= xmippLib.Image()      
    I.setData( data )
    I.write(fname)
     
except ImportError:
  #Import default libraries to read/write files  
  import mrcfile
  from skimage.io import imsave, imread

  warnings.filterwarnings("ignore", "Unrecognised machine stamp" )
  warnings.filterwarnings("ignore","Map ID string not found")

  def loadMic(fname):
    print(fname)
    if os.path.basename(fname).split(".")[-1].startswith("mrc"):
      with mrcfile.open(fname, permissive=True) as mrc:
        micData= np.squeeze( mrc.data)
    else:
      micData= np.squeeze( imread(fnameIn))
    return  micData

  def writeMic(fname, data):
    if data.dtype== np.float64:
      data= data.astype(np.float32)
    print(data.shape)
    if os.path.basename(fname).split(".")[-1].startswith("mrc"):
      with mrcfile.new(fname, overwrite=True) as mrc:
        mrc.set_data(data)
    else:
      imsave(fname, data)
      
  
def loadCoordsPandas(fname):
  with open(fname) as f:
    line= f.readline()
  if ("x" in line and "y" in line) or ("xcoor" and "ycoor" in line):
    colNames=True
  else:
    colNames=False
  if colNames==True:      
    coords= pd.read_csv(fname, sep="\s+", header=0)
  else:
    coords= pd.read_csv(fname, sep="\s+", header=None)
    print("No header found in %f, assuming first column is x and y column is y"%(fname) )
    coords.columns= ["xcoor", "ycoord"]+["c%d"%i for i in range(coords.shape[1]-2)]
  return coords
  

def parseStr(i):
  try:
    return int(i)
  except ValueError:
    return float(i)
    
def loadCoordsPos_Star(fname):
  newSectionCounter=-1
  coordsColumns=[None, None]
  dataHeader=[]
  coords=[]
  with open(fname) as f:
    for line in f:
      if line.strip().startswith("loop_"):
        newSectionCounter=0
        dataHeader=[]
      elif newSectionCounter>=0:
        splitLine= line.split()
        if len(splitLine)==0:
          continue
        elif len(splitLine)>=2 and splitLine[0][0].isdigit():
          coords.append( [ parseStr(elem) for elem in splitLine] )
        else:
          dataHeader.append(line.strip().strip("_"))      
        newSectionCounter+=1
  assert "ycoor" in dataHeader or "y" in dataHeader or "rlnCoordinateY #2" in dataHeader, "Error, input format not understood for %s"%(fname)
  coords= pd.DataFrame(coords)
  coords.columns= dataHeader
  return coords
  
def loadCoords(fname, downFactor):
  if fname.endswith(".pos") or fname.endswith(".star"):
    coordsDf= loadCoordsPos_Star(fname)
  else:
    coordsDf= loadCoordsPandas(fname)
  if downFactor!=1:
    coordsColNames= getCoordsColNames(coordsDf)
    coordsDf[coordsColNames]/=downFactor
  return coordsDf
  

def writeCoords(fname, coordsDf, upFactor):

  if upFactor!=1:
    coordsColNames= getCoordsColNames(coordsDf)
    coordsDf[coordsColNames]*=upFactor

  if fname.endswith(".pos"):
    coordsDf[coordsColNames]= coordsDf[coordsColNames].round().astype(int)
    writeCoordsPos_Star(fname,  coordsDf, xmipp_instead_relion=True  )
  elif fname.endswith(".star"):
    writeCoordsPos_Star(fname,  coordsDf, xmipp_instead_relion=False  )
  else:
    writeCoordsPandas(fname,  coordsDf)
  return coordsDf
  
def writeCoordsPandas(fname,  coordsDf):
  coordsDf.to_csv(fname, sep="\t", index=False, header=True)



def writeCoordsPos_Star(fname, coordsDf, xmipp_instead_relion=True):

  xmipp_header = """# XMIPP_STAR_1 *
#
data_header
loop_
 _pickingMicrographState
Auto
data_particles
loop_
"""

  relion_header = """# RELION; version 3.0-beta-2

data_

loop_
"""
  if xmipp_instead_relion:
    s= xmipp_header
    pattern= "\t%s"
  else:
    s= relion_header
    pattern= "%13s"
  template=""
  colNames= list(coordsDf.columns)
  for colName in colNames:
    s+="_"+str(colName)+"\n"
    template+=pattern
  template+="\n"
  if coordsDf.shape[0]>0:
    with open(fname, "w") as f:
      f.write(s)
      for row in coordsDf.itertuples():
        row= [ "%4.6f"%y if isinstance(y, float) else y for y in row[1:]  ]
        f.write(template%tuple(row) )
        
def getCoordsColNames(coordsDf):
  if"xcoor" in coordsDf:
    return ["xcoor","ycoor"]
  elif "x" in coordsDf:
    return ["x","y"]
  else:
    return coordsDf.columns[:2]


if __name__=="__main__":
  coords= loadCoords("/home/rsanchez/tmp/20170629_00021_frameImage_aligned_mic_autopick.star", 1)
  print( coords.head())
  writeCoords("/home/rsanchez/tmp/new_20170629_00021_frameImage_aligned_mic_autopick.star", coords, 1)
