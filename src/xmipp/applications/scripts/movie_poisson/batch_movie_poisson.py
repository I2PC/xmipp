#!/usr/bin/env python2


import numpy as np
import sys
import xmippLib
import glob
from scipy.stats import chisquare, poisson


if __name__=="__main__":
    dirMov = sys.argv[1]
    extMov = sys.argv[2]
    dosePerFrame = sys.argv[3]
    samplingRate = sys.argv[4]

    listMov = glob.glob(str(dirMov)+'*.'+str(extMov))
    listMov.sort()
    numBins=20

    for i,fnMov in enumerate(listMov): 
        print("Processing ", fnMov)       
        mov = xmippLib.Image()
        mov.read(fnMov)
        movnp = mov.getData()
        frames, _, x, y = movnp.shape
        if i==0:
            histTot = np.zeros((numBins, frames*len(listMov)),dtype=int)

        Inp=np.zeros((x,y),dtype=int)
        lambdaEst = float(dosePerFrame) * (float(samplingRate) ** 2)
        for f in range(frames):
            Inp[:,:] = movnp[f,:,:,:]
            hist, bins = np.histogram(Inp, bins=range(0, numBins))
            histTot[0:len(hist),i*frames+f] = hist
            lambdaExp = float(sum(hist*bins[0:-1]))/float(sum(hist))
            h, p = chisquare(hist/float(sum(hist)), f_exp=poisson.pmf(range(numBins-1), lambdaExp))
            if lambdaExp<lambdaEst-(lambdaEst*0.25) or lambdaExp>lambdaEst+(lambdaEst*0.25):
                print("Anormal dose: check frame %i in movie %s " %(f,fnMov))
                print("Estimated lambda %f, experimental lambda %f" % (lambdaEst, lambdaExp))
            if p<0.05:
                print("The experimental data does not follow a Poisson distribution. Frame %i in movie %s " %(f,fnMov))
                print("h %f, p %f"%(h, p))
    np.savetxt(dirMov+'/histMatrix.csv', histTot, delimiter=' ')
