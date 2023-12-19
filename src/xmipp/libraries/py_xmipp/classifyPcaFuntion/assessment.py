#!/usr/bin/env python3
"""/***************************************************************************
 *
 * Authors:    Erney Ramirez-Aportela
 *
  ***************************************************************************/
"""
import numpy as np
import pandas as pd
import starfile
import torch

class evaluation:
    
    def __init__(self):
        torch.cuda.is_available()
        torch.cuda.current_device()
        self.cuda = torch.device('cuda:0')
    
        #for experimental images with starfile module
    def updateExpStar(self, expStar, refClas, angle, shiftVec, output):
        
        star = starfile.read(expStar)
        new = output + "_images.star"  
    
        star.loc[:,"ref"] = refClas.int()+1
        
        columns = ["anglePsi", "angleRot", "angleTilt", "shiftX", "shiftY", "shiftZ"]
        for column in columns:
            # if column not in star.columns:
            star[column] = 0.0
    
        # Updating columns in the dataframe
        star.loc[:, "anglePsi"] = angle
        star.loc[:, "shiftX"] = shiftVec[:,0].cpu().detach().numpy()
        star.loc[:, "shiftY"] = shiftVec[:,1].cpu().detach().numpy()
    
        starfile.write(star, new, overwrite=True)
        
        
        
    def createClassesStar(self, numClasses, classFile, count, output):
        
        data = {"ref": range(1, numClasses + 1),
                "image": [f"{i:06d}@{classFile}" for i in range(1, numClasses + 1)],
                "classCount": count}
        
        df = pd.DataFrame(data)

        newStar = output + "_classes.star" 
        starfile.write(df, newStar, overwrite=True)
        
        

        
        
        


 