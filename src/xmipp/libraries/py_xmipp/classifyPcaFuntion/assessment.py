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
    def updateExpStar(self, expStar, refClas, output):
        
        star = starfile.read(expStar)
        new = output + "_images.star"  
    
        # star.loc[:, "ref"] = refClas
        star.loc[:,"ref"] = refClas.int()+1
    
        starfile.write(star, new, overwrite=True)
        
        
    def createClassesStar(self, numClasses, classFile, count, output):
        
        data = {"ref": range(1, numClasses + 1),
                "image": [f"{i:06d}@{classFile}" for i in range(1, numClasses + 1)],
                "classCount": count}
        
        df = pd.DataFrame(data)

        newStar = output + "_classes.star" 
        starfile.write(df, newStar, overwrite=True)
        
        
    def createClassesStar3(self, numClasses, classFile, count, output):
        # Crea un DataFrame para almacenar los datos
        
        file = output + "_images.star" 
        star = starfile.read(file)
        
        data1 = [
            ["# class xmipp python"],
            ["data_classes"],
            ["loop_"],
            ["_class"],   
            ["_image"],
            ["_classCount"],
        ]
        
        df1 = pd.DataFrame(data1)
        
        data2 = {"class": range(1, numClasses + 1),
                "image": [f"{i:06d}@{classFile}" for i in range(1, numClasses + 1)],
                "classCount": count}
        
        df2 = pd.DataFrame(data2)
        
        df3 = pd.DataFrame(star)
        selected_rows = []
        
        for i in range(1, numClasses + 1):
            # column_names = ['_' + col for col in df3.columns]
            # header_df = pd.DataFrame(column_names)
            
            # Agregar "loop_" al principio de cada conjunto
            loop_df = pd.DataFrame([["loop_"]])
            
            data_line = [f"data_class{i:06d}_images"]
            data_line_df = pd.DataFrame([data_line])
            
            # Agregar las filas seleccionadas al DataFrame
            filtered_rows = df3[df3['ref'] == i]
            
            # Combinar todos los DataFrames en el orden correcto
            # selected_df = pd.concat([data_line_df, loop_df, header_df, filtered_rows], ignore_index=True)
            selected_df = pd.concat([data_line_df, loop_df, filtered_rows], ignore_index=False)

            # Agregar el DataFrame seleccionado al conjunto total
            selected_rows.append(selected_df)
    
        # Combinar los DataFrames usando pd.concat()
        # combined_df = pd.concat([df1, df2] + selected_rows, ignore_index=True)
        combined_df = pd.concat([df1, df2] + selected_rows, ignore_index=False)

        
        # Escribe el DataFrame en un archivo STAR
        newStar = output + "_classes.star"
        combined_df.to_csv(newStar, sep='\t', index=False, header=False)
        
        
        
        
    def createClassesStar2(self, numClasses, classFile, count, output):
        # Crea un DataFrame para almacenar los datos
        
        file = output + "_images.star" 
        star = starfile.read(file)

        
        data = {"class": range(1, numClasses + 1),
                "image": [f"{i:06d}@{classFile}" for i in range(1, numClasses + 1)],
                "classCount": count}
        
        df = pd.DataFrame(data)
        
        filtered_rows = []  
        for i in range(1, numClasses + 1):
            rows_for_i = star[star['ref'] == i]
            filtered_rows.append(rows_for_i)

        combined_df = ([df]) + filtered_rows
        # resultado = ([df2, filtered_rows])

        newStar = output + "_classes.star" 
        starfile.write(combined_df, newStar, overwrite=True)

 