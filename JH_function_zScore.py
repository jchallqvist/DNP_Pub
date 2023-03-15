# -*- coding: utf-8 -*-
"""
@author: Jenny Hallqvist
"""
import pandas as pd
import scipy.stats as ss

def zScore(DF, fCol): # Row = 1, Column = 0
         
    features = DF.iloc[:,fCol:].astype(float) # features to z-score
    zScored = pd.DataFrame(index = DF.index)
    clinData = DF.iloc[:,:fCol] # clinical data etc
    
    i = 0
    for var in features.columns:
        z_var = ss.zscore(features[var], axis=0, ddof=1, nan_policy='omit')
        zScored.insert(i, value = z_var, column = var)
        i = i + 1
        
    zScored_Final = pd.concat([clinData, zScored], axis = 1, ignore_index = False)
    return zScored_Final
