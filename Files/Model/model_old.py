# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 08:38:23 2015

@author: istvan
"""

import numpy as np
from abc import ABCMeta, abstractmethod



class StaticModel():
    '''
    Static model class
    '''
    
    __metaclass__ = ABCMeta
    
    
    @abstractmethod   
    def LogObs(self,param,y,z):
        '''           
        Caclulates the natural logarithm of the likelihood at param
        with covariate z \n
        ------------------------------------------------------  \n      
        Input: \n            
        param - numTheta x dimParam numpy array \n 
        y - 1 x dimY numpy array \n
        z - 1 x dimZ numpy array \n
        ------------------------------------------------------ \n          
        Returns: \n   
        logObs - numTheta x 1 numpy array \n   
        '''
        raise NotImplementedError("Should implement LogObsDens")


    def LogLike(self, param,y,z):
        '''           
        Caclulates the natural logarithm of the likelihood at param
        with covariate z \n
        ------------------------------------------------------  \n      
        Input: \n            
        param - numTheta x dimParam numpy matrix \n 
        y - 1 x dimY numpy matrix \n
        z - 1 x dimZ numpy matrix \n
        ------------------------------------------------------ \n          
        Returns: \n   
        logLike - numTheta x 1 numpy array \n   
        '''
        numY=np.size(y,0)
        logLike=0
        if(numY==1):
            logLike=logLike+self.LogObs(param,y,z)
        else:
            for i in range(0,numY):
                logLike=logLike+self.LogObs(param,y[i],z[i])            
            
        return logLike