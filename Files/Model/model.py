# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 08:38:23 2015

@author: istvan
"""

import numpy as np
from abc import ABCMeta, abstractmethod
import matplotlib.pyplot as plt


class StaticModel():
    '''
    Static model class
    '''
    
    __metaclass__ = ABCMeta
    
    
    @abstractmethod   
    def LogObs(self,param,state,y,z):
        '''           
        Caclulates the natural logarithm of the observation density 
        at param with covariate z \n
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

   
    def InitialState(self,param):
        '''           
        Calculates the initial state  \n
        ------------------------------------------------------  \n      
        Input: \n            
        param - numTheta x dimParam numpy array \n
        ------------------------------------------------------ \n          
        Returns: \n   
        state - numTheta x dimState numpy array \n   
        '''
        numTheta=np.size(param,0)
        return np.zeros((numTheta,1))
        
     
    def StateTransition(self,param,statePrev,yPrev,zPrev):
        '''           
        Caclulates the next state based on the deterrministic   \n
        transition equation given the previous state and previous obs
        ------------------------------------------------------  \n      
        Input: \n            
        param - numTheta x dimParam numpy array \n 
        statePrev - numTheta x dimState numpy array \n 
        y - 1 x dimY numpy array \n
        z - 1 x dimZ numpy array \n
        ------------------------------------------------------ \n          
        Returns: \n   
        state - numTheta x dimState numpy array \n   
        '''
        return statePrev
        
    def LogLike(self, param, y,z):
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
        numTheta=np.size(param,0)
        logLike=np.zeros((numTheta,1))
        state=self.InitialState(param)  

        for i in range(0,numY):

            logLike=logLike+self.LogObs(param,state,y[i],z[i])
            state=self.StateTransition(param,state,y[i],z[i])        
        

        return logLike,state