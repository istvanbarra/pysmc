# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 08:40:06 2015

@author: istvan
"""

import numpy as np
from abc import ABCMeta, abstractmethod

def ParamPos(x):
    return np.log(x)

def ParamPosBack(x):
    return np.exp(x)
    
def Param01(x):
    return np.exp(x)/(1+np.exp(x))  
    
def Param01Back(x):
    return np.log(x)-np.log(1-x)


class Prior:
    '''
    Parameter class
    Implements the evaluation of the log prior density
    and transformation of the variables base on parameter restrictions
    '''

    __metaclass__ = ABCMeta
   
    def __init__(self, rest):
        for i in range(0,len(rest)):
            if(rest[i] not in ['', 'pos', '01']):
                print('Invalid restriction only! Use pos, 01 or empty string')                
                break
        self.rest=rest
        self.dimParam=len(rest)
    
    @abstractmethod   
    def LogPrior(self, param):
        '''           
        Caclulates the natural logarithm of the prior at param \n
        ------------------------------------------------------  \n      
        Input: \n            
        param - numTheta x numDim numpy array \n   
        ------------------------------------------------------ \n          
        Returns: \n   
        logPrior - numTheta x 1 numpy array \n   
        '''
        raise NotImplementedError("Should implement LogPrior")
    @abstractmethod   
    def SamplePrior(self, numTheta):
        '''           
        Caclulates samples form the prior  \n
        ------------------------------------------------------  \n      
        Input: \n            
        numTheta - number of prior samples  \n   
        ------------------------------------------------------ \n          
        Returns: \n   
        param - numTheta x dimTheta numpy array \n   
        '''
        raise NotImplementedError("Should implement LogPrior")    
        
    def TransParam(self, param):
        '''          
        Transforms param based on the restrictions 
        in res  \n
        ------------------------------------------------------ \n  
        Input: \n            
        param - numTheta x numDim numpy array \n
        ------------------------------------------------------ \n  
        Returns: \n
        paramTrans - numTheta x numDim numpy array of the transformed param
        '''
       
        paramTrans=param.copy()  
        numTheta=np.shape(paramTrans)[0]
        for i in range(0,numTheta):
            for j in range(0, self.dimParam):
                if(self.rest[j]=='pos'):
                    paramTrans[i][j]=ParamPos(param[i][j])
                elif(self.rest[j]=='01'):
                    paramTrans[i][j]=Param01(param[i][j])
                
        return paramTrans        

    def TransParamBack(self, paramTrans):
        '''            
        Transforms back paramTrans based on the restrictions   
        in res. \n 
        ------------------------------------------------------ \n        
        Input:  \n           
        paramTrans - numTheta x numDim numpy array \n 
        ------------------------------------------------------ \n        
        Returns:\n 
        param - numTheta x numDim numpy array of the transformed param
        ''' 
        
        param=paramTrans.copy()  
        
        numTheta=np.shape(paramTrans)[0]
        for i in range(0,numTheta):
            for j in range(0, self.dimParam):
                if(self.rest[j]=='pos'):
                    param[i][j]=ParamPosBack(paramTrans[i][j])
                elif(self.rest[j]=='01'):
                    param[i][j]=Param01Back(paramTrans[i][j])
                    
        return param    