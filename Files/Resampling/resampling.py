# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 09:39:21 2015

@author: istvan
"""
import numpy as np
from abc import ABCMeta, abstractmethod

class Resampling():
    __metaclass__ = ABCMeta  
    @abstractmethod
    def Resample(self,paramCumW):
        pass


class SystemicResampling(Resampling):
    def Resample(self,paramCumW):
        '''
        Systemic resampling \n 
        ------------------------------------------- \n
        Inputs: \n
        paramCumW - cumulative weights
        ------------------------------------------ \n
        Result
        
        '''
        num=len(paramCumW)
        index=[0]*num
        u=np.random.uniform(0,1,1)
        k=0
        for i in range(0,num):
            p=(u+i)/num
            while(paramCumW[k]<p):
                k=k+1
            index[i]=k
         
        return index  