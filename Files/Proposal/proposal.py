# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 09:42:47 2015

@author: istvan
"""

import numpy as np
import math
from scipy.stats import norm

from abc import ABCMeta, abstractmethod

class Proposal():
    __metaclass__ = ABCMeta  
    @abstractmethod
    def Propose(self,param, Prior):
        pass


def norm_pdf_multivariate(x, mu, sigma):
    size = len(x)
    if size == len(mu) and (size, size) == np.shape(sigma):
        det = np.linalg.det(sigma)
        if det == 0:
            raise NameError("The covariance matrix can't be singular")

        norm_const = 1.0/ ( math.pow((2*np.pi),float(size)/2) * math.pow(det,1.0/2) )
        x_mu = np.matrix(x - mu)
        inv = np.linalg.pinv(sigma)    
        result = math.pow(math.e, -0.5 * (x_mu * inv * x_mu.T))
        return norm_const * result
    else:
        raise NameError("The dimensions of the input don't match")

def LogMultiNorm(x, mu,sigma):
    num=len(x)
    result=np.zeros((num,1))
    for i in range(0,num):
        result[i]=np.log(norm_pdf_multivariate(x[i], mu, sigma))

    return result
    
class IndependentProp(Proposal):
    def Propose(self, param,Prior):
        numTheta=np.shape(param)[0]
        ''' Construct proposal '''
        paramTrans=Prior.TransParam(param)
        meanTrans=np.mean(paramTrans,0)
        varTrans=np.cov(paramTrans, y=None,rowvar=0)
        paramTransNew= np.random.multivariate_normal(meanTrans,varTrans,numTheta)
        paramNew=Prior.TransParamBack(paramTransNew)

        ''' Calculate proposal '''     
        logProp=LogMultiNorm(paramTrans, meanTrans,varTrans)
        logPropNew=LogMultiNorm(paramTransNew, meanTrans,varTrans)    

        return paramNew, logProp, logPropNew
        
        
class RandomWalkProp(Proposal):
    def Propose(self, param,Prior):
        (numTheta,numDim)=np.shape(param)
        ''' Construct proposal '''
        paramTrans=Prior.TransParam(param)
        varTrans=2.38*2.38*np.cov(paramTrans, y=None,rowvar=0)/numDim
        
        paramTransNew= paramTrans+np.random.multivariate_normal(np.zeros(numDim),varTrans,numTheta)
        paramNew=Prior.TransParamBack(paramTransNew)

        ''' Calculate proposal '''
        logProp=np.zeros((numTheta,1))
        logPropNew=np.zeros((numTheta,1))
        
        return paramNew, logProp, logPropNew
   

      