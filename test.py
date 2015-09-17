# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 09:14:46 2015

@author: istvan
"""
import numpy as np
from scipy.stats import norm
from scipy.stats import truncnorm
from pysmc import *
import matplotlib.pyplot as plt

np.random.seed(seed=2)  

restOLS=['', '', '', 'pos']



class OLSPrior(Prior):
    def LogPrior(self, param):
        num=len(param)
        result=np.zeros((num,1)) 
        for i in range(0,num):
            result[i]=np.log(norm.pdf(param[i][0]))+np.log(norm.pdf(param[i][1]))+np.log(norm.pdf(param[i][2]))-param[i][3]                    
                     
        return result
             
        
    def SamplePrior(self,numTheta):
        result=np.zeros((numTheta,4))
        for i in range(0,numTheta):
            result[i][0]=np.random.normal(0, 1)
            result[i][1]=np.random.normal(0, 1)
            result[i][2]=np.random.normal(0, 1)
            result[i][3]=-np.log(np.random.uniform(0,1) ) 
       
        return result 

class OLSModel(StaticModel):
    def LogObs(self, param,state, y,z):
        numTheta=np.size(param,0)
        return (norm.logpdf(np.divide(y-param[:,0]-param[:,1]*z[0]-param[:,2]*z[1],np.sqrt(param[:,3])))-0.5*np.log(param[:,3])).reshape((numTheta,1))
   
   
   
   
restGARCH=['pos','pos','pos']
   
class GARCHPrior(Prior):
    def LogPrior(self,param):
        num=len(param)
        result=np.zeros((num,1)) 
        for i in range(0,num):
#            result[i]=-param[i][0]-param[i][1]-param[i][2] 
            a, b = (0 - 0.9) / 0.5, (1 - 0.9) / 0.5
            result[i]=np.log(7)-7*param[i][2]+truncnorm.logpdf(param[i][1],a, b, loc=0.9, scale=0.5)+np.log(5)-5*param[i][2]
#            result[i]=-param[i][0]-param[i][1]
        return result    
        
    def SamplePrior(self,numTheta):
        result=np.zeros((numTheta,3))
        for i in range(0,numTheta):
            result[i][0]=-np.log(np.random.uniform(0,1) )/7 
#            result[i][1]=-np.log(np.random.uniform(0,1) ) 
            a, b = (0 - 0.9) / 0.5, (1 - 0.9) / 0.5
            result[i][1]=truncnorm.rvs(a, b, loc=0.9, scale=0.5)
            result[i][2]=-np.log(np.random.uniform(0,1) )/5 
        return result 

class GARCHModel(StaticModel):
    def LogObs(self, param,state, y,z):
        numTheta=np.size(param,0)
        logObs=(norm.logpdf(np.divide(y,np.sqrt(state)))-0.5*np.log(state)).reshape((numTheta,1))
        for i in range(0,np.size(param,0)):
            if(param[i][1]+param[i][2]>=1):
                logObs[i]=-np.Inf
       
        return logObs
        
    def InitialState(self,param):
        numTheta=np.size(param,0)
        return np.divide(param[:,0], np.ones((1,numTheta))-param[:,1]-param[:,2]).reshape((numTheta,1))        
        
        
    def StateTransition(self,param,statePrev,yPrev,zPrev):
        numTheta=np.size(param,0)
        return (param[:,0]+np.multiply(statePrev.reshape((1,numTheta)),param[:,1])+yPrev*yPrev*param[:,2]).reshape((numTheta,1))
    
paramTrue=[0.04, 0.8, 0.1]
numSim=2000
y=np.zeros((numSim,1))
f=[]

for i in range(0,numSim):
    if(i==0):
        s=paramTrue[0]/(1-paramTrue[1]-paramTrue[2])
    else:
        
        s=paramTrue[0]+paramTrue[1]*sPrev+paramTrue[2]*y[-1]*y[-1]
    f.append(s)    
    y[i]=np.random.normal(0, 1)*np.sqrt(s)
    sPrev=s

garchPrior=GARCHPrior(restGARCH)
garchModel=GARCHModel()
garchData=Data()
garchData.y=np.array(y).reshape((2000,1))
garchData.numY=2000
garchData.dimY=1
garchData.z=np.zeros((2000,1))
garchData.numZ=2000
garchData.dimZ=1




#start=0.93
#end=0.965
#n=100
#ll=np.zeros((n+1,1))
#xx=np.zeros((n+1,1))
#k=0;
#for i in [start+x*(end-start)/n for x in range(0,n+1)]:
#    param=paramTrue
#    param[1]=i
#    param=np.array([param])
#    xx[k]=i
#    logL, state= garchModel.LogLike(param,garchData.y,garchData.z)
#    ll[k]=logL
#    k+=1
#print ll   
#ll=np.array(ll)   
#plt.plot(xx,ll)   
#plt.show() 



#plt.subplot(2,1,1) 
#plt.plot(np.power(y,2))
#plt.subplot(2,1,2) 
#plt.plot(f,color='red')
#plt.show()
   
#olsPrior=OLSPrior(restOLS)
#olsModel=OLSModel()
#olsData=Data()
#olsData.Read('y','ols_y.csv')
#olsData.Read('z','ols_z.csv')
#
#
#proposal=RandomWalkProp()
##proposal=IndependentProp()
#resampling=SystemicResampling()
#
#
#olsIBIS=IBIS(1000,olsModel, olsPrior, resampling,proposal)
##olsIBIS.Update(olsData.y[0], olsData.z[0])
##olsIBIS.Update(olsData.y[1], olsData.z[1])
##olsIBIS.PlotPath()
#
#
#
#
#for t in range(0,olsData.numY):
#    print '========= iteration '+str(t)+' ============='
#    olsIBIS.Update(olsData.y[t], olsData.z[t])
#    print olsIBIS.ess[-1]
#    print olsIBIS.median[-1]
#    print olsIBIS.ar[-1]
#olsIBIS.PlotPath()
#olsIBIS.PlotDensity()



proposal=IndependentProp()
#proposal=RandomWalkProp()
resampling=SystemicResampling()


garchIBIS=IBIS(1000,garchModel, garchPrior, resampling,proposal)
print np.mean(garchIBIS.param, axis=0)
#garchIBIS.Update(garchData.y[0], garchData.z)
#print garchIBIS.median[-1]
#garchIBIS.Update(garchData.y[1], garchData.z)
#print garchIBIS.median[-1]
#garchIBIS.Update(garchData.y[2], garchData.z)
#print garchIBIS.median[-1]
#garchIBIS.Update(garchData.y[3], garchData.z)
#print garchIBIS.median[-1]




for t in range(0,garchData.numY):
    print '========= iteration '+str(t)+' ============='
    garchIBIS.Update(garchData.y[t], garchData.z[t])

    print garchIBIS.median[-1]
    print garchIBIS.ess[-1]
    print garchIBIS.ar[-1]
    print np.mean(garchIBIS.state, axis=0)
    
garchIBIS.PlotPath()
garchIBIS.PlotDensity()

#logObs=np.array([1, np.nan,2])
#print np.isnan(logObs)
#logObs=np.where(np.isnan(logObs) == 0, logObs, -np.Inf)
#print logObs