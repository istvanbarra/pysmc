# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 13:43:10 2015

@author: istvan
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.grid_search import GridSearchCV


  
def kde_sklearn(x, x_grid, bandwidth, **kwargs):
    """Kernel Density Estimation with Scikit-learn"""
    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(x[:, np.newaxis])
    log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])

    return np.exp(log_pdf)


def WeightedPercentile(data, weights, percentile):
    '''
    Calculates weighted p in [0,100] percentile
    '''
    p=float(percentile)/100

    numCol=np.shape(data)[1]
    result=[0]*numCol
    
    for i in range(0, numCol):
        ind=np.argsort(data[:,i])
        d=data[ind,i]
        w=weights[ind,0]
        sumw=w[0]
        j=0;
        while( sumw<p):
            j+=1
            sumw+=w[j]
        
        if(j==0):
            result[i]=d[j]
        else:
            sumwPrev=np.sum(w[:j])
            if(sumw==p or d[j]==d[j-1]):
                result[i]=d[j]
            else:
                result[i]=d[j-1]+np.exp(np.log(p-sumwPrev)+np.log(d[j]-d[j-1])-np.log(sumw-sumwPrev))
    return result  

def ToArray(x):
    num=len(x)
    dim=len(x[0])
    y=np.zeros((num,dim))
    for i in range(0,num):
        y[i]=x[i]
        
    return y


class Estimation():
    
    def PlotPath(self):
        low=np.vstack(self.low)
        high=np.vstack(self.high)
        median=np.vstack(self.median)
        x=range(1,len(low)+1)
        for i in range(0,self.dimTheta):   
            fig, ax1 = plt.subplots(subplot_kw={'axisbg':'#EEEEEE','axisbelow':True})
            ax1.grid(color='white', linestyle='-', linewidth=2)    
            lns1=ax1.plot(x, median[:,i], linewidth=1,color='r',  label='median')
            lns2=ax1.plot(x, low[:,i], linewidth=1, color='b',label='95% credible set') 
            ax1.plot(x, high[:,i], linewidth=1, color='b') 
            ax1.fill_between(x, low[:,i],high[:,i], color='b', alpha=.25)
            lns=lns1+lns2
            labs=[l.get_label() for l in lns]        
            ax1.legend(lns,labs, loc=0,prop={'size':10})
            ax1.set_xlabel('Iteration',fontsize=10)
            ax1.set_xlim(1,len(low))
            ax1.set_ylabel('param '+str(i), fontsize=10)
            for spine in ax1.spines.values():
                spine.set_color('#BBBBBB')
            plt.title('Parameter '+str(i))  
            fig.savefig('param_path_'+str(i)+'.png', bbox_inches=0)
            fig.savefig('param_path_'+str(i)+'.pdf', bbox_inches=0)  
    
    def PlotDensity(self):
        for i in range(0,self.dimTheta):     
            thetaGrid = np.linspace(np.amin(self.param[:,i]), np.amax(self.param[:,i]), self.numTheta)
            hpdGrid = np.linspace(self.low[-1][i], self.high[-1][i], self.numTheta)
            bandWith=1.06*np.std(self.param[:,i])*np.power(self.numTheta,-1.0/5) 
            pdf=kde_sklearn(self.param[:,i], thetaGrid, bandWith)
            hpd=kde_sklearn(self.param[:,i], hpdGrid, bandWith)
            pdfMedian=kde_sklearn(self.param[:,i], np.array(self.median[-1][i]).reshape((1 )), bandWith)
            pdfLow=kde_sklearn(self.param[:,i], np.array(self.low[-1][i]).reshape((1 )), bandWith)
            pdfHigh=kde_sklearn(self.param[:,i], np.array(self.high[-1][i]).reshape((1 )), bandWith)
            fig, ax1 = plt.subplots(subplot_kw={'axisbg':'#EEEEEE','axisbelow':True})
            ax1.grid(color='white', linestyle='-', linewidth=2)
            numBins=np.round(1+np.log2(self.numTheta))
            numBins=np.round(2*pow(self.numTheta,1.0/3))
            ax1.hist(self.param[:,i], numBins, fc='gray', histtype='stepfilled', alpha=0.3, normed=True)
            ax1.plot(thetaGrid, pdf, linewidth=1, alpha=0.5,color='black')
            lns1=ax1.axvline(x=self.median[-1][i], ymin=0, ymax=pdfMedian/ax1.get_ylim()[1], color='red',  label='median')
            lns2=ax1.axvline(x=self.low[-1][i], ymin=0, ymax=pdfLow/ax1.get_ylim()[1], color='blue',label='95% credible set')
            ax1.axvline(x=self.high[-1][i], ymin=0, ymax=pdfHigh/ax1.get_ylim()[1], color='blue')
            ymin=ax1.get_ylim()[0]
            ymax=ax1.get_ylim()[1]
            ax1.fill_between(hpdGrid, 0, hpd, hpd > 0, color='b', alpha=.25,label='hpd region')
            ax1.set_ylim([ymin,ymax]) 
            lns=[lns1,lns2]
            labs=[lns1.get_label() ,lns2.get_label() ]        
            ax1.legend(lns,labs, loc=0,prop={'size':10})
            plt.title('Posterior density of parameter '+str(i))  
            ax1.set_xlabel('parameter'+str(i),fontsize=14)
            ax1.tick_params(axis='both', which='major',labelsize=10)
            for spine in ax1.spines.values():
                spine.set_color('#BBBBBB')
            fig.savefig('param_dens_'+str(i)+'.png', bbox_inches=0)
            fig.savefig('param_dens_'+str(i)+'.pdf', bbox_inches=0)

    
class IBIS(Estimation):
    '''
    Iterated bach importance sampling procedure \n 
    ------------------------------------------- \n
    Inputs: \n
    numTheta - number of \theta particles \n
    data - data class \n 
    model - model class \n 
    param - param class  \n
    ------------------------------------------ \n
    Results
    '''
    
    def __init__(self, numTheta,Model, Prior,Resampling,Proposal):
        
        self.numTheta=numTheta        
        
        self.Model=Model
        self.Prior=Prior
        self.Resampling=Resampling
        self.Proposal=Proposal        
        
        self.param=Prior.SamplePrior(numTheta)
        self.paramLogW=np.zeros((numTheta,1))
        self.paramLogLike=np.zeros((numTheta,1))
        self.paramNormW=np.zeros((numTheta,1))
        
        self.dimTheta=np.shape(self.param)[1]   
        
        self.state=Model.InitialState(self.param)
      
        self.dimState=np.shape(self.state)[1]  
        
        
        self.high=[]
        self.low=[]
        self.median=[]
        self.ess=[]
        self.ar=[]
        
        self.yStored=[]
        self.zStored=[]
        

    def ResampleMove(self):
        print 'Resample move step'
        ''' Resample '''
        paramCumW=np.cumsum(self.paramNormW,axis=0)
        index=self.Resampling.Resample(paramCumW)
        self.param=self.param[index]
        self.state=self.state[index]
        self.paramLogLike=self.paramLogLike[index]
        self.paramLogW=np.zeros((self.numTheta,1))
        ''' Move '''
        ''' Proposal '''        
        paramNew, logProp, logPropNew=self.Proposal.Propose(self.param, self.Prior)
        ''' Calculate prior '''
        logPrior=self.Prior.LogPrior(self.param)
        logPriorNew=self.Prior.LogPrior(paramNew)

        
    
        ''' Calculate likelihood '''                            
        y=ToArray(self.yStored)
        z=ToArray(self.zStored)
        paramLogLikeNew, stateNew=self.Model.LogLike(paramNew, y, z)          
        paramLogLikeNew=np.where(np.isnan(paramLogLikeNew)==0, paramLogLikeNew, -np.Inf)

         
        ''' Accept reject '''
        accept=np.zeros((self.numTheta,1))
        logU=np.log(np.random.uniform(0,1,self.numTheta)) 
        for i in range(0,self.numTheta):
            ''' Calculate acceptenc probability '''
            logProb=(paramLogLikeNew[i]+logPriorNew[i]-logPropNew[i])-(self.paramLogLike[i]+logPrior[i]-logProp[i])
            logProb=min(logProb,1)
            if(logU[i] <logProb):
                self.param[i]=paramNew[i]
                self.state[i]=stateNew[i]
                self.paramLogLike[i]=paramLogLikeNew[i]
                accept[i]=1
        self.ar[-1]=np.mean(accept,0)   
        
    def Update(self,y,z):
        
        
        ''' Update weights '''
        logObs=self.Model.LogObs(self.param, self.state, y, z)
        logObs=np.where(np.isnan(logObs) == 0, logObs, -np.Inf)
        self.paramLogW=self.paramLogW+logObs
        self.paramLogLike=self.paramLogLike+logObs
        

       
        
        ''' Update state '''
        self.state=self.Model.StateTransition(self.param,self.state,y,z)
        
        
        ''' Calculate normalized weights '''
        self.paramNormW=np.exp(self.paramLogW-np.amax(self.paramLogW,0))/np.sum(np.exp(self.paramLogW-np.amax(self.paramLogW,0)),0)

        ''' Store stuff '''
        self.ess.append(1/np.sum(np.power(self.paramNormW,2),0))        
        self.high.append(WeightedPercentile(self.param,self.paramNormW,97.5))   
        self.low.append(WeightedPercentile(self.param,self.paramNormW,2.5))  
        self.median.append(WeightedPercentile(self.param,self.paramNormW,50))
        self.ar.append(np.nan)
        self.yStored.append(y)
        self.zStored.append(z)

        ''' Resample move  if necessary '''
        
        if(self.ess[-1]<self.numTheta/2):
            self.ResampleMove()
            
    
