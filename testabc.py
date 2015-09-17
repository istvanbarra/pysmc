# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 08:15:13 2015

@author: istvan
"""

from abc import ABCMeta, abstractmethod

class StaticModel():
    __metaclass__ = ABCMeta
    @abstractmethod   
    def LogObs(self,y):
        pass

class DynamicModel(StaticModel):
    __metaclass__ = ABCMeta
    @abstractmethod   
    def LogObs(self,y,x):
        pass
    @abstractmethod   
    def LogTrans(self,x):
        pass    
      
class StaticModelExample(StaticModel):
    def LogObs(self,y):
        print 'StaticModel '+str(y)

class DynamicModelExample(DynamicModel): 
    def LogTrans(self,x):
        return x+1    
    
    def LogObs(self,y,x):
        print 'DynamicModel '+str(y)+' '+str(x)  

class EstimationMethod():
    def __init__(self,model,y):
        self.model=model
        self.y=y
        if(str(model.__class__.__base__).find('DynamicModel')!=-1):
            print 'It is a dynamic model'            
            self.x=0
        elif(str(self.model.__class__.__base__).find('StaticModel')!=-1):
            print 'It is a static model'
    
    def Estimate(self):
        if(str(self.model.__class__.__base__).find('StaticModel')!=-1):
            self.model.LogObs(self.y)
        elif(str(self.model.__class__.__base__).find('DynamicModel')!=-1):
            self.model.LogObs(self.y,self.x)
            self.x=self.model.LogTrans(self.x)
            print 'new x '+str(self.x)
        else:
            'model probably inherited from wrong class'
        
        

        
y=1        
modelStatic=StaticModelExample()
estimationStatic=EstimationMethod(modelStatic,y)
estimationStatic.Estimate()

modelDynamic=DynamicModelExample()
estimationDynamic=EstimationMethod(modelDynamic,y)
estimationDynamic.Estimate()


