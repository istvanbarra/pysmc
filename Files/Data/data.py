# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 08:38:22 2015

@author: istvan
"""

import numpy as np

class Data:
    def Read(self,var,filename):
        '''
        Reading in from CSV file the oobservations and the covariates 
        and storing them as well as the size of them. The first row of 
        the csv file is ingnored \n
        ------------------------------------------------------------\n
        Inputs: \n
        var - string 'y' for the observation or 'z' the covariates \n
        ------------------------------------------------------------\n
        Retruns: \n 
        
        '''
        temp=[] 
        j=0
        with open(filename, 'r') as file:
            for record in file:
                if(j>0):
                    #print record
                    line=record.split(',')
                    size=len(line)
                    line[size-1]=line[size-1].strip()
                    for i in range(0,size):
                        line[i]=float(line[i])
                    temp.append(line)
                j=j+1
        
        if(var=='y'):
            self.y=np.array(temp)
            self.numY=np.size(self.y,0)
            self.dimY=np.size(self.y,1)
        elif(var=='z'):
            self.z=np.array(temp)
            self.numZ=np.size(self.z,0)
            self.dimZ=np.size(self.z,1)