# -*- coding: utf-8 -*-
"""
Created on Thu May 24 20:48:14 2018

ScatData - a data class for opearating scattering data recorded primaily at BioCARS (APS)



@author: Denis Leshchev

todo:
- Make valCheck to demand correct input
- Make valCheck spit out proper Warnings/Messages
- Make the code accept multiple runs
- if only multiple logFiles provided with only one InDir, InDir should be the same for all of the logFiles
- get a temporary mask reader
"""
from pathlib import Path
import time

import numpy as np
import pandas as pd

import pyFAI
import fabio

from matplotlib import pyplot as plt

class ScatData:
    
    def __init__(self, logFile, dataInDir, dataOutDir, maskPath, energy, distance, pixelSize, centerX, centerY):
                
        self.logFile = logFile
        self.dataInDir = dataInDir
        self.dataOutDir = dataOutDir
        self.maskPath = maskPath        
        
        self.energy = energy
        self.wavelength = 12.3984/self.energy*1e-10 # in m
        self.distance = distance*1e-3 # in m
        self.pixelSize = pixelSize
        self.centerX = centerX
        self.centerY = centerY
        
        self.valCheck()
        
        self.getLogData()        
        
        self.getLUT()
        
    def valCheck(self):
        if not Path(self.logFile).is_file():
            print("log file does not exist")
        if not Path(self.dataInDir).is_dir():
            print("input directory does not exist")
        if not Path(self.dataOutDir).is_dir():
            print("output directory does not exist")


    def getLogData(self):
        self.logData = pd.read_csv(self.logFile,
                                       sep = '\t',
                                       header = 18)
        self.logData.rename(columns = {'#date time':'date time'}, inplace = True)
        # Add a column that indicates existance of the files
        self.logData['flagExist'] = False        
        # For those files that exist, we put True flag
        for i, row in self.logData.iterrows():
            if Path(self.dataInDir + self.logData.ix[i,'file']).is_file():
                self.logData.set_value(i,'flagExist' , True)
        
        

    def getLUT(self):
        self.ai = pyFAI.AzimuthalIntegrator(dist = self.distance,
                                            poni1 = self.centerY*self.pixelSize,
                                            poni2 = self.centerX*self.pixelSize,
                                            pixel1 = self.pixelSize,
                                            pixel2 = self.pixelSize,
                                            rot1=0,rot2=0,rot3=0,
                                            wavelength = self.wavelength)
        
        test_image = fabio.open(self.dataInDir + self.logData.ix[0,'file']).data
        mask = fabio.open(self.maskPath).data        
        print('ScatData: LUT initialization')
        startInitTime = time.clock()
        q, dummyI = self.ai.integrate1d(test_image,1000,
                                 correctSolidAngle=True,
                                 polarization_factor=1,
                                 method='lut',
                                 mask = mask,
                                 unit="q_A^-1")
        print('ScatData: LUT initialization took', '%.0f' % round(((time.clock()-startInitTime)*1000), 0),'ms')
        plt.plot(q, dummyI)
        
        
A = ScatData(logFile = '/media/denis/Data/work/Experiments/2017/Ubiquitin/10.log',
             dataInDir = '/media/denis/Data/work/Experiments/2017/Ubiquitin/',
             dataOutDir = '/media/denis/Data/work/Experiments/2017/Ubiquitin/',
             maskPath = '/media/denis/Data/work/Experiments/2017/Ubiquitin/mask_aug2017.tif',
             energy = 11.63,
             distance = 362,
             pixelSize = 82e-6,
             centerX = 1990,
             centerY = 1967)