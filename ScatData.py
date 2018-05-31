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
import ntpath
import time
from datetime import datetime
from collections import namedtuple

import numpy as np
import pandas as pd

import pyFAI
import fabio

from matplotlib import pyplot as plt

class ScatData:
    
    def __init__(self, logFile, dataInDir, dataOutDir, maskPath, energy, distance, pixelSize, centerX, centerY, toff_str = '-5us'):
                
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

        self.toff_str = toff_str
        self.toff = self.time_str2num(toff_str)
        
        self.valCheck()
        
        self.getLogData()    
#        self.getDelaysNum()
#        self.getTimeStamps()

#        self.getDifferences()

        
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
        
        

    def getAIGeometry(self):
        # First, lets get the geometry parameters for integration
        self.AIGeometry = namedtuple('AIGeometry', 'ai mask nPt radialRange')
        self.AIGeometry.ai = pyFAI.AzimuthalIntegrator(dist = self.distance,
                                                       poni1 = self.centerY*self.pixelSize,
                                                       poni2 = self.centerX*self.pixelSize,
                                                       pixel1 = self.pixelSize,
                                                       pixel2 = self.pixelSize,
                                                       rot1=0,rot2=0,rot3=0,
                                                       wavelength = self.wavelength)
        self.AIGeometry.mask = fabio.open(self.maskPath).data
        self.AIGeometry.nPt = 400
        self.AIGeometry.radialRange = [0.0, 4.0]
        
        
    def integrate(self):
        # Now lets get the integration going
        if not hasattr(self, 'AIGeometry'):
            self.getAIGeometry()
            self.total = namedtuple('total','S S_raw normInt delay delay_str timeStamp timeStamp_str scanStamp')        
            nFiles = self.logData.shape[0]        
            self.total.S = np.zeros([self.AIGeometry.nPt, nFiles])
            self.total.S_raw = np.zeros([self.AIGeometry.nPt, nFiles])
            self.total.delay, self.total.delay_str = self.getDelays()
            self.total.timeStamp, self.total.timeStamp_str = self.getTimeStamps()
            self.total.scanStamp = [ntpath.splitext(ntpath.basename(self.logFile))[0]]*nFiles
        
        idxIm = 1
        print('*** Integration ***')
        for i,file in enumerate(self.logData['file']):
            path = (self.dataInDir + file)
            startReadTime = time.clock()         
            image = fabio.open(path).data
            readTime = time.clock() - startReadTime
            
            startIntTime = time.clock()
            q, self.total.S[:,i] = self.AIGeometry.ai.integrate1d(
                                    image,
                                    self.AIGeometry.nPt,
                                    radial_range = self.AIGeometry.radialRange,
                                    correctSolidAngle = True,
                                    polarization_factor = 1,
                                    method = 'lut',
                                    mask = self.AIGeometry.mask,
                                    unit = "q_A^-1")
            intTime = time.clock() - startIntTime
            print('Integration of image', idxIm, '(of', nFiles, ') took', '%.0f' % (intTime*1e3), 'ms of which', '%.0f' % (readTime*1e3), 'ms was spent on readout')
            idxIm += 1
        print('*** Integration done ***')
        self.q = q
        plt.plot(self.q, self.total.S)


        
    def getDelays(self):           
        delay_str = self.logData['delay'].tolist()
        delay = []
        for t_str in delay_str:
            delay.append(self.time_str2num(t_str))            
        delay = np.array(delay)
        return delay, delay_str


        
    def time_str2num(self, t_str):
        try:
            t = float(t_str)
        except ValueError:
            if 'ps' in t_str:
                t = float(t_str[0:-2])*1e-12
            elif 'ns' in t_str:
                t = float(t_str[0:-2])*1e-9
            elif 'us' in t_str:
                t = float(t_str[0:-2])*1e-6
            elif 'ms' in t_str:
                t = float(t_str[0:-2])*1e-3
        return t


                        
    def getTimeStamps(self):
        timeStamp_str = self.logData['date time'].tolist()
        timeStamp = []
        for t in timeStamp_str:
            timeStamp.append(datetime.strptime(t,'%d-%b-%y %H:%M:%S').timestamp())
        timeStamp = np.array(timeStamp)
        return timeStamp, timeStamp_str

    
    def getDifferences(self):
        
        idx_on = np.abs(self.delays-self.toff)>1e-12
        idx_off = np.abs(self.delays-self.toff)<1e-12
        S_on = self.S[idx_on]
        S_off = self.S[idx_off]
        delays_on = self.delays[idx_on]
#        delays_off = self.delays[idx_off]   
        timeStamps_on = self.timeStamps[idx_on]
        timeStamps_off = self.timeStamps[idx_off]
        
        self.dS = np.zeros([np.shape(self.q)[0], np.shape(delays_on)[0]])
        
        for i,t in enumerate(delays_on):
            # find the index of closest ref curve
            idx_closest = np.argmin(timeStamps_on[i] - timeStamps_off)
            self.dS[:,i] = S_on[:,i] - S_off[:,idx_closest]

        
        
# check the data
A = ScatData(logFile = '/media/denis/Data/work/Experiments/2017/Ubiquitin/10.log',
             dataInDir = '/media/denis/Data/work/Experiments/2017/Ubiquitin/',
             dataOutDir = '/media/denis/Data/work/Experiments/2017/Ubiquitin/',
             maskPath = '/media/denis/Data/work/Experiments/2017/Ubiquitin/mask_aug2017.tif',
             energy = 11.63,
             distance = 362,
             pixelSize = 82e-6,
             centerX = 1990,
             centerY = 1967,
             toff_str = '-5us')