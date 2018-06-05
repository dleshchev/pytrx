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
- optimize mask usage memory (now you keep it everywhere for no need)
- add subtraction
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
    
    def __init__(self, logFile = None, dataInDir = None, dataOutDir = None):
#    , maskPath, energy, 
#                 distance, pixelSize, centerX, centerY, qNormRange,
#                 toff_str = '-5us'):
                
        self.logFile = logFile
        self.dataInDir = dataInDir
        self.dataOutDir = dataOutDir

#        self.maskPath = maskPath
#        
#        self.energy = energy
#        self.wavelength = 12.3984/self.energy*1e-10 # in m
#        self.distance = distance*1e-3 # in m
#        self.pixelSize = pixelSize
#        self.centerX = centerX
#        self.centerY = centerY
#
#        self.toff_str = toff_str
#        self.toff = self.time_str2num(toff_str)
#        self.qNormRange = qNormRange
#        
        assert Path(self.logFile).is_file(), 'log file not found'
        assert Path(self.dataInDir).is_dir(), 'input directory not found'
        assert Path(self.dataOutDir).is_dir(), 'output directory not found'
        
        self.logData = pd.read_csv(logFile, sep = '\t', header = 18)
        self.logData.rename(columns = {'#date time':'date time'}, inplace = True)
        
        # remove files that for whatever reason do not exist
        idxToDel = []
        for i, row in self.logData.iterrows():
            if not Path(self.dataInDir + self.logData.ix[i,'file']).is_file():
                idxToDel.append(i)
        self.logData = self.logData.drop(idxToDel)
        
        # get the number of time delays and number of files
        print('Found %s files' % (len(self.logData['delay'].tolist())))
        print('Found %s time delays' % (len(np.unique(self.logData['delay'].tolist()))))



    def getAIGeometry(self, energy, distance, pixelSize, centerX, centerY, qRange, nqpt, qNormRange):
        wavelen = 12.3984/energy*1e-10 # in m
        self.AIGeometry = namedtuple('AIGeometry', 'ai nqpt qRange qNormRange')
        self.AIGeometry.ai = pyFAI.AzimuthalIntegrator(dist = distance*1e-3,
                                                       poni1 = centerY*pixelSize,
                                                       poni2 = centerX*pixelSize,
                                                       pixel1 = pixelSize,
                                                       pixel2 = pixelSize,
                                                       rot1=0,rot2=0,rot3=0,
                                                       wavelength = wavelen)
        self.AIGeometry.qNormRange = qNormRange
        
        
        
    def integrate(self, energy=12, distance=365, pixelSize=80e-6, centerX=1900, centerY=1900, qRange=[0.0,4.0], nqpt=400, qNormRange=[1.9,2.1], maskPath=None):
        self.getAIGeometry(energy, distance, pixelSize, centerX, centerY, qRange, nqpt, qNormRange)
        if maskPath:
            assert Path(maskPath).is_file(), 'mask file not found'
            maskImage = fabio.open(maskPath).data

        self.total = namedtuple('total','s s_raw normInt delay delay_str t t_str timeStamp timeStamp_str scanStamp')        
        nFiles = self.logData.shape[0]        
        self.total.s = np.zeros([nqpt, nFiles])
        self.total.s_raw = np.zeros([nqpt, nFiles])
        self.total.normInt = np.zeros(nFiles)
        self.total.delay, self.total.delay_str = self.getDelays()
        self.total.t = np.unique(self.total.delay)
        self.total.t_str = np.unique(self.total.delay_str)
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
            q, self.total.s_raw[:,i] = self.AIGeometry.ai.integrate1d(image, nqpt,
                                    radial_range = qRange,
                                    correctSolidAngle = True,
                                    polarization_factor = 1,
                                    method = 'lut',
                                    mask = maskImage,
                                    unit = "q_A^-1")
            intTime = time.clock() - startIntTime
            print('Integration of image', idxIm, '(of', nFiles, ') took', '%.0f' % (intTime*1e3), 'ms of which', '%.0f' % (readTime*1e3), 'ms was spent on readout')
            idxIm += 1
        
            qNormRangeSel = (q>=qNormRange[0]) & (q<=qNormRange[1])
            self.total.normInt[i] = np.trapz(self.total.s_raw[qNormRangeSel,i], 
                                                        q[qNormRangeSel])
            self.total.s[:,i] = self.total.s_raw[:,i]/self.total.normInt[i]
        
        print('*** Integration done ***')
        self.q = q
        plt.plot(self.q, self.total.s)



    def getDelays(self):           
        delay_str = self.logData['delay'].tolist()
        delay = []
        for t_str in delay_str:
            delay.append(self.time_str2num(t_str))            
        delay_str = np.array(delay_str)
        delay = np.array(delay)
        
        return delay, delay_str


        
    def time_str2num(self, t_str):
        try:
            t = float(t_str)
        except ValueError:
            t_number = float(t_str[0:-2])
            if 'ps' in t_str:
                t = t_number*1e-12
            elif 'ns' in t_str:
                t = t_number*1e-9
            elif 'us' in t_str:
                t = t_number*1e-6
            elif 'ms' in t_str:
                t = t_number*1e-3
        return t


                        
    def getTimeStamps(self):
        timeStamp_str = self.logData['date time'].tolist()
        timeStamp = []
        for t in timeStamp_str:
            timeStamp.append(datetime.strptime(t,'%d-%b-%y %H:%M:%S').timestamp())
        timeStamp = np.array(timeStamp)
        return timeStamp, timeStamp_str

    
    
    def getDifferences(self):
        
        self.diff = namedtuple('ds', 'ds delay t timeStamp')
        self.diff.ds = []
        self.diff.delay = []
        self.diff.t = self.total.t
        self.diff.t_str = self.total.t_str
        self.diff.timeStamp = []
        
        delaySelIDRef = self.total.delay_str == self.toff_str
        s_ref = self.total.s[:, delaySelIDRef]
        timeStamp_ref = self.total.timeStamp[delaySelIDRef]
        timeStampThresh = np.median(timeStamp_ref)*1.1
        
        for specificDelay in self.diff.t:
            if self.diff.t_str == self.toff_str:
                pass
                
            else:
                delaySelID = self.total.delay == specificDelay
                s_loc = self.total.s[delaySelID]
                timeStamp_loc = self.total.timeStamp[delaySelID]
                timeStampDif_loc = np.abs(timeStamp_loc[newaxis].T - timeStamp_ref)
                timeStampDif_ID = timeStampDif_loc<=timeStampThresh
                s_ref_tbs = np.mean(s_ref[:,timeStampDif_ID], axis = 1)
                ds_dummy = s_loc - s_ref_tbs
                
        
        
        
#        delays_off = self.delays[idx_off]   
        
        self.dS = np.zeros([np.shape(self.q)[0], np.shape(delays_on)[0]])
        
        for i,t in enumerate(delays_on):
            # find the index of closest ref curve
            idx_closest = np.argmin(timeStamps_on[i] - timeStamps_off)
            self.dS[:,i] = S_on[:,i] - S_off[:,idx_closest]

        
        
# check the data
A = ScatData(logFile = '/media/denis/Data/work/Experiments/2017/Ubiquitin/10.log',
             dataInDir = '/media/denis/Data/work/Experiments/2017/Ubiquitin/',
             dataOutDir = '/media/denis/Data/work/Experiments/2017/Ubiquitin/')
             
A.integrate(energy = 11.63,
            distance = 362,
            pixelSize = 82e-6,
            centerX = 1990,
            centerY = 1967,
            qRange = [0.0, 4.0],
            nqpt=400,
            qNormRange = [1.9,2.1],
            maskPath = '/media/denis/Data/work/Experiments/2017/Ubiquitin/mask_aug2017.tif')