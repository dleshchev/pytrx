# -*- coding: utf-8 -*-
"""
Created on Thu May 24 20:48:14 2018

ScatData - a data class for opearating scattering data recorded primaily at BioCARS (APS)



@author: Denis Leshchev

todo:

- add threshFactor keyarg to getDifferences method
- add average calculation
"""
from pathlib import Path
import ntpath
import time
from datetime import datetime
from collections import namedtuple

import numpy as np
import pandas as pd
import sympy

import pyFAI
import fabio

from matplotlib import pyplot as plt

class ScatData:
    '''
    This is a class for processing, storage and readout of time resolved x-ray solution scattering data
    
    '''
    def __init__(self, logFile = None, dataInDir = None, dataOutDir = None):
        
        # check if logFile is of correct type and if they exist
        self.assertCorrectInput(logFile, dataInDir, dataOutDir)
       
        # read the log data into a pandas table
        if isinstance(logFile ,str):
            self.logData = pd.read_csv(logFile, sep = '\t', header = 18)
            self.logData['Scan'] = ntpath.splitext(ntpath.basename(logFile))[0]
            self.logData['dataInDir'] = dataInDir
            self.logData.rename(columns = {'#date time':'date time'}, inplace = True)
        else:
            logDataAsList = []
            for i,item in enumerate(logFile):
                logDataAsList.append(pd.read_csv(item, sep = '\t', header = 18))
                logDataAsList[i]['Scan'] = ntpath.splitext(ntpath.basename(item))[0]
                if isinstance(dataInDir ,str):
                    logDataAsList[i]['dataInDir'] = dataInDir
                else:
                    logDataAsList[i]['dataInDir'] = dataInDir[i]
            self.logData = pd.concat(logDataAsList, ignore_index=True)
                
        # remove files that for whatever reason do not exist from logData
        idxToDel = []
        for i, row in self.logData.iterrows():
            filePath = self.logData.loc[i,'dataInDir'] + self.logData.loc[i,'file']
            if not Path(filePath).is_file():
                idxToDel.append(i)
                print(filePath, ' does not exist')
        self.logData = self.logData.drop(idxToDel)
        
        # get the number of time delays and number of files
        print('***')
        print('Found %s files' % (len(self.logData['delay'].tolist())))
        print('Found %s time delays' % (len(np.unique(self.logData['delay'].tolist()))))
        print('***')



    def assertCorrectInput(self, logFile, dataInDir, dataOutDir):
        if isinstance(logFile, str):
            assert Path(logFile).is_file(), 'log file not found'
            assert isinstance(dataInDir, str), 'if log file is string, the dataInDir should be string too'
        else:
            assert isinstance(logFile, list), 'Provide a string or a list of strings as log file'
            for item in logFile:
                assert isinstance(item, str), 'log files paths should be strings'
                assert Path(item).is_file(), item+' not found'
        
        if isinstance(dataInDir, str):
            assert Path(dataInDir).is_dir(), 'input directory not found'
        else:
            assert isinstance(dataInDir, list), 'Provide a string or a list of strings as input directory'
            assert len(dataInDir)==len(logFile), 'If you provide a list of input directories, they should be the same size and correspondingly ordered'
            for item in dataInDir:
                assert Path(item).is_dir(), item+' not found'
        
        assert isinstance(dataOutDir, str), 'provide data output directory as a string'
        assert Path(dataOutDir).is_dir(), 'output directory not found'
    


# 1. Integration of images


        
    def integrate(self, energy=12, distance=365, pixelSize=80e-6, centerX=1900, centerY=1900, qRange=[0.0,4.0], nqpt=400, qNormRange=[1.9,2.1], maskPath=None, plotting = True):
        self.getAIGeometry(energy, distance, pixelSize, centerX, centerY, qRange, nqpt, qNormRange)
        if maskPath:
            assert Path(maskPath).is_file(), maskPath+' file (mask) not found'
            maskImage = fabio.open(maskPath).data

        self.total = namedtuple('total','s s_raw normInt delay delay_str t t_str timeStamp timeStamp_str scanStamp imageAv isOutlier')        
        nFiles = self.logData.shape[0]        
        self.total.s = np.zeros([nqpt, nFiles])
        self.total.s_raw = np.zeros([nqpt, nFiles])
        self.total.normInt = np.zeros(nFiles)
        self.total.delay, self.total.delay_str = self.getDelays()
        self.total.t = np.unique(self.total.delay)
        self.total.t_str = np.unique(self.total.delay_str)
        self.total.timeStamp, self.total.timeStamp_str = self.getTimeStamps()
        self.total.scanStamp = np.array(self.logData['Scan'].tolist())
        self.imageAv = np.zeros(maskImage.shape)
        self.isOutlier = np.zeros(nFiles, dtype = bool)
        
        idxIm = 1
        print('*** Integration ***')
        for i, row in self.logData.iterrows():
            path = row['dataInDir'] + row['file']
            startReadTime = time.clock()         
            image = fabio.open(path).data
            readTime = time.clock() - startReadTime
            startIntTime = time.clock()
#            q, self.total.s_raw[:,i] = self.AIGeometry.ai.integrate1d(image, nqpt,
#                                        radial_range = qRange,
#                                        correctSolidAngle = True,
#                                        polarization_factor = 1,
#                                        mask = maskImage,
#                                        unit = "q_A^-1")
#            q, self.total.s_raw[:,i] = self.AIGeometry.ai.medfilt1d(image, nqpt,
#                                        percentile = 50,
#                                        correctSolidAngle = True,
#                                        polarization_factor = 1,
#                                        mask = maskImage,
#                                        unit = "q_A^-1")
            s_raw_phi, q, phi = self.AIGeometry.ai.integrate2d(image, nqpt, npt_azim = 512, 
                                                                    radial_range = qRange, 
                                                                    correctSolidAngle = True, 
                                                                    mask=maskImage,
                                                                    polarization_factor = 1,
                                                                    unit = "q_A^-1")
            self.total.s_raw[:,i] = np.mean(getMedianSelection(s_raw_phi.T, 0.96), axis=1)
            
            intTime = time.clock() - startIntTime
            print('Integration of image', idxIm, '(of', nFiles, ') took', '%.0f' % (intTime*1e3), 'ms of which', '%.0f' % (readTime*1e3), 'ms was spent on readout')
            idxIm += 1
        
            qNormRangeSel = (q>=qNormRange[0]) & (q<=qNormRange[1])
            self.total.normInt[i] = np.trapz(self.total.s_raw[qNormRangeSel,i], 
                                                        q[qNormRangeSel])
            self.total.s[:,i] = self.total.s_raw[:,i]/self.total.normInt[i]
            self.imageAv += image
#            if idxIm>1:
#                break
            
        print('*** Integration done ***')
        self.q = q
        self.imageAv = self.imageAv/(idxIm-1)
        self.imageAv[maskImage==1] = 0
        self.imageAv_int = np.mean(self.total.s_raw, axis=1)
        print('*** Integration quality check ***')
        self.imageAv_int_phiSlices = self.AIGeometry.ai.integrate2d(self.imageAv, nqpt, npt_azim = 360, 
                                                                    radial_range = qRange, 
                                                                    correctSolidAngle = True, 
                                                                    mask=maskImage,
                                                                    polarization_factor = 1,
                                                                    unit = "q_A^-1")
        self.imageAv_med = self.AIGeometry.ai.medfilt1d(self.imageAv, nqpt, 
                                                                    percentile = (0,99),
                                                                    correctSolidAngle = True, 
                                                                    mask=maskImage,
                                                                    polarization_factor = 1,
                                                                    unit = "q_A^-1")
        if plotting:
            plt.figure(figsize=(12,12))
            plt.clf()
            
            plt.subplot(221)
            plt.imshow(self.imageAv)
            plt.title('Average image')
            plt.subplot(222)
            plt.plot(self.q, self.imageAv_int_phiSlices[0].T)
            plt.plot(self.q, self.imageAv_int,'r.-')
            plt.xlabel('q, A^-1')
            plt.ylabel('Intensity, counts')
            plt.title('Integrated average & sliced integrated average')
#            
            plt.subplot(223)
            plt.plot(self.q, self.total.s_raw)
            plt.xlabel('q, A^-1')
            plt.ylabel('Intensity, counts')
            plt.title('All integrated curves')
#            
            plt.subplot(224)
            plt.plot(self.q, self.total.s)
            plt.xlabel('q, A^-1')
            plt.ylabel('Intensity, a.u.')
            plt.title('All integrated curves (normalized)')
  
    
    
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
        self.AIGeometry.qRange = qRange
        self.AIGeometry.qNormRange = qNormRange
        self.AIGeometry.nqpt = nqpt
    
    
    
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
        timeStamp_str = np.array(timeStamp_str)
        return timeStamp, timeStamp_str    
    


# 2. Total outlier rejection
        
    
    
    def identifyTotalOutliers(self, fraction=0.9, chisqThresh=5, q_break=None, chisqThresh_lowq=5, chisqThresh_highq=5, plotting=True, chisqHistMax = 10):
        self.total.isOutlier, chisqTot, chisqTot_lowq, chisqTot_highq = identifyOutliers(self.q, self.total.s, 
                                                                                         fraction, 
                                                                                         chisqThresh, 
                                                                                         q_break, 
                                                                                         chisqThresh_lowq, 
                                                                                         chisqThresh_highq)
        if plotting:
            plt.figure(figsize = (12,6))
            plt.clf()
            nsubs = 1
            subidx = 0
            plotOutliers(nsubs, subidx, self.q, self.total.s, self.total.isOutlier, 
                         chisqTot, chisqThresh,
                         q_break, chisqTot_lowq, chisqThresh_lowq, chisqTot_highq, chisqThresh_highq,
                         chisqHistMax)
            
            
                
# 3. Difference calculation


    
    def getDifferences(self, toff_str = '-5us', subtractFlag = 'MovingAverage'):
        
        self.diff = namedtuple('differnces', 'mapDiff ds delay delay_str timeStamp timeStamp_str t t_str ')

        stampDiff = self.total.timeStamp[np.newaxis].T-self.total.timeStamp
        stampDiff[stampDiff==0] = stampDiff.max()
        stampDiff[:, self.total.delay_str!=toff_str] = stampDiff.max()
        stampDiff[:, self.total.isOutlier] = stampDiff.max()
        stampDiff[self.total.isOutlier, :] = stampDiff.max()
        mapDiff = np.eye(stampDiff.shape[0])
        
        if subtractFlag == 'Closest':
            offsTBS = np.argmin(np.abs(stampDiff), axis=1)
            mapDiff[np.arange(mapDiff.shape[0]), offsTBS] = -1
            mapDiff = mapDiff[sympy.Matrix(mapDiff).T.rref()[1],:]
        
        elif subtractFlag == 'MovingAverage':
            stampThresh = np.median(np.diff(self.total.timeStamp[self.total.delay_str == toff_str]))*1.1
            offsTBS = np.abs(stampDiff)<stampThresh
            mapDiff = getWeights(self.total.timeStamp, offsTBS, mapDiff)
        
        elif subtractFlag == 'Previous':
            stampDiff[stampDiff<0] = stampDiff.max()
            offsTBS = np.argmin(np.abs(stampDiff), axis=1)
            mapDiff[np.arange(mapDiff.shape[0]), offsTBS] = -1
            mapDiff = np.tril(mapDiff)
            
        elif subtractFlag == 'Next':
            stampDiff[stampDiff>0] = stampDiff.max()
            stampDiff = np.abs(stampDiff)
            offsTBS = np.argmin(np.abs(stampDiff), axis=1)
            mapDiff[np.arange(mapDiff.shape[0]), offsTBS] = -1
            mapDiff = np.triu(mapDiff)
            
        mapDiff = mapDiff[(np.abs(np.sum(mapDiff, axis=1))<1e-4) & ( ~self.total.isOutlier),:]
        idxDiff = np.where(mapDiff==1)[1]
        
        self.diff.mapDiff = mapDiff
        self.diff.ds = np.dot(mapDiff,self.total.s.T).T
        self.diff.delay = self.total.delay[idxDiff]
        self.diff.delay_str = self.total.delay_str[idxDiff]
        self.diff.timeStamp = self.total.timeStamp[idxDiff]
#        self.diff.timeStamp_str = [self.total.timeStamp_str[i] for i in idxDiff] # ... indexing of lists is such a pain ...
        self.diff.timeStamp_str = self.total.timeStamp_str[idxDiff]
        self.diff.t = np.unique(self.diff.delay)
        self.diff.t_str = np.unique(self.diff.delay_str)
        self.diff.isOutlier = np.zeros(self.diff.delay.shape, dtype = bool)



    def getDiffAverages(self, fraction=0.9, chisqThresh=1.5, q_break=None, chisqThresh_lowq=1.5, chisqThresh_highq=1.5, plotting=True, histBins = 10):
        self.diff.ds_av = np.zeros((self.q.size, self.diff.t.size))
        self.diff.ds_err = np.zeros((self.q.size, self.diff.t.size))
        if plotting:
            nsubs = A.diff.t_str.size
            plt.figure(figsize = (nsubs*3,6))
            plt.clf()
        
        for i, t_point in enumerate(A.diff.t_str):
            delay_selection = A.diff.delay_str==t_point
            ds_loc = self.diff.ds[:, delay_selection]
            isOutlier_loc, chisqDiff, chisqDiff_lowq, chisqDiff_highq = identifyOutliers(self.q, ds_loc, fraction, chisqThresh, 
                                                                                         q_break, chisqThresh_lowq, chisqThresh_highq)
            self.diff.isOutlier[delay_selection] = isOutlier_loc
            self.diff.ds_av[:,i] = np.mean(ds_loc[:,isOutlier_loc], axis = 1)
            self.diff.ds_err[:,i] = np.std(ds_loc[:,isOutlier_loc], axis = 1)/np.sqrt(np.sum(isOutlier_loc))
            
            if plotting:
                subidx = i
                plotOutliers(nsubs, subidx, self.q, ds_loc, isOutlier_loc, 
                             chisqDiff, chisqThresh,
                             q_break, chisqDiff_lowq, chisqThresh_lowq, chisqDiff_highq, chisqThresh_highq,
                             histBins)


        

#%% Auxillary functions
        
        

def getWeights(timeStamp, offsTBS, mapDiff):
    # auxillary function 
    for i, stampOn in enumerate(timeStamp):
        offsTBS_loc = offsTBS[i,:]
        stampOffs = timeStamp[offsTBS_loc]
        stampRange = np.diff(stampOffs)
        if stampRange.size==0:
            mapDiff[i,offsTBS_loc] = -1
        elif stampRange.size==1:
            stampDiffs = abs(stampOn - stampOffs)
            weights = -(stampDiffs/stampRange)[::-1]
            mapDiff[i,offsTBS_loc] = weights
        else:
            raise ValueError('Variable stampThresh is too large. Decrease the multiplication factor.')
    return mapDiff



def identifyOutliers(q_orig, x_orig, fraction, chisqThresh, q_break, chisqThresh_lowq, chisqThresh_highq):
    q = q_orig.copy()
    x = x_orig.copy()

    xSel = getMedianSelection(x, fraction)
    xSel_av = np.mean(xSel, axis = 1)
    xSel_std = np.std(xSel, axis = 1)
    errsq = ((x - xSel_av[:,np.newaxis])/xSel_std[:,np.newaxis])**2/q.size
    
    if not q_break:
        chisq = np.nansum(errsq, axis=0)
        isOutlier = chisq>chisqThresh
        chisq_lowq = None
        chisq_highq = None
    else:
        chisq_lowq = np.nansum(errsq[q<q_break,:], axis=0)
        chisq_highq = np.nansum(errsq[q>=q_break,:], axis=0)
        isOutlier = (chisq_lowq>=chisqThresh_lowq) | (chisq_highq>=chisqThresh_highq)
        chisq = None
        
    return isOutlier, chisq, chisq_lowq, chisq_highq



def getMedianSelection(x_orig, frac):
    x = x_orig.copy()
    x = np.sort(x, axis=1)
    ncols = x.shape[1]
    low = np.int(np.round((1-frac)/2*ncols))
    high = np.int(np.round((1+frac)/2*ncols))
#    print(low, high)
    x = x[:,low:high]
    return x  



def plotOutliers(nsubs, subidx, q, x, isOutlier, chisq, chisqThresh, q_break, chisq_lowq, chisqThresh_lowq, chisq_highq, chisqThresh_highq, chisqHistMax):
    plt.subplot(nsubs,2, subidx*2+1)
    if any(isOutlier):
        plt.plot(q, x[:, isOutlier], 'b-')
    if any(~isOutlier):
        plt.plot(q, x[:,~isOutlier], 'k-')
        x_mean = np.mean(x[:,~isOutlier], axis=1)
        plt.plot(q, x_mean,'r')
#        plt.ylim(x_mean.min(), x_mean.max())
    plt.xlabel('q, A^-1')
    plt.ylabel('Intentsity, a.u.')
#    plt.legend('Outliers','Accepted Data')

    chisqBins = np.concatenate((np.arange(0,4,0.5), np.arange(4,chisqHistMax+0.5,0.5), np.array(np.inf)[np.newaxis]))
    chisqWidths = np.diff(chisqBins)
    chisqWidths[-1] = 1
    if not q_break:
        heights,_ = np.histogram(chisq, bins=chisqBins)
        
        plt.subplot(nsubs,2,subidx*2+2)
        plt.bar(chisqBins[:-1], heights, width=chisqWidths, align='edge')
        plt.plot(chisqThresh*np.array([1,1]), plt.ylim(),'k-')
        plt.xlabel('\chi^2')
        plt.ylabel('n. occurances')
        
    else:
        heights_lowq,_ = np.histogram(chisq_lowq, bins=chisqBins)
        heights_highq,_ = np.histogram(chisq_highq, bins=chisqBins)
    
        plt.subplot(nsubs*2,2,subidx*4+2)
        plt.bar(chisqBins[:-1], heights_lowq, width=chisqWidths, align='edge')
        plt.plot(chisqThresh_lowq*np.array([1,1]), plt.ylim())
        plt.xlabel('\chi_lowq^2')
        plt.ylabel('n. occurances')
        
        plt.subplot(nsubs*2,2,subidx*4+4)
        plt.bar(chisqBins[:-1], heights_highq, width=chisqWidths, align='edge')
        plt.plot(chisqThresh_highq*np.array([1,1]), plt.ylim())
        plt.xlabel('\chi_highq^2')
        plt.ylabel('n. occurances')



                  


    
    
#%%        
# Dirty Checking: integration
A = ScatData(logFile = 'D:\\leshchev_1708\\Ubiquitin\\45.log',
             dataInDir = 'D:\\leshchev_1708\\Ubiquitin\\',
             dataOutDir = 'D:\\leshchev_1708\\Ubiquitin\\')
          
A.integrate(energy = 11.63,
            distance = 364,
            pixelSize = 82e-6,
            centerX = 1987,
            centerY = 1965,
            qRange = [0.0, 4.0],
            nqpt=400,
            qNormRange = [1.9,2.1],
            maskPath = 'D:\\leshchev_1708\\Ubiquitin\\MASK_UB.edf')
#%%
# idnetify outliers in total curves
A.identifyTotalOutliers(fraction=0.9, chisqThresh=5)

#%% difference calculation
A.getDifferences(toff_str = '-5us', 
                 subtractFlag = 'MovingAverage')

#%%

#A.getDiffAverages(fraction=0.9, chisqThresh=2.5)
A.getDiffAverages(fraction=0.9, q_break=2, chisqThresh_lowq=2.5, chisqThresh_highq=3)


#%%
#ds_av = np.zeros((A.AIGeometry.nqpt, A.diff.t.size))
#ds_err = np.zeros(ds_av.shape)
#
#chisqThresh = 1
#for i, t_point in enumerate(A.diff.t_str):
#    ds_loc = A.diff.ds[:, A.diff.delay_str==t_point]
#    ds_loc_med = np.median(ds_loc, axis = 1)
##    ds_loc_med = np.mean(ds_loc, axis = 1)
#    ds_loc_std = np.std(ds_loc, axis = 1)
#    errsq = ((ds_loc - ds_loc_med[:,np.newaxis])/ds_loc_std[:,np.newaxis])**2
#    chisqOut = np.nansum(errsq, axis=0)/A.AIGeometry.nqpt
#    selectedCurves = chisqOut<chisqThresh
#    ds_loc_sel = ds_loc[:, selectedCurves]
#    ds_av[:,i] = np.mean(ds_loc_sel, axis = 1)
#    ds_err[:,i] = np.std(ds_loc_sel, axis = 1)/np.sqrt(np.sum(selectedCurves))
#    
#plt.figure(1)
#plt.clf()
#
#plt.subplot(121)
##plt.plot(A.q, ds_loc, 'b-')
##plt.plot(A.q, ds_loc_sel, 'k-')
#plt.plot(A.q, ds_err[:,1], 'r-')
#
#plt.subplot(122)
#plt.hist(chisqOut,20)
#
##s = A.total.s
#idx = 20


#%%

#s = A.diff.ds[:,A.diff.delay>0]
#
#
#
#s1 = getMedianSelection(s, frac = 0.9)
#plt.figure(1)
#plt.clf()
#plt.plot(s,'k')
#plt.plot(s1,'r')


#%%
#
#def hampel(vals_orig, k=7, t0=3):
#    '''
#    vals: pandas series of values from which to remove outliers
#    k: size of window (including the sample; 7 is equal to 3 on either side of value)
#    '''
#    #Make copy so original not edited
#    vals=vals_orig.copy()
#    vals = pd.DataFrame(vals)
#    vals = vals[vals!=0]
#    
#    #Hampel Filter
#    L= 1.4826
#    rolling_median = vals.rolling(k, center=True, min_periods=1).median()
#    difference = np.abs(rolling_median-vals)
#    median_abs_deviation = difference.rolling(k, center=True, min_periods=1).median()
#    threshold= t0 *L * median_abs_deviation
#    outlier_idx = difference>threshold
##    vals[outlier_idx] = np.nan
#    vals[outlier_idx] = rolling_median[outlier_idx]
#    return(vals)
#
#
#
#s1 = hampel(s[:,idx])
#
#sdiff = np.diff(s[:,idx])
#s1diff = hampel(sdiff)
#
###%%
##s1 = s[:].copy()
##
##s1 = pd.DataFrame(s1)
##k=27
##t0=3
##L=1.4826
##
##s1 = s1[s1>0]
##rolling_median = s1.rolling(k, center=True).median()
##difference = np.abs(rolling_median-s1)
##median_abs_deviation = difference.rolling(k, center=True, min_periods=1).median()
##threshold= t0 *L * median_abs_deviation
##outlier_idx = difference>threshold
##s2 = s1.copy()
##s2[outlier_idx] = rolling_median[outlier_idx]
##
##
##plt.figure(1)
##plt.clf()
##plt.plot(np.diff(s1, axis=0))
##plt.plot(rolling_median,'.-')
##plt.plot(difference)
##plt.plot(median_abs_deviation)
#
##plt.plot(s1[outlier_idx],'rx')
###plt.plot(s2)
###plt.plot(median_abs_deviation)
##
###%%
#plt.figure(1)
#plt.clf()
##plt.plot(s[:,idx])
##plt.plot(s1)
#plt.plot(sdiff)
#plt.plot(s1diff)
#
#
#
