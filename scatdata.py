# -*- coding: utf-8 -*-
"""
Created on Thu May 24 20:48:14 2018

scatdata - a module for performing scattering data reduction. It is primarily
intended to be used for BioCARS (APS) data. It is comprised of a class for reduction, 
storage and reading of scattering data and a set of auxillary functions.
See class description to follow the workflow, as well as method docstrings to
understand input/output.

@author: Denis Leshchev

todo:

- add threshFactor keyarg to getDifferences method

"""
from pathlib import Path
import ntpath
import time
from datetime import datetime
from collections import namedtuple

import numpy as np
import pandas as pd
import sympy
from matplotlib import pyplot as plt
from scipy.ndimage import median_filter

import pyFAI
import fabio

class ScatData:
    
    ''' This is a class for processing, storage and readout of time resolved
        x-ray scattering data. The methods invoked during the workflow update the
        attributes of the class instead of just providing returned values. This
        might seem a bit counter-intuitive, but this approach results in a 
        convenient handling of multiple datasets in a uniform fashion down the
        line during the analysis.
        
        The workflow is as follows:
            
        0. Declare the object (ex: A) where you want to store everything and
        provide it with information on log data, input directory and output
        directory. The initiation of the object will assert the correctness of 
        the input to reduce the number of bugs down the line.
        
        1. Integrate the scattering images by using A.integrate(<parameters>)
        method which requires you to declare the experimental geometry. This will
        give you a named tuple A.total.<data> that contains integrated scattering
        data.
        
        2. Remove nasty outliers by using A.identifyTotalOutliers(<parameters>)
        from total scattering originating from bubbles, hair in the air, and other
        issues in the experimental setup. This will give you a flag array
        A.total.isOurlier that will indicate which curves will not be used to
        calculate differences or for any other analysis.
        
        NB: For static data (not time-resolved), you can stop here and do the 
        analysis you want to perform.
        
        3. Calculate differences by using A.getDifferences(<parameters>). This
        method updates the A.diff.<data>; the curves marked as A.total.isOutlier
        are not used in the difference calculation. Unlike in previous versions
        (for instance, in Matlab), to get differences we generate a mapDiff.
        This matrix helps to visualize how we calculate differences and also 
        allows to optimize the way we calculate them.
        
        4. Calculate average differences by using A.getDiffAverages(<parameters>).
        
        5. Save data using A.saveData(<parameters>). *in progress*
        
        6. Load data using A.loadData(<parameters>). *in progress*
        
        *to see the meaning of <data> and <parameters> refer to methods' docstrings.
    '''
    
    def __init__(self, logFile = None, dataInDir = None, dataOutDir = None):
        ''' To initialize the class, you will need:
            logFile - name of the log file. Should be a string or a list of
            strings. Should contain ".log" extension.
            dataInDir - name of the directory which contains data. Should be a
            string or a list of strings. If logFile is a string then this
            argument should also be a string. If logFile is a list of strings,
            this argument should be a list of strings, ordered in the same way
            as logFile list.
            dataOutDir - name of the output directory. Always a string.
            
            The initialization adds a logData attribute (self.logData), which 
            is a pandas table containing all the log information.
            
            Successful initialization will tell you how many images and how many
            unique time delays were measured in a given data set.
        '''
        # check if input is correct:
        self._assertCorrectInput(logFile, dataInDir, dataOutDir) 
       
        # read the log data into logData pandas table and update it with storage details
        if isinstance(logFile ,str):
            self.logData = pd.read_csv(logFile, sep = '\t', header = 18)
            self.logData['Scan'] = ntpath.splitext(ntpath.basename(logFile))[0] # scan name without extension
            self.logData['dataInDir'] = dataInDir
            self.logData.rename(columns = {'#date time':'date time'},
                                inplace = True)
        else: # if the log data is a list, we need a bit of handling:
            logDataAsList = []
            for i,item in enumerate(logFile):
                logDataAsList.append(pd.read_csv(item, sep = '\t', header = 18))
                logDataAsList[i]['Scan'] = ntpath.splitext(ntpath.basename(item))[0]
                if isinstance(dataInDir ,str):
                    logDataAsList[i]['dataInDir'] = dataInDir
                else:
                    logDataAsList[i]['dataInDir'] = dataInDir[i]
            self.logData = pd.concat(logDataAsList, ignore_index=True)
                
        # clean logData from files that do not exist for whatever reason
        idxToDel = []
        for i, row in self.logData.iterrows():
            filePath = self.logData.loc[i,'dataInDir'] + self.logData.loc[i,'file']
            if not Path(filePath).is_file():
                idxToDel.append(i)
                print(filePath, ' does not exist and therefore is out of the analysis')
        self.logData = self.logData.drop(idxToDel)
        
        # print out the number of time delays and number of files
        nFiles = len(self.logData['delay'].tolist())
        nDelays = len(np.unique(self.logData['delay'].tolist()))
        print('Successful initialization:')
        print('Found %s files' % nFiles)
        print('Found %s time delays' % nDelays)



    def _assertCorrectInput(self, logFile, dataInDir, dataOutDir):
        ''' This method asserts the right input according to the logic described
        in __init__ method.
        '''
        if isinstance(logFile, str):
            assert Path(logFile).is_file(), 'log file not found'
            assert isinstance(dataInDir, str), \
            'if log file is a string, the dataInDir should be a string too'
        else:
            assert isinstance(logFile, list), \
            'Provide a string or a list of strings as log file(s)'
            for item in logFile:
                assert isinstance(item, str), \
                'log files paths should be strings'
                assert Path(item).is_file(), item+' not found'
        
        if isinstance(dataInDir, str):
            assert Path(dataInDir).is_dir(), 'input directory not found'
        else:
            assert isinstance(dataInDir, list), \
            'Provide a string or a list of strings as input directory'
            assert len(dataInDir)==len(logFile), \
            'If you provide a list of input directories, they should be the ' \
            'same size and correspondingly ordered as the list of log files'
            for item in dataInDir:
                assert Path(item).is_dir(), item+' not found'
        
        assert isinstance(dataOutDir, str), \
        'provide data output directory as a string'
        assert Path(dataOutDir).is_dir(), 'output directory not found'



# 1. Integration of images



    def integrate(self, energy=12, distance=365, pixelSize=80e-6,
                  centerX=1900, centerY=1900, qRange=[0.0,4.0], nqpt=400, 
                  qNormRange=[1.9,2.1], maskPath=None,
                  dezinger=False, plotting = True):
        ''' This method integrates images given the geometry parameters.
            
            You will need:
            energy - x-ray energy used in the experiment (in keV)
            distance - sample-to-detector distance (in mm)
            pixelSize - size of the pixel (in m)
            centerX - center of the image along the horizontal axis (in pixels)
            centerY - center of the image along the vertical axis (in pixels)
            qRange - a list with two numbers indicating minimum and maximum
            values of transfered momentum used for integration (in A^-1)
            nqpt - number of q points
            qNormRange - normalization range for the integrated curves (in A^-1)
            maskPath - path to mask image (string). It is strongly advised to
            use masks! To produce mask you can use pyFAI drawMask tool.
            dezinger - whether you want to dezinger images  (boolean)
            plotting - whether you want to plot the output results (boolean)
            
            Output:
            self.q - transferred momentum in A^{-1}
            self.total.s_raw - raw integrated (total) curve
                       s - total curve normalized using trapz in qNormRange
                       normInt - normalization value
                       delay - numerical time delay (in s)
                       delay_str - sting time delay (ex: '5us', '100ns')
                       t - unique numerical time delays (in s)
                       t_str - unique string time delays
                       timeStamp - time when the image was measured (epoch)
                       timeStamp_str - time when the image was measured (string)
                       scanStamp - scan (logFile) name
                       imageAv - average image from all the scans
                       isOutlier - creates isOutlier bool array. At this point 
                       all the curves are NOT outliers.
            
        '''
        # Get all the ingredients for integration:
        self._getAIGeometry(energy, distance, pixelSize, centerX, centerY,
                           qRange, nqpt, qNormRange)
        if maskPath:
            assert isinstance(maskPath, str), 'maskPath should be string'
            assert Path(maskPath).is_file(), maskPath + ' file (mask) not found'
            maskImage = fabio.open(maskPath).data
        else:
            maskImage = None
        
        # Declaration of all the necessary attributes/fields:
        self.total = namedtuple('total','s s_raw normInt delay delay_str t t_str '
                                'timeStamp timeStamp_str scanStamp imageAv isOutlier')
        nFiles = self.logData.shape[0]
        self.total.s = np.zeros([nqpt, nFiles])
        self.total.s_raw = np.zeros([nqpt, nFiles])
        self.total.normInt = np.zeros(nFiles)
        self.total.delay, self.total.delay_str = self._getDelays()
        self.total.t = np.unique(self.total.delay)
        self.total.t_str = np.unique(self.total.delay_str)
        self.total.timeStamp, self.total.timeStamp_str = self._getTimeStamps()
        self.total.scanStamp = np.array(self.logData['Scan'].tolist())
        self.imageAv = np.zeros(maskImage.shape)
        self.isOutlier = np.zeros(nFiles, dtype = bool)
        
        # Do the procedure!
        print('*** Integration ***')
        for i, row in self.logData.iterrows():
            impath = row['dataInDir'] + row['file']
            startReadTime = time.clock()         
            image = fabio.open(impath).data
            readTime = time.clock() - startReadTime
            
            startIntTime = time.clock()
            if dezinger:
                image = medianDezinger(image, maskImage)
            q, self.total.s_raw[:,i] = self.AIGeometry.ai.integrate1d(
                                                    image,
                                                    nqpt,
                                                    radial_range = qRange,
                                                    correctSolidAngle = True,
                                                    polarization_factor = 1,
                                                    mask = maskImage,
                                                    unit = "q_A^-1"  )
            #get the region for normalization on the first iteration:
            if i==0: qNormRangeSel = (q>=qNormRange[0]) & (q<=qNormRange[1])
            self.total.normInt[i] = np.trapz(self.total.s_raw[qNormRangeSel,i], 
                                                        q[qNormRangeSel])
            self.total.s[:,i] = self.total.s_raw[:,i]/self.total.normInt[i]
            self.imageAv += image
            intTime = time.clock() - startIntTime
            
            print(i+1, '|',
                 row['file'], ':',
                 'readout: %.0f' % (readTime*1e3),
                 'ms | integration: %.0f' % (intTime*1e3),
                 'ms | total: %.0f' % ((intTime+readTime)*1e3), 'ms')

############ debugging comment:
#            if i>1:
#                break
            
        print('*** Integration done ***')
        self.q = q
        self.imageAv = self.imageAv/(i+1)
        self.imageAv[maskImage==1] = 0
        # check the quality of mask via taking 360 azimuthal slices:
        imageAv_int = np.sum(self.total.s_raw, axis=1)/(i+1)
        imageAv_int_phiSlices, phi, _ = self.AIGeometry.ai.integrate2d(
                                        self.imageAv, nqpt, npt_azim = 360, 
                                        radial_range = qRange, 
                                        correctSolidAngle = True, 
                                        mask=maskImage,
                                        polarization_factor = 1,
                                        unit = "q_A^-1")
        # to keep the figure below "clear":
        imageAv_int_phiSlices[imageAv_int_phiSlices==0] = np.nan
        
        if plotting:
            # get min and max of scale for average image
            vmin, vmax = (imageAv_int[imageAv_int!=0].min(), imageAv_int.max())
            
            plt.figure(figsize=(12,12))
            plt.clf()
            
            plt.subplot(221)
            plt.imshow(self.imageAv, vmin=vmin, vmax=vmax)
            plt.title('Average image')
            
            plt.subplot(222)
            plt.plot(self.q, imageAv_int_phiSlices.T)
            plt.plot(self.q, imageAv_int,'r.-')
            plt.xlabel('q, $\AA^{-1}$')
            plt.ylabel('Intensity, counts')
            plt.title('Integrated average & sliced integrated average')
            
            plt.subplot(223)
            plt.plot(self.q, self.total.s_raw)
            plt.xlabel('q, $\AA^{-1}$')
            plt.ylabel('Intensity, counts')
            plt.title('All integrated curves')
            
            plt.subplot(224)
            plt.plot(self.q, self.total.s)
            plt.xlabel('q, $\AA^{-1}$')
            plt.ylabel('Intensity, a.u.')
            plt.title('All integrated curves (normalized)')



    def _getAIGeometry(self, energy, distance, pixelSize, centerX, centerY,
                      qRange, nqpt, qNormRange):
        '''Method for storing the geometry parameters from the input to 
        self.integrate() method.
        '''
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



    def _getDelays(self):
        ''' Method for getting delays from the logData in numerical and string
            forms.
            This function uses the attributes of the class and does a return of
            time delays in numerical and string formats. This is done for
            the readability of the code  in the main function.
        '''
        delay_str = self.logData['delay'].tolist()
        delay = []
        for t_str in delay_str:
            delay.append(time_str2num(t_str))            
        delay_str = np.array(delay_str)
        delay = np.array(delay)
        return delay, delay_str



    def _getTimeStamps(self):
        ''' Method for getting time stamps in a standard date-time and epoch formats.
            This function uses the attributes of the class and does a return of
            time delays in numerical and string formats. This is done for
            the readability of the code in the main function.
        '''
        timeStamp_str = self.logData['date time'].tolist()
        timeStamp = []
        for t in timeStamp_str:
            timeStamp.append(datetime.strptime(t,'%d-%b-%y %H:%M:%S').timestamp())
        timeStamp = np.array(timeStamp)
        timeStamp_str = np.array(timeStamp_str)
        return timeStamp, timeStamp_str    



# 2. Total curve averaging



    def getTotalAverages(self, fraction=0.9, chisqThresh=5,
                              q_break=None, chisqThresh_lowq=5, chisqThresh_highq=5,
                              plotting=True, chisqHistMax=10):
        self.total.s_av, self.total.s_err, self.total.isOutlier = \
        getAverage(self.q, self.total.delay_str, self.total.s, self.total.t_str,
                   fraction, chisqThresh,
                   q_break, chisqThresh_lowq, chisqThresh_highq,
                   plotting, chisqHistMax)



# 3. Difference calculation


    
    def getDifferences(self, toff_str='-5us', subtractFlag='MovingAverage'):
        
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
        ds = np.dot(mapDiff,self.total.s.T).T
        
        self.diff.mapDiff = mapDiff
        self.diff.ds = ds
        self.diff.delay = self.total.delay[idxDiff]
        self.diff.delay_str = self.total.delay_str[idxDiff]
        self.diff.timeStamp = self.total.timeStamp[idxDiff]
        self.diff.timeStamp_str = self.total.timeStamp_str[idxDiff]
        self.diff.t = np.unique(self.diff.delay)
        self.diff.t_str = np.unique(self.diff.delay_str)
        self.diff.isOutlier = np.zeros(self.diff.delay.shape, dtype = bool)



# 4. Difference averaging



    def getDiffAverages(self, fraction=0.9, chisqThresh=1.5,
                        q_break=None, chisqThresh_lowq=1.5, chisqThresh_highq=1.5,
                        plotting=True, chisqHistMax = 10):
        self.diff.ds_av, self.diff.ds_err, self.diff.isOutlier = \
        getAverage(self.q, self.diff.delay_str, self.diff.ds, self.diff.t_str,
                   fraction, chisqThresh,
                   q_break, chisqThresh_lowq, chisqThresh_highq,
                   plotting, chisqHistMax)


        

#%% Auxillary functions
        
        

def medianDezinger(img_orig, mask):
    img = img_orig.copy()
    img_blur = median_filter(img, size=(3,3))
    img_diff = (img.astype(float)-img_blur.astype(float))
    threshold = np.std(img_diff[mask.astype(bool)])*10
    hot_pixels = np.abs(img_diff)>threshold
    hot_pixels[mask.astype(bool)] = 0
    img[hot_pixels] = img_blur[hot_pixels]
    
#    x,y = np.nonzero(hot_pixels)
    
#    plt.figure(1)
#    plt.clf()
#    plt.imshow(img, vmin = 0, vmax = 3500)
#    plt.plot(y,x,'r.')
##    plt.ylim(1750,1769)
##    plt.xlim(770,795)
##    plt.imshow(hot_pixels)
    
    return img



def time_str2num(t_str):
    ''' Method for converting time delay strings to numerical format (in s)
        Input: time delay string
        Output: time in s
    '''
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



def getAverage(q, delay_str, x, t_str,
               fraction, chisqThresh,
               q_break, chisqThresh_lowq, chisqThresh_highq,
               plotting, chisqHistMax):
    
    x_av = np.zeros((q.size, t_str.size))
    x_err = np.zeros((q.size, t_str.size))
    isOutlier = np.zeros(delay_str.size, dtype=bool)
    if plotting:
        nsubs = t_str.size
        plt.figure(figsize = (nsubs*3,6))
        plt.clf()
    
    for i, delay_point in enumerate(t_str):
        delay_selection = delay_str==delay_point
        x_loc = x[:, delay_selection]
        isOutlier_loc, chisq, chisq_lowq, chisq_highq = \
                identifyOutliers(q, x_loc, fraction, chisqThresh, 
                                 q_break, chisqThresh_lowq, chisqThresh_highq)
        isOutlier[delay_selection] = isOutlier_loc
        x_av[:,i] = np.mean(x_loc[:,isOutlier_loc], axis = 1)
        x_err[:,i] = np.std(x_loc[:,isOutlier_loc], axis = 1)/np.sqrt(np.sum(isOutlier_loc))
        
        if plotting:
            subidx = i
            plotOutliers(nsubs, subidx,
                         q, x_loc, isOutlier_loc, 
                         chisq, chisqThresh,
                         q_break, chisq_lowq, chisqThresh_lowq, chisq_highq, chisqThresh_highq,
                         chisqHistMax)
    
    return x_av, x_err, isOutlier



def identifyOutliers(q_orig, y_orig, fraction, chisqThresh,
                     q_break, chisqThresh_lowq, chisqThresh_highq):
    q = q_orig.copy()
    y = y_orig.copy()

    ySel = getMedianSelection(y, fraction)
    ySel_av = np.mean(ySel, axis = 1)
    ySel_std = np.std(ySel, axis = 1)
    errsq = ((y - ySel_av[:,np.newaxis])/ySel_std[:,np.newaxis])**2/q.size
    
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



def getMedianSelection(z_orig, frac):
    z = z_orig.copy()
    z = np.sort(z, axis=1)
    ncols = z.shape[1]
    low = np.int(np.round((1-frac)/2*ncols))
    high = np.int(np.round((1+frac)/2*ncols))
#    print(low, high)
    z = z[:,low:high]
    return z



def plotOutliers(nsubs, subidx,
                 q, x, isOutlier, chisq,
                 chisqThresh, q_break,
                 chisq_lowq, chisqThresh_lowq, chisq_highq, chisqThresh_highq,
                 chisqHistMax):
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

    chisqBins = np.concatenate((np.arange(0,chisqHistMax+0.5,0.5),
                                np.array(np.inf)[np.newaxis]))
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
if __name__ == '__main__':
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
                nqpt = 400,
                qNormRange = [1.9,2.1],
                maskPath = 'D:\\leshchev_1708\\Ubiquitin\\MASK_UB.edf',
                dezinger=False,
                plotting=True)
#%%
# idnetify outliers in total curves
A.getTotalAverages(fraction=0.9, chisqThresh=6)

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
#img = A.imageAv
#
#img_blur = median_filter(img, size=2)
#img_diff = (img-img_blur)
#threshold = np.std(img_diff)*10
#
#hot_pixels = np.abs(img_diff)>threshold
#
#plt.figure(3)
#plt.clf()
##plt.imshow(img, vmin = 100, vmax = 2400)
##plt.imshow(img_blur, vmin = 100, vmax = 2400)
#plt.imshow(np.abs(img_diff), vmin = 0, vmax = 100)
##plt.hist(np.ravel(img_diff),100)
#
#    
    
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
