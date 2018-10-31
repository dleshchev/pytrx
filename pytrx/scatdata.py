# -*- coding: utf-8 -*-
"""
Created on Thu May 24 20:48:14 2018

scatdata - a module for performing scattering data reduction. It is primarily
intended to be used for BioCARS (APS) data. It is comprised of a class for reduction, 
storage and reading of scattering data, and a set of auxillary functions.
See class description to follow the workflow, as well as method docstrings to
understand input/output.

@author: Denis Leshchev

todo:
- each visualization should be a separate function
- visualization of outlier rejection should be decluttered
- add read/load methods

"""
from pathlib import Path
import ntpath
import time
from datetime import datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.ndimage import median_filter
import scipy.signal as signal
import h5py

import pyFAI
import fabio


class ScatData:
    
    ''' This is a class for processing, storage, and loading of time resolved
        x-ray scattering data.
        
        The workflow:
        
        0. Declare the data object (ex: A):
        A = ScatData(<parameters*>)
            
        Provide the class with information on log data, input directory and output
        directory. The initiation of the object will assert the correctness of 
        the input to reduce the number of bugs down the line. For details see
        self.__init__ docstring.
        
        1. Integrate the scattering images:
        A.integrate(<parameters>)
        Provide the method with the experimental geometry details. This will
        give you a named tuple A.total.<data> that contains integrated scattering
        data.
        
        2. Identify nasty outliers and get total scattering averages: 
        A.getTotalAverages(<parameters>)
        Provide the method with rejection thresholds to obtain
        averaged curves. This method also marks the total curves that will not be
        used for calculation of averages or differences down the line.
        NB: For non-time-resolved data, you can stop here and do the analysis
        you want to perform.
        
        3. Calculate differences:
        A.getDifferences(<parameters>)
        This method updates the A.diff.<data> with difference data calculated
        according to provided parameters.
        
        4. Calculate average differences:
        A.getDiffAverages(<parameters>)
        Provide the method with rejection thresholds.
        
        5. Save data using A.saveData(<parameters>). *in progress*
        
        6. Load data using A.loadData(<parameters>). *in progress*
        
        The methods invoked during the workflow update the
        attributes of the class instead of just providing returned values. This
        might seem a bit counter-intuitive, but this approach results in a 
        convenient handling of multiple datasets in a uniform fashion down the
        line during the analysis.
        
        *to see the meaning of <parameters> refer to methods' docstrings.
    '''
    
    def __init__(self, logFile=None, dataInDir=None, logFileStyle='biocars',
                 ignoreFirst=False, nFiles=None):
        ''' To initialize the class, you will need:
            
            logFile - name of the log file. Should be a string or a list of
            strings. Should contain ".log" extension.
            dataInDir - name of the directory which contains data. Should be a
            string or a list of strings. If logFile is a string then this
            argument should also be a string. If logFile is a list of strings,
            this argument should be a list of strings, ordered in the same way
            as logFile list.
            logFileStyle - style of logFile set to 'biocars' by default. Other
            possible value is 'id09'.
            ignoreFirst - if True, the log reader will remove first image from
            each run frim the data analysis.
            nFiles - number of images you want to integrate in each run.
            
            The initialization adds a logData attribute (self.logData), which 
            is a pandas table containing all the log information.
            
            Successful initialization will tell you how many images and how many
            unique time delays were measured in a given data set.
        '''
        # check if input is correct:
        self._assertCorrectInput(logFile, dataInDir, logFileStyle) 
        
        if isinstance(logFile ,str):
            logFile = [logFile] # convert to list for smooth handling
        
        print('*** Reading log files ***')
        logDataAsList = []
        for i,item in enumerate(logFile):
            if logFileStyle == 'biocars':
                logDataAsList.append(pd.read_csv(item, sep = '\t', header = 18))
                logDataAsList[i].rename(columns = {'#date time':'timeStamp_str', 'delay':'delay_str'}, inplace = True)
                logDataAsList[i]['delay'] = logDataAsList[i]['delay_str'].apply(lambda x: time_str2num(x))
            
            elif logFileStyle == 'id09':
                logDataAsList.append(pd.read_csv(item, skiprows=1, skipfooter=1, sep='\t',
                                                 engine='python', names=_get_id09_columns(),
                                                 skipinitialspace=True))
                logDataAsList[i]['timeStamp_str'] = logDataAsList[i]['date'] + ' ' + logDataAsList[i]['time']
                logDataAsList[i]['delay_str'] = logDataAsList[i]['delay'].apply(lambda x: time_num2str(x))
            
            logDataAsList[i]['timeStamp'] = logDataAsList[i]['timeStamp_str'].apply(
                    lambda x: datetime.strptime(x,'%d-%b-%y %H:%M:%S').timestamp())
            
            if ignoreFirst:
                logDataAsList[i] = logDataAsList[i][1:]
            if nFiles:
                logDataAsList[i] = logDataAsList[i][:nFiles]
                
            logDataAsList[i]['Scan'] = ntpath.splitext(ntpath.basename(item))[0]
            if isinstance(dataInDir ,str):
                logDataAsList[i]['dataInDir'] = dataInDir
            else:
                logDataAsList[i]['dataInDir'] = dataInDir[i]
        
        self.logData = pd.concat(logDataAsList, ignore_index=True)
                
        # clean logData from files that do not exist for whatever reason
        print('Checking if all the files/images exist')
        idxToDel = []
        for i, row in self.logData.iterrows():
            if logFileStyle == 'id09':
                self.logData.loc[i,'file'] = self.logData.loc[i,'file'].replace('ccdraw','edf')
            filePath = self.logData.loc[i,'dataInDir'] + self.logData.loc[i,'file']
            if not Path(filePath).is_file():
                idxToDel.append(i)
                print(filePath, 'does not exist and will be excluded from analysis')
        self.logData = self.logData.drop(idxToDel)
        
        # print out the number of time delays and number of files
        print('Done with checking\nSummary:')
        self.nFiles = len(self.logData.index)
        self.nDelays = self.logData['delay_str'].nunique()
        print('Found %s files' % self.nFiles)
        print('Found %s time delays' % self.nDelays)
        print('Details:\ndelay \t # files')
        print(self.logData['delay_str'].value_counts())



    def _assertCorrectInput(self, logFile, dataInDir, logFileStyle):
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
        
        assert (logFileStyle == 'biocars') or (logFileStyle == 'id09'), \
        'logFileStyle can be either "biocars" or "id09"'



# 1. Integration of images



    def integrate(self, energy=12, distance=365, pixelSize=80e-6,
                  centerX=1900, centerY=1900, qRange=[0.0,4.0], nqpt=400, 
                  qNormRange=[1.9,2.1], maskPath=None,
                  dezinger=False, plotting=True, nMax=None):
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
            nMax - number of Images you want to integrate. All imges will be 
            integrates if nMax is None.
            
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
        
        # Declaration of all the necessary attributes/fields in self.total:
        self.total = DataContainer()
        self.total.s = np.zeros([nqpt, self.nFiles])
        self.total.s_raw = np.zeros([nqpt, self.nFiles])
        self.total.normInt = np.zeros(self.nFiles)
        self.total.delay = self.logData['delay'].values
        self.total.delay_str = self.logData['delay_str'].values
        self.total.t = np.unique(self.total.delay)
        self.total.t_str = np.unique(self.total.delay_str)
        self.total.timeStamp = self.logData['timeStamp'].values
        self.total.timeStamp_str = self.logData['timeStamp_str'].values
        self.total.scanStamp = self.logData['Scan'].values
        self.total.isOutlier = np.zeros(self.nFiles, dtype = bool)
        
        # Do the procedure!
        print('*** Integration ***')
        for i, row in self.logData.iterrows():
            impath = row['dataInDir'] + row['file']
            startReadTime = time.clock()         
            image = fabio.open(impath).data
            readTime = time.clock() - startReadTime
            
            startIntTime = time.clock()
            if dezinger: image = medianDezinger(image, maskImage)
            q, self.total.s_raw[:,i] = self.AIGeometry.ai.integrate1d(
                                                    image,
                                                    nqpt,
                                                    radial_range = qRange,
                                                    correctSolidAngle = True,
                                                    polarization_factor = 1,
                                                    mask = maskImage,
                                                    unit = "q_A^-1"  )
            self.total.s[:,i] = self.total.s_raw[:,i]/self.total.normInt[i]
            
            if i==0: self.imageAv = image
            else: self.imageAv += image
            
            intTime = time.clock() - startIntTime
            print(i+1, '|',
                  row['file'], ':',
                  'readout: %.0f' % (readTime*1e3),
                  'ms | integration: %.0f' % (intTime*1e3),
                  'ms | total: %.0f' % ((intTime+readTime)*1e3), 'ms')
            
            if nMax:
                if (i+1) >= nMax:
                    break
            
        print('*** Integration done ***')
        self.q = q
        self.total.normInt, self.total.s = normalizeQ(q, self.total.s_raw, qNormRange)
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
        '''Method for storing the geometry parameters in self.AIGeometry from
        the input to self.integrate() method.
        '''
        wavelen = 12.3984/energy*1e-10 # in m
        self.AIGeometry = DataContainer()
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
        ''' Method calculates the total averages and gets rid of nasty outliers.
        It uses a chisq-based method for detecting outliers (see getAverage aux
        method).
        
        You need:
        fraction - amount of data used for getting the effective average/std to detect
        outliers. By default it is 0.9 which means that the average/std are calculated
        using data between 0.05 and 0.95 percentiles.
        
        chisqThresh - Threshold value of chisq, above which the data will be marked
        as ourtliers.
        
        q_break - in case if you want to have a separate diagnostics for small and
        high q values, you can use q_break.
        chisqThresh_lowq - Threshold for chisq calculated for q<q_break
        chisqThresh_highq - Threshold for chisq calculated for q>=q_break
        
        plotting - True if you want to see the results of the outlier rejection
        chisqHistMax - maximum value of chisq you want to plot histograms
        '''
        print('*** Averaging the total curves ***')
        self.total.s_av, self.total.s_err, self.total.isOutlier = \
          getAverage(self.q, self.total.delay_str, self.total.s, self.total.t_str,
                     fraction, chisqThresh,
                     q_break, chisqThresh_lowq, chisqThresh_highq,
                     plotting, chisqHistMax)
        print('*** Done ***')


# 3. Difference calculation


    
    def getDifferences(self, toff_str='-5us', subtractFlag='MovingAverage', 
                       dezinger=True, renormalize=False, qNormRange=None):
        ''' Method for calculating differences.
        
        You will need:
        toff_str - time delay which you use for subtraction
        subtractFlag - the way you want to calculate the differences. It can have
        following values:
                'MovingAverage' - calculates a weighted average between (at most)
                                  two closest reference curves. This combined curve
                                  is used to calculate differences.
                'Closest'       - calculates difference using the closest
                                  reference curve.
                'Previous'      - uses previous reference curve.
                'Next'          - uses next reference curve..
        
        NB: the difference calculation ignores curves marked True in
        self.total.isOutlier
        
        The method updates/adds following fields:
        self.diff.ds - difference curves
                  delay - delay (in s)
                  delay_str - delay (string format)
                  timeStamp - when the (laser-on) image was measured (epoch)
                  timeStamp_str - the same as timeStamp, but in string format
                  t - unique time delays in s
                  t_str - unique time delays in string format
                  isOutlier - this is the flag for outliers for *differences*. At
                        this point the isOutlier is an array of Falses, because
                        no rejection has been implented yet
        '''
        print('*** Calculating the difference curves ***')
        if renormalize:
            self.AIGeometry.qNormRange = qNormRange
            self.total.normInt, self.total.s = normalizeQ(self.q, self.total.s_raw, qNormRange)
            
        self.diff = DataContainer()
        assert toff_str in self.total.delay_str, 'toff_str is not found among recorded time delays'
        self.diff.toff_str = toff_str
        self.diff.ds = np.zeros((self.q.size, self.nFiles))
        self.diff.delay = np.array([])
        self.diff.delay_str = np.array([])
        self.diff.timeStamp = np.array([])
        self.diff.timeStamp_str = np.array([])
        
        idx_to_use = [] # indexes for clean-up
        
        for i in range(self.nFiles):
            
            s_tbs = None
            idx_next = self._findOffIdx(i, 'Next')
            idx_prev = self._findOffIdx(i, 'Prev')
            
            if subtractFlag == 'Next':
                if idx_next:
                    s_tbs = self.total.s[:, idx_next]
            
            elif subtractFlag == 'Previous':
                if idx_prev:
                    s_tbs = self.total.s[:, idx_prev]
            
            elif subtractFlag == 'Closest':
                if self.total.delay_str[i] == self.diff.toff_str: # this is to avoid getting the same differences
                    if idx_next:
                        s_tbs = self.total.s[:, idx_next]
                else:
                    if (idx_next) and (idx_prev):
                        timeToNext = np.abs(self.total.timeStamp[i] - self.total.timeStamp[idx_next])
                        timeToPrev = np.abs(self.total.timeStamp[i] - self.total.timeStamp[idx_prev])
                        if timeToNext <= timeToPrev:
                            s_tbs = self.total.s[:, idx_next]
                        else:
                            s_tbs = self.total.s[:, idx_prev]
                    elif (idx_next) and (not idx_prev):
                        s_tbs = self.total.s[:, idx_next]
                    elif (not idx_next) and (idx_prev):
                        s_tbs = self.total.s[:, idx_prev]
                        
            elif subtractFlag == 'MovingAverage':
                if (idx_next) and (idx_prev):
                    timeToNext = np.abs(self.total.timeStamp[i] - 
                                        self.total.timeStamp[idx_next])
                    timeToPrev = np.abs(self.total.timeStamp[i] - 
                                        self.total.timeStamp[idx_prev])
                    timeDiff = np.abs(self.total.timeStamp[idx_next] - 
                                      self.total.timeStamp[idx_prev])
                    w_next = timeToPrev/timeDiff
                    w_prev = timeToNext/timeDiff
                    s_tbs = (w_next*self.total.s[:, idx_next] + 
                             w_prev*self.total.s[:, idx_prev])
                elif (idx_next) and (not idx_prev):
                    s_tbs = self.total.s[:, idx_next]
                elif (not idx_next) and (idx_prev):
                    s_tbs = self.total.s[:, idx_prev]
                    
            if s_tbs is not None:
                idx_to_use.append(i)
                ds_loc = self.total.s[:, i] - s_tbs
                if dezinger:
                    ds_loc = medianDezinger1d(ds_loc)
                self.diff.ds[:,i] = ds_loc
        
        self.diff.ds = self.diff.ds[:, idx_to_use] # clean-up
        
        self.diff.delay = self.total.delay[idx_to_use]
        self.diff.delay_str = self.total.delay_str[idx_to_use]
        self.diff.timeStamp = self.total.timeStamp[idx_to_use]
        self.diff.timeStamp_str = self.total.timeStamp_str[idx_to_use]
        self.diff.scanStamp = self.total.scanStamp[idx_to_use]
        self.diff.t = np.unique(self.diff.delay)
        self.diff.t_str = np.unique(self.diff.delay_str)
        self.diff.isOutlier = np.zeros(self.diff.delay.shape, dtype = bool)                
        print('')  
        print('*** Done with the difference curves ***')
    
    
    
    def _findOffIdx(self, idx, direction):
        idx_start = idx
        while True:
            if direction == 'Next':
                idx += 1
            elif direction == 'Prev':
                idx -= 1
            else:
                raise ValueError('direction must be "Next" or "Prev"')
            if (idx<0) or idx>(self.nFiles-1):
                return None
            if ((self.total.delay_str[idx] == self.diff.toff_str) and # find next/prev reference
                (self.total.scanStamp[idx] == self.total.scanStamp[idx_start])): # should be in the same scan
                return idx



# 4. Difference averaging



    def getDiffAverages(self, fraction=0.9, chisqThresh=1.5,
                        q_break=None, chisqThresh_lowq=1.5, chisqThresh_highq=1.5,
                        plotting=True, chisqHistMax = 10):
        ''' Method to get average differences. It works in the same way as
        getTotalAverages, so refer to the information on input/output in the
        getTotalAverages docstring.
        '''
        print('*** Averaging the difference curves ***')
        self.diff.ds_av, self.diff.ds_err, self.diff.isOutlier = \
        getAverage(self.q, self.diff.delay_str, self.diff.ds, self.diff.t_str,
                   fraction, chisqThresh,
                   q_break, chisqThresh_lowq, chisqThresh_highq,
                   plotting, chisqHistMax)
        print('*** Processing done ***')


        
# 5. Saving
    
    
    
    def saveToHDF(self, savePath=None):
        
        assert isinstance(savePath, str), \
        'provide data output directory as a string'
        
        print('*** Saving ***')
        f = h5py.File(savePath, 'w')
        for attr in dir(self):
            if not (attr.startswith('_') or
                    attr.startswith('get') or
                    attr.startswith('save') or
                    attr.startswith('load') or
                    attr.startswith('integrate')):
                
                if ((attr == 'total') or
                    (attr == 'diff') or
                    (attr == 'AIGeometry')):
                    obj = self.__getattribute__(attr)
                    for subattr in dir(obj):
                        if not (subattr.startswith('_') or
                                subattr.startswith('ai')):
                            data_to_record = obj.__getattribute__(subattr)
                            if type(data_to_record) is not int:
                                if ((type(data_to_record)==str) or
                                    (type(data_to_record[0])==str)):
                                    data_to_record = '|'.join([i for i in data_to_record])
                            
                            f.create_dataset(attr+'/'+subattr, data=data_to_record)
                            print(attr+'/'+subattr, 'was successfully saved')
                
                elif attr == 'logData':
                    self.logData.to_hdf(savePath, key='logData', mode='r+')
                    print(attr, 'was successfully saved')
                    
                else:
                    data_to_record = self.__getattribute__(attr)
                    f.create_dataset(attr, data=data_to_record)
                    print(attr, 'was successfully saved')
                    
        f.close()
        print('*** Done with saving ***')
        

#%% Auxillary functions
# This functions are put outside of the class as they can be used in broader
# contexts and such implementation simplifies access to them.
        

class DataContainer:
    # dummy class to store the data
    def __init__(self):
        pass        
        
        
def _get_id09_columns():
    id09_columns = ['date', 'time', 'file',
                    'delay', 'delay_act', 'delay_act_std', 'delay_act_min', 'delay_act_max',
                    'laser', 'laser_std', 'laser_min', 'laser_max', 'laser_n',
                    'xray',  'xray_std',  'xray_min',  'xray_max',  'xray_n',
                    'n_pulses']
    return id09_columns

def medianDezinger(img_orig, mask):
    ''' Function for image dezingering.
    
    You need:
        img_orig - the image you want to dezinger
        mask - the mask for this image (can be None)
    
    Output:
        img - dezingered image
    '''
    img = img_orig.copy()
    img_blur = median_filter(img, size=(3,3))
    img_diff = (img.astype(float)-img_blur.astype(float))
    if ~mask:
        threshold = np.std(img_diff[mask.astype(bool)])*10
        hot_pixels = np.abs(img_diff)>threshold
        hot_pixels[mask.astype(bool)] = 0
    else:
        threshold = np.std(img_diff)*10
        hot_pixels = np.abs(img_diff)>threshold
    img[hot_pixels] = img_blur[hot_pixels]
    return img



def medianDezinger1d(x_orig, fraction=0.9, kernel_size=5, thresh=5):
    ''' Function for dezingering 1D curves using median filter.
    
    You need:
        x_orig - 1D curve
        fraction - fraction of data used for calculation of STD (default=0.9)
        kernel_size - kernel size for median filtering (default=5)
        thresh - STD multiplier for determination of the filtering threshold (default=5)
        
    Output:
        x - dezingered data
    '''
    x = x_orig.copy()
    idx_nnz = x!=0
    x_nnz = x[idx_nnz]
    x_nnz_filt = signal.medfilt(x_nnz, kernel_size=kernel_size)
    dx = x_nnz - x_nnz_filt
    dx_sel = getMedianSelection(dx[dx!=0][np.newaxis], fraction)
    threshold = np.std(dx_sel)*thresh
    hot_pixels = np.abs(dx)>threshold
    x_nnz_dez = x_nnz
    x_nnz_dez[hot_pixels] = x_nnz_filt[hot_pixels]
    x[idx_nnz] = x_nnz_dez
    return x
    


def normalizeQ(q, s_raw, qNormRange):
    qNormRangeSel = (q>=qNormRange[0]) & (q<=qNormRange[1])
    normInt = np.trapz(s_raw[qNormRangeSel, :], q[qNormRangeSel], axis=0)
    return normInt, s_raw/normInt
    


def time_str2num(t_str):
    ''' Function for converting time delay strings to numerical format (in s)
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



def time_num2str(t):
    ''' Function for converting time delays to string format
        Input: time delay in s
        Output: time string
    '''
    A = np.log10(np.abs(t))
    if (A < -12):
        t_str = str(round(t*1e15)) + 'fs'
    elif (A >= -12) and (A < -9):
        t_str = str(round(t*1e12)) + 'ps'
    elif (A >= -9) and (A < -6):
        t_str = str(round(t*1e9)) + 'ns'
    elif (A >= -6) and (A < -3):
        t_str = str(round(t*1e6)) + 'us'
    elif (A >= -3) and (A < 0):
        t_str = str(round(t*1e3)) + 'ms'
    else:
        t_str = str(int(t))
    return t_str
    



def getWeights(timeStamp, offsTBS, mapDiff):
    ''' Function which calculates weights for the next and previous reference
    curves.
    
    If the given curve is measured at time t_k, the previous reference
    curve is measured at t_i, and the next is measured at t_j, then the weight
    for the *previous* is abs(t_k-t_j)/abs(t_i-t_j) while for the *next* one
    the weight is abs(t_k-t_i)/abs(t_i-t_j). This way the largest weight is 
    given to the reference curve that is closer in time to the k-th curve.
    '''
    for i, stampOn in enumerate(timeStamp):
        offsTBS_loc = offsTBS[i,:]
        stampOffs = timeStamp[offsTBS_loc]
        stampRange = np.diff(stampOffs)
        if stampRange.size == 0:
            mapDiff[i,offsTBS_loc] = -1
        elif stampRange.size == 1:
            stampDiffs = abs(stampOn - stampOffs)
            weights = -(stampDiffs/stampRange)[::-1]
            mapDiff[i,offsTBS_loc] = weights
        else:
            print('Cannot compute difference for', i, 'curve')
            print('diagnostics: i:', i, '; stampOn:', stampOn,
                  '; stampOffs:', stampOffs,
                  '; stampDiffs:', abs(stampOn - stampOffs),
                  '; offs TBS: ', np.where(offsTBS_loc))
            raise ValueError('Variable stampThresh is too large. '
                             'Decrease the multiplication factor.')
    return mapDiff



def getAverage(q, delay_str, x, t_str,
               fraction, chisqThresh,
               q_break, chisqThresh_lowq, chisqThresh_highq,
               plotting, chisqHistMax):
    ''' Function for calculating averages and standard errors of data sets.
    For details see ScatData.getTotalAverages method docstring.
    '''
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
        x_av[:,i] = np.mean(x_loc[:,~isOutlier_loc], axis = 1)
        x_err[:,i] = np.std(x_loc[:,~isOutlier_loc], axis = 1)/np.sqrt(np.sum(~isOutlier_loc))
        
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
    ''' Function for identification of outliers in a given data set.
    
    The function calculates the average and the standard deviation using fraction
    of the data (see below) and uses these values to evaluate chisq for each curve
    in the given data like so:
        for k-th curve chisq_k = sum_q ((y_k-y_av)/(y_std))**2
    if the chisq_k is larger than chisqThresh, then the curve is deemed outlier.
    
    Sometimes one needs to evaluate outliers across different q-regions and to do
    so one needs to introduce q_break. Then the above sum splits into:
        chisq_lowq_k = sum_(q<q_break) ((y_k-y_av)/(y_std))**2
        chisq_highq_k = sum_(q>=q_break) ((y_k-y_av)/(y_std))**2
    To find outliers the function evaluates whether any of chisq_lowq_k or
    chisq_highq_k are higher than chisqThresh_lowq or chisqThresh_highq, respectively.
    
    You will need:
        q_orig - q values of the data
        y_orig - data with axis=0 in q space
        fraction - fraction of the data you want to use to calculate trial average
            and deviation. If it is 0.5, these values will be calculated using the
            curves between 0.25 and 0.75 percentiles.
        chisqThresh - threshold chisq value, above which the data is deemed to be
            outlier.
        q_break - value determening the regions where chisq_lowq and chisq_highq
            will be evaluated.
        chisqThresh_lowq, chisqThresh_highq - chisq threshold values for low and
            high q parts of the data.
    '''
    q = q_orig.copy()
    y = y_orig.copy()

    ySel = getMedianSelection(y, fraction)
    ySel_av = np.mean(ySel, axis = 1)
    ySel_std = np.std(ySel, axis = 1)
    nnzStd = ySel_std!=0
    errsq = ((y[nnzStd,:] - ySel_av[nnzStd,np.newaxis])/
              ySel_std[nnzStd,np.newaxis])**2/q.size
    q_errsq = q[nnzStd]
    
    if not q_break:
        chisq = np.nansum(errsq, axis=0)
        isOutlier = chisq>chisqThresh
        chisq_lowq = None
        chisq_highq = None
    else:
        chisq_lowq = np.nansum(errsq[q_errsq<q_break,:], axis=0)
        chisq_highq = np.nansum(errsq[q_errsq>=q_break,:], axis=0)
        isOutlier = (chisq_lowq>=chisqThresh_lowq) | (chisq_highq>=chisqThresh_highq)
        chisq = None
        
    return isOutlier, chisq, chisq_lowq, chisq_highq



def getMedianSelection(z_orig, frac):
    ''' Function to get selection of data from symmetric percentiles determined
    by fraction. For example if fraction is 0.9, then the output data is the
    selection between 0.05 and 0.95 percentiles.
    '''
    z = z_orig.copy()
    z = np.sort(z, axis=1)
    ncols = z.shape[1]
    low = np.int(np.round((1-frac)/2*ncols))
    high = np.int(np.round((1+frac)/2*ncols))
    z = z[:,low:high]
    return z



def plotOutliers(nsubs, subidx,
                 q, x, isOutlier, chisq,
                 chisqThresh, q_break,
                 chisq_lowq, chisqThresh_lowq, chisq_highq, chisqThresh_highq,
                 chisqHistMax):
    ''' Fucntion to plot data and corresponding chisq histograms.
    '''
    plt.subplot(nsubs,2, subidx*2+1)
    if any(isOutlier):
        plt.plot(q, x[:, isOutlier], 'b-')
    if any(~isOutlier):
        plt.plot(q, x[:,~isOutlier], 'k-')
        x_mean = np.mean(x[:,~isOutlier], axis=1)
        plt.plot(q, x_mean,'r.')
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
    #%% idnetify outliers in total curves
    A.getTotalAverages(fraction=0.9, chisqThresh=6)
    
    #%% difference calculation
    A.getDifferences(toff_str = '-5us', 
                     subtractFlag = 'MovingAverage',
                     dezinger=True)
    
    #%%
    
    A.getDiffAverages(fraction=0.9, chisqThresh=2.5)
    #A.getDiffAverages(fraction=0.9, q_break=2, chisqThresh_lowq=2.5, chisqThresh_highq=3)
    