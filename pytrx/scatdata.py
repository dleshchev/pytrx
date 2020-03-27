# -*- coding: utf-8 -*-
"""
Created on Thu May 24 20:48:14 2018

scatdata - a module for performing scattering data reduction. It is primarily
intended to be used for synchrotron  BioCARS (APS) or ID09 (ESRF) data. It is comprised of a class for reduction,
storage and reading of scattering data, and a set of auxillary functions.
See class description to follow the workflow, as well as method docstrings to
understand input/output.

@author: Denis Leshchev

"""
from pathlib import Path
import ntpath
import time
from datetime import datetime
from math import pi

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.ndimage import median_filter
import scipy.signal as signal
from scipy import sparse
from covar import cov_shrink_ss

import h5py

import pyFAI
import fabio


class ScatData:
    ''' This is a class for processing, storage, and loading of time resolved
        x-ray scattering data.
        
        The workflow:
        
        0. Declare the data object (ex: A):
        A = ScatData()
        
        0a. Read log file A.readLog(<parameters>)
        0b. Load log fine A.load(path)
            
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
        
        5. Save data using A.save(path)
        
        6. Load data using A.load(path)
        
        The methods invoked during the workflow update the
        attributes of the class instead of just providing returned values. This
        might seem a bit counter-intuitive, but this approach results in a 
        convenient handling of multiple datasets in a uniform fashion down the
        line during the analysis.
        
        *to see the meaning of <parameters> refer to methods' docstrings.
    '''

    def __init__(self, logFile, logFileStyle='biocars', ignoreFirst=False, nFirstFiles=None, dataInDir=None):
        '''
        To read the file:

        logFile - name of the log file. Should be a string or a list of
        strings. Should contain ".log" extension.
        dataInDir - name of the directory which contains data. Should be a
        string or a list of strings. If logFile is a string then this
        argument should also be a string. If logFile is a list of strings,
        this argument should be a list of strings, ordered in the same way
        as logFile list.
        logFileStyle - style of logFile set to 'biocars' by default. Other
        possible value is 'id09_old' and 'id09', which correspond to id09
        styles from before 2015 and after.
        ignoreFirst - if True, the log reader will remove first image from
        each run frim the data analysis.
        nFiles - number of images you want to integrate in each run. (mostly for tests)

        The initialization adds a logData attribute (self.logData), which
        is a pandas table containing all the log information.

        Successful initialization will tell you how many images and how many
        unique time delays were measured in a given data set.

        '''
        if dataInDir is None:
            dataInDir = str(Path(logFile).parent.absolute()) + '\\'
        self.logFile = logFile
        self.dataInDir = dataInDir
        self.logFileStyle = logFileStyle
        self.ignoreFirst = ignoreFirst
        self.nFirstFiles = nFirstFiles
        self._getLogData()
        self._identifyExistingFiles()
        self.logSummary()


    def _getLogData(self):
        '''
        Read log file(s) and convert them into a pandas dataframe
        '''

        self._assertCorrectInput()

        if isinstance(self.logFile, str):
            self.logFile = [self.logFile]  # convert to list for smooth handling

        print('*** Reading log files ***')
        logDataAsList = []
        for i, item in enumerate(self.logFile):
            print('reading', item)
            if self.logFileStyle == 'biocars':
                logDataAsList.append(pd.read_csv(item, sep='\t', header=18))
                logDataAsList[i].rename(columns={'#date time': 'timeStamp_str', 'delay': 'delay_str'}, inplace=True)
                logDataAsList[i]['delay'] = logDataAsList[i]['delay_str'].apply(lambda x: time_str2num(x))

            elif self.logFileStyle == 'id09_old':
                logDataAsList.append(pd.read_csv(item, skiprows=1, skipfooter=1, sep='\t',
                                                 engine='python', names=_get_id09_columns_old(),
                                                 skipinitialspace=True))

                logDataAsList[i]['timeStamp_str'] = logDataAsList[i]['date'] + ' ' + logDataAsList[i]['time']
                logDataAsList[i]['delay_str'] = logDataAsList[i]['delay'].apply(lambda x: time_num2str(x))

            elif self.logFileStyle == 'id09':
                logDataAsList.append(pd.read_csv(item, skiprows=1, skipfooter=1, sep='\t',
                                                 engine='python', names=_get_id09_columns(),
                                                 skipinitialspace=True))

                logDataAsList[i]['timeStamp_str'] = logDataAsList[i]['date'] + ' ' + logDataAsList[i]['time']
                logDataAsList[i]['delay_str'] = logDataAsList[i]['delay'].apply(lambda x: time_num2str(x))

            logDataAsList[i]['timeStamp'] = logDataAsList[i]['timeStamp_str'].apply(
                lambda x: datetime.strptime(x, '%d-%b-%y %H:%M:%S').timestamp())

            if self.ignoreFirst:
                logDataAsList[i] = logDataAsList[i][1:]
            if self.nFirstFiles:
                logDataAsList[i] = logDataAsList[i][:self.nFirstFiles]

            logDataAsList[i]['Scan'] = ntpath.splitext(ntpath.basename(item))[0]

            if isinstance(self.dataInDir, str):
                logDataAsList[i]['dataInDir'] = self.dataInDir
            else:
                logDataAsList[i]['dataInDir'] = self.dataInDir[i]

        self.logData = pd.concat(logDataAsList, ignore_index=True)
        print('*** Done ***\n')


    def _assertCorrectInput(self):
        '''
        This method asserts the right input according to the logic described
        in __init__ method.
        '''
        if isinstance(self.logFile, str):
            assert Path(self.logFile).is_file(), 'log file not found'
            assert isinstance(self.dataInDir, str), \
                'if log file is a string, the dataInDir should be a string too'
        else:
            assert isinstance(self.logFile, list), \
                'Provide a string or a list of strings as log file(s)'
            for item in self.logFile:
                assert isinstance(item, str), \
                    'log files paths should be strings'
                assert Path(item).is_file(), item + ' not found'

        if isinstance(self.dataInDir, str):
            assert Path(self.dataInDir).is_dir(), 'input directory not found'
        else:
            assert isinstance(self.dataInDir, list), \
                'Provide a string or a list of strings as input directory'
            assert len(self.dataInDir) == len(self.logFile), \
                'If you provide a list of input directories, they should be the ' \
                'same size and correspondingly ordered as the list of log files'
            for item in self.dataInDir:
                assert Path(item).is_dir(), item + ' not found'

        assert ((self.logFileStyle == 'biocars') or
                (self.logFileStyle == 'id09_old') or
                (self.logFileStyle == 'id09')), \
            'logFileStyle can be either "biocars" or "id09"'


    def _identifyExistingFiles(self):
        '''
        goes through the files listed in log files and checks if they exist
        '''
        idxToDel = []
        for i, row in self.logData.iterrows():
            if self.logFileStyle == 'id09_old':
                self.logData.loc[i, 'file'] = self.logData.loc[i, 'file'].replace('ccdraw', 'edf')
            elif self.logFileStyle == 'id09':
                self.logData.loc[i, 'file'] += '.edf'

            filePath = self.logData.loc[i, 'dataInDir'] + self.logData.loc[i, 'file']
            if not Path(filePath).is_file():
                idxToDel.append(i)
                print(filePath, 'does not exist and will be excluded from analysis')
        self.logData = self.logData.drop(idxToDel)


    def logSummary(self):
        '''
        Print the log information:
        number of files
        number of delays
        number of images per time delay
        '''
        print('*** Summary ***')

        if not hasattr(self, 'nFiles'):
            self.nFiles = len(self.logData.index)
        if not hasattr(self, 'nDelay'):
            self.nDelays = self.logData['delay_str'].nunique()
        if not hasattr(self, 't'):
            self.t = self.logData['delay'].unique()
        if not hasattr(self, 't_str'):
            self.t_str = self.logData['delay_str'].unique()
            self.t_str = self.t_str[np.argsort(self.t)]
            self.t = np.sort(self.t)

        print('Found %s files' % self.nFiles)
        print('Found %s time delays' % self.nDelays)
        print('Details:\ndelay \t # files')
        #        print(self.logData['delay_str'].value_counts()) # unsorted output
        for t_str in self.t_str:
            print(t_str, '\t', np.sum(self.logData['delay_str'] == t_str))

        print('*** End of summary ***\n')


    # 1. Integration of images


    def integrate(self, energy=12, distance=365, pixelSize=80e-6,
                  centerX=1900, centerY=1900, qRange=[0.0, 4.0], nqpt=400,
                  qNormRange=[1.9, 2.1], maskPath=None,
                  correctPhosphor=False, muphos=228, lphos=75e-4,
                  correctSample=False, musample=0.49, lsample=300e-6,
                  plotting=True, nFiles=None):
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
            orrectPhosphor - detector absorption correction
            muphos - absorption coef mu for phosphor screen of the detector
            lphos - thickness of the detector phosphor screen
            correctSample - correct for sample absorption in a flat liquid sheet
            musample - absorption coefficient of the sample
            lsample - thickness of the flat sheet
            dezinger - whether you want to dezinger images  (boolean) - depreciated
            plotting - whether you want to plot the output results (boolean)
            nMax - number of Images you want to integrate. All imges will be 
            integrates if nMax is None.
            
            Method updates the following fields:
            self.q - transferred momentum in A^{-1}
            self.total.s_raw - raw integrated (total) curve
                       s - total curve normalized using trapz in qNormRange
                       normInt - normalization value
                       delay - numerical time delay (in s)
                       delay_str - sting time delay (ex: '5us', '100ns')
                       timeStamp - time when the image was measured (epoch)
                       timeStamp_str - time when the image was measured (string)
                       scanStamp - scan (logFile) name
                       imageAv - average image from all the scans
                       isOutlier - creates isOutlier bool array. At this point 
                       all the curves are NOT outliers.
            
        '''
        # Get all the ingredients for integration:
        if not hasattr(self, 'AIGeometry'):
            self._getAIGeometry(energy, distance, pixelSize, centerX, centerY,
                                qRange, nqpt, qNormRange)
        if maskPath:
            assert isinstance(maskPath, str), 'maskPath should be string'
            assert Path(maskPath).is_file(), maskPath + ' file (mask) not found'
            maskImage = fabio.open(maskPath).data
        else:
            maskImage = None

        self.total = DataContainer()
        self.total.s = np.zeros([nqpt, self.nFiles])
        self.total.s_raw = np.zeros([nqpt, self.nFiles])
        self.total.normInt = np.zeros(self.nFiles)
        self.total.delay = self.logData['delay'].values
        self.total.delay_str = self.logData['delay_str'].values
        self.total.timeStamp = self.logData['timeStamp'].values
        self.total.timeStamp_str = self.logData['timeStamp_str'].values
        self.total.scanStamp = self.logData['Scan'].values
        self.total.isOutlier = np.zeros(self.nFiles, dtype=bool)

        print('*** Integration ***')
        for i, row in self.logData.iterrows():
            impath = row['dataInDir'] + row['file']
            startReadTime = time.perf_counter()
            image = fabio.open(impath).data
            readTime = time.perf_counter() - startReadTime

            startIntTime = time.perf_counter()
            q, self.total.s_raw[:, i] = self.AIGeometry.ai.integrate1d(
                image,
                nqpt,
                radial_range=qRange,
                correctSolidAngle=True,
                polarization_factor=1,
                mask=maskImage,
                unit="q_A^-1")

            if i == 0:
                self.imageAv = image
            else:
                self.imageAv += image

            intTime = time.perf_counter() - startIntTime
            print(i + 1, '|',
                  row['file'], ':',
                  'readout: %.0f' % (readTime * 1e3),
                  'ms | integration: %.0f' % (intTime * 1e3),
                  'ms | total: %.0f' % ((intTime + readTime) * 1e3), 'ms')

            if nFiles:
                if (i + 1) >= nFiles:
                    break

        print('*** Integration done ***\n')
        self.q = q
        self.tth = 2 * np.arcsin(self.AIGeometry.wavelength * 1e10 * self.q / (4 * pi)) / pi * 180

        Corrections = np.ones(q.shape)
        if correctPhosphor:
            Tphos = self._getPhosphorAbsorptionCorrection(muphos, lphos, self.tth / 180 * pi)
            Corrections *= Tphos
        if correctSample:
            Tsample = self._getSampleAbsorptionCorrection(musample, lsample, self.tth / 180 * pi)
            Corrections *= Tsample
        #        print(Corrections)
        #        self.total.s = self.total.s*Corrections[:, np.newaxis]
        self.total.s_raw *= Corrections[:, np.newaxis]

        self.total.normInt, self.total.s = normalizeQ(q, self.total.s_raw, qNormRange)
        self.imageAv = self.imageAv / (i + 1)
        self.imageAv[maskImage == 1] = 0

        if plotting:
            self.plotIntegrationResult()


    def _getAIGeometry(self, energy, distance, pixelSize, centerX, centerY,
                       qRange, nqpt, qNormRange):
        '''Method for storing the geometry parameters in self.AIGeometry from
        the input to self.integrate() method.
        '''

        self.AIGeometry = DataContainer()
        self.AIGeometry.energy = energy
        self.AIGeometry.wavelength = 12.3984 / energy * 1e-10  # in m
        self.AIGeometry.distance = distance
        self.AIGeometry.pixelSize = pixelSize
        self.AIGeometry.centerX = centerX
        self.AIGeometry.centerY = centerY
        self.AIGeometry.qRange = qRange
        self.AIGeometry.nqpt = nqpt
        self.AIGeometry.qNormRange = qNormRange
        self.AIGeometry.ai = self._getai()


    def _getai(self):
        return pyFAI.AzimuthalIntegrator(
            dist=self.AIGeometry.distance * 1e-3,
            poni1=self.AIGeometry.centerY * self.AIGeometry.pixelSize,
            poni2=self.AIGeometry.centerX * self.AIGeometry.pixelSize,
            pixel1=self.AIGeometry.pixelSize,
            pixel2=self.AIGeometry.pixelSize,
            rot1=0, rot2=0, rot3=0,
            wavelength=self.AIGeometry.wavelength)


    def _getTimeStamps(self):
        ''' Method for getting time stamps in a standard date-time and epoch formats.
            This function uses the attributes of the class and does a return of
            time delays in numerical and string formats. This is done for
            the readability of the code in the main function.
        '''
        timeStamp_str = self.logData['date time'].tolist()
        timeStamp = []
        for t in timeStamp_str:
            timeStamp.append(datetime.strptime(t, '%d-%b-%y %H:%M:%S').timestamp())
        timeStamp = np.array(timeStamp)
        timeStamp_str = np.array(timeStamp_str)
        return timeStamp, timeStamp_str


    def plotIntegrationResult(self):
        # get min and max of scale for average image
        vmin, vmax = np.percentile(self.imageAv[self.imageAv != 0], (5, 95))

        plt.figure(figsize=(12, 5))
        plt.clf()

        plt.subplot(131)
        x, y = self.imageAv.shape
        extent = [-self.AIGeometry.centerX,
                  -self.AIGeometry.centerX + x,
                  -self.AIGeometry.centerY + y,
                  -self.AIGeometry.centerY]
        plt.imshow(self.imageAv, vmin=vmin, vmax=vmax, extent=extent, cmap='Greys')
        plt.hlines(0, extent[2], extent[3], colors='r')
        plt.vlines(0, extent[0], extent[1], colors='r')
        plt.xlim(extent[:2])
        plt.ylim(extent[2:])
        plt.colorbar()
        plt.title('Average image')

        plt.subplot(132)
        plt.plot(self.q, self.total.s_raw)
        plt.xlabel('q, $\AA^{-1}$')
        plt.ylabel('Intensity, counts')
        plt.title('All integrated curves')

        plt.subplot(133)
        plt.plot(self.q, self.total.s)
        plt.xlabel('q, $\AA^{-1}$')
        plt.ylabel('Intensity, a.u.')
        plt.title('All integrated curves (normalized)')

        plt.tight_layout()

    def _getPhosphorAbsorptionCorrection(self, mu, l, tth):
        cv = np.cos(tth)  # cos value
        cph = mu * l  # coef phosphor
        Tphos = (1 - np.exp(-cph)) / (1 - np.exp(-cph / cv))
        return Tphos


    def _getSampleAbsorptionCorrection(self, mu, l, tth):
        cv = np.cos(tth)  # cos value
        csa = mu * l
        T = 1 / csa * cv / (1 - cv) * (np.exp(-csa) - np.exp(-csa / cv))
        T0 = np.exp(-csa)
        return T0 / T


    # 2. Total curve averaging


    def getTotalAverages(self, fraction=0.9, chisqThresh=5, q_break=None,
                         dezinger=True, dezingerThresh=5, covShrinkage=0,
                         plotting=True, chisqHistMax=10, y_offset=None):
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
        
        dezinger - flag in case if you want to dezinger the 1d curves
        dezingerThresh - threshold for dezingering (# of stds)
        estimateCov - estimate covariance of total curves (usually not needed) 
        useCovShrinkage - use regularized method for covariance estimation
        covShrinkage - amount of shrinkage; if None, will be determined
        automatically (slow)
        
        plotting - True if you want to see the results of the outlier rejection
        chisqHistMax - maximum value of chisq you want to plot histograms
        y_offset - amount of offset you want in the plot to show different time
        delays
        '''
        print('*** Averaging the total curves ***')
        self.total.s_av, _, self.total.isOutlier, _, _, self.total.chisq = \
            getAverage(self.q, self.total.s, None, self.total.isOutlier, self.total.delay_str, self.t_str, None,
                       fraction, chisqThresh, q_break,
                       dezinger=dezinger, dezingerThresh=dezingerThresh, covShrinkage=covShrinkage,
                       plotting=plotting, chisqHistMax=chisqHistMax, y_offset=y_offset)
        print('*** Done ***')

    # 3. Difference calculation

    def getDifferences(self, toff_str='-5us', subtractFlag='MovingAverage',
                       renormalize=False, qNormRange=None):
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
        
        renormalize - flag for renormalization of total curves. If True, will
        request keyword qNormRange and will use it to update total.s field.
        qNormRange - range for renormalization
        
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

        self.diff.Adiff = sparse.eye(self.nFiles).tolil()

        for i in range(self.nFiles):

            idx_next = self._findOffIdx(i, 'Next')
            idx_prev = self._findOffIdx(i, 'Prev')

            if subtractFlag == 'Next':
                if idx_next:
                    self.diff.Adiff[i, idx_next] = -1

            elif subtractFlag == 'Previous':
                if idx_prev:
                    self.diff.Adiff[i, idx_prev] = -1

            elif subtractFlag == 'Closest':
                if self.total.delay_str[i] == self.diff.toff_str:  # this is to avoid getting the same differences
                    if idx_next:
                        self.diff.Adiff[i, idx_next] = -1
                else:
                    if (idx_next) and (idx_prev):
                        timeToNext = np.abs(self.total.timeStamp[i] - self.total.timeStamp[idx_next])
                        timeToPrev = np.abs(self.total.timeStamp[i] - self.total.timeStamp[idx_prev])
                        if timeToNext <= timeToPrev:
                            self.diff.Adiff[i, idx_next] = -1
                        else:
                            self.diff.Adiff[i, idx_prev] = -1
                    elif (idx_next) and (not idx_prev):
                        self.diff.Adiff[i, idx_next] = -1
                    elif (not idx_next) and (idx_prev):
                        self.diff.Adiff[i, idx_prev] = -1

            elif subtractFlag == 'MovingAverage':
                if (idx_next) and (idx_prev):
                    timeToNext = np.abs(self.total.timeStamp[i] -
                                        self.total.timeStamp[idx_next])
                    timeToPrev = np.abs(self.total.timeStamp[i] -
                                        self.total.timeStamp[idx_prev])
                    timeDiff = np.abs(self.total.timeStamp[idx_next] -
                                      self.total.timeStamp[idx_prev])
                    self.diff.Adiff[i, idx_next] = -timeToPrev / timeDiff
                    self.diff.Adiff[i, idx_prev] = -timeToNext / timeDiff

                elif (idx_next) and (not idx_prev):
                    self.diff.Adiff[i, idx_next] = -1

                elif (not idx_next) and (idx_prev):
                    self.diff.Adiff[i, idx_prev] = -1

        self.diff.ds = (self.diff.Adiff @ self.total.s.T).T

        self.diff.delay = self.total.delay
        self.diff.delay_str = self.total.delay_str
        self.diff.timeStamp = self.total.timeStamp
        self.diff.timeStamp_str = self.total.timeStamp_str
        self.diff.scanStamp = self.total.scanStamp

        self.diff.isOutlier = np.ravel((np.sum(self.diff.Adiff,
                                               axis=1) > 1e-6))  # argument of np.ravel is of matrix type which has ravel method working differently from np.ravel; we need np.ravel!
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
            if (idx < 0) or idx > (self.nFiles - 1):
                return None
            if ((self.total.delay_str[idx] == self.diff.toff_str) and  # find next/prev reference
                    (self.total.scanStamp[idx] == self.total.scanStamp[idx_start])):  # and # should be in the same scan
                #                (not self.total.isOutlier[idx])): # should not be an outlier
                return idx

    # 4. Difference averaging

    def getDiffAverages(self, weightedAveraging=False, fraction=0.9, chisqThresh=1.5,
                        q_break=None, chisqThresh_lowq=1.5, chisqThresh_highq=1.5,
                        plotting=False, chisqHistMax=10, y_offset=None,
                        dezinger=True, dezingerThresh=5,
                        estimateCov=True, useCovShrinkage=True, covShrinkage=None):
        ''' Method to get average differences. It works in the same way as
        getTotalAverages, so refer to the information on input/output in the
        getTotalAverages docstring.
        '''
        print('*** Averaging the difference curves ***')

        if weightedAveraging:
            weights = 1 / self.total.normInt
            weights /= np.median(weights)
            self.diff.covii = self.diff.Adiff @ np.diag(weights) @ self.diff.Adiff.T
        else:
            self.diff.covii = self.diff.Adiff @ self.diff.Adiff.T

        (self.diff.ds_av, self.diff.ds_err, self.diff.isOutlier,
         self.diff.covtt, self.diff.covqq, self.diff.chisq) = \
            getAverage(self.q, self.diff.ds, self.diff.covii,
                       self.diff.isOutlier, self.diff.delay_str, self.t_str, self.diff.toff_str,
                       fraction, chisqThresh,
                       q_break, chisqThresh_lowq, chisqThresh_highq,
                       plotting=plotting, chisqHistMax=chisqHistMax, y_offset=y_offset,
                       dezinger=dezinger, dezingerThresh=dezingerThresh,
                       estimateCov=estimateCov, useCovShrinkage=useCovShrinkage,
                       covShrinkage=covShrinkage)
        print('*** Done ***')

    # 5. Saving

    def save(self, savePath=None):

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
                    print('Group:', attr)
                    for subattr in dir(obj):
                        if not (subattr.startswith('_') or
                                subattr.startswith('ai')):
                            data_to_record = obj.__getattribute__(subattr)
                            #                            print(data_to_record)
                            if not ((type(data_to_record) == int) or
                                    (type(data_to_record) == float)):
                                if ((type(data_to_record) == str) or
                                        (type(data_to_record[0]) == str)):
                                    data_to_record = '|'.join([i for i in data_to_record])

                            if not sparse.issparse(data_to_record):
                                f.create_dataset(attr + '/' + subattr, data=data_to_record)
                                print('\t' + subattr, 'saved')

                elif attr == 'logData':
                    self.logData.to_hdf(savePath, key='logData', mode='r+')
                    print(attr, 'saved')

                else:
                    data_to_record = self.__getattribute__(attr)
                    if not ((type(data_to_record) == int) or
                            (type(data_to_record) == float)):
                        if ((type(data_to_record) == str) or
                                (type(data_to_record[0]) == str)):
                            data_to_record = '|'.join([i for i in data_to_record])
                    f.create_dataset(attr, data=data_to_record)
                    print(attr, 'saved')

        f.close()

        print('*** Done ***')

    def load(self, loadPath):

        assert isinstance(loadPath, str), \
            'Provide data output directory as a string'
        assert Path(loadPath).is_file(), \
            'The file has not been found'

        print('*** Loading ***')
        f = h5py.File(loadPath, 'r')

        for key in f.keys():
            if type(f[key]) == h5py.Dataset:
                if type(f[key].value) == str:
                    data_to_load = np.array(f[key].value.split('|'))
                else:
                    data_to_load = f[key].value
                self.__setattr__(key, data_to_load)
                print(key, 'loaded')

            elif (type(f[key]) == h5py.Group) and (key != 'logData'):
                print('Group:', key)
                self.__setattr__(key, DataContainer())
                for subkey in f[key]:
                    if type(f[key][subkey].value) == str:
                        data_to_load = np.array(f[key][subkey].value.split('|'))
                    else:
                        data_to_load = f[key][subkey].value
                    self.__getattribute__(key).__setattr__(subkey, data_to_load)
                    print('\t', subkey, 'loaded')

            elif (key == 'logData'):
                self.logData = pd.read_hdf(loadPath, key=key)
                print(key, 'loaded')

        f.close()

        if hasattr(self, 'AIGeometry'):
            self.AIGeometry.ai = self._getai()

        print('*** Loading finished ***')


# %% Auxillary functions
# This functions are put outside of the class as they can be used in broader
# contexts and such implementation simplifies access to them.


class DataContainer:
    # dummy class to store the data
    def __init__(self):
        pass


def _get_id09_columns_old():
    return ['date', 'time', 'file',
            'delay', 'delay_act', 'delay_act_std', 'delay_act_min', 'delay_act_max',
            'laser', 'laser_std', 'laser_min', 'laser_max', 'laser_n',
            'xray', 'xray_std', 'xray_min', 'xray_max', 'xray_n',
            'n_pulses']


def _get_id09_columns():
    return ['date', 'time', 'file',
            'delay', 'delay_act', 'delay_act_std', 'delay_act_min', 'delay_act_max', 'delay_n',
            'laser', 'laser_std', 'laser_min', 'laser_max', 'laser_n',
            'xray', 'xray_std', 'xray_min', 'xray_max', 'xray_n',
            'n_pulses']


def medianDezinger(img_orig, mask):
    ''' Function for image dezingering.
    
    You need:
        img_orig - the image you want to dezinger
        mask - the mask for this image (can be None)
    
    Output:
        img - dezingered image
    '''
    img = img_orig.copy()
    img_blur = median_filter(img, size=(3, 3))
    img_diff = (img.astype(float) - img_blur.astype(float))
    if ~mask:
        threshold = np.std(img_diff[mask.astype(bool)]) * 10
        hot_pixels = np.abs(img_diff) > threshold
        hot_pixels[mask.astype(bool)] = 0
    else:
        threshold = np.std(img_diff) * 10
        hot_pixels = np.abs(img_diff) > threshold
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
    idx_nnz = x != 0
    x_nnz = x[idx_nnz]
    x_nnz_filt = signal.medfilt(x_nnz, kernel_size=kernel_size)
    dx = x_nnz - x_nnz_filt
    dx_sel = getMedianSelection(dx[dx != 0][np.newaxis], fraction)
    threshold = np.std(dx_sel) * thresh
    hot_pixels = np.abs(dx) > threshold
    x_nnz_dez = x_nnz
    x_nnz_dez[hot_pixels] = x_nnz_filt[hot_pixels]
    x[idx_nnz] = x_nnz_dez
    return x


def normalizeQ(q, s_raw, qNormRange):
    qNormRangeSel = (q >= qNormRange[0]) & (q <= qNormRange[1])
    normInt = np.trapz(s_raw[qNormRangeSel, :], q[qNormRangeSel], axis=0)
    return normInt, s_raw / normInt


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
            t = t_number * 1e-12
        elif 'ns' in t_str:
            t = t_number * 1e-9
        elif 'us' in t_str:
            t = t_number * 1e-6
        elif 'ms' in t_str:
            t = t_number * 1e-3
    return t


def time_num2str(t):
    ''' Function for converting time delays to string format
        Input: time delay in s
        Output: time string
    '''

    def convertToString(t, factor):
        t_r0 = round(t * factor)
        t_r3 = round(t * factor, 3)
        if t_r3 == t_r0:
            return str(int(t_r0))
        else:
            return str(t_r3)

    if t == 0: return '0'
    A = np.log10(np.abs(t))
    if (A < -12):
        t_str = convertToString(t, 1e15) + 'fs'
    elif (A >= -12) and (A < -9):
        t_str = convertToString(t, 1e12) + 'ps'
    elif (A >= -9) and (A < -6):
        t_str = convertToString(t, 1e9) + 'ns'
    elif (A >= -6) and (A < -3):
        t_str = convertToString(t, 1e6) + 'us'
    elif (A >= -3) and (A < 0):
        t_str = convertToString(t, 1e3) + 'ms'
    else:
        t_str = str(round(t, 3))
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
        offsTBS_loc = offsTBS[i, :]
        stampOffs = timeStamp[offsTBS_loc]
        stampRange = np.diff(stampOffs)
        if stampRange.size == 0:
            mapDiff[i, offsTBS_loc] = -1
        elif stampRange.size == 1:
            stampDiffs = abs(stampOn - stampOffs)
            weights = -(stampDiffs / stampRange)[::-1]
            mapDiff[i, offsTBS_loc] = weights
        else:
            print('Cannot compute difference for', i, 'curve')
            print('diagnostics: i:', i, '; stampOn:', stampOn,
                  '; stampOffs:', stampOffs,
                  '; stampDiffs:', abs(stampOn - stampOffs),
                  '; offs TBS: ', np.where(offsTBS_loc))
            raise ValueError('Variable stampThresh is too large. '
                             'Decrease the multiplication factor.')
    return mapDiff


def getAverage(q, x_orig, covii, isOutlier, delay_str, t_str, toff_str,
               fraction, chisqThresh,
               q_break=None,
               dezinger=True, dezingerThresh=5,
               covShrinkage=0,
               plotting=False, chisqHistMax=10, y_offset=None):
    ''' Function for calculating averages and standard errors of data sets.
    For details see ScatData.getTotalAverages method docstring.
    '''

    x = x_orig.copy()
    if covii is None:
        covii = np.eye(delay_str.size)

    x_av = np.zeros((q.size, t_str.size))

    chisqThresh = np.array(chisqThresh)
    if chisqThresh.size>1:
        q_break = np.array(q_break)
        assert q_break.size == (chisqThresh.size - 1)
        q_break = np.hstack((q.min()-1, q_break, q.max()+1))
    else:
        chisqThresh = chisqThresh[None]
        q_break = np.array([q.min()-1, q.max()+1])

    chisq = np.zeros((chisqThresh.size, delay_str.size))

    # For propoper average estimation we will need Amean operator
    # Amean is defined such that x_mean @ Amean = X
    Amean = np.zeros((t_str.size, delay_str.size))

    print('Identifying outliers ... ', end='');
    outlierStartTime = time.perf_counter()

    for i, delay_point in enumerate(t_str):
        delay_selection = (delay_str == delay_point) & ~isOutlier
        x_loc = x[:, delay_selection]

        isOutlier_loc, chisq_loc, isHotPixel_loc = identifyOutliers(
            q, x_loc, fraction, chisqThresh, q_break, dezingerThresh=dezingerThresh)

        isOutlier[delay_selection] = isOutlier_loc
        chisq[:, delay_selection] = chisq_loc

        if dezinger:
            x_loc_med = np.tile(np.median(x_loc, axis=1)[:, None], x_loc.shape[1])
            x_loc[isHotPixel_loc] = x_loc_med[isHotPixel_loc]
            x[:, delay_selection] = x_loc

        Amean[i, delay_selection & ~isOutlier] = 1

    print('done ( %3.f' % ((time.perf_counter() - outlierStartTime) * 1000), 'ms )')

    print('Averaging ... ', end='');
    averageStartTime = time.perf_counter()

    if sparse.issparse(covii): covii = covii.toarray()

    # an inversion of the covii matrix is expensive, so for computing of the
    # weighed average we approximate it with the inverse of the diagonal:
    covii_diag_inv = np.diag(1 / np.diag(covii))

    H = np.linalg.pinv(Amean @ covii_diag_inv @ Amean.T) @ Amean @ covii_diag_inv
    x_av = (H @ x.T).T
    print('done ( %3.f' % ((time.perf_counter() - averageStartTime) * 1000), 'ms )')

    print('Uncertainty propagation  ... ', end='');
    covtt = H @ covii @ H.T
    covarStartTime = time.perf_counter()

    dx = (x - x_av @ Amean) @ np.sqrt(covii_diag_inv)
    dx = dx[:, ~isOutlier & (delay_str != toff_str)]

    if covShrinkage is None:
        _, covShrinkage = cov_shrink_ss(dx.T.copy(order='C'))

    dxdxt = dx @ dx.T
    dxdxt_diag = np.diag(np.diag(dxdxt))

    covqq = (dxdxt * (1 - covShrinkage) + dxdxt_diag * covShrinkage) / (dx.shape[1] - t_str.size + 1)

    x_err = np.sqrt(np.diag(covqq)[:, None] * np.diag(covtt)[None, :])

    print('done ( %3.f' % ((time.perf_counter() - covarStartTime) * 1000), 'ms )')


    if plotting:
        pass
    #        if y_offset is None:
    #            y_offset = (np.max(np.percentile(x, 95, axis=1)) -
    #                        np.min(np.percentile(x,  5, axis=1)))*1.1
    #        else:
    #            assert type(y_offset)==float, 'y_offset should be either float or None'

    #        plt.figure(figsize = (t_str.size*2,4))
    #        plt.clf()
    #
    #        if plotting:
    #            subidx = i
    #            plotOutliers(q, x_loc, isOutlier_loc,
    #                         chisq, chisqThresh,
    #                         q_break, chisq_lowq, chisqThresh_lowq, chisq_highq, chisqThresh_highq,
    #                         chisqHistMax, subidx, y_offset)

    return x_av, x_err, isOutlier, covtt, covqq, chisq


def identifyOutliers(q_orig, y_orig, fraction, chisqThresh,
                     q_break, dezingerThresh=5):
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
    
    The function also provides estimation of hot pixels
    
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
    chisq = np.zeros((chisqThresh.size, y.shape[1]))

    ySel = getMedianSelection(y, fraction)
    ySel_av = np.mean(ySel, axis=1)
    ySel_std = np.std(ySel, axis=1)
    nnzStd = ySel_std != 0
    errsq = ((y[nnzStd, :] - ySel_av[nnzStd, np.newaxis]) /
             ySel_std[nnzStd, np.newaxis]) ** 2
    q_errsq = q[nnzStd]

    for idx, each_chisqThresh in enumerate(chisqThresh):
        q_sel = (q_errsq > q_break[idx]) & (q_errsq <= q_break[idx+1])
        chisq[idx, :] = np.nansum(errsq[q_sel, :], axis=0)/np.sum(q_sel)

    isOutlier = np.any(chisq >= chisqThresh[:, None], axis=0)
    isHotPixel = np.abs(y - ySel_av[:, None]) > dezingerThresh * ySel_std[:, None]

    return isOutlier, chisq, isHotPixel


def getMedianSelection(z_orig, frac):
    ''' Function to get selection of data from symmetric percentiles determined
    by fraction. For example if fraction is 0.9, then the output data is the
    selection between 0.05 and 0.95 percentiles.
    '''
    z = z_orig.copy()
    z = np.sort(z, axis=1)
    ncols = z.shape[1]
    low = np.int(np.round((1 - frac) / 2 * ncols))
    high = np.int(np.round((1 + frac) / 2 * ncols))
    z = z[:, low:high]
    return z


def plotOutliers(q, x, isOutlier, chisq,
                 chisqThresh, q_break,
                 chisqHistMax, y_offset):
    ''' Fucntion to plot data and corresponding chisq histograms.
    '''


    plt.subplot(121)



    plt.plot([q[0], q[-1]], np.array([0, 0]) - subidx * y_offset, '-', color=(0.5, 0.5, 0.5))
    if any(isOutlier):
        plt.plot(q, x[:, isOutlier] - subidx * y_offset, 'b-')
    if any(~isOutlier):
        x_mean = np.mean(x[:, ~isOutlier], axis=1)
        plt.plot(q, x[:, ~isOutlier] - subidx * y_offset, 'k-')
        plt.plot(q, x_mean - subidx * y_offset, 'r-')

    plt.xlabel('q, A^-1')
    plt.ylabel('Intentsity, a.u.')
    plt.title('Curve selection. \n Blue - outliers, Black - Selected data, Red - average')
    #    plt.legend()

    plt.subplot(122)
    chisqBins = np.concatenate((np.arange(0, chisqHistMax + 0.5, 0.5),
                                np.array(np.inf)[np.newaxis]))
    chisqWidths = np.diff(chisqBins)
    chisqWidths[-1] = 1
    if not q_break:
        heights, _ = np.histogram(chisq, bins=chisqBins)
        heights = heights / np.max(heights) * y_offset * 0.9
        plt.plot([chisqBins[0], chisqBins[-2]], np.array([0, 0]) - subidx * y_offset, '-', color=(0.5, 0.5, 0.5))
        plt.plot(chisqBins[:-1], heights - subidx * y_offset, 'k.-')
        plt.plot(chisqThresh * np.array([1, 1]), np.array([0, 0.9]) * y_offset - subidx * y_offset, 'k--')
        plt.xlabel('\chi^2')
        plt.ylabel('n. occurances')
        plt.title('\chi^2 occurances')

    else:
        heights_lowq, _ = np.histogram(chisq_lowq, bins=chisqBins)
        heights_highq, _ = np.histogram(chisq_highq, bins=chisqBins)
        heights_lowq = heights_lowq / np.max(heights_lowq) * y_offset * 0.9
        heights_highq = heights_highq / np.max(heights_highq) * y_offset * 0.9

        plt.plot([chisqBins[0], chisqBins[-2]], np.array([0, 0]) - subidx * y_offset, '-', color=(0.5, 0.5, 0.5))
        plt.plot(chisqBins[:-1], heights_lowq - subidx * y_offset, 'b.-')
        plt.plot(chisqBins[:-1], heights_highq - subidx * y_offset, 'r.-')
        plt.plot(chisqThresh_lowq * np.array([1, 1]), np.array([0, 0.9]) * y_offset - subidx * y_offset, 'b--')
        plt.plot(chisqThresh_highq * np.array([1, 1]), np.array([0, 0.9]) * y_offset - subidx * y_offset, 'r--')
        plt.xlabel('\chi_lowq^2')
        plt.ylabel('n. occurances')
        plt.title('\chi^2 occurances\nBlue - low q, Red - high q')


def rescaleQ(q_old, wavelength, dist_old, dist_new):
    tth_old = 2 * np.arcsin(wavelength * q_old / 4 / pi)
    r = np.arctan(tth_old) * dist_old
    tth_new = np.tan(r / dist_new)
    return 4 * pi / wavelength * np.sin(tth_new / 2)


# %%
if __name__ == '__main__':
    # Dirty Checking: integration
    A = ScatData(r'C:\work\Experiments\2017\Ubiquitin\10.log',
              r'C:\work\Experiments\2017\Ubiquitin\\')

    # %%
    A.integrate(energy=11.63,
                distance=364,
                pixelSize=82e-6,
                centerX=1987,
                centerY=1965,
                qRange=[0.0, 4.0],
                nqpt=400,
                qNormRange=[1.9, 2.1],
                maskPath=r'C:\work\Experiments\2017\Ubiquitin\MASK_UB.edf',
                plotting=True)
    # %% idnetify outliers in total curves
    A.getTotalAverages(fraction=0.9, chisqThresh=6)
#
#    # %% difference calculation
#    A.getDifferences(toff_str='-5us',
#                     subtractFlag='MovingAverage')
#
#    # %%
#
#    #    A.getDiffAverages(fraction=0.9, chisqThresh=2.5)
#    A.getDiffAverages(fraction=0.9, q_break=2, chisqThresh_lowq=2.5, chisqThresh_highq=3)
