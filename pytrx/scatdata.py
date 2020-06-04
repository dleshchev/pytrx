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
import scipy.io

from pytrx.utils import DataContainer, _get_id09_columns_old, _get_id09_columns, time_str2num, time_num2str, \
    invert_banded

from matplotlib.lines import Line2D

custom_lines = [Line2D([0], [0], color='k', lw=1),
                Line2D([0], [0], color='b', lw=1),
                Line2D([0], [0], color='r', lw=1)]

ignore_these_fields = ['covii', 'imageAv', 'logData']


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

    def __init__(self, inputFile, logFileStyle='biocars', ignoreFirst=False, nFirstFiles=None, dataInDir=None,
                 smallLoad=False):
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
        extension = None
        if type(inputFile) is str:
            extension = Path(inputFile).suffix
        elif type(inputFile) is list:
            extension = Path(inputFile[0]).suffix
        #
        if extension == '.log':
            self.initializeLogFile(inputFile, logFileStyle, ignoreFirst, nFirstFiles, dataInDir)
        elif extension == '.h5':
            self.initializeFromH5(inputFile, smallLoad)

        if inputFile is None:
            self.initializeEmpty()

    def initializeLogFile(self, logFile, logFileStyle, ignoreFirst, nFirstFiles, dataInDir):
        if dataInDir is None:
            if type(logFile) is str:
                dataInDir = str(Path(logFile).parent.absolute()) + '\\'
            else:
                dataInDir = [(str(Path(i).parent.absolute()) + '\\') for i in logFile]
        self.logFile = logFile
        self.dataInDir = dataInDir
        self.logFileStyle = logFileStyle
        self.ignoreFirst = ignoreFirst
        self.nFirstFiles = nFirstFiles

        self._assertCorrectInput()
        self._getLogData()
        self._identifyExistingFiles()

        self.logSummary()

    def _getLogData(self):
        '''
        Read log file(s) and convert them into a pandas dataframe
        '''
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
        if not hasattr(self, 'aiGeometry'):
            self._getaiGeometry(energy, distance, pixelSize, centerX, centerY,
                                qRange, nqpt, qNormRange)
        if maskPath:
            assert isinstance(maskPath, str), 'maskPath should be string'
            assert Path(maskPath).is_file(), maskPath + ' file (mask) not found'
            maskImage = fabio.open(maskPath).data
        else:
            maskImage = None

        self.total = IntensityContainer()
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
            q, self.total.s_raw[:, i] = self.aiGeometry.ai.integrate1d(
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
        self.tth = 2 * np.arcsin(self.aiGeometry.wavelength * 1e10 * self.q / (4 * pi)) / pi * 180

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

        weights = 1 / self.total.normInt
        weights /= np.median(weights)
        self.total.covii = np.diag(weights)

        if plotting:
            self.plotIntegrationResult()

    def _getaiGeometry(self, energy, distance, pixelSize, centerX, centerY, qRange, nqpt, qNormRange):
        '''Method for storing the geometry parameters in self.aiGeometry from
        the input to self.integrate() method.
        '''
        self.aiGeometry = AIGeometry(energy, distance, pixelSize, centerX, centerY, qRange, nqpt, qNormRange)
        self.aiGeometry.getai()

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
        vmin, vmax = np.percentile(self.imageAv[self.imageAv != 0], (1, 99))

        plt.figure(figsize=(12, 5))
        plt.clf()

        plt.subplot(131)
        x, y = self.imageAv.shape
        extent = [-self.aiGeometry.centerX,
                  -self.aiGeometry.centerX + x,
                  -self.aiGeometry.centerY + y,
                  -self.aiGeometry.centerY]
        plt.imshow(self.imageAv, vmin=vmin, vmax=vmax, extent=extent, cmap='Greys')
        plt.vlines(0, extent[2], extent[3], colors='r')
        plt.hlines(0, extent[0], extent[1], colors='r')
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
        self.total.s_av, self.total.s_err, self.total.isOutlier, self.total.covtt, self.total.covqq, self.total.chisq = \
            getAverage(self.q, self.total.s, self.total.covii, self.total.isOutlier,
                       self.total.delay_str, self.t_str, None,
                       fraction, chisqThresh, q_break,
                       dezinger=dezinger, dezingerThresh=dezingerThresh, covShrinkage=covShrinkage,
                       plotting=plotting, chisqHistMax=chisqHistMax, y_offset=y_offset)
        print('*** Done ***\n')

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
        self.diff.s - difference curves
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
            self.aiGeometry.qNormRange = np.array(qNormRange)
            self.total.normInt, self.total.s = normalizeQ(self.q, self.total.s_raw, qNormRange)

        self.diff = IntensityContainer()
        self.diff.delay = self.total.delay
        self.diff.delay_str = self.total.delay_str
        self.diff.timeStamp = self.total.timeStamp
        self.diff.timeStamp_str = self.total.timeStamp_str
        self.diff.scanStamp = self.total.scanStamp

        assert toff_str in self.total.delay_str, 'toff_str is not found among recorded time delays'
        self.diff.toff_str = toff_str

        Adiff = sparse.eye(self.nFiles).tolil()

        for i in range(self.nFiles):

            idx_next = self._findOffIdx(i, 'Next')
            idx_prev = self._findOffIdx(i, 'Prev')

            if subtractFlag == 'Next':
                if idx_next:
                    Adiff[i, idx_next] = -1

            elif subtractFlag == 'Previous':
                if idx_prev:
                    Adiff[i, idx_prev] = -1

            elif subtractFlag == 'Closest':
                if self.total.delay_str[i] == self.diff.toff_str:  # this is to avoid getting the same differences
                    if idx_next:
                        Adiff[i, idx_next] = -1
                else:
                    if (idx_next) and (idx_prev):
                        timeToNext = np.abs(self.total.timeStamp[i] - self.total.timeStamp[idx_next])
                        timeToPrev = np.abs(self.total.timeStamp[i] - self.total.timeStamp[idx_prev])
                        if timeToNext <= timeToPrev:
                            Adiff[i, idx_next] = -1
                        else:
                            Adiff[i, idx_prev] = -1
                    elif (idx_next) and (not idx_prev):
                        Adiff[i, idx_next] = -1
                    elif (not idx_next) and (idx_prev):
                        Adiff[i, idx_prev] = -1

            elif subtractFlag == 'MovingAverage':
                if (idx_next) and (idx_prev):
                    timeToNext = np.abs(self.total.timeStamp[i] -
                                        self.total.timeStamp[idx_next])
                    timeToPrev = np.abs(self.total.timeStamp[i] -
                                        self.total.timeStamp[idx_prev])
                    timeDiff = np.abs(self.total.timeStamp[idx_next] -
                                      self.total.timeStamp[idx_prev])
                    Adiff[i, idx_next] = -timeToPrev / timeDiff
                    Adiff[i, idx_prev] = -timeToNext / timeDiff

                elif (idx_next) and (not idx_prev):
                    Adiff[i, idx_next] = -1

                elif (not idx_next) and (idx_prev):
                    Adiff[i, idx_prev] = -1

        self.diff.s = (Adiff @ self.total.s.T).T
        self.diff.ds = np.copy(self.diff.s)
        self.diff.covii = Adiff @ self.total.covii @ Adiff.T
        self.Adiff = Adiff
        self.diff.isOutlier = np.ravel((np.sum(Adiff,
                                               axis=1) > 1e-6))  # argument of np.ravel is of matrix type which has ravel method working differently from np.ravel; we need np.ravel!

        if subtractFlag == 'MovingAverage':  # handle the rank deficiency for covii
            last_off = np.where((self.diff.delay_str == self.diff.toff_str) & ~self.diff.isOutlier)[0][-1]
            self.diff.isOutlier[last_off] = True
            a = self.diff.covii[last_off, last_off]
            self.diff.covii[last_off, :] = 0
            self.diff.covii[:, last_off] = 0
            self.diff.covii[last_off, last_off] = a

        print('*** Done ***\n')

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

    def getDiffAverages(self, fraction=0.9, chisqThresh=1.5, q_break=None,
                        dezinger=True, dezingerThresh=5, covShrinkage=None,
                        plotting=True, chisqHistMax=10, y_offset=None):
        ''' Method to get average differences. It works in the same way as
        getTotalAverages, so refer to the information on input/output in the
        getTotalAverages docstring.
        '''
        print('*** Averaging the difference curves ***')

        (self.diff.s_av, self.diff.s_err, self.diff.isOutlier,
         self.diff.covtt, self.diff.covqq, self.diff.chisq) = \
            getAverage(self.q, self.diff.s, self.diff.covii,
                       self.diff.isOutlier, self.diff.delay_str, self.t_str, self.diff.toff_str,
                       fraction, chisqThresh, q_break,
                       dezinger=dezinger, dezingerThresh=dezingerThresh, covShrinkage=covShrinkage,
                       plotting=plotting, chisqHistMax=chisqHistMax, y_offset=y_offset)
        self.diff.ds_av = np.copy(self.diff.s_av)
        print('*** Done ***\n')

    def plotDiffAverages(self, fig=None, y_offset=None, x_txt=None, y_txt=None, qpower=0):
        if fig is None:
            plt.figure()
        else:
            plt.figure(fig)
        plotAvData(self.q, self.diff.s_av, self.t_str, y_offset=y_offset, x_txt=x_txt, y_txt=y_txt, qpower=qpower)

    # 5. Saving

    def save(self, savePath=None):

        assert isinstance(savePath, str), \
            'provide data output directory as a string'

        print('*** Saving ***')
        f = h5py.File(savePath, 'w')

        self._save_group(savePath, f, self.__dict__)
        self._save_group(savePath, f, self.aiGeometry.__dict__, indent='\t', group='aiGeometry')
        self._save_group(savePath, f, self.total.__dict__, indent='\t', group='total')
        self._save_group(savePath, f, self.diff.__dict__, indent='\t', group='diff')

        f.close()
        print('*** Saving finished ***')

    def _save_group(self, savepath, f, dict, indent='', group=None):
        if group:
            print('Saving the following group:', group)
            fpath = group + '/'
        else:
            print('Saving following attributes')
            fpath = ''

        for key in dict.keys():
            if (dict[key] is None
                    or (type(dict[key]) == IntensityContainer)
                    or (type(dict[key]) == AIGeometry)
                    or (key == 'ai')
                    or (key == 'Adiff')):
                continue

            print(indent, key, '\t', end=' ')

            if type(dict[key]) == pd.core.frame.DataFrame:
                try:
                    dict[key].to_hdf(savepath, key='logData', mode='r+')
                    print('success')
                except:
                    'failed'
            else:
                try:
                    if ((type(dict[key]) == list)
                            or ((type(dict[key]) == np.ndarray) and (
                                    (type(dict[key][0]) == str) or (type(dict[key][0]) == np.str_)))):
                        data = '|'.join(dict[key])
                    else:
                        data = dict[key]
                    f.create_dataset(fpath + key, data=data)

                    print('success')
                except:
                    print('failed')

    def initializeFromH5(self, loadPath, smallLoad):

        assert isinstance(loadPath, str), \
            'Provide data output directory as a string'
        assert Path(loadPath).is_file(), \
            'The file has not been found'

        print('*** Loading ***')
        f = h5py.File(loadPath, 'r')

        for key in f.keys():

            if (smallLoad) and (key in ignore_these_fields):
                print(f'{key} skipped to conserve memory')
                continue

            # print(type(f[key].value))  #################################

            if type(f[key]) == h5py.Dataset:
                if type(f[key].value) == str:
                    data_to_load = np.array(f[key].value.split('|'))
                else:
                    data_to_load = f[key].value
                self.__setattr__(key, data_to_load)
                print(key, 'success')

            elif (type(f[key]) == h5py.Group) and (key != 'logData'):

                print('Loading group', key)
                if (key == 'diff') or (key == 'total'):
                    self.__setattr__(key, IntensityContainer())
                elif key == 'aiGeometry':
                    self.__setattr__(key, AIGeometry())
                for subkey in f[key]:

                    if (smallLoad) and (subkey in ignore_these_fields):
                        print('\t', f'{subkey} skipped to conserve memory')
                        continue

                    # print(type(f[key][subkey]))  #################################
                    if type(f[key][subkey].value) == str:
                        data_to_load = np.array(f[key][subkey].value.split('|'))
                    else:
                        data_to_load = f[key][subkey].value
                    self.__getattribute__(key).__setattr__(subkey, data_to_load)
                    print('\t', subkey, 'success')

            elif (key == 'logData'):
                self.logData = pd.read_hdf(loadPath, key=key)
                print(key, 'success')

        f.close()

        if hasattr(self, 'aiGeometry'):
            self.aiGeometry.getai()

        print('*** Loading finished ***')

    def initializeEmpty(self):
        self.q = None
        self.tth = None
        self.t = None
        self.t_str = None
        self.dataInDir = None
        self.ignoreFirst = None
        self.imageAv = None
        self.logFile = None
        self.logFileStyle = None
        self.nDelays = None
        self.nFiles = None
        self.aiGeometry = AIGeometry()
        self.diff = IntensityContainer()
        self.total = IntensityContainer()


class AIGeometry:
    def __init__(self, energy=None, distance=None, pixelSize=None, centerX=None, centerY=None, qRange=None, nqpt=None,
                 qNormRange=None):
        self.energy = energy
        if energy:
            self.wavelength = 12.3984 / energy * 1e-10  # in m
        self.distance = distance
        self.pixelSize = pixelSize
        self.centerX = centerX
        self.centerY = centerY
        self.qRange = np.array(qRange)
        self.nqpt = nqpt
        self.qNormRange = np.array(qNormRange)
        # self.ai = self.getai()

    def getai(self):
        self.ai = pyFAI.AzimuthalIntegrator(
            dist=self.distance * 1e-3,
            poni1=self.centerY * self.pixelSize,
            poni2=self.centerX * self.pixelSize,
            pixel1=self.pixelSize,
            pixel2=self.pixelSize,
            rot1=0, rot2=0, rot3=0,
            wavelength=self.wavelength)


class IntensityContainer:
    def __init__(self, s_raw=None, s=None, s_av=None, s_err=None,
                 s2=None, s2_av=None, s2_err=None,  # These are anisotropy terms
                 normInt=None, covii=None, covqq=None, covtt=None,
                 chisq=None, isOutlier=None,
                 delay=None, delay_str=None, toff_str=None,
                 timeStamp=None, timeStamp_str=None, scanStamp=None):
        self.s_raw = s_raw
        self.s = s
        self.ds = s
        self.s_av = s_av
        self.ds_av = s_av
        self.s_err = s_err
        self.s2 = s2  # DJH 20/05/29, for adding anisotropic intensity
        self.s2_av = s2_av
        self.s2_err = s2_err
        self.normInt = normInt
        self.covii = covii
        self.covqq = covqq
        self.covtt = covtt
        self.chisq = chisq
        self.isOutlier = isOutlier
        self.delay = delay
        self.delay_str = delay_str
        self.toff_str = toff_str
        self.timeStamp = timeStamp
        self.timeStamp_str = timeStamp_str
        self.scanStamp = scanStamp

    def scale_by(self, scale):
        if self.s is not None: self.s *= scale
        if self.s_av is not None: self.s_av *= scale
        if self.s_err is not None: self.s_err *= scale
        if self.covqq is not None: self.covqq *= scale ** 2


# %% Auxillary functions
# This functions are put outside of the class as they can be used in broader
# contexts and such implementation simplifies access to them.


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

    if sparse.issparse(covii): covii = covii.toarray()
    if covii is None:
        covii = np.eye(delay_str.size)

    chisqThresh = np.array(chisqThresh)
    if chisqThresh.size > 1:
        q_break = np.array(q_break)
        assert q_break.size == (chisqThresh.size - 1)
        q_break = np.hstack((q.min() - 1e-6, q_break, q.max() + 1e-6))
    else:
        chisqThresh = chisqThresh[None]
        q_break = np.array([q.min() - 1e-6, q.max() + 1e-6])

    isOutlier, chisq = identifyOutliers_all(q, x, delay_str, t_str, isOutlier, fraction, chisqThresh, q_break,
                                            dezinger, dezingerThresh)

    print('Averaging ... ', end='')
    averageStartTime = time.perf_counter()

    # For propoper average estimation we will need Amean operator
    # Amean is defined such that x_mean @ Amean = X
    Amean = np.zeros((t_str.size, delay_str.size))
    for i, delay_point in enumerate(t_str):
        delay_selection = (delay_str == delay_point)
        Amean[i, delay_selection] = 1

    # an inversion of the covii matrix is expensive, so for computing of the
    # weighed average we approximate it with the inverse of the diagonal:
    covii_diag_inv = np.diag(1 / np.diag(covii))

    H = np.linalg.pinv(Amean @ covii_diag_inv @ Amean.T) @ Amean @ covii_diag_inv
    x_av = (H @ x.T).T
    print('done ( %3.f' % ((time.perf_counter() - averageStartTime) * 1000), 'ms )')

    print('Uncertainty propagation  ... ', end='')
    covtt = H @ covii @ H.T
    covarStartTime = time.perf_counter()

    dx = (x - x_av @ Amean) @ np.sqrt(covii_diag_inv)
    dx = dx[:, ~isOutlier & (delay_str != toff_str)]

    if covShrinkage is None:
        _, covShrinkage = cov_shrink_ss(dx.T.copy(order='C'))

    dxdxt = dx @ dx.T
    dxdxt_diag = np.diag(np.diag(dxdxt))

    covqq = (dxdxt * (1 - covShrinkage) + dxdxt_diag * covShrinkage) / (dx.shape[1] - t_str.size)

    x_err = np.sqrt(np.diag(covqq)[:, None] * np.diag(covtt)[None, :])

    print('done ( %3.f' % ((time.perf_counter() - covarStartTime) * 1000), 'ms )')

    if plotting:
        plotOutliers(q, x, delay_str, t_str, isOutlier, chisq,
                     chisqThresh, q_break,
                     chisqHistMax, y_offset)

    return x_av, x_err, isOutlier, covtt, covqq, chisq


def identifyOutliers_all(q, x, delay_str, t_str, isOutlier, fraction, chisqThresh, q_break, dezinger, dezingerThresh):
    print('Identifying outliers ... ')
    outlierStartTime = time.perf_counter()

    chisq = np.zeros((chisqThresh.size, delay_str.size)) + 10

    for i, delay_point in enumerate(t_str):
        delay_selection = (delay_str == delay_point) & ~isOutlier
        x_loc = x[:, delay_selection]

        isOutlier_loc, chisq_loc, isHotPixel_loc = identifyOutliers_one(
            q, x_loc, fraction, chisqThresh, q_break, dezingerThresh=dezingerThresh)
        print(f'Acceptance for {delay_point}: {np.sum(~isOutlier_loc)}/{np.sum((delay_str == delay_point))}')
        isOutlier[delay_selection] = isOutlier_loc
        chisq[:, delay_selection] = chisq_loc

        if dezinger:
            x_loc_med = np.tile(np.median(x_loc, axis=1)[:, None], x_loc.shape[1])
            x_loc[isHotPixel_loc] = x_loc_med[isHotPixel_loc]
            x[:, delay_selection] = x_loc

    print('... done ( %3.f' % ((time.perf_counter() - outlierStartTime) * 1000), 'ms )')
    return isOutlier, chisq


def identifyOutliers_one(q_orig, y_orig, fraction, chisqThresh,
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
        q_sel = (q_errsq > q_break[idx]) & (q_errsq <= q_break[idx + 1])
        chisq[idx, :] = np.nansum(errsq[q_sel, :], axis=0) / np.sum(q_sel)

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


def plotOutliers(q, x, delay_str, t_str, isOutlier, chisq,
                 chisqThresh, q_break,
                 chisqHistMax, y_offset):
    ''' Fucntion to plot data and corresponding chisq histograms.
    '''

    if len(chisq.shape) == 1:
        chisq = chisq[None, :]

    if y_offset is None:
        y_offset = np.abs(np.max(x[:, ~isOutlier]) - np.min(x[:, ~isOutlier]))

    plt.figure(figsize=(12, 7))
    plt.subplot(121)
    for i, each_t in enumerate(t_str):
        y_offset_i = y_offset * i
        sel_loc = delay_str == each_t
        x_loc = x[:, sel_loc]
        isOutlier_loc = isOutlier[sel_loc]
        x_av_loc = np.mean(x_loc[:, ~isOutlier_loc], axis=1)
        plt.plot(q, x_loc[:, ~isOutlier_loc] - y_offset_i, 'k-', alpha=0.33, zorder=5)
        if x_loc[:, isOutlier_loc].size > 0:
            plt.plot(q, x_loc[:, isOutlier_loc] - y_offset_i, 'b-', zorder=10)
        plt.plot(q, x_av_loc - y_offset_i, 'r-', zorder=20)
        # print(np.sum(~isOutlier_loc), isOutlier_loc)
        msg = f'{each_t}\n{np.sum(~isOutlier_loc)}/{len(isOutlier_loc)}'
        plt.text(q.min() + (q.max() - q.min()) * 0.75, y_offset * 0.25 - y_offset_i, msg, va='center', ha='center',
                 backgroundcolor='w', zorder=3)

    plt.hlines(-np.arange(chisqThresh.size) * y_offset, q.min(), q.max())
    plt.legend(custom_lines, ['Data', 'Outliers', 'Mean'])
    plt.xlim(q.min(), q.max())
    plt.xlabel('q, A$^{-1}$')
    plt.ylabel('scaled S, a.u.')
    plt.title('Averaging result')

    YLIM = plt.ylim()
    chisqBins = np.concatenate((np.arange(0, chisqHistMax + 0.5, 0.5),
                                np.array(np.inf)[np.newaxis]))

    plt.subplot(122)
    for i, each_thresh in enumerate(chisqThresh):
        y_offset_i = y_offset * i
        chisq_loc = chisq[i, :]
        heights, _ = np.histogram(chisq_loc, bins=chisqBins)
        heights = heights / np.max(heights) * y_offset * 0.95
        plt.bar(chisqBins[:-1], heights, bottom=-y_offset_i, color='k')
        plt.plot([each_thresh, each_thresh], [-y_offset_i, -y_offset_i + y_offset], 'k--')
        plt.text(each_thresh, y_offset * 0.9 - y_offset_i, 'Threshold', va='center', ha='center', backgroundcolor='w')
        plt.text(chisqBins.min() + (chisqBins[:-2].max() - chisqBins.min()) * 0.75,
                 y_offset * 0.5 - y_offset_i, f'chisq for {np.round(q_break[i], 2)}<q<{np.round(q_break[i + 1], 2)}',
                 va='center', ha='center', backgroundcolor='w')

    plt.hlines(-np.arange(chisqThresh.size) * y_offset, chisqBins[0], chisqBins[-2])
    plt.xlim(chisqBins[0], chisqBins[-2])
    plt.ylim(y_offset_i, y_offset)
    plt.ylabel('Occurances')
    plt.xlabel('chisq value')


def plotAvData(q, x, t_str, y_offset=None, x_txt=None, y_txt=None, qpower=0):
    x_mult = x * q[:, None] ** qpower

    if y_offset is None:
        y_offset = np.abs(x_mult.max() - x_mult.min())
    if x_txt is None:
        x_txt = q.min() + 0.8 * (q.max() - q.min())
    if y_txt is None:
        y_txt = y_offset * 0.15

    for i, each_t in enumerate(t_str):
        y_offset_i = y_offset * i
        plt.plot(q, x_mult[:, i] - y_offset_i, 'k.-')
        plt.text(x_txt, y_txt - y_offset_i, each_t)
    plt.hlines(-np.arange(len(t_str)) * y_offset, q.min(), q.max(), colors=[0.5, 0.5, 0.5], linewidths=0.5)

    plt.xlabel('q, 1/A')
    if qpower == 0:
        plt.ylabel('S, a.u.')
    else:
        plt.ylabel(f'q^{qpower}*S, a.u.')
    plt.xlim(q.min(), q.max())


def rescaleQ(q_old, wavelength, dist_old, dist_new):
    tth_old = 2 * np.arcsin(wavelength * q_old / 4 / pi)
    r = np.arctan(tth_old) * dist_old
    tth_new = np.tan(r / dist_new)
    return 4 * pi / wavelength * np.sin(tth_new / 2)


def distribute_Mat2ScatData(matfile):
    data = ScatData(None)
    matdata = scipy.io.loadmat(matfile)
    data.q = matdata['data']['q'][0][0].squeeze()
    data.t = matdata['data']['t'][0][0].squeeze()
    data.t_str = np.array([time_num2str(i * 1e-12) for i in data.t])
    data.diff.s = matdata['data']['ds0'][0][0]
    data.diff.ds = np.copy(data.diff.s)  # Allow both s and ds be used
    data.diff.s2 = matdata['data']['ds2'][0][0]
    data.diff.covqq = matdata['data']['ds0_covqq'][0][0]
    data.diff.covtt = matdata['data']['covtt'][0][0]
    data.total.s_av = np.mean(matdata['data']['soff'][0][0], 1)[:, None]
    return data


if __name__ == '__main__':
    A = ScatData([r'C:\work\Experiments\2015\Ru-Rh\Ru=Co_data\Ru_Co_rigid_25kev\run1\diagnostics.log',
                  r'C:\work\Experiments\2015\Ru-Rh\Ru=Co_data\Ru_Co_rigid_25kev\run2\diagnostics.log'],
                 logFileStyle='id09_old',
                 nFirstFiles=2500)

#    # %%
#    A.integrate(energy=25.2,
#                distance=44.75,
#                pixelSize=0.000104388,
#                centerX=469.25,
#                centerY=599.5,
#                qRange=[0.39, 12.01],
#                nqpt=465,
#                qNormRange=[5.00, 10.00],
#                maskPath=r"C:\work\Experiments\2015\Ru-Rh\Ru=Co_data\Ru_Co_rigid_25kev\ru=co_mask.edf",
#                correctPhosphor=True, muphos=92.8, lphos=75e-4,
#                correctSample=True, musample=0.29, lsample=300e-4,
#                plotting=False)
#    # %% idnetify outliers in total curves
#    A.getTotalAverages(fraction=0.9, chisqThresh=[6, 4], q_break=5, plotting=False)
#    #
#    # %% difference calculation
#    A.getDifferences(toff_str='-3ns',
#                     subtractFlag='MovingAverage')

# %%
#    A = ScatData(r'C:\pyfiles\pytrx_testing\bla.h5')
#    A.getDiffAverages(fraction=0.9, chisqThresh=2.5, plotting=False, covShrinkage=0.01)
#    A.getDiffAverages(fraction=0.9, chisqThresh=3, plotting=True)
#
#    # %%
##
#    A.save(r'C:\pyfiles\pytrx_testing\bla.h5')
#    B = ScatData(r'C:\pyfiles\pytrx_testing\bla.h5')