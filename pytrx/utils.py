# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 21:10:16 2020

@author: dleshchev
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solveh_banded



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


def convert_banded(x):
    n, m = x.shape
    assert n == m, 'matrix must be squared'

    #    d = np.diag(x)
    diags = []
    for i in range(n):
        read = np.diag(x, i)
        if not np.all(read == 0):
            diags.append(read)
        else:
            break

    n_diags = len(diags)
    out = np.zeros((n_diags, n))

    for i in range(0, n_diags):
        d = diags[i]
        idx = d.size
        out[i, :idx] = d

    return out, n


def invert_banded(x):
    x_conv, n = convert_banded(x)
    x_inv = solveh_banded(x_conv, np.eye(n), lower=True)
    return x_inv

