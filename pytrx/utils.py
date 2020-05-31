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

def cov2corr(C):
    d = np.diag(1/np.sqrt(np.diag(C)))
    return d @ C @ d


def z_num2str(z):
    return ElementString()[z - 1]


def z_str2num(z):
    for i, el in enumerate(ElementString()):
        if el == z:
            return i + 1


def ElementString():
    ElementString = 'H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og'
    return ElementString.split()

def AtomicMass():
    AtomicMass = np.array([1.00797, 4.0026, 6.941, 9.01218, 10.81, 12.011, 14.0067, 15.9994, 18.998403, 20.179, 22.98977, 24.305, 26.98154, 28.0855, 30.97376, 32.06, 35.453, 39.0983, 39.948, 40.08, 44.9559, 47.9, 50.9415, 51.996, 54.938, 55.847, 58.7, 58.9332, 63.546, 65.38, 69.72, 72.59, 74.9216, 78.96, 79.904, 83.8, 85.4678, 87.62, 88.9059, 91.22, 92.9064, 95.94, -98, 101.07, 102.9055, 106.4, 107.868, 112.41, 114.82, 118.69, 121.75, 126.9045, 127.6, 131.3, 132.9054, 137.33, 138.9055, 140.12, 140.9077, 144.24, -145, 150.4, 151.96, 157.25, 158.9254, 162.5, 164.9304, 167.26, 168.9342, 173.04, 174.967, 178.49, 180.9479, 183.85, 186.207, 190.2, 192.22, 195.09, 196.9665, 200.59, 204.37, 207.2, 208.9804, 209, 210, 222, 223, 226.0254, 227.0278, 231.0359, 232.0381, 237.0482, 238.029])
    return AtomicMass
