# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 21:10:16 2020

@author: dleshchev
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solveh_banded
# from pytrx.transformation import *



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

    if type(t_str) == float:
        return t_str

    try:
        t = float(t_str)
    except ValueError:
        t_number = float(t_str[0:-2])
        if 'fs' in t_str:
            t = t_number * 1e-15
        elif 'ps' in t_str:
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

    if (type(t) == str) or (type(t) == np.str_):
        return t

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

def weighted_mean(y, K):

    nt = y.shape[0]
    A = np.ones((nt, 1))

    K_inv = np.linalg.inv(K)
    H = np.linalg.inv(A.T @ K_inv @ A) @ A.T @ K_inv
    y_av = H @ y
    K_av = H @ K @ H.T
    return y_av, K_av


def bin_operator(q_out, q_in):
    assert q_in.size >= q_out.size, 'the input array must be sampled more densely compared to output array'
    assert q_in.min() <= q_out.min(), 'the input array must have minimum value smaller than the minimum value of output array'
    assert q_in.max() >= q_out.max(), 'the input array must have maximum value larger than the maximum value of output array'

    q_edges = q_out[:-1] + np.diff(q_out)/2
    q_edges = np.hstack((q_out[0] - (q_out[1] - q_out[0])/2,
                         q_edges,
                         q_out[-1] + (q_out[-1] - q_out[-2])/2))
    Abin = np.zeros((q_out.size, q_in.size))
    for i in range(q_out.size):
        q_mask_i = (q_in >= q_edges[i]) & (q_in <= q_edges[i + 1])
        Abin[i, q_mask_i] = 1 / np.sum(q_mask_i)
    return Abin


def bin_vector_with_covmat(q_out, q_in, x, C):
    if C is None:
        C_out = None
        if x.ndim == 1:
            x_out = np.interp(q_out, q_in, x)
        else:
            x_out = []
            for each_x in x.T:
                x_out.append( np.interp(q_out, q_in, each_x) )
            x_out = np.array(x_out).T
    else:
        Abin = bin_operator(q_out, q_in)
        x_out = Abin @ x
        C_out = Abin @ C @ Abin.T
    return x_out, C_out



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


def convolutedExpDecay(t, tau, tzero, fwhm):
   t = t - tzero
   sigma = fwhm/2.355
   val = sigma**2 - tau*t
   return ( 1/2 * np.exp( (sigma**2 - 2*tau*t) / (2*tau**2) )*
           (1 + (np.sign(-val) * erf(np.abs(val) / (np.sqrt(2)*sigma*tau)))) )



def convolutedStep(t, tzero, fwhm):
   t = t - tzero
   sigma = fwhm/2.355
   val = t / (np.sqrt(2)*sigma)
   return (1/2 * (1 + erf(val)))


def AtomColor():
    AtomColors = np.array([(0, 0, 0),  # Avoid atomic number to index conversion
                        (255, 255, 255), (217, 255, 255), (204, 128, 255),
                        (194, 255, 0), (255, 181, 181), (144, 144, 144),
                        (48, 80, 248), (255, 13, 13), (144, 224, 80),
                        (179, 227, 245), (171, 92, 242), (138, 255, 0),
                        (191, 166, 166), (240, 200, 160), (255, 128, 0),
                        (255, 255, 48), (31, 240, 31), (128, 209, 227),
                        (143, 64, 212), (61, 225, 0), (230, 230, 230),
                        (191, 194, 199), (166, 166, 171), (138, 153, 199),
                        (156, 122, 199), (224, 102, 51), (240, 144, 160),
                        (80, 208, 80), (200, 128, 51), (125, 128, 176),
                        (194, 143, 143), (102, 143, 143), (189, 128, 227),
                        (225, 161, 0), (166, 41, 41), (92, 184, 209),
                        (112, 46, 176), (0, 255, 0), (148, 255, 255),
                        (148, 224, 224), (115, 194, 201), (84, 181, 181),
                        (59, 158, 158), (36, 143, 143), (10, 125, 140),
                        (0, 105, 133), (192, 192, 192), (255, 217, 143),
                        (166, 117, 115), (102, 128, 128), (158, 99, 181),
                        (212, 122, 0), (148, 0, 148), (66, 158, 176),
                        (87, 23, 143), (0, 201, 0), (112, 212, 255),
                        (255, 255, 199), (217, 225, 199), (199, 225, 199),
                        (163, 225, 199), (143, 225, 199), (97, 225, 199),
                        (69, 225, 199), (48, 225, 199), (31, 225, 199),
                        (0, 225, 156), (0, 230, 117), (0, 212, 82),
                        (0, 191, 56), (0, 171, 36), (77, 194, 255),
                        (77, 166, 255), (33, 148, 214), (38, 125, 171),
                        (38, 102, 150), (23, 84, 135), (0, 255, 0), #Pt
                        (255, 209, 35), (184, 184, 208), (166, 84, 77),
                        (87, 89, 97), (158, 79, 181), (171, 92, 0),
                        (117, 79, 69), (66, 130, 150), (66, 0, 102),
                        (0, 125, 0), (112, 171, 250), (0, 186, 255),
                        (0, 161, 255), (0, 143, 255), (0, 128, 255),
                        (0, 107, 255), (84, 92, 242), (120, 92, 227),
                        (138, 79, 227), (161, 54, 212), (179, 31, 212),
                        (179, 31, 186), (179, 13, 166), (189, 13, 135),
                        (199, 0, 102), (204, 0, 89), (209, 0, 79),
                        (217, 0, 69), (224, 0, 56), (230, 0, 46),
                        (235, 0, 38), (255, 0, 255), (255, 0, 255),
                        (255, 0, 255), (255, 0, 255), (255, 0, 255),
                        (255, 0, 255), (255, 0, 255), (255, 0, 255),
                        (255, 0, 255)], dtype=np.float32)/255.0

    return AtomColors

def vdWradius():
    vdWradii = np.array([0,  # Avoid atomic number to index conversion
                       230, 930, 680, 350, 830, 680, 680, 680, 640,
                       1120, 970, 1100, 1350, 1200, 750, 1020, 990,
                       1570, 1330, 990, 1440, 1470, 1330, 1350, 1350,
                       1340, 1330, 1500, 1520, 1450, 1220, 1170, 1210,
                       1220, 1210, 1910, 1470, 1120, 1780, 1560, 1480,
                       1470, 1350, 1400, 1450, 1500, 1590, 1690, 1630,
                       1460, 1460, 1470, 1400, 1980, 1670, 1340, 1870,
                       1830, 1820, 1810, 1800, 1800, 1990, 1790, 1760,
                       1750, 1740, 1730, 1720, 1940, 1720, 1570, 1430,
                       1370, 1350, 1370, 1320, 1500, 1500, 1700, 1550,
                       1540, 1540, 1680, 1700, 2400, 2000, 1900, 1880,
                       1790, 1610, 1580, 1550, 1530, 1510, 1500, 1500,
                       1500, 1500, 1500, 1500, 1500, 1500, 1600, 1600,
                       1600, 1600, 1600, 1600, 1600, 1600, 1600, 1600,
                       1600, 1600, 1600, 1600, 1600, 1600],
                      dtype=np.float32) / 1000.0
    return vdWradii


def DrawMolecule(mol, draw_par=False, scaling=1.15, fignum=10, overlay=False, ax=None, shownum=True):
    dist_mat = mol.calcDistMat(return_mat=True)
    dist_thres = (vdWradius()[mol.Z_num][:, None] + vdWradius()[mol.Z_num]) * scaling * scaling

    bonds = dist_thres > dist_mat
    bonds_pair = []
    for i in range(len(mol.Z_num)):
        for j in range(i + 1, len(mol.Z_num)):
            if bonds[i, j]:
                bonds_pair.append([i, j])
    bonds_pair = np.array(bonds_pair)
    print(mol)

    # mol = c.mol_es
    fig = plt.figure(fignum)
    if not overlay:
        plt.clf()
    if ax is None:
        ax = fig.add_subplot(111, projection='3d', proj_type='ortho')
        ax.view_init(elev=60, azim=60)
        ax.set_facecolor((0, 0, 0))
        plt.xlim((np.min(mol.xyz_ref[:, 0]), np.max(mol.xyz_ref[:, 0])))
        plt.ylim((np.min(mol.xyz_ref[:, 1]), np.max(mol.xyz_ref[:, 1])))
        ax.set_zlim(np.min(mol.xyz_ref[:, 2]), np.max(mol.xyz_ref[:, 2]))

    ax.scatter(mol.xyz[:, 0], mol.xyz[:, 1], mol.xyz[:, 2],
               s=vdWradius()[mol.Z_num] * 15,
               c=AtomColor()[mol.Z_num])
    ax.scatter(mol.xyz[:, 0], mol.xyz[:, 1], mol.xyz[:, 2],
               s=vdWradius()[mol.Z_num] * 15)

    plt.axis('off')
    for row in bonds_pair:
        ax.plot([mol.xyz[row[0], 0], mol.xyz[row[1], 0]],
                [mol.xyz[row[0], 1], mol.xyz[row[1], 1]],
                [mol.xyz[row[0], 2], mol.xyz[row[1], 2]],
                c=(0.5, 0.5, 0.5),
                linewidth=1)
    if draw_par: # For testing, don't use
        for t in mol._associated_transformation:
            if issubclass(type(t), Transformation_distance):
                G2_mean = t.group2_mean
                G1_mean = t.group1_mean
                # axis = mol._associated_transformation[1].axis
                ax.quiver(G1_mean[0], G1_mean[1], G1_mean[2],
                          G2_mean[0] - G1_mean[0],
                          G2_mean[1] - G1_mean[1],
                          G2_mean[2] - G1_mean[2],
                          color='r', linewidth=2)
    return ax



#     def applyPolyCorrection(self, q_poly, E, I, E_eff='auto'):
#         self.s_poly = self._PolyCorrection(self.q, self.s, q_poly, E, I, E_eff=E_eff)
#         self.polyFlag = True
#
#     def _PolyCorrection(self, q_mono, s_mono, q_poly, E, I, E_eff):
#         I = I[np.argsort(E)]
#         E = np.sort(E)
#
#         if E_eff == 'auto':
#             E_eff = E[np.argmax(I)]
#
#         I = I / np.sum(I)
#
#         W = 12.3984 / E
#         W_eff = 12.3984 / E_eff
#
#         tth_mono = 2 * np.arcsin(q_mono[:, np.newaxis] * W[np.newaxis] / (4 * pi)) * 180 / (pi)
#         tth_poly = 2 * np.arcsin(q_poly * W_eff / (4 * pi)) * 180 / (pi)
#
#         if not np.all(tth_poly[0] > tth_mono[0, 0]):
#             raise ValueError('Input q range is too narrow: Decrease q_mono min.')
#         if not tth_poly[-1] < tth_mono[-1, -1]:
#             raise ValueError('Input q range is too narrow: Increase q_mono max.')
#
#         if len(s_mono.shape) == 1:
#             s_mono = s_mono[:, np.newaxis]
#
#         nCurves = s_mono.shape[1]
#         s_poly = np.zeros((q_poly.size, nCurves))
#         for i in range(nCurves):
#             for j, e in enumerate(E):
#                 s_poly[:, i] += I[j] * np.interp(tth_poly, tth_mono[:, j], s_mono[:, i])
#
#         return s_poly