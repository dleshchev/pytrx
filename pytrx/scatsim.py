# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 18:02:17 2016

@author: denis
"""

from math import pi
from itertools import islice
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
from pytrx.utils import z_str2num, z_num2str
import pkg_resources

from pytrx import hydro
from pytrx.transformation import *
from numba import njit, prange


class Molecule:
    def __init__(self, Z, xyz,
                 calc_gr=False, rmin=0, rmax=25, dr=0.01,
                 associated_transformation=None):
        '''
            associated_transformation will be either a transformation class or
            a list of transformations
        '''
        if type(Z) == str:
            Z = np.array([Z])
        self.Z = Z
        self.Z_num = np.array([z_str2num(z) for z in Z])

        self.xyz = xyz.copy()
        self.xyz_ref = xyz.copy()

        print("Running initial check up for associated_transformation")
        if associated_transformation is None:
            self._associated_transformation = None
        elif type(associated_transformation) is list:
            print("associated_transformation is a list. Examining elements...")
            for t in associated_transformation:
                print(f'Checking {t}')
                # print(f'{type(t)}')
                assert issubclass(type(t), Transformation), 'List element is not a Transformation class'
            self._associated_transformation = associated_transformation
        elif issubclass(type(associated_transformation), Transformation):
            self._associated_transformation = [associated_transformation]
        else:
            raise TypeError('Supplied transformations must be None, a transformation class, or a list of it')

        if self._associated_transformation is not None:
            for transform in self._associated_transformation:
                transform.prepare(self.xyz, self.Z_num)
            self.n_par = len(self._associated_transformation)
        else:
            self.n_par = 0

        if calc_gr: self.calcGR(rmin=rmin, rmax=rmax, dr=dr)

    def calcDistMat(self, return_mat=False):
        self.dist_mat = np.sqrt(np.sum((self.xyz[None, :, :] -
                                        self.xyz[:, None, :]) ** 2, axis=2))
        if return_mat: return self.dist_mat

    def calcGR(self, rmin=0, rmax=25, dr=0.01):
        self.calcDistMat()
        self.gr = GR(self.Z, rmin=rmin, rmax=rmax, dr=dr)
        self.r = self.gr.r
        for pair in self.gr.el_pairs:
            el1, el2 = pair
            idx1, idx2 = (el1 == self.Z, el2 == self.Z)
            self.gr[pair] += np.histogram(self.dist_mat[np.ix_(idx1, idx2)].ravel(),
                                          self.gr.r_bins)[0]

    def calcDens(self):
        self.gr.calcDens()
        self.dens = self.gr.dens

    def transform(self, par=None):

        '''
        Transforms xyz based on the transformation supplied in the _associated_transformation.
        Also takes the par which should be either None or a list that is the same length as the
        number of transformations.
        reprep: recalculate associated vectors, COMs, etc. after each step (as they might shift)
                by calling the prepare() methods within each class.
        '''
        if (par is not None) and (self._associated_transformation is not None):
            # Resets the coordinate set to be transformed
            # self.xyz = copy.deepcopy(self.xyz_ref)
            self.xyz = self.xyz_ref.copy() # as a numpy array we can just use the array's method

            assert (len(par) == len(self._associated_transformation)), \
                "Number of parameters not matching number of transformations"
            for p, t in zip(par, self._associated_transformation):
                # print(t)
                self.xyz = t.transform(self.xyz, self.Z_num, p)

    def s(self, q, pars=None):
        self.transform(pars)
        # TODO: compute form-factors and store them in Molecule and pass them to Debye
        return Debye(q, self)


    def clash(self):
        # Check for clash by whether min distances between two atom types are shorter than 80 % of original (tentative)
        pass


    # def sum_parameters(self):
    #     if self._associated_transformation is not None:
    #         return len(self._associated_transformation)


class GR:
    def __init__(self, Z, rmin=0, rmax=25, dr=0.01, r=None, el_pairs=None):

        self.Z = np.unique(Z)
        if el_pairs is None:
            self.el_pairs = [(z_i, z_j) for i, z_i in enumerate(self.Z) for z_j in self.Z[i:]]
        else:
            self.el_pairs = el_pairs

        if r is None:
            #            self.r = np.arange(rmin, rmax+dr, dr)
            self.r = np.linspace(rmin, rmax, int((rmax - rmin) / dr) + 1)
        else:
            self.r = r
            rmin, rmax, dr = r.min(), r.max(), r[1] - r[0]
        #        self.r_bins = np.arange(rmin-0.5*dr, rmax+1.5*dr, dr)
        self.r_bins = np.linspace(rmin - 0.5 * dr, rmax + 0.5 * dr, (rmax - rmin) / dr + 2)

        self.gr = {}
        for pair in self.el_pairs:
            self.gr[frozenset(pair)] = np.zeros(self.r.size)

    def __setitem__(self, key, data):
        key = frozenset(key)
        self.gr[key] = data

    def __getitem__(self, key):
        key = frozenset(key)
        return self.gr[key]

    def __add__(self, gr_other):
        gr_out = GR(self.Z, r=self.r, el_pairs=self.el_pairs)
        for pair in self.el_pairs:
            gr_out[pair] = self[pair] + gr_other[pair]
        return gr_out

    def __sub__(self, gr_other):
        gr_out = GR(self.Z, r=self.r, el_pairs=self.el_pairs)
        for pair in self.el_pairs:
            gr_out[pair] = self[pair] - gr_other[pair]
        return gr_out

    def __mul__(self, factor):
        gr_out = GR(self.Z, r=self.r, el_pairs=self.el_pairs)
        for pair in self.el_pairs:
            gr_out[pair] = self[pair] * factor
        return gr_out

    def calcDens(self):
        self.dens = np.zeros(self.r.shape)
        for pair in self.el_pairs:
            el1, el2 = pair
            z1 = z_str2num(el1)
            z2 = z_str2num(el2)
            self.dens += z1 * z2 * self.gr[frozenset(pair)]


### UTILS


def formFactor(q, Elements):
    '''
    Calculates atomic form-factor at value q
    q - np.array of scattering vector values
    Elements - np.array or list of elements. May be a string if one wants to
    compute form-factor for only one element.

    returns a dict of form factors

    Examples:

    q = np.arange(10)
    f = formFactor(q, 'Si')
    print(f['Si'])

    Elements = ['Si', 'O']
    f = formFactor(q, Elements)
    print(f['Si'], f['O'])
    '''

    Elements = np.unique(Elements)

    fname = pkg_resources.resource_filename('pytrx', './f0_WaasKirf.dat')
    with open(fname) as f:
        content = f.readlines()

    s = q / (4 * pi)
    formFunc = lambda sval, a: np.sum(a[None, :5] * np.exp(-a[None, 6:] * sval[:, None] ** 2), axis=1) + a[5]

    f = {}
    for i, x in enumerate(content):
        if x[0:2] == '#S':
            atom = x.split()[-1]
            if any([atom == x for x in Elements]):
                coef = np.fromstring(content[i + 3], sep='\t')
                f[atom] = formFunc(s, coef)

    return f


def Debye(q, mol, f=None, atomOnly=False, debug=False):
    if f is None:
        f = formFactor(q, mol.Z)
    if debug:
        print(f)
    Scoh = np.zeros(q.shape)
    mol.calcDistMat()
    natoms = mol.Z.size
    # Baseline speed - 164 ms
    # for idx1 in range(natoms):
    #     if not atomOnly:
    #         for idx2 in range(idx1 + 1, natoms):
    #             r12 = mol.dist_mat[idx1, idx2]
    #             qr12 = q * r12
    #             Scoh += 2 * f[mol.Z[idx1]] * f[mol.Z[idx2]] * np.sin(qr12) / qr12
    #     Scoh += f[mol.Z[idx1]] ** 2

    # Trial 1 use predefined f list - 155 ms
    # FFtable = np.zeros((natoms,len(q)))
    # for idx in range(natoms):
    #     FFtable[idx] = f[mol.Z[idx]]
    # for idx1 in range(natoms):
    #     if not atomOnly:
    #         for idx2 in range(idx1 + 1, natoms):
    #             r12 = mol.dist_mat[idx1, idx2]
    #             qr12 = q * r12
    #             Scoh += 2 * FFtable[idx1] * FFtable[idx2] * np.sin(qr12) / qr12
    #     Scoh += f[mol.Z[idx1]] ** 2

    # Trial 2 use predefined f list and broadcast - 124 ms
    # FFtable = np.zeros((natoms, len(q)))
    # for idx in range(natoms):
    #     FFtable[idx] = f[mol.Z[idx]]
    # for idx1 in range(natoms):
    #     if atomOnly:
    #         Scoh += f[mol.Z[idx1]] ** 2
    #     else:
    #         r12 = mol.dist_mat[idx1][:,None]
    #         # print(r12.shape, q.shape)
    #         qr12 = q[None,:] * r12
    #         # print(qr12.shape,FFtable[idx1][None,:].shape, FFtable.shape, np.sinc(qr12/np.pi).shape)
    #         Scoh += (FFtable[idx1][None,:] * FFtable * np.sinc(qr12 / np.pi)).sum(0)

    # Trial 3 decrease usage of large matrices - 118 ms
    # FFtable = np.zeros((natoms, len(q)))
    # for idx in range(natoms):
    #     FFtable[idx] = f[mol.Z[idx]]
    # for idx1 in range(natoms):
    #     if atomOnly:
    #         Scoh += f[mol.Z[idx1]] ** 2
    #     else:
    #         r12 = mol.dist_mat[idx1][:, None]
    #         # print(r12.shape, q.shape)
    #         # qr12 = q[None, :] * r12
    #         FFqr12 = np.sum(FFtable * np.sinc(q[None, :] * r12 / np.pi), axis=0)
    #         # FFqr12 = np.nansum(FFtable * np.sin(qr12) / qr12, axis=0)
    #         # FFqr12 = FFqr12calc(FFtable, q, r12)
    #         # print(FFqr12.shape)
    #         # print(qr12.shape,FFtable[idx1][None,:].shape, FFtable.shape, np.sinc(qr12/np.pi).shape)
    #         Scoh += FFtable[idx1] * FFqr12

    # Trial 4 use numba - 52 ms for Scoh_calc and 17 ms for Scoh_calc2
    FFtable = np.zeros((natoms, len(q)))
    for idx in range(natoms):
        FFtable[idx] = f[mol.Z[idx]]
    if atomOnly:
        for idx1 in range(natoms):
            Scoh += f[mol.Z[idx1]] ** 2
    else:
        Scoh = Scoh_calc2(FFtable, q, mol.dist_mat, natoms)

    if debug:
        print(Scoh)

    return Scoh

@njit
def Scoh_calc(FF, q, r, natoms):
    Scoh = np.zeros(q.shape)
    for idx1 in range(natoms):
        for idx2 in range(idx1 + 1, natoms):
            r12 = r[idx1, idx2]
            qr12 = q * r12
            Scoh += 2 * FF[idx1] * FF[idx2] * np.sin(qr12) / qr12
        Scoh += FF[idx1] ** 2
    return Scoh

@njit(parallel=True)
def Scoh_calc2(FF, q, r, natoms):
    # Scoh = np.zeros(q.shape)
    Scoh2 = np.zeros((natoms, len(q)))
    for idx1 in prange(natoms):
        for idx2 in range(idx1 + 1, natoms):
            r12 = r[idx1, idx2]
            qr12 = q * r12
            Scoh2[idx1] += 2 * FF[idx1] * FF[idx2] * np.sin(qr12) / qr12
        Scoh2[idx1] += FF[idx1] ** 2
    return np.sum(Scoh2, axis=0)


def DebyeFromGR(q, gr, f=None, rmax=None, cage=False):
    if f is None:
        f = formFactor(q, gr.Z)
    if rmax is None: rmax = gr.r.max()

    Scoh = np.zeros(q.shape)
    rsel = gr.r < rmax
    qr = q[:, None] * gr.r[None, rsel]
    qr[qr < 1e-6] = 1e-6
    Asin = np.sin(qr) / qr

    for pair in gr.el_pairs:
        el1, el2 = pair
        #        print(Asin.shape, gr[pair].shape)
        pair_scat = f[el1] * f[el2] * (Asin @ gr[pair][rsel])
        if el1 == el2:
            if cage:
                Scoh += 2 * pair_scat
            else:
                Scoh += pair_scat
        else:
            Scoh += 2 * pair_scat

    return Scoh


def ScatFromDens(q, gr):
    gr.calcDens()
    qr = q[:, None] * gr.r[None, :]
    qr[qr < 1e-6] = 1e-6
    Asin = np.sin(qr) / qr
    return Asin @ gr.dens


def Compton(z, q):
    fname_lowz = pkg_resources.resource_filename('pytrx', './Compton_lowZ.dat')
    fname_highz = pkg_resources.resource_filename('pytrx', './Compton_highZ.dat')

    data_lowz = pd.read_csv(fname_lowz, sep='\t')
    data_highz = pd.read_csv(fname_highz, sep='\t')
    data_lowz['Z'] = data_lowz['Z'].apply(lambda x: z_num2str(x))
    data_highz['Z'] = data_highz['Z'].apply(lambda x: z_num2str(x))

    Scoh = formFactor(q, z)[z] ** 2
    z_num = z_str2num(z)

    if z in data_lowz['Z'].values:
        M, K, L = data_lowz[data_lowz['Z'] == z].values[0, 1:4]
        S_inc = (z_num - Scoh / z_num) * (1 - M * (np.exp(-K * q / (4 * pi)) - np.exp(-L * q / (4 * pi))))
    #        S(idx_un(i),:) = (Z_un(i)-Scoh(idx_un(i),:)/Z_un(i)).*...
    #                         (1-M*(exp(-K*Q/(4*pi))-exp(-L*Q/(4*pi))));
    elif z in data_highz['Z'].values:
        A, B, C = data_highz[data_highz['Z'] == z].values[0, 1:4]
        S_inc = z_num * (1 - A / (1 + B * q / (4 * pi)) ** C)
    #        S(idx_un(i),:) = Z_un(i)*(1-A./(1+B*Q/(4*pi)).^C);

    elif z == 'H':
        S_inc = np.zeros(q.shape)
    else:
        S_inc = np.zeros(q.shape)
        print(z, 'not found')
    return S_inc


def fromXYZ(filename, n_header=0, associated_transformation=None):
    Z = []
    xyz = []
    with open(filename) as f:
        for line in f.readlines():
            values = line.split()
            if (line[0] != '#') and (len(values) == 4):
                try:
                    z = z_num2str(int(values[0]))
                except ValueError:
                    z = values[0]
                Z.append(z)
                xyz.append([float(i) for i in values[1:]])
    xyz = np.array(xyz)
    Z = np.array(Z)
    return Molecule(Z, xyz, associated_transformation=associated_transformation)


def FiletoZXYZ(filepath):
    ZXYZ = pd.read_csv(filepath, names=['Z', 'x', 'y', 'z'], sep='\s+')
    return ZXYZ


def totalScattering(q, mol, atomOnly=False):
    s_debye = Debye(q, mol, atomOnly=atomOnly)

    s_inc = np.zeros(q.shape)
    for z in mol.Z:
        s_inc += Compton(z, q)

    return s_debye + s_inc


def Solvent(name_str):
    return Molecule(hydro.solvent_data[name_str].Z, hydro.solvent_data[name_str].xyz)


