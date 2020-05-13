# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 18:02:17 2016

@author: denis
"""

from math import pi
from itertools import islice
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pytrx.utils import z_str2num, z_num2str
import pkg_resources

from pytrx import hydro



class Molecule:
    def __init__(self, Z, xyz,
                 calc_gr=False, rmin=0, rmax=25, dr=0.01):
        if type(Z) == str:
            Z = np.array([Z])
        self.Z = Z
        self.Z_num =np.array([z_str2num(z) for z in Z])
        
        self.xyz = xyz
        
        if calc_gr: self.calcGR(rmin=rmin, rmax=rmax, dr=dr)
            
        
    def calcDistMat(self):
        self.dist_mat = np.sqrt(np.sum((self.xyz[None, :, :] -
                                        self.xyz[:, None, :])**2, axis=2))
        
        
    def calcGR(self, rmin=0, rmax=25, dr=0.01):
        self.calcDistMat()
        self.gr = GR(self.Z, rmin=rmin, rmax=rmax, dr=dr)
        self.r = self.gr.r
        for pair in self.gr.el_pairs:
            el1, el2 = pair
            idx1, idx2 = (el1==self.Z, el2==self.Z)
            self.gr[pair] += np.histogram(self.dist_mat[np.ix_(idx1, idx2)].ravel(),
                                          self.gr.r_bins)[0]
            
    def calcDens(self):
        self.gr.calcDens()
        self.dens = self.gr.dens
        
   

class GR:
    def __init__(self, Z, rmin=0, rmax=25, dr=0.01, r=None, el_pairs=None):
        
        self.Z = np.unique(Z)
        if el_pairs is None:
            self.el_pairs = [(z_i, z_j) for i, z_i in enumerate(self.Z) for z_j in self.Z[i:]]
        else:
            self.el_pairs = el_pairs
        
        if r is None:
#            self.r = np.arange(rmin, rmax+dr, dr)
            self.r = np.linspace(rmin, rmax, int((rmax - rmin)/dr) + 1)
        else:
            self.r = r
            rmin, rmax, dr = r.min(), r.max(), r[1]-r[0]
#        self.r_bins = np.arange(rmin-0.5*dr, rmax+1.5*dr, dr)
        self.r_bins = np.linspace(rmin-0.5*dr, rmax+0.5*dr, (rmax - rmin)/dr + 2)
        
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
            self.dens += z1*z2*self.gr[frozenset(pair)]



    
        
        
        
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
    
    s = q/(4*pi)
    formFunc = lambda sval, a: np.sum(a[None, :5]*np.exp(-a[None, 6:]*sval[:, None]**2), axis=1) + a[5]
    
    f = {}
    for i,x in enumerate(content):
        if x[0:2]=='#S':
            atom = x.split()[-1]
            if any([atom==x for x in Elements]):
                coef = np.fromstring(content[i+3], sep='\t')
                f[atom] = formFunc(s, coef)
    
    return f


def Debye(q, mol, f=None, atomOnly=False):
    if f is None:
        f = formFactor(q, mol.Z)
    Scoh = np.zeros(q.shape)
    mol.calcDistMat()
#    for el1 in f.keys():
#        idx1 = el1 == mol.Z
#        
#        for el2 in f.keys():
#            idx2 = el2 == mol.Z
#            r12 = mol.dist_mat[np.ix_(idx1, idx2)].ravel()
#            qr12 = q[:, None]*r12[None, :]
#            qr12[qr12<1e-6] = 1e-6 # to deal with r an q == 0
#            Scoh += f[el1]*f[el2]*np.sum(np.sin(qr12)/qr12, axis=1)
    natoms = mol.Z.size
    for idx1 in range(natoms):
        if not atomOnly:
            for idx2 in range(idx1+1, natoms):
                r12 = mol.dist_mat[idx1, idx2]
                qr12 = q*r12
                Scoh += 2 * f[mol.Z[idx1]] * f[mol.Z[idx2]] * np.sin(qr12)/qr12
        Scoh += f[mol.Z[idx1]]**2
    
    return Scoh
        


def DebyeFromGR(q, gr, f=None, rmax=None, cage=False):
    if f is None:
        f = formFactor(q, gr.Z)
    if rmax is None: rmax = gr.r.max()
    
    Scoh = np.zeros(q.shape)
    rsel = gr.r<rmax
    qr = q[:, None]*gr.r[None, rsel]
    qr[qr<1e-6] = 1e-6
    Asin = np.sin(qr)/qr
    
    for pair in gr.el_pairs:
        el1, el2 = pair
#        print(Asin.shape, gr[pair].shape)
        pair_scat = f[el1] * f[el2] * (Asin @ gr[pair][rsel])
        if el1==el2:
            if cage:
                Scoh += 2*pair_scat
            else:
                Scoh += pair_scat
        else:
            Scoh += 2*pair_scat
        
    
    return Scoh


def ScatFromDens(q, gr):
    gr.calcDens()
    qr = q[:, None]*gr.r[None, :]
    qr[qr<1e-6] = 1e-6
    Asin = np.sin(qr)/qr
    return Asin @ gr.dens
    
    

def Compton(z, q):
    fname_lowz = pkg_resources.resource_filename('pytrx', './Compton_lowZ.dat')
    fname_highz = pkg_resources.resource_filename('pytrx', './Compton_highZ.dat')
    
    data_lowz = pd.read_csv(fname_lowz, sep='\t')
    data_highz = pd.read_csv(fname_highz, sep='\t')
    data_lowz['Z'] = data_lowz['Z'].apply(lambda x: z_num2str(x))
    data_highz['Z'] = data_highz['Z'].apply(lambda x: z_num2str(x))
    
    Scoh = formFactor(q, z)[z]**2
    z_num = z_str2num(z)
    
    if z in data_lowz['Z'].values:
        M, K, L = data_lowz[data_lowz['Z']==z].values[0,1:4]
        S_inc = (z_num - Scoh/z_num) * (1-M*(np.exp(-K*q/(4*pi))-np.exp(-L*q/(4*pi))))
#        S(idx_un(i),:) = (Z_un(i)-Scoh(idx_un(i),:)/Z_un(i)).*...
#                         (1-M*(exp(-K*Q/(4*pi))-exp(-L*Q/(4*pi))));
    elif z in data_highz['Z'].values:
        A, B, C = data_highz[data_highz['Z']==z].values[0,1:4]
        S_inc = z_num*(1 - A/(1+B*q/(4*pi))**C)
#        S(idx_un(i),:) = Z_un(i)*(1-A./(1+B*Q/(4*pi)).^C);
    
    elif z == 'H':
        S_inc = np.zeros(q.shape)
    else:
        S_inc = np.zeros(q.shape)
        print(z, 'not found')
    return S_inc


def fromXYZ(filename, n_header=0):
    Z = []
    xyz = []
    with open(filename) as f:
        for line in f.readlines():
            values = line.split()
            if (line[0] != '#') and (len(values)==4):
                try: 
                    z = z_num2str(int(values[0]))
                except ValueError:
                    z = values[0]
                Z.append(z)
                xyz.append([float(i) for i in values[1:]])
    xyz = np.array(xyz)
    Z = np.array(Z)
    return Molecule(Z, xyz)
            
    
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
    if name_str in hydro.acetonitrile_keys:
       Z = np.array(['C', 'C', 'H', 'H', 'H', 'N'])
       xyz = np.array([[  0.000000,    0.000000,    0.006313],
                       [  0.000000,    0.000000,    1.462539],
                       [  1.024583,    0.000000,   -0.370908],
                       [ -0.512291,   -0.887315,   -0.370908],
                       [ -0.512291,    0.887315,   -0.370908],
                       [  0.000000,    0.000000,    2.615205]])
    elif name_str in hydro.cyclohexane_keys:
        Z = np.array(['C']*6 + ['H']*12)
        xyz = np.array([[-0.7613, -0.7573,  0.9857],               
                        [ 0.7614, -0.7575,  0.9855],               
                        [-1.2697,  0.5686,  0.4362],               
                        [ 1.2693, -0.5690, -0.4379],               
                        [-0.7627,  0.7570, -0.9872],               
                        [ 0.7603,  0.7568, -0.9883],               
                        [-1.1313, -0.8928,  2.0267],               
                        [-1.1314, -1.5893,  0.3455],               
                        [ 1.1315, -1.7280,  1.3860],               
                        [ 1.1315,  0.0762,  1.6235],               
                        [-2.3828,  0.5666,  0.4355],               
                        [-0.8972,  1.4009,  1.0747],               
                        [ 2.3828, -0.5672, -0.4356],               
                        [ 0.8980, -1.4027, -1.0759],               
                        [-1.1338,  1.7273, -1.3870],               
                        [-1.1329, -0.0768, -1.6251],               
                        [ 1.1290,  0.8924, -2.0303],               
                        [ 1.1300,  1.5904, -0.3493]])
    elif name_str in hydro.tetrahydrofuran_keys:
        # structure is taken from SI of Cryst. Growth Des., 2015, 15 (3), pp 1073–1081 DOI: 10.1021/cg501228w
        Z = np.array(['O']*1 + ['C']*4 + ['H']*8)
        xyz = np.array([[ 0.000760889, -0.000738525, -1.456851081],
                        [-0.652804737, -0.977671000, -0.666797608],
                        [-0.135276924, -0.754196802,  0.753993299],
                        [ 0.652965710,  0.977170235, -0.666602790],
                        [ 0.134622392,  0.755233684,  0.754352491],
                        [-1.737556125, -0.829010551, -0.720475115],
                        [-0.415216782, -1.967675331, -1.059040344],
                        [-0.850969390, -1.066076288,  1.514860077],
                        [ 0.796865925, -1.302852121,  0.910687964],
                        [ 1.737518733,  0.828788014, -0.720974308],
                        [ 0.414866234,  1.967002482, -1.058148213],
                        [ 0.850649795,  1.065510448,  1.514996321],
                        [-0.796426720,  1.304515754,  0.911888645]])
    elif name_str in hydro.water_keys:
        Z = np.array(['O'] * 1 + ['H'] * 2)
        xyz = np.array([[ 0.   ,  0.   ,  0.   ],
                        [ 0.928,  0.013,  0.246],
                        [-0.263, -0.899, -0.21 ]])
    else:
        return None
    return Molecule(Z, xyz)
        
        




#def convolutedExpDecay(t, tau, tzero, fwhm):
#    t = t - tzero
#    sigma = fwhm/2.355
#    val = sigma**2 - tau*t
#    return ( 1/2 * np.exp( (sigma**2 - 2*tau*t) / (2*tau**2) )*
#            (1 + (np.sign(-val) * erf(np.abs(val) / (np.sqrt(2)*sigma*tau)))) )
#
#
#
#def convolutedStep(t, tzero, fwhm):
#    t = t - tzero
#    sigma = fwhm/2.355
#    val = t / (np.sqrt(2)*sigma)
#    return (1/2 * (1 + erf(val)))




# if __name__ == '__main__':
#     np.random.seed(100)
#     import timeit
#
# #    Z = np.array(['I', 'I', 'I', 'Br'])
# #
# #    xyz = np.array([[0.00, 0.0, 0.0],
# #                    [2.67, 0.0, 0.0],
# #                    [6.67, 0.0, 0.0],
# #                    [0.00, 3.0, 0.0]])
# #
# #    Z = np.array(['I', 'I', 'Br'])
# #
# #    xyz = np.array([[0.00, 0.0, 0.0],
# #                    [2.67, 0.0, 0.0],
# #                    [6.67, 0.0, 0.0]])
#
# #    mol1 = Molecule(Z, xyz, calc_gr=True, dr=0.01)
# #    fname1 = r'C:\work\Experiments\2015\Ru_Dimers\Theory\Ru=Co\DFT\RuCo-LS-opt-PBE-TZVP-COSMO.xyz'
# #    fname2 = r'C:\work\Experiments\2015\Ru_Dimers\Theory\Ru=Co\DFT\RuCo-HS-opt-PBE-TZVP-COSMO.xyz'
#
#     fname1 = r"D:\lcls_dec2018\UserScripts\structures\pt2g4\pt2g24_singlet_b3lyp.xyz"
#     fname2 = r"D:\lcls_dec2018\UserScripts\structures\pt2g4\pt2g24_triplet_b3lyp.xyz"
#
#     mol1 = fromXYZ(fname1)
#     mol2 = fromXYZ(fname2)
#
#     n_mol=100
#     ens1 = Ensemble(mol1, n_mol=n_mol)
#     ens1.perturb(0.1)
#     ens1.calcGR(dr=0.01)
#
#     ens2 = Ensemble(mol2, n_mol=n_mol)
#     ens2.perturb(0.1)
#     ens2.calcGR(dr=0.01)
#
#     ens1.calcStructDeviation()
#
#
#     q = np.linspace(3, 12, 901)
#     ff = formFactor(q, ens1.Z)
#
#     DebyeFromGR(q, ens1.gr, f=ff)
#
#
#     plt.figure(1)
#     plt.clf()
#     plt.subplot(211)
#     plt.plot(ens1.r, ens1.gr[('Co','N')])
#     plt.plot(ens2.r, ens2.gr[('Co','N')])
#
#
#
#     plt.subplot(212)
#
#     plt.plot(q, (Debye(q, mol2) - Debye(q, mol1)), '.-')
#     plt.plot(q, 1/n_mol*(DebyeFromGR(q, ens2.gr) - DebyeFromGR(q, ens1.gr)))
#
    
    