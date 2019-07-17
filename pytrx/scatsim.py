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
import os
import pkg_resources

class Solute:
    
    def __init__(self, inputStyle='XYZ_file', inputObj=None,
                 qRange=[0,10], nqpt=1001, modelLabels=None, printing=False):
        
        assert ((inputStyle == 'XYZ_file') or
                (inputStyle == 'XYZ_list') or
                (inputStyle == 'PDB_file') or
                (inputStyle == 'PDB_entry')), \
                'To initiate MolecularStructure class you need to provide one of'\
                'the following input styles: XYZ_file, XYZ_list, PDB_file, PDB_entry'
        
        assert ((type(qRange) == list) and
                (len(qRange)==2) and
                ((type(qRange[0])==float) or (type(qRange[0])==int)) and
                ((type(qRange[1])==float) or (type(qRange[1])==int))), \
                'Provide correct qRange'
                
        assert type(nqpt) == int, 'nqpt should be an int'
        
        if modelLabels:
            assert(len(inputObj)==len(modelLabels)), 'provide the same number' \
            ' of models as modelLabels'
        else:
            modelLabels = [i for i in range(len(inputObj))]
        
        self.q = np.linspace(qRange[0], qRange[1], nqpt)
        listOfModels = inputObj
        self.s = np.zeros((self.q.size, len(listOfModels)))
        self.f_self = np.zeros((self.q.size, len(listOfModels)))
        self.f_sharp = np.zeros((self.q.size, len(listOfModels)))
        self.gr = []
        
        if inputStyle == 'XYZ_file':
            for i, filepath in enumerate(listOfModels):
                if printing: print('Calculating scattering for model', modelLabels[i])
                model = self.FiletoZXYZ(filepath)
                self.s[:, i], self.f_self[:, i], self.f_sharp[:, i] = self.DebyeScat_fromZXYZ(model, self.q)
                self.gr.append(self.ZXYZtoGR(model))
        
        elif inputStyle == 'XYZ_list':
            listOfModels = inputObj
            self.s = np.zeros((self.q.size, len(listOfModels)))
            for i, model in enumerate(listOfModels):
                if printing: print('Calculating scattering for model', modelLabels[i])
                self.s[:, i], self.f_self[:, i], self.f_sharp[:, i] = self.DebyeScat_fromZXYZ(model, self.q)
                self.gr.append(self.ZXYZtoGR(model))
                
        elif inputStyle == 'PDB_file':
            pass
        elif inputStyle == 'PDB_list':
            pass
        

    
    def FiletoZXYZ(self, filepath, n_head = 0):
        with open(filepath) as f:
            content = f.readlines()
        ZXYZ = [[x.split()[0],
                 float(x.split()[1]),
                 float(x.split()[2]),
                 float(x.split()[3]) ]
                for x in content[n_head:]]
        return ZXYZ


    
    def DebyeScat_fromZXYZ(self, ZXYZ, q):
        
        Elements = self.getElements(ZXYZ)
        atomForm = self.getAtomicFormFactor(Elements, q)
                
        S = np.zeros(q.shape)
        f_self = np.zeros(q.shape)
        f_sharp = np.zeros(q.shape)
        for i,item in enumerate(ZXYZ):
            xyz_i = np.array(item[1:])
            f_i = atomForm[item[0]]
            
            S += f_i**2
            f_self += f_i**2
            
            for jtem in ZXYZ[:i]:
                xyz_j = np.array(jtem[1:])
                r_ij = np.sqrt(np.sum((xyz_i - xyz_j)**2))                
                f_j = atomForm[jtem[0]]
                
#                print(r_ij)
#                S += 2 * f_i * f_j * np.sin( q * r_ij ) / ( q * r_ij )
                S[q!=0] += 2*f_i[q!=0]*f_j[q!=0]*np.sin(q[q!=0]*r_ij)/(q[q!=0]*r_ij)
                S[q==0] += 2*f_i[q==0]*f_j[q==0]
                f_sharp += 2*f_i*f_j
        
        return S, f_self, f_sharp
        
    
    def ZXYZtoGR(self, ZXYZ, Rmax = 1e2, dR = 1e-2):
        
        Elements = self.getElements(ZXYZ)
        Rpts = Rmax/dR
        
        r = np.linspace(0,Rmax,Rpts+1)
        r_bins = np.linspace(-dR/2, Rmax+dR/2, Rpts+2)
        
        gr = {}
        for i,item in enumerate(Elements):
            xyz_i = np.array(list(x[1:] for x in ZXYZ if x[0]==item))
            for j,jtem in enumerate(Elements[:i+1]):
                xyz_j = np.array(list(x[1:] for x in ZXYZ if x[0]==jtem))
#                print(xyz_i,xyz_j)
                dist = np.sqrt(np.subtract(xyz_i[:,[0]],xyz_j[:,[0]].T)**2 + 
                               np.subtract(xyz_i[:,[1]],xyz_j[:,[1]].T)**2 + 
                               np.subtract(xyz_i[:,[2]],xyz_j[:,[2]].T)**2).flatten()
                
#                print(r_bins.min(), r_bins.max())
                gr_ij = np.histogram(dist,r_bins)[0]
                if item!=jtem:
                    gr[item+'-'+jtem] = 2*gr_ij
                else:
                    gr[item+'-'+jtem] = gr_ij
                
        return r, gr
                        


    def DebyeScat_fromGR(self, r, gr, q):
        Elements = list(set(x[:x.index('-')] for x in gr))
        atomForm = self.getAtomicFormFactor(Elements,q)    
        
        QR = q[np.newaxis].T*r[np.newaxis]
        Asin = np.sin(QR)/QR
        Asin[QR==0] = 1;
        
        S = np.zeros(q.shape)
        for atomPair, atomCorrelation in gr.items():
            sidx = atomPair.index('-') # separator index
            El_i, El_j = atomPair[:sidx], atomPair[sidx+1:]
            f_i = atomForm[El_i][np.newaxis]
            f_j = atomForm[El_j][np.newaxis]
            S += np.squeeze(f_i.T*f_j.T*np.dot(Asin, atomCorrelation[np.newaxis].T))
        
        return S
            
    
    def getSR(self, r, alpha):
        self.r = r.copy()
        dq = self.q[1] - self.q[0]
        x = self.q[None, :]*self.r[:, None]
        self.rsr =  ((np.sin(x)*dq) @ ((self.s-self.f_self)/self.f_sharp * 
                                       self.q[:, None] * 
                                       np.exp(-alpha*self.q[:, None]**2)))
    
    
    def getAtomicFormFactor(self,Elements,q):
        if type(Elements) == str: Elements = [Elements]
        
        s=q/(4*pi)
        formFunc = lambda s,a: (np.sum(np.reshape(a[:5],[5,1])*
                                       np.exp(-a[6:,np.newaxis]*s**2),axis=0) + a[5])
		
        fname = pkg_resources.resource_filename('pytrx', './f0_WaasKirf.dat')
        with open(fname) as f:
            content = f.readlines()    
        
        atomData = list()
        for i,x in enumerate(content):
            if x[0:2]=='#S':
                atomName = x.rstrip().split()[-1]
                if any([atomName==x for x in Elements]):
                    atomCoef = content[i+3].rstrip()
                    atomCoef = np.fromstring(atomCoef, sep=' ')
                    atomData.append([atomName, atomCoef])
    
        atomData.sort(key=lambda x: Elements.index(x[0]))
        atomForm = {}
        for x in atomData:
            atomForm[x[0]] = formFunc(s,x[1])
    
        return atomForm



    def getElements(self,ZXYZ):    
        return list(set(x[0] for x in ZXYZ))
        
    
    
    def applyPolyCorrection(self, q_poly, E, I, E_eff='auto'):
        self.s_poly = self._PolyCorrection(self.q, self.s, q_poly, E, I, E_eff=E_eff)
        self.polyFlag = True
            
            
        
    def _PolyCorrection(self, q_mono, s_mono, q_poly, E, I, E_eff):
        I = I[np.argsort(E)]
        E = np.sort(E)
        
        if E_eff == 'auto':
            E_eff = E[np.argmax(I)]
        
        I = I/np.sum(I)
        
        W = 12.3984/E
        W_eff = 12.3984/E_eff
    
        tth_mono = 2*np.arcsin(q_mono[:, np.newaxis]*W[np.newaxis]/(4*pi))*180/(pi)
        tth_poly = 2*np.arcsin(q_poly*W_eff/(4*pi))*180/(pi)
        
        if not np.all(tth_poly[0]>tth_mono[0,0]):
            raise ValueError('Input q range is too narrow: Decrease q_mono min.')
        if not tth_poly[-1]<tth_mono[-1,-1]:
            raise ValueError('Input q range is too narrow: Increase q_mono max.')
        
        if len(s_mono.shape)==1:
            s_mono = s_mono[:, np.newaxis]
        
        nCurves = s_mono.shape[1]
        s_poly = np.zeros((q_poly.size, nCurves))
        for i in range(nCurves):
            for j, e in enumerate(E):
                s_poly[:, i] += I[j]*np.interp(tth_poly, tth_mono[:, j], s_mono[:, i])
        
        return s_poly





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
    def __init__(self, Z, rmin=0, rmax=25, dr=0.01):
        self.Z = np.unique(Z)
        
        self.r = np.arange(rmin, rmax+dr, dr)
        self.r_bins = np.arange(rmin-0.5*dr, rmax+1.5*dr, dr)
        
        self.gr = {}
        self.el_pairs = [(z_i, z_j) for i, z_i in enumerate(self.Z) for z_j in self.Z[i:]]
        for pair in self.el_pairs:
            self.gr[frozenset(pair)] = np.zeros(self.r.size)
        
        
    def __setitem__(self, key, data):
        key = frozenset(key)
        self.gr[key] = data
    
    
    def __getitem__(self, key):
        key = frozenset(key)
        return self.gr[key]
    
    
    def calcDens(self):
        self.dens = np.zeros(self.r.shape)
        for pair in self.el_pairs:
            el1, el2 = pair
            z1 = z_str2num(el1)
            z2 = z_str2num(el2)
            self.dens += z1*z2*self.gr[frozenset(pair)]


class Ensemble:
    def __init__(self, targetMolecule, n_mol=100):
        self.Z = targetMolecule.Z
        self.xyz_t = targetMolecule.xyz
        
        self.n_atom = self.xyz_t.shape[0]
        self.n_mol = n_mol
        
        self.xyz = np.tile(self.xyz_t, (n_mol, 1, 1))
        
        
    def calcDistMat(self, subset=None):
        if subset is None: subset = np.arange(self.n_mol)
        self.dist_mat = np.sqrt(np.sum((self.xyz[subset, None, :, :] - 
                                        self.xyz[subset, :, None, :])**2, axis=3))
        
        
    def _computeGR(self, rmin=0, rmax=25, dr=0.01, subset='all'):
        self.calcDistMat(subset=subset)
        n_subset = self.dist_mat.shape[0]
        gr = GR(self.Z, rmin=rmin, rmax=rmax, dr=dr)
        if not hasattr(self, 'r'): self.r = gr.r
        
        for pair in gr.el_pairs:
            el1, el2 = pair
            idx1, idx2 = (el1==self.Z, el2==self.Z)
            idx_grid = np.ix_(np.ones(n_subset, dtype='bool'), idx1, idx2)
            gr[pair] += np.histogram(self.dist_mat[idx_grid].ravel(), gr.r_bins)[0]
        
        return gr
    
    
    def calcGR(self, rmin=0, rmax=25, dr=0.01):
        self.gr = self._computeGR(rmin=rmin, rmax=rmax, dr=dr, subset=None)
        
    
    def perturb(self, amplitude, idx_mol=None, idx_atom=None):
        if idx_mol is None: idx_mol = np.arange(self.n_mol)
        if idx_atom is None: idx_atom = np.arange(self.n_atom)
        
        m_mol = idx_mol.size
        m_atom = idx_atom.size
        dxyz = np.random.randn(m_mol, m_atom, 3)*amplitude/np.sqrt(3)
#        idx = np.ix_()
#        print(self.xyz.shape, idx_mol.shape, idx_atom.shape)
#        print(self.xyz[idx_mol[], idx_atom, :].shape)
        self.xyz[idx_mol[:, None], idx_atom, :] += dxyz
    
    
    def calcStructDeviation(self, subset=None):
        if subset is None: subset = np.arange(self.n_mol)
        
        return np.sum((self.xyz[subset, :, :] - self.xyz_t[None, :, :])**2)
    
#    def calcDens(self):
#        self.gr.calcDens()
#        self.dens = self.gr.dens    
    
        





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


def Debye(q, mol, f=None):
    if f is None:
        f = formFactor(q, mol.Z)
    Scoh = np.zeros(q.shape)
    mol.calcDistMat()
    for el1 in f.keys():
        idx1 = el1 == mol.Z
        
        for el2 in f.keys():
            idx2 = el2 == mol.Z
            r12 = mol.dist_mat[np.ix_(idx1, idx2)].ravel()
            qr12 = q[:, None]*r12[None, :]
            qr12[qr12<1e-6] = 1e-6 # to deal with r an q == 0
            Scoh += f[el1]*f[el2]*np.sum(np.sin(qr12)/qr12, axis=1)
            
    return Scoh
        

def DebyeFromGR(q, gr, f=None):
    if f is None:
        f = formFactor(q, gr.Z)
    Scoh = np.zeros(q.shape)
    qr = q[:, None]*gr.r[None, :]
    qr[qr<1e-6] = 1e-6
    Asin = np.sin(qr)/qr
    
    for pair in gr.el_pairs:
        el1, el2 = pair
        pair_scat = f[el1] * f[el2] * (Asin @ gr[pair])
        if el1==el2:
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
    
    Scoh = Solute.getAtomicFormFactor(None, z, q)[z]
    z_num = z_str2num(z)
    
    if z in data_lowz['Z'].values:
        M, K, L = data_lowz[data_lowz['Z']==z].values[0,1:4]
        S_inc = (z_num - Scoh)/z_num * (1-M*(np.exp(-K*q/(4*pi))-np.exp(-L*q/(4*pi))))
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
                Z.append(values[0])
                xyz.append([float(i) for i in values[1:]])
    xyz = np.array(xyz)
    Z = np.array(Z)
    return Molecule(Z, xyz)
            
    
def FiletoZXYZ(filepath):
    ZXYZ = pd.read_csv(filepath, names=['Z', 'x', 'y', 'z'], sep='\s+')
    return ZXYZ

def z_num2str(z):
    return ElementString()[z-1]


def z_str2num(z):
    for i, el in enumerate(ElementString()):
        if el == z:
            return i+1

   
def ElementString():
    ElementString = 'H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og'
    return ElementString.split()
    

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




if __name__ == '__main__':
    np.random.seed(100)
    import timeit
    
#    Z = np.array(['I', 'I', 'I', 'Br'])
#    
#    xyz = np.array([[0.00, 0.0, 0.0],
#                    [2.67, 0.0, 0.0],
#                    [6.67, 0.0, 0.0],
#                    [0.00, 3.0, 0.0]])
#    
#    Z = np.array(['I', 'I', 'Br'])
#    
#    xyz = np.array([[0.00, 0.0, 0.0],
#                    [2.67, 0.0, 0.0],
#                    [6.67, 0.0, 0.0]])

#    mol1 = Molecule(Z, xyz, calc_gr=True, dr=0.01)    
    fname1 = r'C:\work\Experiments\2015\Ru_Dimers\Theory\Ru=Co\DFT\RuCo-LS-opt-PBE-TZVP-COSMO.xyz'
    fname2 = r'C:\work\Experiments\2015\Ru_Dimers\Theory\Ru=Co\DFT\RuCo-HS-opt-PBE-TZVP-COSMO.xyz'
    
    mol1 = fromXYZ(fname1)
    mol2 = fromXYZ(fname2)
    
    n_mol=100
    ens1 = Ensemble(mol1, n_mol=n_mol)
    ens1.perturb(0.1)
    ens1.calcGR(dr=0.01)
    
    ens2 = Ensemble(mol2, n_mol=n_mol)
    ens2.perturb(0.1)
    ens2.calcGR(dr=0.01)
    
    ens1.calcStructDeviation()
    
    
#    mol1.calcGR(dr=0.01)
#    mol2.calcGR(dr=0.01)
#
#    mol1.calcDens()
#    mol2.calcDens()
#    
    q = np.linspace(3, 12, 901)
    ff = formFactor(q, ens1.Z)
    
    DebyeFromGR(q, ens1.gr, f=ff)
    
    
    plt.figure(1)
    plt.clf()
    plt.subplot(211)
    plt.plot(ens1.r, ens1.gr[('Co','N')])
    plt.plot(ens2.r, ens2.gr[('Co','N')])
    
    
#    plt.plot(mol1.r, mol1.gr[('C', 'C')])
##    plt.plot(mol1.r, mol1.dens)
##    plt.plot(mol1.r, mol2.dens)
#    plt.plot(mol1.r, mol2.dens-mol1.dens)
#    
#    
    plt.subplot(212)
#    plt.plot(mol1.gr.r, mol1.gr.edens)
##    plt.plot(q, Debye(q, mol1), '.-')
##    plt.plot(q, DebyeFromGR(q, mol1.gr))
#    
    plt.plot(q, (Debye(q, mol2) - Debye(q, mol1)), '.-')
    plt.plot(q, 1/n_mol*(DebyeFromGR(q, ens2.gr) - DebyeFromGR(q, ens1.gr)))
#    plt.plot(q, (ScatFromDens(q, mol2.gr) - ScatFromDens(q, mol1.gr)))
    
    
    
    
    
# Test for 2D lattice    
#    d = 3
#    
#    nx, ny, nz = 50, 50, 5
#    
#    xyz = np.array([np.arange(nx), np.zeros(nx), np.zeros(nx)]).T
#    xyz_single_array = xyz.copy()
#    
#    for i in range(1,ny):
#        dy = np.array([0, 1, 0])
#        xyz_new_y = xyz_single_array + dy[None,:]*i
#        
#        xyz = np.vstack((xyz, xyz_new_y))
#        
#    xyz_single_sheet = xyz.copy()
#    for i in range(1,nz):
#        dy = np.array([0, 0, 1])
#        xyz_new_y = xyz_single_sheet + dy[None,:]*i
#        
#        xyz = np.vstack((xyz, xyz_new_y))
#    
#    xyz *= d
#    
#    xyz -= np.mean(xyz, axis=0)
#    Z = np.array(['Si']*xyz.shape[0])
#    
#    mol1 = Molecule(Z, xyz, calc_gr=True, dr=0.05, rmax=1000)
#    
#    q = np.linspace(0, 10, 1001)
#    
#    from mpl_toolkits.mplot3d import Axes3D
#    fig = plt.figure(1)
#    fig.clf()
#    ax1 = fig.add_subplot(221, projection='3d')
#    
#    ax1.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], '.', color='k')
#    
#    ax1.set_xlim(-15*d, 15*d)
#    ax1.set_ylim(-15*d, 15*d)
#    ax1.set_zlim(-15*d, 15*d)
#    
#    ax2 = fig.add_subplot(222)
#    ax2.plot(mol1.r, mol1.gr[('Si', 'Si')])
#    
#    s_coh = DebyeFromGR(q, mol1.gr)
#    
#    ax3 = fig.add_subplot(223)
#    ax3.semilogy(q, s_coh)
#    ax3.set_xlim(1e-3, 1)
#    
#    ax4 = fig.add_subplot(224)
#    ax4.plot(q, s_coh)
#    ax4.set_xlim(1, 4)
#    ax4.set_ylim(0, s_coh[q>1].max())
#    
    
    