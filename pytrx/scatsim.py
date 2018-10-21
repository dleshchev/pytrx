# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 18:02:17 2016

@author: denis
"""

from math import pi
from itertools import islice
import numpy as np
import matplotlib.pyplot as plt
import os
import pkg_resources

class molecular_structure:
    
    def __init__(self):
        pass
        
        
    def DebyeScat_fromZXYZ(self, ZXYZ, q):
        
        Elements = self.getElements(ZXYZ)
        atomForm = self.getAtomicFormFactor(Elements,q)
                
        S = np.zeros(q.shape)
        for i,item in enumerate(ZXYZ):
            xyz_i = np.array(item[1:])
            f_i = atomForm[item[0]]
            
            S += f_i**2
            
            for jtem in ZXYZ[:i]:
                xyz_j = np.array(jtem[1:])
                r_ij = np.sqrt(np.sum((xyz_i - xyz_j)**2))                
                f_j = atomForm[jtem[0]]
                
#                print(r_ij)
#                S += 2 * f_i * f_j * np.sin( q * r_ij ) / ( q * r_ij )
                S[q!=0] += 2*f_i[q!=0]*f_j[q!=0]*np.sin(q[q!=0]*r_ij)/(q[q!=0]*r_ij)
                S[q==0] += 2*f_i[q==0]*f_j[q==0]
        
        return S
        
    
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
                dist = np.sqrt(np.subtract(xyz_i[:,[0]],xyz_j[:,[0]].T)**2 + \
                               np.subtract(xyz_i[:,[1]],xyz_j[:,[1]].T)**2 + \
                               np.subtract(xyz_i[:,[2]],xyz_j[:,[2]].T)**2).flatten()
                
                
#                print(dist)                
                gr_ij = np.histogram(dist,r_bins)[0]
                if item!=jtem:
                    gr[item+'-'+jtem] = 2*gr_ij
                else:
                    gr[item+'-'+jtem] = gr_ij
                
        return r, gr
                        
    
    def ZXYZtoGR_cage(self, ZXYZ, n_solute, Rmax = 1e2, dR = 1e-2):
        
#        print(ZXYZ)
        ZXYZ_solute, ZXYZ_solvent = ZXYZ[:n_solute], ZXYZ[n_solute:]
        Elements_solute = self.getElements(ZXYZ_solute)
        Elements_solvent = self.getElements(ZXYZ_solvent)
        Rpts = Rmax/dR
        
        r = np.linspace(0,Rmax,Rpts+1)
        r_bins = np.linspace(-dR/2, Rmax+dR/2, Rpts+2)
        
        gr = {}
#        print()
        for i,item in enumerate(Elements_solute):
            xyz_i = np.array(list(x[1:] for x in ZXYZ_solute if x[0]==item))
            for j,jtem in enumerate(Elements_solvent):
                xyz_j = np.array(list(x[1:] for x in ZXYZ_solvent if x[0]==jtem))
#                print(xyz_i,xyz_j)
                dist = np.sqrt(np.subtract(xyz_i[:,[0]],xyz_j[:,[0]].T)**2 + \
                               np.subtract(xyz_i[:,[1]],xyz_j[:,[1]].T)**2 + \
                               np.subtract(xyz_i[:,[2]],xyz_j[:,[2]].T)**2).flatten()
                
                gr_ij = np.histogram(dist,r_bins)[0]
                gr[item+'-'+jtem] = gr_ij
                
        return r, gr    
    


    def DebyeScat_fromGR(self, r, gr, q):
        Elements = list(set(x[:x.index('-')] for x in gr))
        atomForm = self.getAtomicFormFactor(Elements,q)    
        
        QR = q[np.newaxis].T*r[np.newaxis]
        Asin = np.sin(QR)/QR
        Asin[QR==0] = 1;
        
        S = np.zeros(q.shape)
#        print(atomForm)
        for atomPair, atomCorrelation in gr.items():
            sidx = atomPair.index('-') # separator index
            El_i, El_j = atomPair[:sidx], atomPair[sidx+1:]
            f_i = atomForm[El_i][np.newaxis]
            f_j = atomForm[El_j][np.newaxis]
            S += np.squeeze(f_i.T*f_j.T*np.dot(Asin, atomCorrelation[np.newaxis].T))
        
        return S
            
        
    
    
    def getAtomicFormFactor(self,Elements,q):
        # Returns a dictionary of atomic formfactors using the element name as a key        
        
        ###        
        # define the scattering vector used for calculations of atomic formfactor:
        s=q/(4*pi)
        
        # The most modern and advanced parameterization of f0 form factors is
        # given by WaasKirf file.
        # fname = 'f0_WaasKirf.dat'
#        fname = './pytrx/f0_CromerMann.dat'
        
        # This is a functional form for this parameterization
        formFunc = lambda s,a: np.sum(np.reshape(a[:5],[5,1])*np.exp(-a[6:,np.newaxis]*s**2),axis=0)+a[5]
		
        fname = pkg_resources.resource_filename('pytrx', './f0_WaasKirf.dat')
#        print(f)
        with open(fname) as f:
            content = f.readlines()    
#        content = f.readlines()
        
		# Read the coefficients for the calculation:
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
#            print(x)
            atomForm[x[0]] = formFunc(s,x[1])
    
        return atomForm


    def getElements(self,ZXYZ):    
        Elements = list(set(x[0] for x in ZXYZ))
        return Elements
        
        
    def TrajectoriesToGR(self, folderpath, n_head, n_solute, n_atoms, Rmax = 1e2, dR = 1e-2):
        n_traj = 0
        r, gr_solute, gr_cage = 0, {}, {}
        for file in os.listdir(folderpath):
            with open(folderpath+file) as f:
                while True:
                    MD_snapshot = list(islice(f, n_head + n_atoms))
                    if not MD_snapshot:
                        break        
                    
                    ZXYZ_all    = list([ x.split()[0], float(x.split()[1]), float(x.split()[2]), float(x.split()[3]) ] for x in MD_snapshot[n_head:])
                    ZXYZ_solute = list([ x.split()[0], float(x.split()[1]), float(x.split()[2]), float(x.split()[3]) ] for x in MD_snapshot[n_head:n_head+n_solute])
                    
                    r, gr_solute_upd = self.ZXYZtoGR(ZXYZ_solute, Rmax, dR)               
                    gr_solute = { k: gr_solute.get(k, 0) + gr_solute_upd.get(k, 0) for k in set(gr_solute) | set(gr_solute_upd) }
                    
                    r, gr_cage_upd = self.ZXYZtoGR_cage(ZXYZ_all, n_solute, Rmax, dR)
                    gr_cage = { k: gr_cage.get(k, 0) + gr_cage_upd.get(k, 0) for k in set(gr_cage) | set(gr_cage_upd) }
                    
                    n_traj += 1
                    print('Number of processed frames: ', n_traj)
                    
        gr_solute = {k: x/n_traj for k,x in gr_solute.items()}
        gr_cage = {k: x/n_traj for k,x in gr_cage.items()}
        return r, gr_solute, gr_cage
                    
        
    def FiletoZXYZ(self, filepath, n_head = 0):
        with open(filepath) as f:
            content = f.readlines()
        ZXYZ    = list([ x.split()[0], float(x.split()[1]), float(x.split()[2]), float(x.split()[3]) ] for x in content[n_head:])
        return ZXYZ
        
        