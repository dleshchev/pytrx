# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 17:33:17 2016

@author: darren

A library of predefined moves for the Molecule class.

"""

import numpy as np
from pytrx.utils import AtomicMass

class Transformation_move_vector:
    # Move a group of atoms along a vector (normalized)
    def __init__(self, group1, vector, amplitude0=0):
        self.group1 = np.array(group1)
        self.vector = np.array(vector)
        self.unit_vector = np.array(vector) / np.linalg.norm(vector)
        self.amplitude0 = amplitude0

    def prepare(self, mol):
        return self

    def transform(self, xyz, amplitude=None):
        assert (np.max(self.group1) <= len(xyz)), \
            "Index out of bound: largest index of group 1 > length of supplied molecule"
        if amplitude is None:
            amplitude = self.amplitude0
        xyz[self.group1] += self.unit_vector * amplitude
        return xyz


class Transformation_distance:
    # Move two groups of atoms closer/further in distance, using simple mean of coordinates as
    # reference centers for each group.
    # Vector is from group1 to group2. Negative amplitude is shrinking.
    def __init__(self, group1, group2, amplitude0=0):
        assert (len(group1) > 0) and (len(group2) > 0), 'Cannot operate on empty set'
        self.group1 = np.array(group1)
        self.group2 = np.array(group2)
        self.amplitude0 = amplitude0

    def prepare(self, mol):
        self.group1_mean = np.mean(mol.xyz[self.group1], 0)
        self.group2_mean = np.mean(mol.xyz[self.group2], 0)
        self.unit_vec = (self.group2_mean - self.group1_mean) / np.linalg.norm(self.group2_mean - self.group1_mean)
        return self

    def transform(self, xyz, amplitude=None):
        assert (np.max(self.group1) <= len(xyz)), \
            "Index out of bound: largest index of group 1 > length of supplied molecule"
        assert (np.max(self.group2) <= len(xyz)), \
            "Index out of bound: largest index of group 2 > length of supplied molecule"

        if amplitude is None:
            amplitude = self.amplitude0

        xyz[self.group1] -= self.unit_vec * amplitude / 2
        xyz[self.group2] += self.unit_vec * amplitude / 2

        return xyz

class Transformation_distance_1side:
    pass

class Transformation_distanceCOM:
    # Move two group of atoms closer/further in distance, using center of mass as ref centers for each group
    # Vector is from group1 to group2. Negative amplitude is shrinking.
    def __init__(self, group1, group2, amplitude0=0):
        assert (len(group1) > 0) and (len(group2) > 0), 'Cannot operate on empty set'
        self.group1 = np.array(group1)
        self.group2 = np.array(group2)
        self.amplitude0 = amplitude0

    def prepare(self, mol):
        self.group1_Mass = np.sum(AtomicMass()[mol.Z_num[self.group1] - 1])
        self.group1_COM = np.sum(mol.xyz[self.group1].T * AtomicMass()[mol.Z_num[self.group1] - 1], 1) / self.group1_Mass
        self.group2_Mass = np.sum(AtomicMass()[mol.Z_num[self.group2] - 1])
        self.group2_COM = np.sum(mol.xyz[self.group2].T * AtomicMass()[mol.Z_num[self.group2] - 1], 1) / self.group2_Mass
        self.unit_vec = (self.group2_COM - self.group1_COM) / np.linalg.norm(self.group2_COM - self.group1_COM)
        self.unit_vec = self.unit_vec.T
        return self

    def transform(self, xyz, amplitude=None):
        assert (np.max(self.group1) <= len(xyz)), \
            "Index out of bound: largest index of group 1 > length of supplied molecule"
        assert (np.max(self.group2) <= len(xyz)), \
            "Index out of bound: largest index of group 2 > length of supplied molecule"

        if amplitude is None:
            amplitude = self.amplitude0

        xyz[self.group1] -= self.unit_vec * amplitude / 2
        xyz[self.group2] += self.unit_vec * amplitude / 2

        return xyz

class Transformation_distanceCOM_1side:
    pass

class Transform_rotation:
    def __init__(self, group1, axis, amplitude=0):
        # A, B, and C can be group of atoms.
        # Centers will be the mean of their coordinates.
        # If axis is length 2 (AB), use vector AB as the rotation axis
        # If axis is length 3 (ABC), use the center of central group as the center,
        # the normal vector of BA and BC as axis.
        # Amplitude is in degrees
        pass

class Transform_rotationCOM:
    def __init__(self, group1, axis, amplitude=0):
        # A, B, and C can be group of atoms.
        # Centers will be the COM of their coordinates.
        # If axis is length 2 (AB), use vector AB as the rotation axis
        # If axis is length 3 (ABC), use the COM of central group as the center,
        # the normal vector of BA and BC as axis
        # Amplitude is in degrees
        pass