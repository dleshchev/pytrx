# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 17:33:17 2016

@author: darren

A library of predefined moves for the Molecule class.

"""

import numpy as np
import math
from pytrx.utils import AtomicMass
from abc import (ABC as _ABC, abstractmethod as _abstractmethod)


class Transformation(_ABC):
    ''' Abstract class for transformations

    '''

    def __init__(self):
        pass

    @_abstractmethod
    def prepare(self, xyz, Z_num):
        '''computes the necessary '''

    @_abstractmethod
    def describe(self):
        '''transformation description '''

    @_abstractmethod
    def transform(self, xyz, Z_num, amplitude=None):
        pass


class Transformation_move_vector(Transformation):
    # Move a group of atoms along a vector (normalized)
    def __init__(self, group1, vector, amplitude0=0, reprep=True):
        super().__init__()
        self.group1 = np.array(group1)
        self.vector = np.array(vector)
        self.unit_vector = np.array(vector) / np.linalg.norm(vector)
        self.amplitude0 = amplitude0
        self.reprep = reprep

    def prepare(self, xyz, Z_num):
        assert (np.max(self.group1) <= len(xyz)), \
            "Index out of bound: largest index of group 1 > length of supplied molecule"
        # return self

    def describe(self):
        print("  Moving group1 along a predefined vector.")
        print(f'    Group 1: {self.group1}')
        print(f'    Vector : {self.unit_vector}')

    def transform(self, xyz, Z_num, amplitude=None):
        if amplitude is None:
            amplitude = self.amplitude0
        if self.reprep:
            self.prepare(xyz, Z_num)
        xyz[self.group1] += self.unit_vector * amplitude
        return xyz

class Transformation_group_vector(Transformation):
    # Move a group of atoms along a vector that is constructed by the center of coordinates of two other groups.
    # Vector will be from vector_group 1 to vector_group 2
    def __init__(self, group1, vector_groups, amplitude0=0, reprep=True):
        super().__init__()
        self.group1 = np.array(group1)
        self.vector_groups = vector_groups
        self.amplitude0 = amplitude0
        self.reprep = reprep

    def prepare(self, xyz, Z_num):
        assert (np.max(self.group1) <= len(xyz)), \
            "Index out of bound: largest index of group 1 > length of supplied molecule"
        assert (np.max(self.vector_groups[0]) <= len(xyz)), \
            "Index out of bound: largest index of group 1 > length of supplied molecule"
        assert (np.max(self.vector_groups[1]) <= len(xyz)), \
            "Index out of bound: largest index of group 1 > length of supplied molecule"
        self.vector = np.mean(xyz[self.vector_groups[1]], 0) - np.mean(xyz[self.vector_groups[0]], 0)
        self.unit_vector = np.array(self.vector) / np.linalg.norm(self.vector)

        # return self

    def describe(self):
        print("  Moving group1 along a vector constructed by two other groups (mean of coordinates).")
        print("  Vector will be from vector_group 1 to vector_group 2")
        print(f'    Group 1: {self.group1}')
        print(f'    Vector_group 1: {self.vector_groups[0]}')
        print(f'    Vector_group 2: {self.vector_groups[1]}')

    def transform(self, xyz, Z_num, amplitude=None):
        if amplitude is None:
            amplitude = self.amplitude0
        if self.reprep:
            self.prepare(xyz, Z_num)
        xyz[self.group1] -= self.unit_vector * amplitude
        return xyz


class Transformation_vibration(Transformation):
    def __init__(self, dxyz, amplitude0=0, reprep=True):
        super().__init__()
        self.dxyz = dxyz
        self.amplitude0 = amplitude0
        self.reprep = reprep

    def prepare(self, xyz, Z_num):
        assert self.dxyz.shape[0] == xyz.shape[0], \
            'number of atoms in transformation and in the molecule must match'
        return self

    def describe(self):
        print("Move all atoms along a predefined vibrational mode.")

    def transform(self, xyz, Z_num, amplitude=None):
        if amplitude is None:
            amplitude = self.amplitude0
        if self.reprep:
            self.prepare(xyz, Z_num)
        return xyz + self.dxyz * amplitude



class Transformation_distance(Transformation):
    # Move two groups of atoms closer/further in distance, using simple mean of coordinates as
    # reference centers for each group.
    # Vector is from group1 to group2. Negative amplitude is shrinking.
    def __init__(self, group1, group2, amplitude0=0, reprep=True):
        super().__init__()
        assert (len(group1) > 0) and (len(group2) > 0), 'Cannot operate on empty set'
        self.group1 = np.array(group1)
        self.group2 = np.array(group2)
        self.amplitude0 = amplitude0
        self.reprep = reprep

    def prepare(self, xyz, Z_num):
        assert (np.max(self.group1) <= len(xyz)), \
            "Index out of bound: largest index of group 1 > length of supplied molecule"
        assert (np.max(self.group2) <= len(xyz)), \
            "Index out of bound: largest index of group 2 > length of supplied molecule"
        self.group1_mean = np.mean(xyz[self.group1], 0)
        self.group2_mean = np.mean(xyz[self.group2], 0)
        self.unit_vec = (self.group2_mean - self.group1_mean) / np.linalg.norm(self.group2_mean - self.group1_mean)
        # return self

    def describe(self):
        print(f'  Increasing / decreasing distance between group1 and group2 using '
              f'simple mean of coordinates as centers.\n'
              f'  Both groups move.')
        print(f'    Group 1: {self.group1}')
        print(f'    Group 2: {self.group2}')

    def transform(self, xyz, Z_num, amplitude=None):
        if amplitude is None:
            amplitude = self.amplitude0
        if self.reprep:
            self.prepare(xyz, Z_num)
        xyz[self.group1] -= self.unit_vec * amplitude / 2
        xyz[self.group2] += self.unit_vec * amplitude / 2

        return xyz


class Transformation_distance_1side(Transformation):
    # Move GROUP 2 toward/away from GROUP 1 in distance, using simple mean of coordinates as
    # reference centers for each group.
    # Vector is from group1 to group2. Negative amplitude is shrinking.
    # GROUP 1 is fixed.
    def __init__(self, group1, group2, amplitude0=0, reprep=True):
        super().__init__()
        assert (len(group1) > 0) and (len(group2) > 0), 'Cannot operate on empty set'
        self.group1 = np.array(group1)
        self.group2 = np.array(group2)
        self.amplitude0 = amplitude0
        self.reprep = reprep

    def prepare(self, xyz, Z_num):
        assert (np.max(self.group1) <= len(xyz)), \
            "Index out of bound: largest index of group 1 > length of supplied molecule"
        assert (np.max(self.group2) <= len(xyz)), \
            "Index out of bound: largest index of group 2 > length of supplied molecule"
        self.group1_mean = np.mean(xyz[self.group1], 0)
        self.group2_mean = np.mean(xyz[self.group2], 0)
        self.unit_vec = (self.group2_mean - self.group1_mean) / np.linalg.norm(self.group2_mean - self.group1_mean)
        # return self

    def describe(self):
        print(f'  Increasing / decreasing distance between group1 and group2 using '
              f'simple mean of coordinates as centers.\n'
              f'  Only group 2 moves.')
        print(f'    Group 1: {self.group1}')
        print(f'    Group 2: {self.group2}')

    def transform(self, xyz, Z_num, amplitude=None):
        if amplitude is None:
            amplitude = self.amplitude0
        if self.reprep:
            self.prepare(xyz, Z_num)
        xyz[self.group2] += self.unit_vec * amplitude / 2

        return xyz

class Transformation_distance_1side(Transformation):
    # Move GROUP 2 toward/away from GROUP 1 in distance, using simple mean of coordinates as
    # reference centers for each group.
    # Vector is from group1 to group2. Negative amplitude is shrinking.
    # GROUP 1 is fixed.
    def __init__(self, group1, group2, amplitude0=0, reprep=True):
        super().__init__()
        assert (len(group1) > 0) and (len(group2) > 0), 'Cannot operate on empty set'
        self.group1 = np.array(group1)
        self.group2 = np.array(group2)
        self.amplitude0 = amplitude0
        self.reprep = reprep

    def prepare(self, xyz, Z_num):
        assert (np.max(self.group1) <= len(xyz)), \
            "Index out of bound: largest index of group 1 > length of supplied molecule"
        assert (np.max(self.group2) <= len(xyz)), \
            "Index out of bound: largest index of group 2 > length of supplied molecule"
        self.group1_mean = np.mean(xyz[self.group1], 0)
        self.group2_mean = np.mean(xyz[self.group2], 0)
        self.unit_vec = (self.group2_mean - self.group1_mean) / np.linalg.norm(self.group2_mean - self.group1_mean)
        # return self

    def describe(self):
        print(f'  Increasing / decreasing distance between group1 and group2 using '
              f'simple mean of coordinates as centers.\n'
              f'  Only group 2 moves.')
        print(f'    Group 1: {self.group1}')
        print(f'    Group 2: {self.group2}')

    def transform(self, xyz, Z_num, amplitude=None):
        if amplitude is None:
            amplitude = self.amplitude0
        if self.reprep:
            self.prepare(xyz, Z_num)
        xyz[self.group2] += self.unit_vec * amplitude / 2

        return xyz


class Transformation_distanceCOM(Transformation):
    # Move two group of atoms closer/further in distance, using center of mass as ref centers for each group
    # Vector is from group1 to group2. Negative amplitude is shrinking.
    def __init__(self, group1, group2, amplitude0=0, reprep=True):
        super().__init__()
        assert (len(group1) > 0) and (len(group2) > 0), 'Cannot operate on empty set'
        self.group1 = np.array(group1)
        self.group2 = np.array(group2)
        self.amplitude0 = amplitude0
        self.reprep = reprep

    def prepare(self, xyz, Z_num):
        assert (np.max(self.group1) <= len(xyz)), \
            "Index out of bound: largest index of group 1 > length of supplied molecule"
        assert (np.max(self.group2) <= len(xyz)), \
            "Index out of bound: largest index of group 2 > length of supplied molecule"

        self.group1_COM = np.sum(xyz[self.group1].T * AtomicMass()[Z_num[self.group1] - 1],
                                 1) / np.sum(AtomicMass()[Z_num[self.group1] - 1])

        self.group2_COM = np.sum(xyz[self.group2].T * AtomicMass()[Z_num[self.group2] - 1],
                                 1) / np.sum(AtomicMass()[Z_num[self.group2] - 1])
        self.unit_vec = (self.group2_COM - self.group1_COM) / np.linalg.norm(self.group2_COM - self.group1_COM)
        self.unit_vec = self.unit_vec.T
        # return self

    def describe(self):
        print(f'  Increasing / decreasing distance between group1 and group2 using centers of masses as centers.\n'
              f'  Both groups move.')
        print(f'    Group 1: {self.group1}')
        print(f'    Group 2: {self.group2}')

    def transform(self, xyz, Z_num, amplitude=None):
        if amplitude is None:
            amplitude = self.amplitude0
        if self.reprep:
            self.prepare(xyz, Z_num)
        xyz[self.group1] -= self.unit_vec * amplitude / 2
        xyz[self.group2] += self.unit_vec * amplitude / 2

        return xyz


class Transformation_distanceCOM_1side(Transformation):
    # Move GROUP 2 toward/away from GROUP 1 in distance, using center of mass as ref centers for each group
    # Vector is from group1 to group2. Negative amplitude is shrinking.
    # GROUP 1 is fixed.
    def __init__(self, group1, group2, amplitude0=0, reprep=True):
        super().__init__()
        assert (len(group1) > 0) and (len(group2) > 0), 'Cannot operate on empty set'
        self.group1 = np.array(group1)
        self.group2 = np.array(group2)
        self.amplitude0 = amplitude0
        self.reprep = reprep

    def prepare(self, xyz, Z_num):
        assert (np.max(self.group1) <= len(xyz)), \
            "Index out of bound: largest index of group 1 > length of supplied molecule"
        assert (np.max(self.group2) <= len(xyz)), \
            "Index out of bound: largest index of group 2 > length of supplied molecule"
        self.group1_COM = np.sum(xyz[self.group1].T * AtomicMass()[Z_num[self.group1] - 1],
                                 1) / np.sum(AtomicMass()[Z_num[self.group1] - 1])
        self.group2_COM = np.sum(xyz[self.group2].T * AtomicMass()[Z_num[self.group2] - 1],
                                 1) / np.sum(AtomicMass()[Z_num[self.group2] - 1])
        self.unit_vec = (self.group2_COM - self.group1_COM) / np.linalg.norm(self.group2_COM - self.group1_COM)
        self.unit_vec = self.unit_vec.T
        # return self

    def describe(self):
        print(f'  Increasing / decreasing distance between group1 and group2 using centers of masses as centers.\n'
              f'  Only group 2 moves.')
        print(f'    Group 1: {self.group1}')
        print(f'    Group 2: {self.group2}')

    def transform(self, xyz, Z_num, amplitude=None):
        if amplitude is None:
            amplitude = self.amplitude0
        if self.reprep:
            self.prepare(xyz, Z_num)
        xyz[self.group2] += self.unit_vec * amplitude

        return xyz


class Transformation_rotation(Transformation):
    def __init__(self, group1, axis_groups, amplitude0=0, reprep=True):
        # A, B, and C can be group of atoms.
        # Centers will be the mean of their coordinates.
        # If axis is length 2 (AB), use vector AB as the rotation axis
        # If axis is length 3 (ABC), use the center of central group as the center,
        # the cross vector of AB and BC as axis.
        # Amplitude is in degrees
        # Rotation is counterclockwise for an observer to whom the axis vector is pointing (right hand rule)
        super().__init__()
        assert (len(group1) > 0) and (len(axis_groups) > 0), 'Cannot operate on empty set'
        assert (len(axis_groups) == 2) or (len(axis_groups) == 3), 'Axis must be defined with 2 or 3 groups'
        for i in np.arange(len(axis_groups)):
            assert (len(axis_groups[i]) > 0), f'Axis group {i} is empty'

        self.group1 = group1
        self.axis_groups = axis_groups
        self.amplitude0 = amplitude0
        self.reprep = reprep

    def prepare(self, xyz, Z_num):
        assert (np.max(self.group1) <= len(xyz)), \
            "Index out of bound: largest index of group 1 > length of supplied molecule"
        for i in np.arange(len(self.axis_groups)):
            assert (np.max(self.axis_groups) <= len(xyz)), \
                "Index out of bound: largest index of group 1 > length of supplied molecule"
        self.A_mean = np.mean(xyz[self.axis_groups[0]], 0)
        self.B_mean = np.mean(xyz[self.axis_groups[1]], 0)
        if len(self.axis_groups) == 3:
            self.C_mean = np.mean(xyz[self.axis_groups[2]], 0)

        if len(self.axis_groups) == 2:  # Then use AB as vector
            self.axis = self.B_mean - self.A_mean

        if len(self.axis_groups) == 3:  # Use cross product of AB and BC as vector
            self.axis = np.cross(self.B_mean - self.A_mean, self.C_mean - self.B_mean)

        # return self

    def describe(self):
        if len(self.axis_groups) == 2:
            print(f'  Rotate group 1 along the axis from center of axis_group 1 to center of axis_group 2.\n')
        elif len(self.axis_groups) == 3:
            print(f'  Rotate group 1 along the axis normal to the plane spanned by\n'
                  f'    center of axis_group 1 to center of axis_group 2 and \n'
                  f'    center of axis_group 2 to center of axis_group 3. \n'
                  f'  Center is defined as simple mean of coordinates in that group.')

        print(f'    Group 1: {self.group1}')
        print(f'    Axis_group 1: {self.axis_groups[0]}')
        print(f'    Axis_group 2: {self.axis_groups[1]}')
        if len(self.axis_groups) == 3:
            print(f'    Axis_group 3: {self.axis_groups[2]}')

    def transform(self, xyz, Z_num, amplitude=None):
        if amplitude is None:
            amplitude = self.amplitude0
        if self.reprep:
            self.prepare(xyz, Z_num)
        # Shift reference frame, rotate, then shift back
        if len(self.axis_groups) == 2:
            xyz[self.group1] = rotation3D((xyz[self.group1] - self.A_mean).T, self.axis, amplitude).T + self.A_mean
        if len(self.axis_groups) == 3:
            xyz[self.group1] = rotation3D((xyz[self.group1] - self.A_mean).T, self.axis, amplitude).T + self.B_mean

        return xyz


class Transformation_rotationCOM(Transformation):
    def __init__(self, group1, axis_groups, amplitude0=0, reprep=True):
        # A, B, and C can be group of atoms.
        # Centers will be the mean of their coordinates.
        # If axis is length 2 (AB), use vector AB as the rotation axis
        # If axis is length 3 (ABC), use the center of central group as the center,
        # the cross vector of AB and BC as axis.
        # Amplitude is in degrees
        # Rotation is counterclockwise for an observer to whom the axis vector is pointing (right hand rule)
        super().__init__()
        assert (len(group1) > 0) and (len(axis_groups) > 0), 'Cannot operate on empty set'
        assert (len(axis_groups) == 2) or (len(axis_groups) == 3), 'Axis must be defined with 2 or 3 groups'
        for i in np.arange(len(axis_groups)):
            assert (len(axis_groups[i]) > 0), f'Axis group {i} is empty'

        self.group1 = group1
        self.axis_groups = axis_groups
        self.amplitude0 = amplitude0
        self.reprep = reprep

    def prepare(self, xyz, Z_num):
        assert (np.max(self.group1) <= len(xyz)), \
            "Index out of bound: largest index of group 1 > length of supplied molecule"
        for i in np.arange(len(self.axis_groups)):
            assert (np.max(self.axis_groups) <= len(xyz)), \
                "Index out of bound: largest index of group 1 > length of supplied molecule"
        self.A_COM = np.sum(xyz[self.axis_groups[0]].T * AtomicMass()[Z_num[self.axis_groups[0]] - 1],
                            1) / np.sum(AtomicMass()[Z_num[self.axis_groups[0]] - 1])
        self.B_COM = np.sum(xyz[self.axis_groups[1]].T * AtomicMass()[Z_num[self.axis_groups[1]] - 1],
                            1) / np.sum(AtomicMass()[Z_num[self.axis_groups[1]] - 1])
        if len(self.axis_groups) == 3:
            self.C_COM = np.sum(xyz[self.axis_groups[2]].T * AtomicMass()[Z_num[self.axis_groups[2]] - 1],
                                1) / np.sum(AtomicMass()[Z_num[self.axis_groups[2]] - 1])

        if len(self.axis_groups) == 2:  # Then use AB as vector
            self.axis = self.B_COM - self.A_COM

        if len(self.axis_groups) == 3:  # Use cross product of AB and BC as vector
            self.axis = np.cross(self.B_COM - self.A_COM, self.C_COM - self.B_COM)

        # return self

    def describe(self):
        if len(self.axis_groups) == 2:
            print(f'  Rotate group 1 along the axis from center of axis_group 1 to center of axis_group 2\n')
        elif len(self.axis_groups) == 3:
            print(f'  Rotate group 1 along the axis normal to the plane spanned by\n'
                  f'    center of axis_group 1 to center of axis_group 2 and \n'
                  f'    center of axis_group 2 to center of axis_group 3. \n'
                  f'  Center is defined as center of mass in that group.')

        print(f'    Group 1: {self.group1}')
        print(f'    Axis_group 1: {self.axis_groups[0]}')
        print(f'    Axis_group 2: {self.axis_groups[1]}')
        if len(self.axis_groups) == 3:
            print(f'    Axis_group 3: {self.axis_groups[2]}')

    def transform(self, xyz, Z_num, amplitude=None):
        if amplitude is None:
            amplitude = self.amplitude0
        if self.reprep:
            self.prepare(xyz, Z_num)
        if len(self.axis_groups) == 2:
            xyz[self.group1] = rotation3D((xyz[self.group1] - self.A_COM).T, self.axis, amplitude).T + self.A_COM
        if len(self.axis_groups) == 3:
            xyz[self.group1] = rotation3D((xyz[self.group1] - self.A_COM).T, self.axis, amplitude).T + self.B_COM

        return xyz


def rotation3D(v, axis, degrees):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians. Using the Euler-Rodrigues formula:
    https://stackoverflow.com/questions/6802577/rotation-of-3d-vector
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    theta = degrees * math.pi / 180
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    rot_mat = np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                        [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                        [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

    return np.dot(rot_mat, v)
