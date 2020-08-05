import numpy as np

from abc import (ABC as _ABC, abstractmethod as _abstractmethod)


class DebyeWaller(_ABC):
    ''' Abstract class for transformations

        '''

    def __init__(self, suffix=None, standard_value=None):
        if type(suffix) != list:
            suffix = [suffix]
        if type(standard_value) != list:
            standard_value = [standard_value]
        self.suffix = suffix
        self.standard_value = standard_value
        # self.par_standard = {}
        # for key, value in zip(self.suffix, self.standard_value):
        #     self.par_standard[key] = value


    @_abstractmethod
    def disperse(self, xyz, Z_num):
        '''computes the necessary '''
        pass




class Gaussian(DebyeWaller):

    def __init__(self, m=25, ns=3, suffix='_sigma', standard_value=0):
        super().__init__(suffix=suffix,
                         standard_value=standard_value)
        self.m = m
        self.ns = ns

    def disperse(self, p_dict, main_key):
        p0 = p_dict[main_key]
        sigma = np.abs(p_dict[main_key + self.suffix[0]])

        if sigma != 0:
            p = p0 + np.linspace(-self.ns, self.ns, self.m) * sigma
            ksi = (p - p0) / sigma
            w = np.exp(-0.5 * ksi**2)
            w /= np.sum(w)
            return p, w
        else:
            return p0, 1


class AssymetricGaussian(DebyeWaller):
    def __init__(self, m=25, ns=3, suffix=['_sigma_left', '_sigma_right'], standard_value=[0, 0]):
        super().__init__(suffix=suffix,
                         standard_value=standard_value)
        self.m = m
        self.ns = ns

    def disperse(self, p_dict, main_key):
        p0 = p_dict[main_key]
        m_side = int((self.m - 1) / 2 + 1)
        sigma_left = np.abs(p_dict[main_key + self.suffix[0]])
        sigma_right = np.abs(p_dict[main_key + self.suffix[1]])
        if sigma_left > 0:
            p_left = p0 + np.linspace(-self.ns, 0, m_side) * sigma_left
            ksi = (p_left - p0) / sigma_left
            w_left = np.exp(-0.5 * ksi ** 2)
        else:
            p_left = p0
            w_left = 1.0

        if sigma_right > 0:
            p_right = p0 + np.linspace(0, self.ns, m_side) * sigma_right
            ksi = (p_right - p0) / sigma_right
            w_right = np.exp(-0.5 * ksi ** 2)
        else:
            p_right = p0
            w_right = 1.0

        p, idx = np.unique(np.hstack((p_left, p_right)), return_index=True)
        w = np.hstack((w_left, w_right))[idx]
        print(p, w, np.sum(w))
        w /= np.sum(w) # normalization should be corrected
        return p, w


