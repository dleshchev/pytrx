import numpy as np

from abc import (ABC as _ABC, abstractmethod as _abstractmethod)


class DebyeWaller(_ABC):
    '''
    Abstract class for Debye-Waller dispersions
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
        sigma = p_dict[main_key + self.suffix[0]]

        if sigma > 0:
            p = p0 + np.linspace(-self.ns, self.ns, self.m) * sigma
            ksi = (p - p0) / sigma
            w = np.exp(-0.5 * ksi**2)
            w /= np.sum(w)
            return p, w
        else:
            return p0, 1


# class AssymetricGaussian:
#     def __init__(self, m=25, ns=3, suffix=['_sigma_left', '_sigma_right']):
#         super().__init__(suffix=suffix)
#         self.m = m
#         self.ns = ns
#
#     def disperse(self, p_dict, main_key):
#         p0 = p_dict[main_key]
#         sigma_left = p_dict[main_key + self.suffix[0]]
#         sigma_right = p_dict[main_key + self.suffix[1]]
        # if sigma > 0: