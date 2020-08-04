import numpy as np




class Gaussian:

    def __init__(self, m=25, ns=3, suffix='_sigma'):
        self.m = m
        self.ns = ns
        self.suffix = suffix
        if type(suffix) != list:
            self.k_par = 1
        else:
            self.k_par = len(suffix)

    def disperse(self, p_dict, main_key):
        p0 = p_dict[main_key]
        sigma = p_dict[main_key + self.suffix]
        if sigma >0:
            p = p0 + np.linspace(-self.ns, self.ns, self.m) * sigma
            ksi = (p - p0) / sigma
            w = np.exp(-0.5 * ksi**2)
            w /= np.sum(w)
            return p, w
        else:
            return p0, 1


