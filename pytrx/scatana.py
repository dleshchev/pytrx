

import numpy as np
import matplotlib.pyplot as plt
from pytrx.scatdata import ScatData
from pytrx import scatsim, hydro

class ScatAna(ScatData):

    def __init__(self, input):
        super().__init__(input, smallLoad=True)


    def scale(self, solvent, qNormRange=None, plotting=True):

        if qNormRange is None:
            qNormRange = self.aiGeometry.qNormRange

        s_th = scatsim.totalScattering(self.q, scatsim.Solvent(solvent))
        q_sel = (self.q >= qNormRange[0]) & (self.q <= qNormRange[1])
        scale = np.trapz(s_th[q_sel], self.q[q_sel]) / np.trapz(self.total.s_av[q_sel, 0], self.q[q_sel])

        self.total.scale_by(scale)
        self.diff.scale_by(scale)


    def solvent_per_solute(self, solvent, concentration):
        data_solv = hydro.data[solvent]
        return data_solv.density / data_solv.molar_mass / concentration


    def 
