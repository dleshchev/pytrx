

import numpy as np
import matplotlib.pyplot as plt
from pytrx.scatdata import ScatData
from pytrx import scatsim, hydro

class SmallMoleculeProject:

    def __init__(self, input_data, **kwargs):
        '''

        Args:
            input_data - .h5 file created using ScatData.save method
            **kwargs - any metadata you like, e.g. concnetration=10, solvent='water', etc
        '''

        if type(input_data) == str:
            self.data = ScatData(input_data, smallLoad=True)
        elif type(input_data) == ScatData:
            self.data = input_data

        self.metadata = Metadata(**kwargs)


    def scale(self, qNormRange=None, plotting=True, fig=None, idx_off=None):

        if qNormRange is None:
            qNormRange = self.data.aiGeometry.qNormRange

        solvent = self.metadata.solvent
        s_th = scatsim.totalScattering(self.data.q, scatsim.Solvent(solvent))

        q = self.data.q
        q_sel = (q >= qNormRange[0]) & (q <= qNormRange[1])
        if idx_off is None:
            idx_off = self.data.t_str == self.data.diff.toff_str
        s_off = self.data.total.s_av[:, idx_off]

        scale = np.trapz(s_th[q_sel], q[q_sel]) / np.trapz(s_off[q_sel, 0], q[q_sel])

        self.data.total.scale_by(scale)
        self.data.diff.scale_by(scale)

        if plotting:
            plt.figure(fig)
            plt.clf()
            plt.plot(q, s_off*scale, 'k-', label=('data (' + str(self.data.diff.toff_str) + ')'))
            plt.plot(q, s_th, 'r--', label='solvent (gas)')
            plt.xlabel('q, 1/A')
            plt.ylabel('S(q), e.u.')
            plt.legend()
            plt.xlim(q.min(), q.max())


    def submit_candidates(self, candidate_list):
        q = self.data.q
        R = self.solvent_per_solute()
        self.ds_solute = np.zeros(q.size, len(candidate_list))
        self.ds_labels = np.array([i.label for i in candidate_list])
        for i, candidate in enumerate(candidate_list):
            self.ds_solute[:, i] = (scatsim.Debye(q, candidate.mol_es) - scatsim.Debye(q, candidate.mol_gs))/R


    def solvent_per_solute(self):
        solvent = self.metadata.solvent
        concentration = self.metadata.concentration
        data = hydro.data[solvent]
        return data.density / data.molar_mass / concentration







class Metadata:

    def __init__(self, solvent=None, concentration=None,
                 sample_thickness=None,
                 epsilon=None, OD=None,
                 power=None, wavelength=None, laser_fwhm=None,
                 xray_fwhm_h=None, xray_fwhm_w=None, xray_polychromatic=None, more=None):
        self.solvent = solvent

        self.sample_thickness = sample_thickness

        self.concentration = concentration
        self.epsilon = epsilon
        if OD:
            self.OD = OD
        else:
            if concentration and epsilon:
                self.OD = concentration * epsilon

        self.power = power
        self.wavelength = wavelength
        self.laser_fwhm = laser_fwhm

        self.xray_fwhm_h = xray_fwhm_h
        self.xray_fwhm_w = xray_fwhm_w
        self.xray_polychromatic = xray_polychromatic

        if more: self.read_more(more)

    def read_more(self, more):
        for pair in more:
            key, value = pair
            self.__setattr__(self, key, value)


    def estimate_esf(self):
        pass




class Solute:

    def __init__(self, xyz_gs=None, xyz_es=None, label=None):
        self.label=label
        self.xyz_gs = xyz_gs
        self.xyz_es = xyz_es
        self.mol_gs = scatsim.fromXYZ(xyz_gs)
        self.mol_es = scatsim.fromXYZ(xyz_es)



# class Solute:
#
#     def __init__(self, inputStyle='XYZ_file', inputObj=None,
#                  qRange=[0, 10], nqpt=1001, modelLabels=None, printing=False):
#
#         assert ((inputStyle == 'XYZ_file') or
#                 (inputStyle == 'XYZ_list') or
#                 (inputStyle == 'PDB_file') or
#                 (inputStyle == 'PDB_entry')), \
#             'To initiate MolecularStructure class you need to provide one of' \
#             'the following input styles: XYZ_file, XYZ_list, PDB_file, PDB_entry'
#
#         assert ((type(qRange) == list) and
#                 (len(qRange) == 2) and
#                 ((type(qRange[0]) == float) or (type(qRange[0]) == int)) and
#                 ((type(qRange[1]) == float) or (type(qRange[1]) == int))), \
#             'Provide correct qRange'
#
#         assert type(nqpt) == int, 'nqpt should be an int'
#
#         if modelLabels:
#             assert (len(inputObj) == len(modelLabels)), 'provide the same number' \
#                                                         ' of models as modelLabels'
#         else:
#             modelLabels = [i for i in range(len(inputObj))]
#
#         self.q = np.linspace(qRange[0], qRange[1], nqpt)
#         listOfModels = inputObj
#         self.s = np.zeros((self.q.size, len(listOfModels)))
#         self.f_self = np.zeros((self.q.size, len(listOfModels)))
#         self.f_sharp = np.zeros((self.q.size, len(listOfModels)))
#         self.gr = []
#
#         if inputStyle == 'XYZ_file':
#             for i, filepath in enumerate(listOfModels):
#                 if printing: print('Calculating scattering for model', modelLabels[i])
#                 model = self.FiletoZXYZ(filepath)
#                 self.s[:, i], self.f_self[:, i], self.f_sharp[:, i] = self.DebyeScat_fromZXYZ(model, self.q)
#                 self.gr.append(self.ZXYZtoGR(model))
#
#         elif inputStyle == 'XYZ_list':
#             listOfModels = inputObj
#             self.s = np.zeros((self.q.size, len(listOfModels)))
#             for i, model in enumerate(listOfModels):
#                 if printing: print('Calculating scattering for model', modelLabels[i])
#                 self.s[:, i], self.f_self[:, i], self.f_sharp[:, i] = self.DebyeScat_fromZXYZ(model, self.q)
#                 self.gr.append(self.ZXYZtoGR(model))
#
#         elif inputStyle == 'PDB_file':
#             pass
#         elif inputStyle == 'PDB_list':
#             pass
#
#     def FiletoZXYZ(self, filepath, n_head=0):
#         with open(filepath) as f:
#             content = f.readlines()
#         ZXYZ = [[x.split()[0],
#                  float(x.split()[1]),
#                  float(x.split()[2]),
#                  float(x.split()[3])]
#                 for x in content[n_head:]]
#         return ZXYZ
#
#     def DebyeScat_fromZXYZ(self, ZXYZ, q):
#
#         Elements = self.getElements(ZXYZ)
#         atomForm = self.getAtomicFormFactor(Elements, q)
#
#         S = np.zeros(q.shape)
#         f_self = np.zeros(q.shape)
#         f_sharp = np.zeros(q.shape)
#         for i, item in enumerate(ZXYZ):
#             xyz_i = np.array(item[1:])
#             f_i = atomForm[item[0]]
#
#             S += f_i ** 2
#             f_self += f_i ** 2
#
#             for jtem in ZXYZ[:i]:
#                 xyz_j = np.array(jtem[1:])
#                 r_ij = np.sqrt(np.sum((xyz_i - xyz_j) ** 2))
#                 f_j = atomForm[jtem[0]]
#
#                 #                print(r_ij)
#                 #                S += 2 * f_i * f_j * np.sin( q * r_ij ) / ( q * r_ij )
#                 S[q != 0] += 2 * f_i[q != 0] * f_j[q != 0] * np.sin(q[q != 0] * r_ij) / (q[q != 0] * r_ij)
#                 S[q == 0] += 2 * f_i[q == 0] * f_j[q == 0]
#                 f_sharp += 2 * f_i * f_j
#
#         return S, f_self, f_sharp
#
#     def ZXYZtoGR(self, ZXYZ, Rmax=1e2, dR=1e-2):
#
#         Elements = self.getElements(ZXYZ)
#         Rpts = Rmax / dR
#
#         r = np.linspace(0, Rmax, Rpts + 1)
#         r_bins = np.linspace(-dR / 2, Rmax + dR / 2, Rpts + 2)
#
#         gr = {}
#         for i, item in enumerate(Elements):
#             xyz_i = np.array(list(x[1:] for x in ZXYZ if x[0] == item))
#             for j, jtem in enumerate(Elements[:i + 1]):
#                 xyz_j = np.array(list(x[1:] for x in ZXYZ if x[0] == jtem))
#                 #                print(xyz_i,xyz_j)
#                 dist = np.sqrt(np.subtract(xyz_i[:, [0]], xyz_j[:, [0]].T) ** 2 +
#                                np.subtract(xyz_i[:, [1]], xyz_j[:, [1]].T) ** 2 +
#                                np.subtract(xyz_i[:, [2]], xyz_j[:, [2]].T) ** 2).flatten()
#
#                 #                print(r_bins.min(), r_bins.max())
#                 gr_ij = np.histogram(dist, r_bins)[0]
#                 if item != jtem:
#                     gr[item + '-' + jtem] = 2 * gr_ij
#                 else:
#                     gr[item + '-' + jtem] = gr_ij
#
#         return r, gr
#
#     def DebyeScat_fromGR(self, r, gr, q):
#         Elements = list(set(x[:x.index('-')] for x in gr))
#         atomForm = self.getAtomicFormFactor(Elements, q)
#
#         QR = q[np.newaxis].T * r[np.newaxis]
#         Asin = np.sin(QR) / QR
#         Asin[QR == 0] = 1;
#
#         S = np.zeros(q.shape)
#         for atomPair, atomCorrelation in gr.items():
#             sidx = atomPair.index('-')  # separator index
#             El_i, El_j = atomPair[:sidx], atomPair[sidx + 1:]
#             f_i = atomForm[El_i][np.newaxis]
#             f_j = atomForm[El_j][np.newaxis]
#             S += np.squeeze(f_i.T * f_j.T * np.dot(Asin, atomCorrelation[np.newaxis].T))
#
#         return S
#
#     def getSR(self, r, alpha):
#         self.r = r.copy()
#         dq = self.q[1] - self.q[0]
#         x = self.q[None, :] * self.r[:, None]
#         self.rsr = ((np.sin(x) * dq) @ ((self.s - self.f_self) / self.f_sharp *
#                                         self.q[:, None] *
#                                         np.exp(-alpha * self.q[:, None] ** 2)))
#
#     def getAtomicFormFactor(self, Elements, q):
#         if type(Elements) == str: Elements = [Elements]
#
#         s = q / (4 * pi)
#         formFunc = lambda s, a: (np.sum(np.reshape(a[:5], [5, 1]) *
#                                         np.exp(-a[6:, np.newaxis] * s ** 2), axis=0) + a[5])
#
#         fname = pkg_resources.resource_filename('pytrx', './f0_WaasKirf.dat')
#         with open(fname) as f:
#             content = f.readlines()
#
#         atomData = list()
#         for i, x in enumerate(content):
#             if x[0:2] == '#S':
#                 atomName = x.rstrip().split()[-1]
#                 if any([atomName == x for x in Elements]):
#                     atomCoef = content[i + 3].rstrip()
#                     atomCoef = np.fromstring(atomCoef, sep=' ')
#                     atomData.append([atomName, atomCoef])
#
#         atomData.sort(key=lambda x: Elements.index(x[0]))
#         atomForm = {}
#         for x in atomData:
#             atomForm[x[0]] = formFunc(s, x[1])
#
#         return atomForm
#
#     def getElements(self, ZXYZ):
#         return list(set(x[0] for x in ZXYZ))
#
#     def applyPolyCorrection(self, q_poly, E, I, E_eff='auto'):
#         self.s_poly = self._PolyCorrection(self.q, self.s, q_poly, E, I, E_eff=E_eff)
#         self.polyFlag = True
#
#     def _PolyCorrection(self, q_mono, s_mono, q_poly, E, I, E_eff):
#         I = I[np.argsort(E)]
#         E = np.sort(E)
#
#         if E_eff == 'auto':
#             E_eff = E[np.argmax(I)]
#
#         I = I / np.sum(I)
#
#         W = 12.3984 / E
#         W_eff = 12.3984 / E_eff
#
#         tth_mono = 2 * np.arcsin(q_mono[:, np.newaxis] * W[np.newaxis] / (4 * pi)) * 180 / (pi)
#         tth_poly = 2 * np.arcsin(q_poly * W_eff / (4 * pi)) * 180 / (pi)
#
#         if not np.all(tth_poly[0] > tth_mono[0, 0]):
#             raise ValueError('Input q range is too narrow: Decrease q_mono min.')
#         if not tth_poly[-1] < tth_mono[-1, -1]:
#             raise ValueError('Input q range is too narrow: Increase q_mono max.')
#
#         if len(s_mono.shape) == 1:
#             s_mono = s_mono[:, np.newaxis]
#
#         nCurves = s_mono.shape[1]
#         s_poly = np.zeros((q_poly.size, nCurves))
#         for i in range(nCurves):
#             for j, e in enumerate(E):
#                 s_poly[:, i] += I[j] * np.interp(tth_poly, tth_mono[:, j], s_mono[:, i])
#
#         return s_poly




