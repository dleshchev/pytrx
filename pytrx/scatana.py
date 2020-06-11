
import numpy as np
import matplotlib.pyplot as plt
from pytrx.scatdata import ScatData
from pytrx import scatsim, hydro


## TODO:
#   - MINOR
#   - we need to access both isotroptic (ds0) and anisotropic (ds2) parts of scattering, so we need to create corresponding fields in IntensityContainer. ds (s) must remain there
#   - it would be great to figure out with ds vs s pointers
#   - move ZXYZ data from scatsim.py to hydro.py by expanding the hydroproperties class. scatsim.Solvent should still be used to calculate scattering (may be more descriptive name could be applied)
#   - MAJOR
#   - * write Solute class as interface for structural optimization (currently it can accept 2 xyz files for ES and GS),
#   - the goal for Solute is to generate a function that computes Debye difference as a function of some parameters, where number of parameters goes from 0 (None) to whatever
#   - it should be able to accept filepath (str), Molecule instance, VibratingMolecule (may be)
#   - Solute output (or method, attribute, it should return) -  f(q, parameters), this f is the difference signal S_es(q,some_parameters) - S_gs(q, some_other_parameters), parameters should be ablot to be 0 or None
#   - * write Cage class - accept dense gridded q, ds, and covariance (emphasize in Q space) OR accept diff_gr (TBD)
#   - * write Solvent Class - TBD
#   - * write fitting routines (regressors) - WLS, GLS, and TLS
#   - More thoughts on Solute class:
#   - user somehow comes up with a function that produces xyz matrix as a function of parameters or (no parameters)
#   - user packs it into a class that is either Molecule or it is VibratingMolecule (see below) or we can call something else
#   - the goal of this is to have an instance with two fields: Z and xyz. Then Solute class has method scattering_signal that converts it from xyz(p) to dS(q, p)




# class VibratingMolecule(Molecule):
#     def __init__(self):
#         super.__init__()


class SmallMoleculeProject:

    def __init__(self, input_data, **kwargs):
        '''
        Args:
            input_data - .h5 file created using ScatData.save method
            **kwargs - any metadata you like, e.g. concnetration=10, solvent='water', etc
        '''
        # print(type(input_data), type(input_data) == ScatData)
        if type(input_data) == str:
            self.data = ScatData(input_data, smallLoad=True)
        elif type(input_data) == ScatData:
            print('Inputting ScatData data')
            self.data = input_data

        self.metadata = Metadata(**kwargs)


    def scale(self, qNormRange=None, plotting=True, fig=None, idx_off=None):
        '''

        Args:
            qNormRange: Two element list such as [2.1 4.1]
            plotting: Flag, T/F
            fig: figure number for plotting
            idx_off: n where the n-th curve is used as the laser-off curve

        Returns:

        '''
        if qNormRange is None:
            qNormRange = self.data.aiGeometry.qNormRange

        assert (self.metadata.solvent is not None), 'Solvent in metadata not specified.'

        solvent = self.metadata.solvent
        s_th = scatsim.totalScattering(self.data.q, scatsim.Solvent(solvent))

        q = self.data.q
        q_sel = (q >= qNormRange[0]) & (q <= qNormRange[1])
        if idx_off is None:
            idx_off = self.data.t_str == self.data.diff.toff_str
        s_off = self.data.total.s_av[:, idx_off].copy()

        scale = np.trapz(s_th[q_sel], q[q_sel]) / np.trapz(s_off[q_sel], q[q_sel])

        self.data.total.scale_by(scale)
        self.data.diff.scale_by(scale)

        if plotting:
            plt.figure(fig)
            plt.clf()
            plt.plot(q, s_off *scale, 'k-', label=('data (' + str(self.data.diff.toff_str) + ')'))
            plt.plot(q, s_th, 'r--', label='solvent (gas)')
            plt.xlabel('q, 1/A')
            plt.ylabel('S(q), e.u.')
            plt.legend()
            plt.xlim(q.min(), q.max())


    def add_solute(self, solute_instance):
        # q = self.data.q
        # R = self.solvent_per_solute()

        self.solute = solute_instance

        # self.ds_solute = np.zeros(q.size, len(candidate_list))
        # self.ds_labels = np.array([i.label for i in candidate_list])
        # for i, candidate in enumerate(candidate_list):
        #     self.ds_solute[:, i] = (scatsim.Debye(q, candidate.mol_es) - scatsim.Debye(q, candidate.mol_gs))/R


    def add_solvent(self, solvent_instance):
        self.solvent = solvent_instance


    def add_cage(self, cage_instance):
        self.cage = cage_instance


    def fit(self, p0, method='gls'):
        pass


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

    def __init__(self, input_gs=None, input_es=None,
                 label=None):

        '''
        input_gs - ground state structure, must be a path to file or Molecule instance or instance based on Molecule class
        '''

        self.label = label
        self.mol_gs = self.parse_input(input_gs)
        # self.mol_gs_ref = self.parse_input(input_gs)
        self.mol_es = self.parse_input(input_es)

        self.n_par_total = self.mol_es.n_par + self.mol_gs.n_par
        # self.mol_es_ref = self.parse_input(input_es)

    def parse_input(self, input):
        if type(input) == str:
            # Takes the xyz file
            print("Using scatsim.fromXYZ will result in no associated_transformation in the returned Molecule.")
            print("You may want to consider creating the Molecule and pass it to the Solute.")
            return scatsim.fromXYZ(input)
        else:
            # Take the Molecule class
            return input


    # def signal(self, q, pars=None):
    #     # pars is a list for BOTH excited state (first) and ground state (second)
    #    if pars is None:
    #        pars_es, pars_gs = None, None

    def s(self, q, pars=None, target='mol_es'):
        '''
        Computes the signal for es/gs molecule using provided parameters and a q-grid
        '''
        if target == 'mol_es':
            if self.mol_es is not None:
                self.mol_es.transform(pars)
                return scatsim.Debye(q, self.mol_es)
            else:
                return np.zeros(q.shape)
        elif target == 'mol_gs':
            if self.mol_gs is not None:
                self.mol_gs.transform(pars)
                return scatsim.Debye(q, self.mol_gs)
            else:
                return np.zeros(q.shape)
        else:
            print("No signal is calculated as no target is specified. None returned.")

    def ds(self, q, pars=None):
        '''
        Originally just 'signal' but changed to 'ds' as this is calculating difference signal
        '''
        # self.mol_es.move(*x) - consider this
        pars_es, pars_gs = deal_pars(pars, self.mol_es.n_par)
        print(f'ES parameters: {pars_es}, GS parameters: {pars_gs}')
        return self.s(q, pars=pars_es, target='mol_es') - self.s(q, pars=pars_gs, target='mol_gs')

    def list_pars(self):
        # Because we pass to signal() a list of parameters which is not intuitive
        print(f'Listing parameters: \n'
              f'There are {self.n_par_total} parameters to be passed '
              f'to the pars argument as a list for ds method\n')
        for i in np.arange(self.mol_es.n_par):
            print(f'Parameter { i +1}: ES, {type(self.mol_es._associated_transformation[i])}')
            self.mol_es._associated_transformation[i].describe()
            print("")
        for i in np.arange(self.mol_gs.n_par):
            print(f'Parameter { i + 1 +self.mol_es.n_par}: ES, {type(self.mol_gs._associated_transformation[i])}')
            self.mol_gs._associated_transformation[i].describe()
            print("")
        # if target == 'mol_es':
        #     return scatsim.Debye(q, self.mol_es)
        # elif target == 'mol_gs':
        #     return scatsim.Debye(q, self.mol_gs)
        # else:
        #     print("No signal is calculated as no target is specified. None returned.")


    # def ds(self, q, pars):
    #     '''
    #     Originally just 'signal' but changed to 'ds' as this is calculating difference signal
    #     '''
    #     # self.mol_es.move(*x) - consider this
    #     pars_es, pars_gs = deal_pars(pars, self.mol_es.n_par)
    #     print(pars_es, pars_gs)
    #     return self.s(q, pars=pars_es, target='mol_es') - self.s(q, pars=pars_gs, target='mol_gs')


            # scatsim.Debye(q, self.mol_es) - scatsim.Debye(q, self.mol_gs)
    #
    # def transform(self, target, par=None):
    #     '''
    #         Interface function
    #     '''
    #     if par is not None:
    #         if target == 'mol_es':
    #             self.mol_es = self.mol_es.transform(par=par)
    #         if target == 'mol_gs':
    #             self.mol_gs = self.mol_gs.transform(par=par)
    #     else:
    #         print("No moves supplied. No transformation happened.")
    #     return self





class Solvent:

    def __init__(self, q, dsdt, dsdr, sigma=None, K=None):
        self.q = q
        self.dsdt = dsdt
        self.dsdr = dsdr
        self.C = read_sigma(sigma)
        self.K = K


    def smooth(self):
        pass



class Cage:

    def __init__(self, q, ds_cage, sigma=None):
        self.q = q
        self.ds_cage = ds_cage
        self.C = read_sigma(sigma)



def read_sigma(sigma):
    '''
    auxillary function for reading covariance matrix.
    Args:
        sigma: can be either
        - one-dimensional numpy ndarray containing standard deviations of the measurement at each point
        - two-dimensional numpy ndarray containing the full covariance matrix for the measurement

    Returns:
        two-dimensional covariance matrix
    '''
    if sigma is not None:
        assert type(sigma) == np.ndarray, 'sigma must be an numpy ndarray'
        if sigma.ndim == 1:
            return np.diag(sigma ** 2)
        elif sigma.ndim == 2:
            return sigma
        else:
            raise ValueError('sigma must be either 1d- or 2d array')
    return None


def deal_pars(pars, n):
    if pars is None:
        pars_es, pars_gs = None, None
    else:
        pars_es, pars_gs = pars[:n], pars[n:]
        if len(pars_es) == 0: pars_es = None
        if len(pars_gs) == 0: pars_gs = None
    return pars_es, pars_gs
