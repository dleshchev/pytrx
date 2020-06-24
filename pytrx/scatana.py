
import numpy as np
import matplotlib.pyplot as plt
from pytrx.scatdata import ScatData
from pytrx import scatsim, hydro
from pytrx.utils import weighted_mean, time_str2num, time_num2str
import lmfit


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

    def __init__(self, input_data, input_model, **kwargs):
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
        else:
            raise ValueError('input_data must be a path to a the h5 file or a ScatData object')

        self.model = input_model

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


    def solvent_per_solute(self):
        solvent = self.metadata.solvent
        concentration = self.metadata.concentration
        data = hydro.solvent_data[solvent]
        return data.density / data.molar_mass / concentration

    def fit(self, qmin=None, qmax=None, t=None, trange=None, tavrg=False, method='gls', prefit=True):

        # check q-range
        if qmin is None: qmin = self.data.q.min()
        if qmax is None: qmax = self.data.q.max()
        q_idx = (self.data.q >= qmin) & (self.data.q <= qmax)
        q_idx = check_range_with_cov_matrix(self.data.q, q_idx, self.data.diff.covqq)

        # check t-range
        if (t is None) and (trange is None):
            t = 'all'

        if type(t) == str:
            if t == 'all':
                t_idx = np.ones(self.data.t.size, dtype=bool)
            else:
                t_idx = self.data.t_str == t
        if type(t) == list:
            t_idx = np.zeros(self.data.t.size, dtype=bool)
            for each_t in t:
                if type(each_t) == str:
                    upd = self.data.t_str == each_t
                else:
                    upd = self.data.t == each_t
                t_idx += upd

        if trange is not None:
            assert (type(trange) == list) and len(trange) == 2, 'trange must be a 2-element list'
            if type(trange[0]) == str:
                trange0 = time_str2num(trange[0])
            else: trange0 = trange[0]
            if type(trange[1]) == str:
                trange1 = time_str2num(trange[1])
            else: trange1 = trange[1]
            t_idx = (self.data.t>=trange0) & (self.data.t<=trange1)

        t_idx = check_range_with_cov_matrix(self.data.t, t_idx, self.data.diff.covtt)

        # get the targets
        qfit = self.data.q[q_idx]
        tfit = self.data.t[t_idx]
        tfit_str = self.data.t_str[t_idx]
        ds_target = self.data.diff.s_av[np.ix_(q_idx, t_idx)]

        C_target = self.data.diff.covqq[np.ix_(q_idx, q_idx)]
        K_target = self.data.diff.covtt[np.ix_(t_idx, t_idx)]

        if tavrg:
            ds_target_T, K_target = weighted_mean(ds_target.T, K_target)
            ds_target = ds_target_T.T

        n_curves = ds_target.shape[1]
        param_labels = list(self.model.params0.keys())
        output = optimizedResult(qfit, tfit, tfit_str, param_labels)

        for i in range(n_curves):
            output[tfit_str[i]] = self.model.fit(qfit, ds_target[:, i], C_target * K_target[i,i],
                                         self.solvent_per_solute(), method=method, prefit=prefit)
        return output
        # return ds_target, C_target, K_target


### output[i] = model.fit(ds_target[:, i], bla bla bla)
###TODO:
###TODO:minor rename C and K to covqq and covtt, may be ds as well
### write wrapper class output__class['esf'] <-- best fit values, should also give access to chisq red chisq, ds_target, ds_taget whatnot
### this output class or SmallMoleculeProject should have plotting functions to produce pretty figures for fitting 2D maps, 1D curves for specfici time delay with decomposition of the model on solute cage and solvent
### one must be able to supply list of models to the SmallMoleculeProject  think about it
### optimize Debye by precomputing f
### Denis keeps working on TLS




def check_range_with_cov_matrix(x, x_range, C):
    a, b = C.shape
    if a == b == np.linalg.matrix_rank(C):
        return x_range
    else:
        x_good = np.abs(np.diag(C)) > np.finfo(float).eps
        return x_good & x_range




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
            self.__setattr__(key, value)


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
        self.n_par_total = 0
        if self.mol_es is not None:
            self.n_par_total += self.mol_es.n_par
        if self.mol_gs is not None:
            self.n_par_total += self.mol_gs.n_par
        self.par_labels, self.par_vals0 = self.list_pars(return_labels=True)
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



    def ds(self, q, pars=None, printing=False):
        '''
        Originally just 'signal' but changed to 'ds' as this is calculating difference signal
        '''
        # self.mol_es.move(*x) - consider this
        if pars is not None:
            assert len(pars) == (self.mol_es.n_par + self.mol_gs.n_par), \
                'nummber of parameteres should match the sum of numbers of parameters for gs and es'
        pars_es, pars_gs = deal_pars(pars, self.mol_es.n_par)
        if printing: print(f'ES parameters: {pars_es}, GS parameters: {pars_gs}')

        if (self.mol_es is not None) and (self.mol_gs is not None):
            return self.mol_es.s(q, pars_es) - self.mol_gs.s(q, pars_gs)
        elif self.mol_gs is not None:
            return - self.mol_gs.s(q, pars_gs)
        elif self.mol_es is not None:
            return self.mol_es.s(q, pars_es)
        else:
            return np.zeros(q.shape)

    def list_pars(self, return_labels=False):
        # Because we pass to signal() a list of parameters which is not intuitive
        labels, standard_values = [], []
        print(f'Listing structural parameters: \n'
              f'There are {self.n_par_total} structural parameters to be passed '
              f'to the pars argument as a list for ds method\n')
        if self.mol_es is not None:
            for i in np.arange(self.mol_es.n_par):
                print(f'Parameter { i +1}: ES, {type(self.mol_es._associated_transformation[i])}')
                self.mol_es._associated_transformation[i].describe()
                print("")
                labels.append(f'par_es_{ i +1}')
                standard_values.append(self.mol_es._associated_transformation[i].amplitude0)
        if self.mol_gs is not None:
            for i in np.arange(self.mol_gs.n_par):
                print(f'Parameter { i + 1 +self.mol_es.n_par}: ES, {type(self.mol_gs._associated_transformation[i])}')
                self.mol_gs._associated_transformation[i].describe()
                print("")
                labels.append(f'par_gs_{i + 1}')
                standard_values.append(self.mol_es._associated_transformation[i].amplitude0)
        if return_labels: return labels, standard_values






class Solvent:

    def __init__(self, input, sigma=None, K=None):
        self.q, self.dsdt_orig, self.dsdr_orig = self.parse_input(input)
        self.C = read_sigma(sigma)
        self.K = K

    def parse_input(self, input):
        if type(input) == tuple:
            q, dsdt_orig, dsdr_orig = input
        elif type(input) == str:
            data = np.genfromtxt(input)
            q = data[:, 0]
            dsdt_orig = data[:, 1]
            dsdr_orig = data[:, 2]
        else:
            raise ValueError('input should a tuple with (q, dsdt, dsdr) elements or a filepath to the file with 3 columns')
        return q, dsdt_orig, dsdr_orig


    def smooth(self):
        pass



class Cage:

    def __init__(self, input, sigma=None):
        self.q, self.ds_orig = self.parse_input(input)
        self.C = read_sigma(sigma)

    def parse_input(self, input):
        if type(input) == tuple:
            q, ds = input
        elif type(input) == str:
            data = np.genfromtxt(input)
            q = data[:, 0]
            ds = data[:, 1]
        else:
            raise ValueError('input should a tuple with (q, ds) elements or a filepath to the file with 2 columns')
        return q, ds




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



_esf_dict= {'value' : 0, 'vary' : True}
_cage_dict = {'value' : 1, 'vary' : False}
_dsdt_dict = {'value' : 0, 'vary' : True}
_dsdr_dict = {'value' : 0, 'vary' : True}






class SolutionScatteringModel:
    def __init__(self, *args,
                 label=None,
                 esf_dict=_esf_dict,
                 cage_dict=_cage_dict,
                 dsdt_dict=_dsdt_dict,
                 dsdr_dict=_dsdr_dict,
                 **kwargs):
        self.solute = None
        self.solvent = None
        self.cage = None
        self.add(*args)
        self.label = label
        self.prepare_parameters(esf_dict=esf_dict,
                                cage_dict=cage_dict,
                                dsdt_dict=dsdt_dict,
                                dsdr_dict=dsdr_dict,
                                **kwargs)


    def add(self, *args):
        for arg in args:
            self._add_item(arg)


    def _add_item(self, item):
        if type(item) == Solute:
            self.solute = item
        elif type(item) == Solvent:
            self.solvent = item
        elif type(item) == Cage:
            self.cage = item
        else:
            raise TypeError ('invalid input type. Must be Solute, Solvent, or Cage')



    def fit(self, q, ds_exp, C, sps, method='gls', prefit=True):
        # out = regressors.fit(Y, C, K, f, method=method) <- actually use this
        q = self.ensure_qrange(q)
        self.prepare_vectors(q)

        if method == 'wls':
            L = np.sqrt(np.diag(np.diag(C)))
        else:
            L = np.linalg.cholesky(np.linalg.inv(C))


        def residual(params, q, y, L, sps):
            dy = self.fit_func(params, q, sps) - y
            return L.T @ dy

        # prefit
        if prefit:
            vary_status = {key : self.params0[key].vary for key in self.params0.keys()}
            for key in self.solute.par_labels:
                self.params0[key].vary = False
            result_pre = lmfit.minimize(residual, self.params0, args=(q, ds_exp, L, sps),
                                 scale_covar=False, method='least_squares')
            self.params0 = result_pre.params
            for key in self.params0.keys():
                self.params0[key].vary = vary_status[key]

        result = lmfit.minimize(residual, self.params0, args=(q, ds_exp, L, sps),
                             scale_covar=False, method='least_squares')

        out = optimizedResultEntry(ds_exp, self.fit_func(result.params, q, sps), result )

        return out


    def fit_func(self, params, q, sps): # need to add q argument
        v = params.valuesdict()

        if len(self.solute.par_labels) != 0:
            pars_structural = [v[p] for p in self.solute.par_labels]
        else:
            pars_structural = None

        return (v['esf']/sps * self.solute.ds(q, pars_structural)
                + v['cage_amp']/sps * self.cage.ds
                + v['dsdt_amp'] * self.solvent.dsdt
                + v['dsdr_amp'] * self.solvent.dsdr)


    def ensure_qrange(self, q_in):

        q_out = q_in.copy()
        qmin = np.max((q_in.min(), self.solvent.q.min(), self.cage.q.min()))
        qmax = np.min((q_in.max(), self.solvent.q.max(), self.cage.q.max()))

        long_print_flag = False
        if qmin > q_in.min():
            print('invalid qmin - ', end='')
            long_print_flag = True

        if qmax < q_in.max():
            print('invalid qmax - ', end='')
            long_print_flag = True

        if long_print_flag:
            print('will perform fit below provided qmax based on the model/data')

        return q_out[(q_out>=qmin) & (q_out<=qmax)]


    def prepare_vectors(self, qfit):
        self.cage.ds = np.interp(qfit, self.cage.q, self.cage.ds_orig)
        self.solvent.dsdt = np.interp(qfit, self.solvent.q, self.solvent.dsdt_orig)
        self.solvent.dsdr = np.interp(qfit, self.solvent.q, self.solvent.dsdr_orig)


    def prepare_parameters(self,
                           esf_dict={ 'value' : 0, 'vary' : True},
                           cage_dict={'value' : 1, 'vary' : False},
                           dsdt_dict={'value' : 0, 'vary' : True},
                           dsdr_dict={'value' : 0, 'vary' : True},
                           **kwargs):
        # TODO: actual interpolation operator with uncertainty propagation

        params0 = lmfit.Parameters()


        if self.solute is None:
            self.solute = Solute() # self.solute.ds(q) returns zero array with q.size
            params0.add('esf', value=0, vary=False)
        else:
            params0.add('esf', **esf_dict)

        if self.cage is None:
            self.cage = Cage((np.array([0, 1e6]), np.array([0, 0])))
            params0.add('cage_amp', value=0, vary=False)
        else:
            params0.add('cage_amp', **cage_dict)

        if self.solvent is None:
            self.solvent = Solvent((np.array([0, 1e6]), np.array([0, 0]), np.array([0, 0])))
            params0.add('dsdt_amp', value=0, vary=False)
            params0.add('dsdr_amp', value=0, vary=False)
        else:
            if self.solvent.dsdt_orig is None:
                self.solvent.dsdt = np.zeros(self.solvent.q.size)
                params0.add('dsdt_amp', value=0, vary=False)
            else:
                params0.add('dsdt_amp', **dsdt_dict)

            if self.solvent.dsdr_orig is None:
                self.solvent.dsdr = np.zeros(self.solvent.q.size)
                params0.add('dsdr_amp', value=0, vary=False)
            else:
                params0.add('dsdr_amp', **dsdr_dict)

        for lab, val in zip(self.solute.par_labels, self.solute.par_vals0):
            print((lab+'_dict'), kwargs.keys(), (lab+'_dict') in kwargs.keys())
            if (lab+'_dict') in list(kwargs.keys()):
                params0.add(lab, **kwargs[lab + '_dict'])
            else:
                params0.add(lab, value=val)

        self.params0 = params0



class optimizedResultEntry:
    def __init__(self, ds_exp, ds_th, minimizerResult : lmfit.minimizer.MinimizerResult):

        self.ds_exp = ds_exp
        self.ds_th = ds_th

        keys = minimizerResult.params.keys()
        self.params = {k : minimizerResult.params[k].value for k in keys}
        self.params_err = {k: minimizerResult.params[k].stderr for k in keys}
        self.chisq = minimizerResult.chisqr
        self.chisq_red = minimizerResult.redchi



class optimizedResult:

    def __init__(self, q, t, t_str, param_labels):
        self.q = q
        self.t = t
        self.t_str = t_str
        self._d = {}
        self.param_labels = param_labels
        self._all_labels = (param_labels +
                            [p+'_err' for p in param_labels]
                            + ['chisq', 'chisq_red'])



    def __setitem__(self, key, entry : optimizedResultEntry):
        key_str = time_num2str(key)


        assert key_str in self.t_str, 'key must be one of the predefined time-delays'
        self._d[key_str] = entry

        try:
            key_num = time_str2num(key)
            self._d[key_num] = entry
        except ValueError:
            pass




    def __getitem__(self, key):

        # if float then treat it as numerical time
        if (type(key) == float) or (type(key) == int):
            which_t = np.argmin(np.abs(self.t - key))
            if not np.isclose(self.t[which_t], key, atol=np.finfo(float).eps):
                print('t not found within range of the data, closest t is', self.t_str[which_t])
            key = self.t_str[np.isclose(self.t, key, atol=np.finfo(float).eps)]

        # if string, check if this is the time
        if key in self._d.keys():
            return self._d[key]

        # otherwise check if this is parameter
        elif key in self.param_labels:
            vals = []
            for each_t in self.t_str:
                vals.append(self._d[each_t].params[key])
            return np.array(vals)

        elif key.endswith('_err'):
            key = key[:-4]
            vals = []
            for each_t in self.t_str:
                vals.append(self._d[each_t].params_err[key])
            return np.array(vals)

        elif key == 'chisq':
            vals = []
            for each_t in self.t_str:
                vals.append(self._d[each_t].chisq)
            return np.array(vals)

        elif key == 'chisq_red':
            vals = []
            for each_t in self.t_str:
                vals.append(self._d[each_t].chisq_red)
            return np.array(vals)


    def __getattr__(self, item):
        return self.__getitem__(item)

    def __dir__(self):
        return ['q', 't', 't_str'] + [p for p in self.param_labels]













