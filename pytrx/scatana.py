
import numpy as np
import matplotlib.pyplot as plt
from pytrx.scatdata import ScatData
from pytrx import scatsim, hydro
from pytrx.utils import weighted_mean, bin_vector_with_covmat, time_str2num, time_num2str
from pytrx.regressors import MainRegressor
import lmfit
import time
import copy


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
        print(type(input_data))
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
        if (self.metadata.solvent is not None) and (self.metadata.concentration is not None):
            solvent = self.metadata.solvent
            concentration = self.metadata.concentration
            data = hydro.solvent_data[solvent]
            return data.density / data.molar_mass / concentration
        else:
            return 1


    def check_qrange_with_model(self, q_idx):
        q_in = self.data.q[q_idx]
        # q_out = q_in.copy()
        qmin = np.max((q_in.min(), self.model.solvent.q.min(), self.model.cage.q.min()))
        qmax = np.min((q_in.max(), self.model.solvent.q.max(), self.model.cage.q.max()))

        long_print_flag = False
        if qmin > q_in.min():
            print('invalid qmin - ', end='')
            long_print_flag = True

        if qmax < q_in.max():
            print('invalid qmax - ', end='')
            long_print_flag = True

        if long_print_flag:
            print('will perform fit q-range based on the available q from the model')

        return (self.data.q>=qmin) & (self.data.q<=qmax)



    def fit(self, qmin=None, qmax=None, t=None, trange=None, tavrg=False, method='gls',
            prefit=True, direction='forward', use_prev_param0=True):

        print('Starting the fitting procedure')
        # check q-range
        print(' ... checking the validity of the q-range')
        if qmin is None: qmin = self.data.q.min()
        if qmax is None: qmax = self.data.q.max()
        q_idx = (self.data.q >= qmin) & (self.data.q <= qmax)
        q_idx = check_range_with_cov_matrix(self.data.q, q_idx, self.data.diff.covqq)
        q_idx = self.check_qrange_with_model(q_idx)

        # check t-range
        print(' ... checking the validity of the t-range')
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

        print('... definining the fitting targets ...')
        # get the targets
        qfit = self.data.q[q_idx]
        tfit = self.data.t[t_idx]
        tfit_str = self.data.t_str[t_idx]

        ds_target = self.data.diff.s_av[np.ix_(q_idx, t_idx)]
        C_target = self.data.diff.covqq[np.ix_(q_idx, q_idx)]
        K_target = self.data.diff.covtt[np.ix_(t_idx, t_idx)]

        if tavrg:
            print('... averaging data along t-axis ...')
            ds_target_T, K_target = weighted_mean(ds_target.T, K_target)
            ds_target = ds_target_T.T
            tfit = np.mean(tfit)
            tfit_str = np.array([time_num2str((tfit))])

        # prepare the model
        print('... preparing the model vectors ...')
        self.model.prepare_model(qfit, self.solvent_per_solute())

        # generate description for the fitting round
        description = {'qmin' : qfit[0],
                       'qmax' : qfit[-1],
                       'tmin' : tfit_str[0],
                       'tmax' : tfit_str[-1],
                       'averaged' : tavrg}

        # fit!
        print('... fitting ...')
        self.result = _fit(qfit, tfit, tfit_str, ds_target, C_target, K_target,
                           self.model.problem_input, self.model.nonlinear_labels, self.model.params0,
                           method=method, prefit=prefit,
                           direction=direction, use_prev_param0=use_prev_param0,
                           description=description)
        print('Done!')

    def output_molecular_movie(self, fname):
        n_pars = len(self.model.params0)-4
        pars = []
        for i in range(n_pars):
            pars.append(self.result[f'par_es_{i+1}'])
        pars = np.array(pars).T
        with open(fname, 'w') as f:
            for idx, t in enumerate(self.result.t_str):
                Z = self.model.solute.mol_es.Z
                XYZ = self.model.solute.mol_es.transform(pars[idx],return_xyz=True)
                f.write(f'{len(Z)}')
                f.write(f'\nOutput of xyz for molecule, time {t}\n')
                for i in range(len(self.model.solute.mol_es.Z)):
                    f.write(f'{Z[i]} {XYZ[i][0]} {XYZ[i][1]} {XYZ[i][2]}\n')
            f.write('\n')


###TODO:
### this output class or SmallMoleculeProject should have plotting functions to produce pretty figures for fitting 2D maps, 1D curves for specfici time delay with decomposition of the model on solute cage and solvent
### one must be able to supply list of models to the SmallMoleculeProject  think about it
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

        # check if the labels are not intersecting
        es_keys = list(self.mol_es.par0.keys())
        gs_keys = list(self.mol_gs.par0.keys())

        n_labels_es = len(es_keys)
        n_labels_gs = len(gs_keys)
        n_labels_all = len(set(es_keys + gs_keys))
        assert  n_labels_es + n_labels_gs == n_labels_all, \
            'parameter labels in mol_es and mol_gs must be different'

        self.par0 = { **self.mol_gs.par0, **self.mol_es.par0}
        # self.par_labels, self.par_vals0 = self.list_pars(return_labels=True)
        # self.mol_es_ref = self.parse_input(input_es)

    def parse_input(self, input):
        if type(input) == str:
            # Takes the xyz file
            print("Using scatsim.fromXYZ will result in no associated_transformation in the returned Molecule.")
            print("You may want to consider creating the Molecule and pass it to the Solute.")
            return scatsim.fromXYZ(input)
        else:
            # Take the Molecule class
            return copy.deepcopy(input)



    def ds(self, q, pars=None, printing=False):
        '''
        Originally just 'signal' but changed to 'ds' as this is calculating difference signal
        '''
        # self.mol_es.move(*x) - consider this
        if pars is not None:
            assert (all([p in pars.keys() for p in self.mol_es.par0.keys()])), \
                'key(s) for mol_es are missing in pars'
            assert (all([p in pars.keys() for p in self.mol_gs.par0.keys()])), \
                'key(s) for mol_gs are missing in pars'
            #
            # len(pars) == (self.mol_es.n_par + self.mol_gs.n_par), \
            #     'nummber of parameteres should match the sum of numbers of parameters for gs and es'
            # pars_es, pars_gs = deal_pars(pars, self.mol_es.n_par)
        # else:
        #     pars_es, pars_gs = None, None
        if printing: print(f'ES parameters: {self.mol_es.par_keys}, GS parameters: {self.mol_gs.par_keys}')

        if (self.mol_es is not None) and (self.mol_gs is not None):
            return self.mol_es.s(q, pars) - self.mol_gs.s(q, pars)
        elif self.mol_gs is not None:
            return - self.mol_gs.s(q, pars)
        elif self.mol_es is not None:
            return self.mol_es.s(q, pars)
        else:
            return np.zeros(q.shape)

    def list_pars(self):

        print(f'Listing structural parameters: \n'
              f'There are {self.n_par_total} structural parameters to be passed '
              f'to the pars argument as a list for ds method\n')
        if self.mol_es is not None:
            # for i in np.arange(self.mol_es.n_par):
            for i, key in enumerate(self.mol_es.par_keys):
                print(f'Parameter {key}: ES, {type(self.mol_es._associated_transformation[i])}')
                self.mol_es._associated_transformation[i].describe()
                print("")
        if self.mol_gs is not None:
            for i in np.arange(self.mol_gs.n_par):
                print(f'Parameter { i + 1 +self.mol_es.n_par}: ES, {type(self.mol_gs._associated_transformation[i])}')
                self.mol_gs._associated_transformation[i].describe()
                print("")







class Solvent:

    def __init__(self, input, sigma=None, K=None):
        self.q, self.dsdt_orig, self.dsdr_orig = self.parse_input(input)
        self.C_orig = read_sigma(sigma)
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
        self.C_orig = read_sigma(sigma)

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


    def prepare_model(self, q, sps):
        self.prepare_vectors(q)
        def ds_solute(p):
            return self.solute.ds(q, p)/sps

        self.problem_input = [ ('esf', 'ds_solute', ds_solute, None, None),
                               ('cage_amp', 'ds_cage', self.cage.ds/sps, self.cage.C, None),
                               ('dsdt_amp', 'dsdt', self.solvent.dsdt, self.solvent.C, None),
                               ('dsdr_amp', 'dsdr', self.solvent.dsdr, None, None)]
        self.nonlinear_labels = self.solute.par0.keys()



    def prepare_vectors(self, qfit):
        # self.cage.ds = np.interp(qfit, self.cage.q, self.cage.ds_orig)
        # self.solvent.dsdt = np.interp(qfit, self.solvent.q, self.solvent.dsdt_orig)
        # self.solvent.dsdr = np.interp(qfit, self.solvent.q, self.solvent.dsdr_orig)
        self.cage.ds, self.cage.C = bin_vector_with_covmat(qfit, self.cage.q,
                                                           self.cage.ds_orig, self.cage.C_orig)
        self.solvent.dsdt, self.solvent.C = bin_vector_with_covmat(qfit, self.solvent.q,
                                                                   self.solvent.dsdt_orig, self.solvent.C_orig)
        self.solvent.dsdr, _              = bin_vector_with_covmat(qfit, self.solvent.q,
                                                                   self.solvent.dsdr_orig, None)


    def prepare_parameters(self,
                           esf_dict={ 'value' : 0, 'vary' : True},
                           cage_dict={'value' : 1, 'vary' : False},
                           dsdt_dict={'value' : 0, 'vary' : True},
                           dsdr_dict={'value' : 0, 'vary' : True},
                           **kwargs):

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
                self.solvent.dsdt_orig = np.zeros(self.solvent.q.size)
                params0.add('dsdt_amp', value=0, vary=False)
            else:
                params0.add('dsdt_amp', **dsdt_dict)

            if self.solvent.dsdr_orig is None:
                self.solvent.dsdr_orig = np.zeros(self.solvent.q.size)
                params0.add('dsdr_amp', value=0, vary=False)
            else:
                params0.add('dsdr_amp', **dsdr_dict)

        for key in self.solute.par0.keys():
            # print((lab+'_dict'), kwargs.keys(), (lab+'_dict') in kwargs.keys())
            if (key+'_dict') in list(kwargs.keys()):
                params0.add(key, **kwargs[key + '_dict'])
            else:
                params0.add(key, value=self.solute.par0[key])

        self.params0 = params0






def _fit(q, t, t_str, Yt, C, K, problem_input, nonlinear_labels, params0, method='gls', prefit=True,
        direction='forward', use_prev_param0=False, exp_labels = ['ds', 'ds_cov', 'ds_err', 'ds_fit', 'ds_fit_cov', 'ds_fit_err'], description=None):
    assert direction in ['forward','backward'], 'direction of fitting must be forward or backward'

    if Yt.ndim == 1: Yt = Yt[:, None]
    if K.ndim == 1: K = K[:, None]

    yt_label, Cyt_label, yt_err_label, y_label, Cy_label, y_err_label = exp_labels

    n_curves = Yt.shape[1]
    n_comp = len(problem_input)

    Y = np.zeros(Yt.shape)
    components_fit = np.zeros(Yt.shape + (n_comp,))

    if direction=='forward': i_generator = range(n_curves)
    elif direction == 'backward': i_generator = range(n_curves-1, -1, -1)

    param_labels = list(params0.keys())
    component_labels = [i[1] for i in problem_input]
    vector_labels = component_labels + exp_labels + ['resid', 'resid_w']
    result = optimizedResult(q, t, t_str, param_labels, vector_labels, description)

    curve_counter = 1
    for i in i_generator:
        starting_time = time.perf_counter()
        print('Fitting time delay', t_str[i],'\tProgress:', curve_counter, '/', n_curves, end=' \t ')
        regressor = MainRegressor(Yt[:, i], C * K[i, i], problem_input,
                                  params0, nonlinear_labels=nonlinear_labels)
        regressor.fit(method=method, prefit=prefit)
        vector_dict = {yt_label : regressor.yt,
                       Cyt_label : regressor.Cyt,
                       yt_err_label : np.sqrt(np.diag(regressor.Cyt)),
                       y_label : regressor.y,
                       Cy_label : regressor.Cy,
                       y_err_label : regressor.y_err,
                       'resid': (regressor.y - regressor.yt)[:q.size],
                       'resid_w' : regressor.result.residual[:q.size]}
        for v_label, p_label in zip(component_labels, param_labels):
            vector_dict[v_label] = regressor.vectors[p_label]['v_fit']

        result[t_str[i]] = optimizedResultEntry(regressor.result, vector_dict)


        if use_prev_param0:
            params0 = regressor.result.params
            prefit = False

        print('took %0.0f' %((time.perf_counter() - starting_time)*1e3), 'ms')
        curve_counter += 1

    return result




class optimizedResultEntry:
    def __init__(self, minimizerResult : lmfit.minimizer.MinimizerResult,
                 vector_dict):

        keys = minimizerResult.params.keys()
        self.params = {k : minimizerResult.params[k].value for k in keys}
        self.params_err = {k: minimizerResult.params[k].stderr for k in keys}
        self.chisq = minimizerResult.chisqr
        self.chisq_red = minimizerResult.redchi
        self.vector_dict = vector_dict



class optimizedResult:

    def __init__(self, q, t, t_str, param_labels, vector_labels, description):
        self.q = q
        self.t = t
        self.t_str = t_str
        self._d = {}
        self.param_labels = param_labels
        self.vector_labels = vector_labels
        self._all_labels = (param_labels +
                            [p+'_err' for p in param_labels] +
                            vector_labels +
                            ['chisq', 'chisq_red'] + ['q', 't', 't_str'])
        self.description_dict = description


    def __repr__(self):
        _d = self.description_dict
        output = ('This is the analysis result for data recorded in\n' +
                  ('%0.2f' %_d["qmin"]) + ' < q < ' + ('%0.2f' %_d["qmax"]) + '\n' +
                  f'{_d["tmin"]} < t < {_d["tmax"]}\n')
        if _d['averaged']:
            output += 'The data has been averaged along time axis'
        return output


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

        elif ('_'.join(key.split('_')[:-1]) in self.param_labels) and (key.endswith('_err')):
            key = key[:-4]
            vals = []
            for each_t in self.t_str:
                vals.append(self._d[each_t].params_err[key])
            return np.array(vals)

        elif key in self.vector_labels:
            vals = []
            for each_t in self.t_str:
                vals.append(self._d[each_t].vector_dict[key])
            return np.array(vals).T

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

        else:
            raise AttributeError(f"'optimizedResult' object has no attriubre '{key}'")


    def __getattr__(self, item):
        return self.__getitem__(item)

    def __dir__(self):
        return self._all_labels









