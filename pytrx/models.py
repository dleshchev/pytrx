import numpy as np
import matplotlib.pyplot as plt
import lmfit
from scipy.interpolate import interp1d


def exp(t, amp, tau):
    s = amp * np.exp(-t/tau)
    s[t < 0] = 0
    return s

def offset(t, amp):
    s = amp * np.ones(t.shape)
    s[t < 0] = 0
    return s

def sine(t, amp, f, phi):
    s = amp * np.sin(2 * np.pi * f * t + phi)
    s[t < 0] = 0
    return s

# def kinetic_model(n_exp, n_offset, n_sine, ):

def assym_gauss(x, amp, x_zero, sigma_L, sigma_R):
    output = np.zeros(x.shape)
    x = x - x_zero
    output[x <= 0] = np.exp(-x[x <= 0] ** 2 / (2 * sigma_L ** 2))
    output[x >= 0] = np.exp(-x[x >= 0] ** 2 / (2 * sigma_R ** 2))
    return output * amp


def _pulse_shape(x, amp1, amp2, amp3, amp4,
               tz1, tz2, tz3, tz4,
               sl1, sl2, sl3, sl4,
               sr1, sr2, sr3, sr4):
    return (assym_gauss(x, amp1, tz1, sl1, sr1) +
            assym_gauss(x, amp2, tz2, sl2, sr2) +
            assym_gauss(x, amp3, tz3, sl3, sr3) +
            assym_gauss(x, amp4, tz4, sl4, sr4))


def pulse_profile(fwhm, shape='id09'):
    # def xrayPulseProfile(fwhm, shape='gauss'):
    if shape == 'gauss':
        sigma = fwhm / 2.355
        t = np.linspace(-1, 1, 151) * 5
        h = np.exp(- (t / (np.sqrt(2))) ** 2)
        t *= sigma
    elif shape == 'id09':


        params = [0.25658488, 0.18297786, 0.76292762, 0.69525572,  # amplitudes
                  -46.08440552, 54.83370414, -23.70876518, 18.33081068,  # time zeros
                  9.5657804, 19.73970962, 16.51307652, 24.68971104,  # left sigmas
                  10.36663466, 17.0784876, 22.77630371, 25.49143015]  # right sigmas
        t = np.linspace(-1, 1.2, 151) * 99.2640 / 2.355 * 4
        h = _pulse_shape(t, *params)
        t = (t + 10.2956) / 99.2640 * fwhm
    else:
        raise ValueError('shape can be only "gauss" or "id09"')
    h /= np.trapz(h, t)
    return t, h


def numericalConvolution(t, s, delays, tzero, fwhm, shape='id09'):
    if len(s.shape) == 1:
        s = s[:, np.newaxis]
    ncurves = s.shape[1]
    s_conv = np.zeros((delays.size, ncurves))
    tmin = t.min()
    t_h, h = pulse_profile(fwhm, shape=shape)
    for j in range(ncurves):
        interpolator = interp1d(t, s[:, j], kind='cubic')
        for i, delay in enumerate(delays):
            t_h_local = t_h + delay - tzero
            # handling crossing of time-zero (this drastically improves the accuracy of convolution):
            if (tmin > t_h_local.min()) and (tmin < t_h_local.max()):
                h_tmin = np.interp(tmin, t_h_local, h)
                t_h_local = np.hstack((t_h_local, tmin))
                h_local = np.hstack((h, h_tmin))
                h_local = h_local[np.argsort(t_h_local)]
                t_h_local = np.sort(t_h_local)
                h_local = h_local / np.trapz(h_local, t_h_local)
            else:
                h_local = h
            h_local = h_local[t_h_local >= tmin]
            t_h_local = t_h_local[t_h_local >= tmin]

            if t_h_local.size > 0:
                s_conv[i, j] = np.trapz(interpolator(t_h_local) * h_local, t_h_local)
    return s_conv




class KineticModel:
    def __init__(self, model_str, irf_str, sampling=10):
        # tsel = np.ones(t.shape, dtype=bool)
        # if tmin: tsel = tsel & (t>=tmin)
        # if tmax: tsel = tsel & (t<=tmax)
        # self.t = t[tsel]
        self.sampling = sampling
        self.irf_str = irf_str
        self.parse_model(model_str)
        # self.x = x[tsel]
        # self.Cx = Cx[np.ix_(tsel, tsel)]

    def parse_irf(self, irf_str):
        def irf:
            return numericalConvolution(t, s, delays, tzero, fwhm, shape=irf_str)

    def parse_model(self, model_str):
        # "exp + exp + offset + exp*sine + exp*sine"
        terms = model_str.split(' + ')
        problem_input = []
        idx_exp = 1
        idx_sine = 1
        idx_offset = 1
        for term in terms:
            if 'offset' in term:
                par_key = 'offset_amp_' + str(num_hint)
                vec_key = 'offset_' + str(num_hint)
                v = self._define_offset(num_hint)
            elif 'exp' in term:
                par_key = 'exp_amp_' + str(num_hint)
                vec_key = 'exp_' + str(num_hint)
                v = self._define_exp(num_hint)
            elif 'sine' in term:
                par_key = 'sine_amp_' + str(num_hint)
                vec_key = 'sine_' + str(num_hint)
                v = self._define_sine(num_hint)
            elif ('sine' in term) and ('exp' in term) and ('*' in term):
                par_key = 'exp_amp_' + str(num_hint)
                vec_key = 'exp_' + str(num_hint)
                v = self._define_exp_sine(num_hint)

            problem_input_appendreturn [par_key, vec_key, v, None, None]

                par_key = 'exp_amp_' + str(idx_exp)
                vec_key = 'exp_' + str(idx_exp)
                v =




    def _get_t_inst(self, t, sigma):
        return np.linspace(t.min() - 5 * sigma,
                           t.max() + 5 * sigma,
                           ((t.max() - t.min()) / sigma + 1) * self.sampling + 1)

    def _define_offset(self, num_hint):
        def _offset(t, param: dict):
            sigma = param['sigma_conv']
            tzero = param['time_zero']
            amp = param['offset_amp_' + str(num_hint)]
            t_inst = self._get_t_inst(t, sigma)
            s_inst = offset(t, amp)
            s_conv = numericalConvolution(t_inst, s_inst, t, tzero, sigma * 2.355, shape=self.irf_str)
            return s_conv

    def _define_exp(self, num_hint, fix_amp=False ):
        def _exp(t, param: dict):
            sigma = param['sigma_conv']
            tzero = param['time_zero']
            if fix_amp: amp = 1
            else: amp = param['exp_amp_' + str(num_hint)]
            tau = param['exp_tau_' + str(num_hint)]

            t_inst = self._get_t_inst(t, sigma)
            s_inst = exp(t_inst, amp, tau)
            s_conv = numericalConvolution(t_inst, s_inst, t, tzero, sigma * 2.355, shape=self.irf_str)
            return s_conv
        return _exp

    def _define_sine(self, num_hint, fix_amp=False ):
        def _sine(t, param: dict):
            sigma = param['sigma_conv']
            tzero = param['time_zero']
            if fix_amp: amp = 1
            else: amp = param['sine_amp_' + str(num_hint)]
            f = param['sine_freq_' + str(num_hint)]
            phi = param['sine_phase_' + str(num_hint)]
            t_inst = self._get_t_inst(t, sigma)
            s_inst = sine(t_inst, amp, f, phi)
            s_conv = numericalConvolution(t_inst, s_inst, t, tzero, sigma * 2.355, shape=self.irf_str)
            return s_conv
        return _sine


    def _define_exp_sine(self, exp_num_hint, sine_num_hint ):
        _exp = self._define_exp(exp_num_hint, self.irf_str)
        _sine = self._define_sine(sine_num_hint, self.irf_str, fix_amp=True)
        def _exp_sine(t, param : dict):
            return _exp(t, param) * _sine(t, param)
        return _exp_sine










