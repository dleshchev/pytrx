import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize

import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize, linalg
import lmfit
from pytrx.utils import  cov2corr
from pytrx.utils import weighted_mean, time_str2num, time_num2str
import copy


class MainRegressor:
    def __init__(self, yt, Cyt, problem_input, params0, nonlinear_labels=None):
    # def __init__(self, yt, Cyt, problem_input, params0):
        # data input
        self.yt = yt
        self.Cyt = Cyt
        self.n = self.yt.shape[0]


        # label handling
        self.nl_labels = nonlinear_labels
        self.lin_labels = []
        self.params0 = copy.deepcopy(params0)


        # problem unpacking
        self.vectors = {}
        for item in problem_input:
            par_key, vec_key, v, Cv, T = item
            self.lin_labels.append(par_key)
            self.vectors[par_key] = {'v' : v, 'Cv' : Cv, 'exact' : (Cv is None), 'T' : T, 'v_key' : vec_key}

        # placeholders for other fields
        self.method = None
        self.Ly = None

        self.Vt = None
        self.CV = None
        self.CV_inv = None
        self.chol_CV_inv = None
        self.CVT = None
        self.CVT_inv = None
        self.chol_CVT_inv = None

        # auxillary flag for tls fitting
        self._V_vary = True


    def vec(self, V):
        return V.T.ravel()

    def bigB(self, B):
        I = np.eye(self.n)
        return np.block([[np.kron(I, b.T)] for b in B.T])

    # def bigB(self, b):
    #     I = np.eye(self.n)
    #
    #     # return np.block([[np.kron(I, b.T)] for b in B.T])
    #     return np.kron(I, b)


    def prepare_nonexact_matrix(self):
        Vt = []
        for key in self.lin_labels:
            if not self.vectors[key]['exact']:
                Vt.append(self.vectors[key]['v'])
        self.m = len(Vt)
        self.Vt = np.array(Vt).T



    def prepare_nonexact_covariances(self):
        cv_list = []
        for k in self.lin_labels:
            if not self.vectors[k]['exact']:
                cv_list.append(self.vectors[k]['Cv'])
        if cv_list is not None:
            cv_inv_list = [np.linalg.pinv(cv) for cv in cv_list]
            chol_cv_inv_list = [np.linalg.cholesky(cv_inv) for cv_inv in cv_inv_list]

            self.CV = linalg.block_diag(*cv_list)
            self.CV_inv = linalg.block_diag(*cv_inv_list)
            self.chol_CV_inv = linalg.block_diag(*chol_cv_inv_list)

            self.CVT = np.zeros((self.m * self.n, self.m * self.n))
            self.CVT_inv = np.zeros((self.m * self.n, self.m * self.n))
            self.chol_CVT_inv = np.zeros((self.m * self.n, self.m * self.n))

            for i in range(self.m):
                self.CVT[i::self.m, i::self.m] = cv_list[i]
                self.CVT_inv[i::self.m, i::self.m] = cv_inv_list[i]
                self.chol_CVT_inv[i::self.m, i::self.m] = chol_cv_inv_list[i]




    def prepare_y_covariance(self):
        if self.method == 'wls':
            self.Cyt_inv = np.diag(1/np.diag(self.Cyt))
            self.Ly = np.sqrt(self.Cyt_inv)
        else:
            self.Cyt_inv = np.linalg.inv(self.Cyt)
            self.Ly = np.linalg.cholesky(self.Cyt_inv)


    def f_exact(self, params):
        p = params.valuesdict()
        # if len(self.nl_labels) != 0:
        #     p_nl = [p[t] for t in self.nl_labels]
        # else:
        #     p_nl = None

        y = np.zeros(self.yt.shape)
        for key in self.lin_labels:
            if self.vectors[key]['exact']:
                v = self.vectors[key]['v']
                if callable(v):
                    # y += p[key] * v(p_nl)
                    y += p[key] * v(p)
                else:
                    y += p[key] * v
        return y



    def f_nonexact(self, params):
        p = params.valuesdict()
        y = np.zeros(self.yt.shape)

        if self.method != 'tls':
            for key in self.lin_labels:
                if not self.vectors[key]['exact']:
                    y += p[key] * self.vectors[key]['v']
                    if 'v_estimated' not in self.vectors[key].keys():
                        self.vectors[key]['v_estimated'] = self.vectors[key]['v']

        else:
            if self._V_vary:
                if self.CV is None: self.prepare_nonexact_covariances()
                if self.Vt is None: self.prepare_nonexact_matrix()
                ## prepare the parameters for estimation

                b = self.get_nonexact_amplitudes(params)
                # form the matrices for estimation
                I = np.eye(self.m * self.n)
                Z = np.block([[self.bigB(b)], [I]])

                Null = np.zeros((self.n * 1, self.n * self.m)) # 1 stands for the number of fitted curves
                Oinv = np.block([[self.Cyt_inv, Null],
                                 [Null, self.CVT_inv]])

                E = np.block([self.yt - self.f_exact(params),
                              self.vec(self.Vt.T)])
                # estimate!
                vecVT = np.linalg.inv(Z.T @ Oinv @ Z) @ Z.T @ Oinv @ E


                self.V = np.reshape(vecVT, (self.n, self.m))

                y = self.V @ b
                idx = 0
                for key in self.lin_labels:
                    if not self.vectors[key]['exact']:
                        self.vectors[key]['v_estimated'] = self.V[:, idx]
                        idx += 1
            else:
                for key in self.lin_labels:
                    if not self.vectors[key]['exact']:
                        y += p[key] * self.vectors[key]['v_estimated']

        return y.ravel()


    def get_nonexact_amplitudes(self, params):
        p = params.valuesdict()
        b = []
        for key in self.lin_labels:
            if not self.vectors[key]['exact']:
                b.append(p[key])
        return np.array(b)[:, None]


    def compute_components(self, params):
        p = params.valuesdict()
        # if len(self.nl_labels) != 0:
        #     p_nl = [p[t] for t in self.nl_labels]
        # else:
        #     p_nl = None

        for key in self.lin_labels:
            if self.vectors[key]['exact']:
                v = self.vectors[key]['v']
                if callable(v):
                    # self.vectors[key]['v_fit'] = p[key] * v(p_nl)
                    self.vectors[key]['v_fit'] = p[key] * v(p)
                else:
                    self.vectors[key]['v_fit'] = p[key] * v
            else:
                v = self.vectors[key]['v_estimated']
                self.vectors[key]['v_fit'] = p[key] * v


    def fit_func(self, params):
        return self.f_exact(params) + self.f_nonexact(params)



    def residual(self, params):
        # print([self.vectors[key]['exact'] for key in self.lin_labels])
        if self.Ly is None: self.prepare_y_covariance()
        dy_w = self.Ly.T @ (self.fit_func(params) - self.yt) # for TLS this updates self.V
        if self.method != 'tls':
            return dy_w
        else:
            dV_w = self.chol_CVT_inv.T @ self.vec((self.V -self.Vt).T)
            return np.hstack((dy_w, dV_w))



    def prefit(self):

        if self.nl_labels:
            method_hold = copy.copy(self.method) # prefit is executed using gls regression
            if method_hold == 'tls':
                self.method = 'gls'

            vary_status = {key: self.params0[key].vary for key in self.params0.keys()}
            for key in self.nl_labels:
                self.params0[key].vary = False
            result_pre = lmfit.minimize(self.residual, self.params0,
                                        scale_covar=False, method='least_squares')
            self.params0 = result_pre.params
            for key in self.params0.keys():
                self.params0[key].vary = vary_status[key]
            self.method = method_hold
        else:
            print('Prefit impossible because of one of the following reasons')
            print('(1) the model does not contain non-linear part')
            print('(2) the non-linear parameter labels are not specified')



    def fit(self, prefit=True, method='gls'):
        self.method = method
        if self.method == 'tls':
            tls_possible = self.check_if_tls_is_possible()
            if not tls_possible:
                print('No model vectors with uncertainty are found, the fitting will be performed using GLS')
                self.method = 'gls'

        if self.method == 'gls':
            gls_possible = self.check_if_gls_is_possible()
            if not gls_possible:
                print('input data covairance matrix is diagonal, the fitting will be performed using WLS')
                self.method = 'wls'

        if prefit: self.prefit()

        self.result = lmfit.minimize(self.residual, self.params0,
                             scale_covar=False, method='least_squares')

        self.y = self.fit_func(self.result.params)
        self.compute_components(self.result.params)

        if self.method != 'tls':
            Jw = self.result.jac # weighted jacobian of residuals with respect to fitting parameters
            L_all_inv = np.linalg.inv(self.Ly.T)
        else:
            self._V_vary = False
            res = lmfit.minimize(self.residual, self.result.params,
                             scale_covar=False, method='least_squares')
            Jw = res.jac
            L_all_inv = linalg.block_diag(*(np.linalg.inv(self.Ly.T), np.linalg.inv(self.chol_CVT_inv.T)))
            b = self.get_nonexact_amplitudes(self.result.params)
            jac_ext_1 = np.hstack(tuple(each_b * self.Ly.T for each_b in b))
            jac_ext = np.vstack((jac_ext_1, self.chol_CVT_inv.T))
            Jw = np.hstack((Jw, jac_ext))
            # Jw[self.n:, :len(self.params0)] = 0


        J = L_all_inv @ Jw
        Hess = Jw.T @ Jw
        self.Cp = np.linalg.inv(Hess)
        p_err = np.sqrt(np.diag(self.Cp))
        # print(self._V_vary)
        # print(J.shape, Hess.shape, p_err.shape, self.params0.keys(), self.result.params.keys())

        idx = 0
        for key in self.result.params.keys():
            if self.result.params[key].vary:
                self.result.params[key].stderr = p_err[idx]
                idx += 1

        self.Cy = (J @ self.Cp @ J.T)[:self.n, :self.n]
        self.y_err = np.sqrt(np.diag(self.Cy))





    def check_if_tls_is_possible(self):
        self.prepare_nonexact_matrix() # computes self.m
        if self.m > 0: return True
        else: return False

    def check_if_gls_is_possible(self):

        C = self.Cyt/np.abs(self.Cyt).max()

        return (not np.all(np.isclose(C, np.diag(np.diag(C)))))




#
#
#
# class NLPTLS:
#     def __init__(self, Y_t, C, K, s_t_list, Q_list, f):
#         #        self.x = x
#         if Y_t.ndim == 1:
#             Y_t = Y_t[:, None]
#         self.Y_t = Y_t
#         self.n, self.k = Y_t.shape
#         self.C = C
#         self.Cdiag = np.diag(np.diag(C))
#         self.Cinv = np.linalg.pinv(C)
#         self.Cdiaginv = np.diag(1 / np.diag(C))
#         self.L = np.linalg.cholesky(self.Cinv)
#
#         if K.ndim == 1:
#             K = np.array([[K]])
#         self.K = K
#         self.Kinv = np.linalg.pinv(K)
#         self.D = np.linalg.cholesky(self.Kinv)
#
#         self.G = np.kron(self.K, self.C)
#         self.Gdiag = np.diag(np.diag(self.G))
#         self.Ginv = np.kron(self.Kinv, self.Cinv)
#         self.Gdiaginv = np.diag(1 / np.diag(self.G))
#         self.H = np.kron(self.D, self.L)
#
#         s_t_list = [np.squeeze(s_t) for s_t in s_t_list]
#         self.S_t = np.array(s_t_list).T
#         self.m = len(s_t_list)
#         self.Q_list = Q_list
#         self.Qinv_list = [np.linalg.pinv(Q) for Q in Q_list]
#         self.V_list = [np.linalg.cholesky(Qinv) for Qinv in self.Qinv_list]
#
#         self.CS = np.zeros((self.m * self.n, self.m * self.n))
#         self.CSinv = np.zeros((self.m * self.n, self.m * self.n))
#         self.W = np.zeros((self.m * self.n, self.m * self.n))
#         for i in range(self.m):
#             self.CS[i::self.m, i::self.m] = self.Q_list[i]
#             self.CSinv[i::self.m, i::self.m] = self.Qinv_list[i]
#             self.W[i::self.m, i::self.m] = self.V_list[i]
#
#         self.f = f
#
#     def split_p(self, P):
#         return P[:self.m, :], P[self.m:, :]
#
#     def F(self, X):
#         F = np.zeros((self.n, self.k))
#         for i in range(self.k):
#             F[:, i] = self.f(X[:, i])
#         return F
#
#     def Y(self, P, S):
#         B, X = self.split_p(P)
#         return S @ B + self.F(X)
#
#     def vec(self, V):
#         return V.T.ravel()
#
#     def unflatten_pars(self, pars):
#         n_par = int(pars.size / self.k)
#         return np.reshape(pars, (n_par, self.k))
#
#     def bigB(self, B):
#         I = np.eye(self.n)
#         return np.block([[np.kron(I, b.T)] for b in B.T])
#
#     def get_S(self, P):
#         B, X = self.split_p(P)
#         I = np.eye(self.m * self.n)
#         Z = np.block([[self.bigB(B)], [I]])
#         Null = np.zeros((self.n * self.k, self.n * self.m))
#         Oinv = np.block([[self.Ginv, Null],
#                          [Null, self.CSinv]])
#         E = np.block([self.vec(self.Y_t - self.F(X)),
#                       self.vec(self.S_t.T)])
#         vecST = np.linalg.pinv(Z.T @ Oinv @ Z) @ Z.T @ Oinv @ E
#         return np.reshape(vecST, (self.n, self.m))
#
#     def resid_Y(self, P, S):
#         return self.H.T @ self.vec(self.Y_t - self.Y(P, S))
#
#     def resid_S(self, S):
#         return self.W.T @ self.vec(S.T - self.S_t.T)
#
#     def chisq(self, P, S):
#         rY = self.resid_Y(P, S)
#         rS = self.resid_S(S)
#         return rY @ rY + rS @ rS
#
#     def resid_tls(self, pars):
#         idx = self.m * self.n
#         P = self.unflatten_pars(pars[:-idx])
#         S = np.reshape(pars[-idx:], (self.n, self.m))
#         return np.hstack((self.resid_Y(P, S), self.resid_S(S)))
#
#     def resid_tls_P(self, pars):
#         P = self.unflatten_pars(pars)
#         S = self.get_S(P)
#         return np.hstack((self.resid_Y(P, S), self.resid_S(S)))
#
#     def resid_gls(self, pars):
#         P = self.unflatten_pars(pars)
#         return self.resid_Y(P, self.S_t)
#
#     def resid_ols(self, pars):
#         P = self.unflatten_pars(pars)
#         return self.vec(self.Y_t - self.Y(P, self.S_t)) / np.diag(self.G)
#
#     def ols_lin(self, X0):
#         np.diag(self.C)
#         a = np.linalg.pinv(self.S_t.T @ self.Cdiaginv @ self.S_t)
#         b = self.S_t.T @ self.Cdiaginv @ (self.Y_t - self.F(X0))
#         return a @ b
#
#     def ols(self, X0, printing=False):
#         P0 = Parameter(np.vstack((self.ols_lin(X0), X0)))
#         opt_ols = optimize.least_squares(self.resid_ols, P0.flat, method='lm')
#         if printing: print('OLS success:', opt_ols['success'])
#
#         J_w = opt_ols['jac']
#         J = np.sqrt(self.Gdiag) @ J_w
#         Hess = J_w.T @ J_w
#         C_all = np.linalg.pinv(Hess)
#         self.P_ols = Parameter(opt_ols['x'], C=C_all,
#                                flat=True, ncols=self.k)
#         self.Y_ols = Parameter(self.Y(self.P_ols.value, self.S_t),
#                                C=(J @ C_all @ J.T))
#
#     def gls_lin(self, X0):
#         a = np.linalg.pinv(self.S_t.T @ self.Cinv @ self.S_t)
#         b = self.S_t.T @ self.Cinv @ (self.Y_t - self.F(X0))
#         return a @ b
#
#     def gls(self, X0, printing=False):
#         P0 = Parameter(np.vstack((self.gls_lin(X0), X0)))
#         opt_gls = optimize.least_squares(self.resid_gls, P0.flat, method='lm')
#         if printing: print('GLS success:', opt_gls['success'])
#
#         J_w = opt_gls['jac']
#         J = np.linalg.pinv(self.H.T) @ J_w
#         Hess = J_w.T @ J_w
#         C_all = np.linalg.pinv(Hess)
#         self.P_gls = Parameter(opt_gls['x'], C=C_all,
#                                flat=True, ncols=self.k)
#         self.Y_gls = Parameter(self.Y(self.P_gls.value, self.S_t),
#                                C=(J @ C_all @ J.T))
#
#     def tls(self, X0, printing=False):
#         P0 = Parameter(np.vstack((self.gls_lin(X0), X0)))
#         opt_tls_P = optimize.least_squares(self.resid_tls_P, P0.flat, method='lm')
#         p_temp = opt_tls_P['x']
#         s_temp = self.get_S(self.unflatten_pars(p_temp)).ravel()
#         temp = np.hstack((p_temp, s_temp))
#         opt_tls = optimize.least_squares(self.resid_tls, temp, method='lm')
#         if printing: print('TLS success:', opt_tls['success'] & opt_tls_P['success'])
#
#         J_w = opt_tls['jac']
#         Lallinv = np.hstack((np.linalg.pinv(self.H.T), np.linalg.pinv(self.W.T)))
#         J = Lallinv @ J_w
#         Hess = J_w.T @ J_w
#         Cov_all = np.linalg.pinv(Hess)
#         idx = P0.value.size
#         Cov_p = Cov_all[:idx, :idx]
#         Cov_s = Cov_all[idx:, idx:]
#         self.P_tls = Parameter(opt_tls['x'][:idx], C=Cov_p,
#                                flat=True, ncols=self.k)
#         self.S_tls = Parameter(opt_tls['x'][idx:], C=Cov_s,
#                                flat=True, ncols=self.m)
#         self.Y_tls = Parameter(self.Y(self.P_tls.value, self.S_tls.value),
#                                C=(J @ Cov_all @ J.T))
#
#
# class Parameter:
#     def __init__(self, P, C=None, flat=False, ncols=False):
#         if flat:
#             self.flat = P
#             nrows = int(P.size / ncols)
#             self.value = np.reshape(P, (nrows, ncols))
#         else:
#             self.value = P
#             self.flat = P.ravel()
#         if C is not None:
#             self.C = C
#             self.err = np.reshape(np.sqrt(np.diag(C)), self.value.shape)
#
#
# class GLS:
#
#
#     def __init__(self, y, Cy, f):
#         '''
#
#         Args:
#             y: target matrix with size n x 1
#             Cy: covariance matrix for Y with size n x n
#             f: non-linear function to be fit that takes paramter vector p and outputs n x 1 vector
#         '''
#
#         self.y = y
#         self.Cy = Cy
#         self.f = f
#
#         self.Cy_inv = np.linalg.pinv(Cy)
#         self.chol_Cy_inv = np.linalg.cholesky(self.Cy_inv)
#
#
#
#     def resid_w(self, p):
#         return self.chol_Cy_inv @ (self.y - self.f(p))
#
#
#     def fit(self, p0, printing=False):
#         result = optimize.least_squares(self.resid_w, p0, method='lm')
#         if printing: print('converged:', result['success'])
#
#         J_w = result['jac']
#         J = np.linalg.pinv(self.chol_Cy_inv.T) @ J_w
#         Hess = J_w.T @ J_w
#
#         self.p = result['x']
#         self.Cp = np.linalg.pinv(Hess)
#         self.p_err = np.sqrt(np.diag(self.Cp))
#
#         self.y_fit = self.f(self.p)
#         self.Cy_fit = J @ self.Cp @ J.T
#         self.y_fit_err = np.sqrt(np.diag(self.Cy_fit))
#
#
#
# class GLS_lin:
#     def __init__(self, y, Cy, H):
#         '''
#
#         Args:
#             y: target matrix with size n x 1
#             Cy: covariance matrix for Y with size n x n
#             H: basis set of functions to be fit
#         '''
#
#         self.y = y
#         self.Cy = Cy
#         self.H = H
#
#         self.Cy_inv = np.linalg.pinv(Cy)
#         self.chol_Cy_inv = np.linalg.cholesky(self.Cy_inv)
#
#
#     def fit(self):
#
#         solver = np.linalg.pinv(self.H.T @ self.Cy_inv @ self.H) @ self.H.T @ self.Cy_inv
#         hat = self.H @ solver
#         self.p = solver @ self.y
#         self.Cp = solver @ self.Cy @ solver.T
#         self.p_err = np.sqrt(np.diag(self.Cp))
#
#         self.y_fit = hat @ self.y
#         self.Cy_fit = hat @ self.Cy @ hat.T
#         self.y_fit_err = np.sqrt(np.diag(self.Cy_fit))
#
#
#



















#
#
# def gls_fit(y, C, f_nl, H, x0, printing=False):
#
#     '''
#
#     Args:
#         y:
#         C:
#         f_nl:
#         H:
#         x0:
#
#     Returns:
#
#     '''
#     n_x = x0.size
#
#     H_ext = np.hstack((f_nl(x0), H))
#     gls_lin = GLS_lin(y, C, H_ext)
#     gls_lin.fit()
#     b0 = gls_lin.p
#     p0 = np.hstack((x0, b0))
#
#     def f(p):
#         x = p[:n_x]
#         b = p[n_x:]
#         return np.hstack((f_nl(x), H)) @ b
#
#     gls = GLS(y, C, f)
#     gls.fit(p0, printing=printing)
#     return gls.p, gls.p_err, gls.Cp, gls.y_fit, gls.y_fit_err, gls.Cy_fit
#
#
# def gls_fit_multitarget(Y, C, K, f_nl, H, x0, printing=False):
#     '''
#
#     Args:
#         Y:
#         C:
#         K:
#         f_nl:
#         H:
#         x0:
#         printing:
#
#     Returns:
#
#     '''
#
#     n_y = Y.shape[1]
#     n_p = x0.size + 1 + H.shape[1]
#     P = np.zeros((n_p, n_y))
#     P_err = np.zeros(P.shape)
#     Y_fit = np.zeros(Y.shape)
#     for i in range(n_y):
#         if printing: print(f'progress {i}/{n_y}', end=': ')
#         P[:, i], _, _, Y_fit[:, i], _, _ = gls_fit(Y[:, i], C, f_nl, H, x0, printing=printing)
#
