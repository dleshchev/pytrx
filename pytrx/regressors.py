import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize

import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize




class NLPTLS:
    def __init__(self, Yt, Cy, Ky, s_t_list, Cs_list, f):
        if Yt.ndim == 1:
            Yt = Yt[:, None]
        self.Yt = Yt
        self.n, self.k = Yt.shape
        self.Cy = Cy
        self.Cy_diag = np.diag(np.diag(Cy))
        self.Cy_inv = np.linalg.pinv(Cy)
        self.Cy_diag_inv = np.diag(1 / np.diag(Cy))
        self.chol_Cy_inv = np.linalg.cholesky(self.Cy_inv)

        if Ky.ndim == 1:
            Ky = np.array([[Ky]])
        self.Ky = Ky
        self.Ky_inv = np.linalg.pinv(Ky)
        self.chol_Ky_inv = np.linalg.cholesky(self.Ky_inv)

        self.Gy = np.kron(self.Ky, self.Cy)
        self.Gy_diag = np.diag(np.diag(self.Gy))
        self.Gy_inv = np.kron(self.Ky_inv, self.Cy_inv)
        self.Gy_diag_inv = np.diag(1 / np.diag(self.Gy))
        self.chol_Gy_inv = np.kron(self.chol_Ky_inv, self.chol_Cy_inv)

        s_t_list = [np.squeeze(s_t) for s_t in s_t_list]
        self.S_t = np.array(s_t_list).T
        self.m = len(s_t_list)

        if Cs_list is not None:
            self.Cs_list = Cs_list
            self.Cs_inv_list = [np.linalg.pinv(Cs) for Cs in Cs_list]
            self.chol_Cs_inv_list = [np.linalg.cholesky(Cs_inv) for Cs_inv in self.Cs_inv_list]

            self.CS = np.zeros((self.m * self.n, self.m * self.n))
            self.CS_inv = np.zeros((self.m * self.n, self.m * self.n))
            self.chol_CS_inv = np.zeros((self.m * self.n, self.m * self.n))
            for i in range(self.m):
                self.CS[i::self.m, i::self.m] = self.Cs_list[i]
                self.CS_inv[i::self.m, i::self.m] = self.Cs_inv_list[i]
                self.chol_CS_inv[i::self.m, i::self.m] = self.chol_Cs_inv_list[i]

        self.f = f

    def split_p(self, P):
        return P[:self.m, :], P[self.m:, :]

    def F(self, X):
        F = np.zeros((self.n, self.k))
        for i in range(self.k):
            F[:, i] = self.f(X[:, i])
        return F

    def Y(self, P, S):
        B, X = self.split_p(P)
        return S @ B + self.F(X)

    def vec(self, V):
        return V.T.ravel()

    def unflatten_pars(self, pars):
        n_par = int(pars.size / self.k)
        return np.reshape(pars, (n_par, self.k))

    def bigB(self, B):
        I = np.eye(self.n)
        return np.block([[np.kron(I, b.T)] for b in B.T])

    def get_S(self, P):
        B, X = self.split_p(P)
        I = np.eye(self.m * self.n)
        Z = np.block([[self.bigB(B)], [I]])
        Null = np.zeros((self.n * self.k, self.n * self.m))
        Oinv = np.block([[self.Gy_inv, Null],
                         [Null, self.Gy_inv]])
        E = np.block([self.vec(self.Yt - self.F(X)),
                      self.vec(self.S_t.T)])
        vecST = np.linalg.pinv(Z.T @ Oinv @ Z) @ Z.T @ Oinv @ E
        return np.reshape(vecST, (self.n, self.m))

    def resid_Y(self, P, S):
        return self.chol_Gy_inv.T @ self.vec(self.Yt - self.Y(P, S))

    def resid_S(self, S):
        return self.chol_CS_inv.T @ self.vec(S.T - self.S_t.T)

    def chisq(self, P, S):
        rY = self.resid_Y(P, S)
        rS = self.resid_S(S)
        return rY @ rY + rS @ rS

    def resid_tls(self, pars):
        idx = self.m * self.n
        P = self.unflatten_pars(pars[:-idx])
        S = np.reshape(pars[-idx:], (self.n, self.m))
        return np.hstack((self.resid_Y(P, S), self.resid_S(S)))

    def resid_tls_P(self, pars):
        P = self.unflatten_pars(pars)
        S = self.get_S(P)
        return np.hstack((self.resid_Y(P, S), self.resid_S(S)))

    def resid_gls(self, pars):
        P = self.unflatten_pars(pars)
        return self.resid_Y(P, self.S_t)

    def resid_ols(self, pars):
        P = self.unflatten_pars(pars)
        return self.vec(self.Yt - self.Y(P, self.S_t)) / np.diag(self.Gy)

    def ols_lin(self, X0):
        # np.diag(self.C)
        a = np.linalg.pinv(self.S_t.T @ self.Cy_diag_inv @ self.S_t)
        b = self.S_t.T @ self.Cy_diag_inv @ (self.Yt - self.F(X0))
        return a @ b

    def ols(self, X0, printing=False):
        P0 = Parameter(np.vstack((self.ols_lin(X0), X0)))
        opt_ols = optimize.least_squares(self.resid_ols, P0.flat, method='lm')
        if printing: print('OLS success:', opt_ols['success'])

        J_w = opt_ols['jac']
        J = np.sqrt(self.Gy_diag) @ J_w
        Hess = J_w.T @ J_w
        C_all = np.linalg.pinv(Hess)
        self.P_ols = Parameter(opt_ols['x'], C=C_all,
                               flat=True, ncols=self.k)
        self.Y_ols = Parameter(self.Y(self.P_ols.value, self.S_t),
                               C=(J @ C_all @ J.T))

    def gls_lin(self, X0):
        a = np.linalg.pinv(self.S_t.T @ self.Cy_inv @ self.S_t)
        b = self.S_t.T @ self.Cy_inv @ (self.Yt - self.F(X0))
        return a @ b

    def gls(self, X0, printing=False):
        P0 = Parameter(np.vstack((self.gls_lin(X0), X0)))
        opt_gls = optimize.least_squares(self.resid_gls, P0.flat, method='lm')
        if printing: print('GLS success:', opt_gls['success'])

        J_w = opt_gls['jac']
        J = np.linalg.pinv(self.chol_Gy_inv.T) @ J_w
        Hess = J_w.T @ J_w
        C_all = np.linalg.pinv(Hess)
        self.P_gls = Parameter(opt_gls['x'], C=C_all,
                               flat=True, ncols=self.k)
        self.Y_gls = Parameter(self.Y(self.P_gls.value, self.S_t),
                               C=(J @ C_all @ J.T))

    def tls(self, X0, printing=False):
        P0 = Parameter(np.vstack((self.gls_lin(X0), X0)))
        opt_tls_P = optimize.least_squares(self.resid_tls_P, P0.flat, method='lm')
        p_temp = opt_tls_P['x']
        s_temp = self.get_S(self.unflatten_pars(p_temp)).ravel()
        temp = np.hstack((p_temp, s_temp))
        opt_tls = optimize.least_squares(self.resid_tls, temp, method='lm')
        if printing: print('TLS success:', opt_tls['success'] & opt_tls_P['success'])

        J_w = opt_tls['jac']
        Lallinv = np.hstack((np.linalg.pinv(self.H.T), np.linalg.pinv(self.W.T)))
        J = Lallinv @ J_w
        Hess = J_w.T @ J_w
        Cov_all = np.linalg.pinv(Hess)
        idx = P0.value.size
        Cov_p = Cov_all[:idx, :idx]
        Cov_s = Cov_all[idx:, idx:]
        self.P_tls = Parameter(opt_tls['x'][:idx], C=Cov_p,
                               flat=True, ncols=self.k)
        self.S_tls = Parameter(opt_tls['x'][idx:], C=Cov_s,
                               flat=True, ncols=self.m)
        self.Ytls = Parameter(self.Y(self.P_tls.value, self.S_tls.value),
                               C=(J @ Cov_all @ J.T))


class Parameter:
    def __init__(self, P, C=None, flat=False, ncols=False):
        if flat:
            self.flat = P
            nrows = int(P.size / ncols)
            self.value = np.reshape(P, (nrows, ncols))
        else:
            self.value = P
            self.flat = P.ravel()
        if C is not None:
            self.C = C
            self.err = np.reshape(np.sqrt(np.diag(C)), self.value.shape)


class GLS:


    def __init__(self, y, Cy, f):
        '''

        Args:
            y: target matrix with size n x 1
            Cy: covariance matrix for Y with size n x n
            f: non-linear function to be fit that takes paramter vector p and outputs n x 1 vector
        '''

        self.y = y
        self.Cy = Cy
        self.f = f

        self.Cy_inv = np.linalg.pinv(Cy)
        self.chol_Cy_inv = np.linalg.cholesky(self.Cy_inv)



    def resid_w(self, p):
        return self.chol_Cy_inv @ (self.y - self.f(p))


    def fit(self, p0, printing=False):
        result = optimize.least_squares(self.resid_w, p0, method='lm')
        if printing: print('converged:', result['success'])

        J_w = result['jac']
        J = np.linalg.pinv(self.chol_Cy_inv.T) @ J_w
        Hess = J_w.T @ J_w

        self.p = result['x']
        self.Cp = np.linalg.pinv(Hess)
        self.p_err = np.sqrt(np.diag(self.Cp))

        self.y_fit = self.f(self.p)
        self.Cy_fit = J @ self.Cp @ J.T
        self.y_fit_err = np.sqrt(np.diag(self.Cy_fit))



class GLS_lin:
    def __init__(self, y, Cy, H):
        '''

        Args:
            y: target matrix with size n x 1
            Cy: covariance matrix for Y with size n x n
            H: basis set of functions to be fit
        '''

        self.y = y
        self.Cy = Cy
        self.H = H

        self.Cy_inv = np.linalg.pinv(Cy)
        self.chol_Cy_inv = np.linalg.cholesky(self.Cy_inv)


    def fit(self):

        solver = np.linalg.pinv(self.H.T @ self.Cy_inv @ self.H) @ self.H.T @ self.Cy_inv
        hat = self.H @ solver
        self.p = solver @ self.y
        self.Cp = solver @ self.Cy @ solver.T
        self.p_err = np.sqrt(np.diag(self.Cp))

        self.y_fit = hat @ self.y
        self.Cy_fit = hat @ self.Cy @ hat.T
        self.y_fit_err = np.sqrt(np.diag(self.Cy_fit))




def gls_fit(y, C, f_nl, H, x0, printing=False):

    '''

    Args:
        y:
        C:
        f_nl:
        H:
        x0:

    Returns:

    '''
    n_x = x0.size

    H_ext = np.hstack((f_nl(x0), H))
    gls_lin = GLS_lin(y, C, H_ext)
    gls_lin.fit()
    b0 = gls_lin.p
    p0 = np.hstack((x0, b0))

    def f(p):
        x = p[:n_x]
        b = p[n_x:]
        return np.hstack((f_nl(x), H)) @ b

    gls = GLS(y, C, f)
    gls.fit(p0, printing=printing)
    return gls.p, gls.p_err, gls.Cp, gls.y_fit, gls.y_fit_err, gls.Cy_fit


def gls_fit_multitarget(Y, C, K, f_nl, H, x0, printing=False):
    '''

    Args:
        Y:
        C:
        K:
        f_nl:
        H:
        x0:
        printing:

    Returns:

    '''

    n_y = Y.shape[1]
    n_p = x0.size + 1 + H.shape[1]
    P = np.zeros((n_p, n_y))
    P_err = np.zeros(P.shape)
    Y_fit = np.zeros(Y.shape)
    for i in range(n_y):
        if printing: print(f'progress {i}/{n_y}', end=': ')
        P[:, i], _, _, Y_fit[:, i], _, _ = gls_fit(Y[:, i], C, f_nl, H, x0, printing=printing)

