import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize


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

