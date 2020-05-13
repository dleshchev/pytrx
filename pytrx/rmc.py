import numpy as np
import matplotlib.pyplot as plt
import h5py
from pytrx.scatsim import Molecule, GR, DebyeFromGR
from pytrx.utils import z_num2str
import os
import time



class Ensemble:
    def __init__(self, targetMolecule, n_mol=100):
        self.Z = targetMolecule.Z
        self.xyz_t = targetMolecule.xyz

        self.n_atom = self.xyz_t.shape[0]
        self.n_mol = n_mol

        self.xyz = np.tile(self.xyz_t, (n_mol, 1, 1))

    def calcDistMat(self, subset=None):
        if subset is None: subset = np.arange(self.n_mol)
        self.dist_mat = np.sqrt(np.sum((self.xyz[subset, None, :, :] -
                                        self.xyz[subset, :, None, :]) ** 2, axis=3))

    def _computeGR(self, rmin=0, rmax=25, dr=0.01, subset=None):
        self.calcDistMat(subset=subset)
        n_subset = self.dist_mat.shape[0]

        if not hasattr(self, 'r'):
            gr = GR(self.Z, rmin=rmin, rmax=rmax, dr=dr)
            self.r = gr.r
        else:
            gr = GR(self.Z, r=self.r)

        for pair in gr.el_pairs:
            el1, el2 = pair
            idx1, idx2 = (el1 == self.Z, el2 == self.Z)
            idx_grid = np.ix_(np.ones(n_subset, dtype='bool'), idx1, idx2)
            gr[pair] += np.histogram(self.dist_mat[idx_grid].ravel(), gr.r_bins)[0]

        return gr

    def calcGR(self, rmin=0, rmax=25, dr=0.01):
        self.gr = self._computeGR(rmin=rmin, rmax=rmax, dr=dr, subset=None)

    def perturb(self, amplitude, idx_mol=None, idx_atom=None):
        if idx_mol is None: idx_mol = np.arange(self.n_mol)
        if idx_atom is None: idx_atom = np.arange(self.n_atom)

        m_mol = idx_mol.size
        m_atom = idx_atom.size
        dxyz = np.random.randn(m_mol, m_atom, 3) * amplitude / np.sqrt(3)
        self.xyz[idx_mol[:, None], idx_atom, :] += dxyz

    def calcStructDeviation(self, subset=None):
        if subset is None: subset = np.arange(self.n_mol)

        return np.sum((self.xyz[subset, :, :] - self.xyz_t[None, :, :]) ** 2)


#    def calcDens(self):
#        self.gr.calcDens()
#        self.dens = self.gr.dens


class diffEnsemble:
    def __init__(self, mol_gs, mol_es, n_mol_gs, n_mol_es):
        #        self.Z = np.hstack((mol_gs.Z, mol_es.Z))
        self.Z = mol_gs.Z.copy()

        self.n_atom = mol_gs.xyz.shape[0]
        #        self.n_atom_es = mol_es.xyz.shape[0]

        self.n_mol_gs = n_mol_gs
        self.n_mol_es = n_mol_es

        self.mol_gs = mol_gs
        self.mol_es = mol_es

        self.xyz = np.concatenate((np.tile(mol_gs.xyz, (n_mol_gs, 1, 1)),
                                   np.tile(mol_es.xyz, (n_mol_es, 1, 1))), axis=0)

    def calcDistMat(self, subset):
        self.dist_mat = np.sqrt(np.sum((self.xyz[subset, None, :, :] -
                                        self.xyz[subset, :, None, :]) ** 2, axis=3))

    def _computeGR(self, rmin=0, rmax=25, dr=0.01, subset=None):
        subset = self._enumerateSubset(subset)
        mask_gs, mask_es = self._divideSubset(subset)
        self.calcDistMat(subset)

        if hasattr(self, 'r'):
            rmin, rmax, dr = self.r.min(), self.r.max(), (self.r[1] - self.r[0])
        gr = GR(self.Z, rmin=rmin, rmax=rmax, dr=dr)
        if not hasattr(self, 'r'): self.r = gr.r

        for pair in gr.el_pairs:
            el1, el2 = pair
            idx1, idx2 = (el1 == self.Z, el2 == self.Z)
            idx_grid_gs = np.ix_(mask_gs, idx1, idx2)
            idx_grid_es = np.ix_(mask_es, idx1, idx2)
            gr[pair] -= np.histogram(self.dist_mat[idx_grid_gs].ravel(), gr.r_bins)[0] * 1 / self.n_mol_gs
            gr[pair] += np.histogram(self.dist_mat[idx_grid_es].ravel(), gr.r_bins)[0] * 1 / self.n_mol_es

        return gr

    def calcGR(self, rmin=0, rmax=25, dr=0.01):
        self.gr_gs = self._computeGR(rmin=rmin, rmax=rmax, dr=dr, subset='gs')
        self.gr_es = self._computeGR(rmin=rmin, rmax=rmax, dr=dr, subset='es')
        self.diff_gr = self.gr_es + self.gr_gs  # note that gr_gs is negative

    def perturb_all(self, amplitude):
        p = Perturbation('normal', amplitude,
                         n_mol=(self.n_mol_gs + self.n_mol_es),
                         n_atom=self.n_atom)
        self.xyz += p.generate_displacement()

    def var(self, subset=None):
        subset = self._enumerateSubset(subset)
        mask_gs, mask_es = self._divideSubset(subset)
        subset_gs = subset[mask_gs]
        subset_es = subset[mask_es]
        return ((np.sum((self.xyz[subset_gs, :, :] -
                         self.mol_gs.xyz[None, :, :]) ** 2) +
                 np.sum((self.xyz[subset_es, :, :] -
                         self.mol_es.xyz[None, :, :]) ** 2)) /
                (self.n_mol_gs + self.n_mol_es))

    def rmsd(self):
        return np.sqrt(self.var() / self.n_atom)

    def _enumerateSubset(self, subset):
        if subset is None:
            subset = np.arange(self.n_mol_gs + self.n_mol_es)
        elif type(subset) == str:
            if subset == 'gs':
                subset = np.arange(self.n_mol_gs)
            elif subset == 'es':
                subset = np.arange(self.n_mol_es) + self.n_mol_gs
        return subset

    def _divideSubset(self, subset):
        return subset < self.n_mol_gs, subset >= self.n_mol_gs


class Perturbation:
    def __init__(self, pert_type, amp, n_atom=None, dxyz_vib=None):
        self.pert_type = pert_type
        if pert_type == 'normal':
            self.amp = amp
            self.dxyz = np.tile(np.eye(n_atom)[:, :, None], (1, 1, 3))
            self.dxyz *= self.amp / np.sqrt(3)
            self.n = n_atom
            self.shape = (1, 3)

        elif pert_type == 'vib':
            self.amp = amp
            self.dxyz = dxyz_vib * self.amp
            self.n = self.dxyz.shape[0]
            self.shape = (1, 1)

    def generate_displacement(self):
        idx = np.random.randint(self.n)
        shake = np.random.randn(*self.shape)
        return (shake * self.dxyz[idx, :, :])


class RMC_Engine:
    def __init__(self, q, ds, sigma, dsdt,
                 diff_ens, perturbation_gs, perturbation_es, reg,
                 optimize_gs=True, optimize_es=True,
                 rmc_temp_init=None, rmc_cool=0.6,
                 qmin=None, qmax=None,
                 dr=0.01, rmax=None, rmax_margin=20,
                 output_file=None):
        if qmin is None: qmin = q.min()
        if qmax is None: qmax = q.max()
        qsel = (q >= qmin) & (q <= qmax)

        self.q = q[qsel]
        self.ds = ds[qsel]
        self.sigma = sigma[np.ix_(qsel, qsel)]
        self.sigma_inv = np.linalg.pinv(self.sigma)
        self.Lq = np.linalg.cholesky(self.sigma_inv)
        self.dsdt = dsdt[qsel]

        self.diff_ens = diff_ens
        self.perturbation_gs = perturbation_gs
        self.perturbation_es = perturbation_es

        self.reg = reg

        self.optimize_gs = optimize_gs
        self.optimize_es = optimize_es
        self._idxMolSpan()

        if rmax is None:
            self.rmax = self._rmax(rmax_margin=rmax_margin)
        else:
            self.rmax = rmax
        self.dr = dr

        self.ds_solu = self.calc_ds_solu()

        self.ds_fit, self.chisq = self._fit(self.ds_solu)
        self.penalty = self.diff_ens.var() / self.reg ** 2
        self.obj_fun = self.chisq + self.penalty

        # define rmc temperature if it is not defined already (for example, from the previous run)

        if rmc_temp_init is None:
            rmc_temp_init = 2 * (1 / self.diff_ens.n_mol_es +
                                 1 / self.diff_ens.n_mol_gs)
        self.rmc_temp = np.array(rmc_temp_init)
        self.rmc_cool = np.array(rmc_cool)

        print('Current rmc_temp: ', self.rmc_temp.ravel())
        print('Current chisq: ', self.chisq.ravel())
        print('Current penalty: ', self.penalty.ravel())
        print('Current obj_fun: ', self.obj_fun.ravel())

        self.output_file = output_file

    def _idxMolSpan(self):
        if self.optimize_gs:
            self.idx_mol_low = 0
        else:
            self.idx_mol_low = self.diff_ens.n_mol_gs
        if self.optimize_es:
            self.idx_mol_high = self.diff_ens.n_mol_gs + self.diff_ens.n_mol_es
        else:
            self.idx_mol_high = self.diff_ens.n_mol_gs

    def _rmax(self, rmax_margin=20):
        self.diff_ens.mol_gs.calcDistMat()
        self.diff_ens.mol_es.calcDistMat()

        all_dist = np.hstack((self.diff_ens.mol_gs.dist_mat.ravel().max(),
                              self.diff_ens.mol_es.dist_mat.ravel().max()))
        return all_dist.max() + rmax_margin

    def calc_ds_solu(self, subset=None):
        gr = self.diff_ens._computeGR(rmax=self.rmax, dr=self.dr, subset=subset)
        return DebyeFromGR(self.q, gr)

    def _fit(self, ds_solu):
        basis_set = np.hstack((ds_solu[:, None],
                               self.dsdt[:, None]))

        coef, chisq, _, _ = np.linalg.lstsq(self.Lq.T @ basis_set,
                                            self.Lq.T @ self.ds[:, None],
                                            rcond=-1)
        ds_fit = basis_set @ coef
        return ds_fit, chisq

    def run(self, n_steps, iter_upd=100, iter_autosave=None, iter_converge=None,
            con_thresh=0.01):
        #        n_steps, iter_upd, iter_autosave, iter_converge = int(n_steps), int(iter_upd), int(iter_autosave), int(iter_converge)

        n_cur = self.chisq.size
        self.chisq = np.hstack((self.chisq, np.zeros(n_steps) * np.nan))
        self.penalty = np.hstack((self.penalty, np.zeros(n_steps) * np.nan))
        self.obj_fun = np.hstack((self.obj_fun, np.zeros(n_steps) * np.nan))
        self.rmc_temp = np.hstack((self.rmc_temp, np.zeros(n_steps) * np.nan))

        # define number of steps sufficient to check convergence
        if iter_converge is None:
            iter_converge = 1 * self.diff_ens.n_atom * (self.diff_ens.n_mol_es +
                                                        self.diff_ens.n_mol_gs)

        startIntTime = time.clock()

        for i in range(n_cur, n_steps + n_cur):

            # progress/timing tracker
            if i % iter_upd == 0:
                interval = time.clock() - startIntTime
                print('Progress: ', i, '/', n_cur + n_steps,
                      '| Average interation time: %.0f' % (interval / iter_upd * 1e3), 'ms')
                startIntTime = time.clock()

            # actual RMC step calculation
            step_result = self.step(self.chisq[i - 1],
                                    self.penalty[i - 1],
                                    self.obj_fun[i - 1],
                                    self.rmc_temp[i - 1])  # inverse RMC temperature
            self.ds_solu, self.ds_fit, self.chisq[i], self.penalty[i], self.obj_fun[i] = step_result

            # convergence checker
            if i % iter_converge == 0:
                if self.isConverged(self.obj_fun[-iter_converge:], con_thresh):
                    #                    print('CONVERGED')
                    break

            # cooling checker
            if i % iter_converge == 0:
                if self.isEquilibrated(self.obj_fun[-iter_converge:]):
                    self.rmc_temp[i] *= self.rmc_cool
                    print('Ensemble equilibrated; RMC temperature now is', self.rmc_temp[i])
            else:
                self.rmc_temp[i] = self.rmc_temp[i - 1]

            # autosaver
            if (iter_autosave is not None) and (self.output_file is not None):
                if i % iter_autosave == 0:
                    self.save(self.output_file)

        self.diff_ens.calcGR()
        if self.output_file is not None:
            self.save(self.output_file)

        print('Converged:', self.isConverged(self.obj_fun[-iter_converge:], con_thresh))

    def step(self, chisq, penalty, obj_fun, rmc_temp):
        idx_mol, flag_mol = self._selectMolecule()

        if flag_mol == 'gs':
            dxyz = self.perturbation_gs.generate_displacement()
        #            print('gs', end='')
        elif flag_mol == 'es':
            dxyz = self.perturbation_es.generate_displacement()
        #            print('es', end='')

        ds_subset = self.calc_ds_solu(subset=idx_mol)
        penalty_subset = self.diff_ens.var(subset=idx_mol) / self.reg ** 2

        #        self.diff_ens.xyz[idx_mol, idx_atom, :] += dxyz
        self.diff_ens.xyz[idx_mol, :, :] += dxyz

        ds_subset_upd = self.calc_ds_solu(subset=idx_mol)
        penalty_subset_upd = self.diff_ens.var(subset=idx_mol) / self.reg ** 2

        ds_solu_upd = self.ds_solu - ds_subset + ds_subset_upd

        ds_fit_upd, chisq_upd = self._fit(ds_solu_upd)
        penalty_upd = penalty - penalty_subset + penalty_subset_upd
        obj_fun_upd = (chisq_upd + penalty_upd)

        dobj_fun = obj_fun_upd - obj_fun

        prob = np.exp(-dobj_fun / rmc_temp / 2)
        x = np.random.rand()

        if x < prob:  # accept
            #            print('accept', end=' ')
            return ds_solu_upd, ds_fit_upd, chisq_upd, penalty_upd, obj_fun_upd
        else:  # reject
            #            print('reject', end=' ')
            #            self.diff_ens.xyz[idx_mol, idx_atom, :] -= dxyz
            self.diff_ens.xyz[idx_mol, :, :] -= dxyz
            return self.ds_solu, self.ds_fit, chisq, penalty, obj_fun

    def _selectMolecule(self):
        idx_mol = np.random.randint(self.idx_mol_low,
                                    high=self.idx_mol_high, size=(1))
        if idx_mol < self.diff_ens.n_mol_gs:
            flag_mol = 'gs'
        else:
            flag_mol = 'es'
        return idx_mol, flag_mol

    def isEquilibrated(self, v, deg=3, minThresh=0.1):
        if (np.abs(v[0] - v[-1]) > minThresh):
            return False
        else:
            x = np.arange(v.size)
            p = np.polyfit(x, v, deg)
            v_fit = np.polyval(p, x)
            r = v - v_fit
            sig = np.std(r)

            if ((v[0] - v[-1]) <= 4 * sig):
                return True
            else:
                return False

    def isConverged(self, v, threshold):
        return ((v.max() - v.min()) < threshold)

    def save(self, filepath):
        try:
            f = h5py.File(filepath, 'w')
            f.create_dataset('q', data=self.q)
            f.create_dataset('ds', data=self.ds)
            f.create_dataset('sigma', data=self.sigma)
            f.create_dataset('dsdt', data=self.dsdt)
            f.create_dataset('reg', data=self.reg)
            f.create_dataset('rmax', data=self.rmax)
            f.create_dataset('dr', data=self.dr)
            f.create_dataset('ds_solu', data=self.ds_solu)
            f.create_dataset('ds_fit', data=self.ds_fit)
            f.create_dataset('chisq', data=self.chisq)
            f.create_dataset('penalty', data=self.penalty)
            f.create_dataset('obj_fun', data=self.obj_fun)
            f.create_dataset('rmc_temp', data=self.rmc_temp)
            f.create_dataset('rmc_cool', data=self.rmc_cool)
            f.create_dataset('optimize_gs', data=self.optimize_gs)
            f.create_dataset('optimize_es', data=self.optimize_es)

            f.create_dataset('diff_ens/n_mol_gs', data=self.diff_ens.n_mol_gs)
            f.create_dataset('diff_ens/n_mol_es', data=self.diff_ens.n_mol_es)
            f.create_dataset('diff_ens/xyz', data=self.diff_ens.xyz)
            f.create_dataset('diff_ens/mol_gs/Z_num', data=self.diff_ens.mol_gs.Z_num)
            f.create_dataset('diff_ens/mol_gs/xyz', data=self.diff_ens.mol_gs.xyz)
            f.create_dataset('diff_ens/mol_es/Z_num', data=self.diff_ens.mol_es.Z_num)
            f.create_dataset('diff_ens/mol_es/xyz', data=self.diff_ens.mol_es.xyz)
            f.create_dataset('diff_ens/r', data=self.diff_ens.r)

            f.create_dataset('perturbation_gs/pert_type', data=self.perturbation_gs.pert_type)
            f.create_dataset('perturbation_gs/amp', data=self.perturbation_gs.amp)
            f.create_dataset('perturbation_gs/dxyz', data=self.perturbation_gs.dxyz)
            f.create_dataset('perturbation_gs/shape', data=self.perturbation_gs.shape)
            f.create_dataset('perturbation_gs/n', data=self.perturbation_gs.n)

            f.create_dataset('perturbation_es/pert_type', data=self.perturbation_es.pert_type)
            f.create_dataset('perturbation_es/amp', data=self.perturbation_es.amp)
            f.create_dataset('perturbation_es/dxyz', data=self.perturbation_es.dxyz)
            f.create_dataset('perturbation_es/shape', data=self.perturbation_es.shape)
            f.create_dataset('perturbation_es/n', data=self.perturbation_es.n)

            rnd_st = np.random.get_state()
            for i in range(len(rnd_st)):
                key = 'random_state/t' + str(i)
                f.create_dataset(key, data=rnd_st[i])

        except OSError as err:
            print(err)

        if 'f' in locals():
            f.close()

    def plot_stats(self, num=None):
        if num is None:
            num = 1001
        plt.figure(num)
        plt.gcf().clf()
        fig, ax1 = plt.subplots(num=num)

        color = 'tab:red'
        ax1.set_ylabel('chisq/obj_fun', color=color)  # we already handled the x-label with ax1
        p1 = ax1.plot(self.chisq, color=color, linestyle=':', label='chisq')
        p2 = ax1.plot(self.obj_fun, color=color, label='obj_fun')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_xlabel('number of RMC steps')

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:blue'
        ax2.set_ylabel('penalty', color=color)  # we already handled the x-label with ax1
        p3 = ax2.plot(self.penalty, color=color, label='penalty')
        ax2.tick_params(axis='y', labelcolor=color)

        # added these three lines
        p = p2 + p1 + p3
        labs = [k.get_label() for k in p]
        ax1.legend(p, labs, loc=5)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped


def load_RMC_Engine(filepath, output_file=None):
    f = h5py.File(filepath, 'r')

    q = f['q'].value
    ds = f['ds'].value
    sigma = f['sigma'].value
    dsdt = f['dsdt'].value
    reg = f['reg'].value
    dr = f['dr'].value
    rmax = f['rmax'].value
    optimize_gs = f['optimize_gs'].value
    optimize_es = f['optimize_es'].value

    # get diff ensemble
    n_mol_gs = f['diff_ens/n_mol_gs'].value
    n_mol_es = f['diff_ens/n_mol_es'].value
    Z_gs = [z_num2str(z) for z in f['diff_ens']['mol_gs/Z_num'].value]
    Z_es = [z_num2str(z) for z in f['diff_ens']['mol_es/Z_num'].value]
    Z_gs = np.array(Z_gs)
    Z_es = np.array(Z_es)
    xyz_gs = f['diff_ens/mol_gs/xyz'].value
    xyz_es = f['diff_ens/mol_es/xyz'].value
    mol_gs = Molecule(Z_gs, xyz_gs)
    mol_es = Molecule(Z_es, xyz_es)
    diff_ens = diffEnsemble(mol_gs, mol_es, n_mol_gs, n_mol_es)
    diff_ens.xyz = f['diff_ens/xyz'].value
    diff_ens.r = f['diff_ens/r'].value
    diff_ens.calcGR()

    # get perturbation_gs
    pert_type = f['perturbation_gs/pert_type'].value
    amp = f['perturbation_gs/amp'].value
    dxyz = f['perturbation_gs/dxyz'].value
    #    shape = f['perturbation_gs/shape'].value
    n = f['perturbation_gs/n'].value
    perturbation_gs = Perturbation(pert_type, amp, n_atom=n, dxyz_vib=dxyz)
    perturbation_gs.dxyz = dxyz

    # get perturbation_es
    pert_type = f['perturbation_es/pert_type'].value
    amp = f['perturbation_es/amp'].value
    dxyz = f['perturbation_es/dxyz'].value
    #    shape = f['perturbation_es/shape'].value
    n = f['perturbation_es/n'].value
    perturbation_es = Perturbation(pert_type, amp, n_atom=n, dxyz_vib=dxyz)
    perturbation_es.dxyz = dxyz

    engine = RMC_Engine(q, ds, sigma, dsdt,
                        diff_ens, perturbation_gs, perturbation_es, reg,
                        optimize_gs=optimize_gs,
                        optimize_es=optimize_es,
                        qmin=None, qmax=None,
                        dr=dr, rmax=rmax,
                        output_file=output_file)

    # fitting stats
    engine.chisq = f['chisq'].value
    engine.penalty = f['penalty'].value
    engine.obj_fun = f['obj_fun'].value
    engine.rmc_temp = f['rmc_temp'].value

    bad_values = np.isnan(engine.chisq)
    engine.chisq = engine.chisq[~bad_values]
    engine.penalty = engine.penalty[~bad_values]
    engine.obj_fun = engine.obj_fun[~bad_values]
    engine.rmc_temp = engine.rmc_temp[~bad_values]

    # restore the state of the random generator for reproducibility
    rnd_st = []
    for i in range(5):
        key = 'random_state/t' + str(i)
        rnd_st.append(f[key].value)
    np.random.set_state(tuple(rnd_st))

    f.close()
    return engine
