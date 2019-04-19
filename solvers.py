import numpy as np
from numpy import ma
from scipy import sparse as sps
from functools import partial
from time import time
from math import exp
import subprocess
import shlex
import ot
import cplex
from copy import copy


def fast_primal_dual_non_bipartite(mc, ms, mcz, msz, pc, ps, lamda_s, lamda_c,
                                       check_every=10**3, max_iter=10**5, epsilon=10**-3):

    if sps.isspmatrix(pc):

        def f(pi_c_k, pi_s_k, pi_cz_k, pi_sz_k):

            ent_s = (pi_s_k.data * np.log((lamda_c_inv_diag.dot(pi_s_k)).data)).sum()
            ent_c = (pi_c_k.data * np.log((pi_c_k.dot(lamda_s_inv_diag)).data)).sum()
            ent_cz = (pi_cz_k * np.log(pi_cz_k/lamda_c)).sum()
            ent_sz = (pi_sz_k * np.log(pi_sz_k/lamda_s)).sum()
            m = (mc.multiply(pi_c_k) + ms.multiply(pi_s_k)).sum()
            mz = ((mcz * pi_cz_k).sum() + (msz * pi_sz_k).sum())
            return ent_c + ent_s + ent_cz + ent_sz + m + mz

        def check_optimality_gap():

            pi_c_eta = nm_c.multiply(eta.expm1() + p_ones)
            pi_s_eta = nm_s.multiply((eta * (-1)).expm1() + p_ones)

            c_normalizer_eta = lamda_c/(cz + pi_c_eta.multiply(pc).sum(axis=1).A.ravel())
            s_normalizer_eta = lamda_s/(sz + pi_s_eta.multiply(ps).sum(axis=0).A.ravel())

            pi_c_eta = sps.diags(c_normalizer_eta).dot(pi_c_eta)
            pi_s_eta = pi_s_eta.dot(sps.diags(s_normalizer_eta))

            pi_cz_eta = cz * c_normalizer_eta
            pi_sz_eta = sz * s_normalizer_eta

            f_pi_hat_k = f(pi_c_hat, pi_s_hat, pi_cz_hat, pi_sz_hat)
            phi_eta_k = f(pi_c_eta, pi_s_eta, pi_cz_eta, pi_sz_eta) - (pi_c_eta - pi_s_eta).sum()
            gap = f_pi_hat_k - phi_eta_k
            gap_pct = np.abs(gap)/np.abs(phi_eta_k)
            return gap, gap_pct

        def check_feasibility_gap():

            gap_k = pi_c_hat - pi_s_hat

            return (gap_k.multiply(gap_k).sum())**0.5

        lamda_c_inv_diag = sps.diags(1.0/lamda_c)
        lamda_s_inv_diag = sps.diags(1.0/lamda_s)

        nc, ns = pc.shape
        L = max((pc.multiply(pc) + ps.multiply(ps)).max(), 2.0)

        cz = (np.exp(-mcz - 1)*lamda_c)#.reshape((nc, 1))
        sz = (np.exp(-msz - 1)*lamda_s)#.reshape((1, ns))

        nm_c = sps.csr_matrix((np.exp(-mc.data - 1), mc.indices, mc.indptr)).dot(sps.diags(lamda_s))
        nm_s = sps.diags(lamda_c).dot(sps.csr_matrix((np.exp(-ms.data - 1), ms.indices, ms.indptr)))

        p_ones = pc.sign().multiply(pc.sign())

        pi_c_hat = 0 * pc
        pi_s_hat = 0 * ps
        pi_cz_hat = 0 * cz
        pi_sz_hat = 0 * sz

        theta = 0 * pc
        zeta = 0 * pc

        for i in np.arange(max_iter):

            if i == 0:

                print 'starting fast primal dual gradient descent'

            if i > 0 and i % check_every == 0:


                print 'iteration',  i
                opt_gap, opt_gap_pct = check_optimality_gap()
                feas_gap = check_feasibility_gap()
                print 'optimality gap pct is: ', opt_gap_pct
                print 'feasiblity gap is: ', feas_gap
                if opt_gap_pct < epsilon and feas_gap < epsilon:

                    break

            pi_c = nm_c.multiply(theta.expm1() + p_ones)
            pi_s = nm_s.multiply((theta*(-1)).expm1() + p_ones)

            c_normalizer = lamda_c/(cz + pi_c.multiply(pc).sum(axis=1).A.ravel())
            s_normalizer = lamda_s/(sz + pi_s.multiply(ps).sum(axis=0).A.ravel())

            pi_c = sps.diags(c_normalizer).dot(pi_c)
            pi_s = pi_s.dot(sps.diags(s_normalizer))

            pi_cz = cz * c_normalizer
            pi_sz = sz * s_normalizer

            alpha = (i + 1.0)/2.0
            tau = 2.0/(i+3.0)

            gap = pi_c - pi_s

            eta = theta - (1.0/L) * gap
            zeta = zeta - (alpha/L) * gap

            pi_c_hat = tau * pi_c + (1.0 - tau) * pi_c_hat
            pi_s_hat = tau * pi_s + (1.0 - tau) * pi_s_hat
            pi_cz_hat = tau * pi_cz + (1.0 - tau) * pi_cz_hat
            pi_sz_hat = tau * pi_sz + (1.0 - tau) * pi_sz_hat

            theta = (tau * zeta) + (1.0 - tau) * eta

        return pi_c_hat, pi_s_hat, pi_cz_hat, pi_sz_hat

    else:

        def f(pi_c_k, pi_s_k, pi_cz_k, pi_sz_k):

            ent_s = (pi_s_k.data * np.log((lamda_c_inv_diag.dot(pi_s_k)).data)).sum()
            ent_c = (pi_c_k.data * np.log((pi_c_k.dot(lamda_s_inv_diag)).data)).sum()
            ent_cz = (pi_cz_k * np.log(pi_cz_k/lamda_c)).sum()
            ent_sz = (pi_sz_k * np.log(pi_sz_k/lamda_s)).sum()
            m = (mc.multiply(pi_c_k) + ms.multiply(pi_s_k)).sum()
            mz = ((mcz * pi_cz_k).sum() + (msz * pi_sz_k).sum())
            return ent_c + ent_s + ent_cz + ent_sz + m + mz

        def check_optimality_gap():

            pi_c_eta = nm_c.multiply(eta.expm1() + p_ones)
            pi_s_eta = nm_s.multiply((eta * (-1)).expm1() + p_ones)

            c_normalizer_eta = lamda_c/(cz + pi_c_eta.multiply(pc).sum(axis=1).A.ravel())
            s_normalizer_eta = lamda_s/(sz + pi_s_eta.multiply(ps).sum(axis=0).A.ravel())

            pi_c_eta = sps.diags(c_normalizer_eta).dot(pi_c_eta)
            pi_s_eta = pi_s_eta.dot(sps.diags(s_normalizer_eta))

            pi_cz_eta = cz * c_normalizer_eta
            pi_sz_eta = sz * s_normalizer_eta

            f_pi_hat_k = f(pi_c_hat, pi_s_hat, pi_cz_hat, pi_sz_hat)
            phi_eta_k = f(pi_c_eta, pi_s_eta, pi_cz_eta, pi_sz_eta) - (pi_c_eta - pi_s_eta).sum()
            gap = f_pi_hat_k - phi_eta_k
            gap_pct = np.abs(gap)/np.abs(phi_eta_k)
            return gap, gap_pct

        def check_feasibility_gap():

            gap_k = pi_c_hat - pi_s_hat

            return (gap_k.multiply(gap_k).sum())**0.5

        #lamda_c_inv_diag = sps.diags(1.0/lamda_c)
        #lamda_s_inv_diag = sps.diags(1.0/lamda_s)

        nc, ns = pc.shape
        L = max((pc*pc + ps*ps).max(), 2.0)

        cz = (np.exp(-mcz - 1)*lamda_c)#.reshape((nc, 1))
        sz = (np.exp(-msz - 1)*lamda_s)#.reshape((1, ns))

        #nm_c = sps.csr_matrix((np.exp(-mc.data - 1), mc.indices, mc.indptr)).dot(sps.diags(lamda_s))
        #nm_s = sps.diags(lamda_c).dot(sps.csr_matrix((np.exp(-ms.data - 1), ms.indices, ms.indptr)))
        nm_c = np.exp(-mc - 1.0)
        nm_s = np.exp(-ms - 1.0)
        p_ones = pc.sign() * pc.sign()

        pi_c_hat = 0 * pc
        pi_s_hat = 0 * ps
        pi_cz_hat = 0 * cz
        pi_sz_hat = 0 * sz

        theta = 0 * pc
        zeta = 0 * pc

        for i in np.arange(max_iter):

            if i == 0:

                print 'starting fast primal dual gradient descent'

            if i > 0 and i % check_every == 0:


                print 'iteration',  i
                opt_gap, opt_gap_pct = check_optimality_gap()
                feas_gap = check_feasibility_gap()
                print 'optimality gap pct is: ', opt_gap_pct
                print 'feasiblity gap is: ', feas_gap
                if opt_gap_pct < epsilon and feas_gap < epsilon:

                    break

            pi_c = nm_c * (theta.expm1() + p_ones)
            pi_s = nm_s * ((theta*(-1)).expm1() + p_ones)

            c_normalizer = lamda_c/(cz + pi_c*pc.sum(axis=1).ravel())
            s_normalizer = lamda_s/(sz + pi_s*ps.sum(axis=0).ravel())

            pi_c = pi_c * c_normalizer
            pi_s = (pi_s.T*s_normalizer).T

            pi_cz = cz * c_normalizer
            pi_sz = sz * s_normalizer

            alpha = (i + 1.0)/2.0
            tau = 2.0/(i+3.0)

            gap = pi_c - pi_s

            eta = theta - (1.0/L) * gap
            zeta = zeta - (alpha/L) * gap

            pi_c_hat = tau * pi_c + (1.0 - tau) * pi_c_hat
            pi_s_hat = tau * pi_s + (1.0 - tau) * pi_s_hat
            pi_cz_hat = tau * pi_cz + (1.0 - tau) * pi_cz_hat
            pi_sz_hat = tau * pi_sz + (1.0 - tau) * pi_sz_hat

            theta = (tau * zeta) + (1.0 - tau) * eta

        return pi_c_hat, pi_s_hat, pi_cz_hat, pi_sz_hat


def fast_primal_dual_algorithm_grid(mc, ms, mcz, msz, pc, ps, lamda_s, lamda_c,
                                    check_every=10**3, max_iter=10**5, epsilon=10**-3):

    if sps.isspmatrix(pc):

        def f(pi_c_k, pi_s_k, pi_cz_k, pi_sz_k):

            ent_s = (pi_s_k.data * np.log((lamda_c_inv_diag.dot(pi_s_k)).data)).sum()
            ent_c = (pi_c_k.data * np.log((pi_c_k.dot(lamda_s_inv_diag)).data)).sum()
            ent_cz = (pi_cz_k * np.log(pi_cz_k/lamda_c)).sum()
            ent_sz = (pi_sz_k * np.log(pi_sz_k/lamda_s)).sum()
            m = (mc.multiply(pi_c_k) + ms.multiply(pi_s_k)).sum()
            mz = ((mcz * pi_cz_k).sum() + (msz * pi_sz_k).sum())
            return ent_c + ent_s + ent_cz + ent_sz + m + mz

        def check_optimality_gap():

            pi_c_eta = nm_c.multiply(eta.expm1() + p_ones)
            pi_s_eta = nm_s.multiply((eta * (-1)).expm1() + p_ones)

            c_normalizer_eta = lamda_c/(cz + pi_c_eta.multiply(pc).sum(axis=1).A.ravel())
            s_normalizer_eta = lamda_s/(sz + pi_s_eta.multiply(ps).sum(axis=0).A.ravel())

            pi_c_eta = sps.diags(c_normalizer_eta).dot(pi_c_eta)
            pi_s_eta = pi_s_eta.dot(sps.diags(s_normalizer_eta))

            pi_cz_eta = cz * c_normalizer_eta
            pi_sz_eta = sz * s_normalizer_eta

            f_pi_hat_k = f(pi_c_hat, pi_s_hat, pi_cz_hat, pi_sz_hat)
            phi_eta_k = f(pi_c_eta, pi_s_eta, pi_cz_eta, pi_sz_eta) - (pi_c_eta - pi_s_eta).sum()
            gap = f_pi_hat_k - phi_eta_k
            gap_pct = np.abs(gap)/np.abs(phi_eta_k)
            return gap, gap_pct

        def check_feasibility_gap():

            gap_k = pi_c_hat - pi_s_hat

            return (gap_k.multiply(gap_k).sum())**0.5

        lamda_c_inv_diag = sps.diags(1.0/lamda_c)
        lamda_s_inv_diag = sps.diags(1.0/lamda_s)

        nc, ns = pc.shape
        L = max((pc.multiply(pc) + ps.multiply(ps)).max(), 2.0)

        cz = (np.exp(-mcz - 1)*lamda_c)#.reshape((nc, 1))
        sz = (np.exp(-msz - 1)*lamda_s)#.reshape((1, ns))

        nm_c = sps.csr_matrix((np.exp(-mc.data - 1), mc.indices, mc.indptr)).dot(sps.diags(lamda_s))
        nm_s = sps.diags(lamda_c).dot(sps.csr_matrix((np.exp(-ms.data - 1), ms.indices, ms.indptr)))

        p_ones = pc.sign().multiply(pc.sign())

        pi_c_hat = 0 * pc
        pi_s_hat = 0 * ps
        pi_cz_hat = 0 * cz
        pi_sz_hat = 0 * sz

        theta = 0 * pc
        zeta = 0 * pc

        for i in np.arange(max_iter):

            if i == 0:

                print 'starting fast primal dual gradient descent'

            if i > 0 and i % check_every == 0:


                print 'iteration',  i
                opt_gap, opt_gap_pct = check_optimality_gap()
                feas_gap = check_feasibility_gap()
                print 'optimality gap pct is: ', opt_gap_pct
                print 'feasiblity gap is: ', feas_gap
                if opt_gap_pct < epsilon and feas_gap < epsilon:

                    break

            pi_c = nm_c.multiply(theta.expm1() + p_ones)
            pi_s = nm_s.multiply((theta*(-1)).expm1() + p_ones)

            c_normalizer = lamda_c/(cz + pi_c.multiply(pc).sum(axis=1).A.ravel())
            s_normalizer = lamda_s/(sz + pi_s.multiply(ps).sum(axis=0).A.ravel())

            pi_c = sps.diags(c_normalizer).dot(pi_c)
            pi_s = pi_s.dot(sps.diags(s_normalizer))

            pi_cz = cz * c_normalizer
            pi_sz = sz * s_normalizer

            alpha = (i + 1.0)/2.0
            tau = 2.0/(i+3.0)

            gap = pi_c - pi_s

            eta = theta - (1.0/L) * gap
            zeta = zeta - (alpha/L) * gap

            pi_c_hat = tau * pi_c + (1.0 - tau) * pi_c_hat
            pi_s_hat = tau * pi_s + (1.0 - tau) * pi_s_hat
            pi_cz_hat = tau * pi_cz + (1.0 - tau) * pi_cz_hat
            pi_sz_hat = tau * pi_sz + (1.0 - tau) * pi_sz_hat

            theta = (tau * zeta) + (1.0 - tau) * eta

        return pi_c_hat, pi_s_hat, pi_cz_hat, pi_sz_hat

    else:

        def f(pi_c_k, pi_s_k, pi_cz_k, pi_sz_k):

            ent_s = (pi_s_k.data * np.log((lamda_c_inv_diag.dot(pi_s_k)).data)).sum()
            ent_c = (pi_c_k.data * np.log((pi_c_k.dot(lamda_s_inv_diag)).data)).sum()
            ent_cz = (pi_cz_k * np.log(pi_cz_k/lamda_c)).sum()
            ent_sz = (pi_sz_k * np.log(pi_sz_k/lamda_s)).sum()
            m = (mc.multiply(pi_c_k) + ms.multiply(pi_s_k)).sum()
            mz = ((mcz * pi_cz_k).sum() + (msz * pi_sz_k).sum())
            return ent_c + ent_s + ent_cz + ent_sz + m + mz

        def check_optimality_gap():

            pi_c_eta = nm_c.multiply(eta.expm1() + p_ones)
            pi_s_eta = nm_s.multiply((eta * (-1)).expm1() + p_ones)

            c_normalizer_eta = lamda_c/(cz + pi_c_eta.multiply(pc).sum(axis=1).A.ravel())
            s_normalizer_eta = lamda_s/(sz + pi_s_eta.multiply(ps).sum(axis=0).A.ravel())

            pi_c_eta = sps.diags(c_normalizer_eta).dot(pi_c_eta)
            pi_s_eta = pi_s_eta.dot(sps.diags(s_normalizer_eta))

            pi_cz_eta = cz * c_normalizer_eta
            pi_sz_eta = sz * s_normalizer_eta

            f_pi_hat_k = f(pi_c_hat, pi_s_hat, pi_cz_hat, pi_sz_hat)
            phi_eta_k = f(pi_c_eta, pi_s_eta, pi_cz_eta, pi_sz_eta) - (pi_c_eta - pi_s_eta).sum()
            gap = f_pi_hat_k - phi_eta_k
            gap_pct = np.abs(gap)/np.abs(phi_eta_k)
            return gap, gap_pct

        def check_feasibility_gap():

            gap_k = pi_c_hat - pi_s_hat

            return (gap_k.multiply(gap_k).sum())**0.5

        #lamda_c_inv_diag = sps.diags(1.0/lamda_c)
        #lamda_s_inv_diag = sps.diags(1.0/lamda_s)

        nc, ns = pc.shape
        L = max((pc*pc + ps*ps).max(), 2.0)

        cz = (np.exp(-mcz - 1)*lamda_c)#.reshape((nc, 1))
        sz = (np.exp(-msz - 1)*lamda_s)#.reshape((1, ns))

        #nm_c = sps.csr_matrix((np.exp(-mc.data - 1), mc.indices, mc.indptr)).dot(sps.diags(lamda_s))
        #nm_s = sps.diags(lamda_c).dot(sps.csr_matrix((np.exp(-ms.data - 1), ms.indices, ms.indptr)))
        nm_c = np.exp(-mc - 1.0)
        nm_s = np.exp(-ms - 1.0)
        p_ones = pc.sign() * pc.sign()

        pi_c_hat = 0 * pc
        pi_s_hat = 0 * ps
        pi_cz_hat = 0 * cz
        pi_sz_hat = 0 * sz

        theta = 0 * pc
        zeta = 0 * pc

        for i in np.arange(max_iter):

            if i == 0:

                print 'starting fast primal dual gradient descent'

            if i > 0 and i % check_every == 0:


                print 'iteration',  i
                opt_gap, opt_gap_pct = check_optimality_gap()
                feas_gap = check_feasibility_gap()
                print 'optimality gap pct is: ', opt_gap_pct
                print 'feasiblity gap is: ', feas_gap
                if opt_gap_pct < epsilon and feas_gap < epsilon:

                    break

            pi_c = nm_c * (theta.expm1() + p_ones)
            pi_s = nm_s * ((theta*(-1)).expm1() + p_ones)

            c_normalizer = lamda_c/(cz + pi_c*pc.sum(axis=1).ravel())
            s_normalizer = lamda_s/(sz + pi_s*ps.sum(axis=0).ravel())

            pi_c = pi_c * c_normalizer
            pi_s = (pi_s.T*s_normalizer).T

            pi_cz = cz * c_normalizer
            pi_sz = sz * s_normalizer

            alpha = (i + 1.0)/2.0
            tau = 2.0/(i+3.0)

            gap = pi_c - pi_s

            eta = theta - (1.0/L) * gap
            zeta = zeta - (alpha/L) * gap

            pi_c_hat = tau * pi_c + (1.0 - tau) * pi_c_hat
            pi_s_hat = tau * pi_s + (1.0 - tau) * pi_s_hat
            pi_cz_hat = tau * pi_cz + (1.0 - tau) * pi_cz_hat
            pi_sz_hat = tau * pi_sz + (1.0 - tau) * pi_sz_hat

            theta = (tau * zeta) + (1.0 - tau) * eta

        return pi_c_hat, pi_s_hat, pi_cz_hat, pi_sz_hat


def strange_shelikhovskii(p, q, a, b, check_every=None, max_iter=10**5, epsilon=10**-5):

    # gets a matrix p with demands as rows and supply as columns
    # a[i] is the total demand of demand node i
    # b[j] is the total supply of supply node j
    # m,n = p.shape

    if check_every is None:
        check_every = 100

    k = 0
    flag = True

    if sps.isspmatrix(p):

        while k < max_iter and flag:

            p = sps.diags(a/p.sum(axis=1).A.ravel()).dot(p)
            p = p.dot(sps.diags(b/(p.multiply(q)).sum(axis=0).A.ravel()))
            if k > 0 and k % check_every == 0:
                print k
                print max(np.max(np.abs((b - (p.multiply(q)).sum(axis=0)))/b[:np.newaxis]),
                          np.max(np.abs((p.sum(axis=1).T - a))/a[:np.newaxis]))
                if max(np.max(np.abs((b - (p.multiply(q)).sum(axis=0)))/b[:np.newaxis]),
                       np.max(np.abs((p.sum(axis=1).T - a))/a[:np.newaxis])) < epsilon:
                    flag = False

            k += 1

    else:

        while k < max_iter and flag:
            p = (p.transpose() * a/p.sum(axis=1)).transpose()
            p = p * b/p.sum(axis=0)
            if k > 0 and k % check_every == 0:
                print k, max(max(abs(b - p.sum(axis=0))/b), max(abs(a - p.sum(axis=1))/a))
                if max(max(abs(b - p.sum(axis=0))/b), max(abs(a - p.sum(axis=1))/a)) < epsilon:
                    flag = False
            k += 1

    return p


def shelikhovskii(p, a, b, check_every=10**3, max_iter=10**6, epsilon=10**-6):

    # gets a matrix p with demands as rows and supply as columns
    # a[i] is the total demand of demand node i
    # b[j] is the total supply of supply node j
    # m,n = p.shape

    if check_every is None:
        check_every = 100

    k = 0
    flag = True

    if sps.isspmatrix(p):

        while k < max_iter and flag:

            p = sps.diags(a/p.sum(axis=1).A.ravel()).dot(p)
            p = p.dot(sps.diags(b/p.sum(axis=0).A.ravel()))
            if k > 0 and k % check_every == 0:
                print k
                print max(np.max(np.abs((b - p.sum(axis=0)))/b[:np.newaxis]),
                       np.max(np.abs((p.sum(axis=1).T - a))/a[:np.newaxis]))
                if max(np.max(np.abs((b - p.sum(axis=0)))/b[:np.newaxis]),
                       np.max(np.abs((p.sum(axis=1).T - a))/a[:np.newaxis])) < epsilon:
                    flag = False

            k += 1

    else:

        while k < max_iter and flag:
            p = (p.transpose() * a/p.sum(axis=1)).transpose()
            p = p * b/p.sum(axis=0)
            if k > 0 and k % check_every == 0:
                print k, max(max(abs(b - p.sum(axis=0))/b), max(abs(a - p.sum(axis=1))/a))
                if max(max(abs(b - p.sum(axis=0))/b), max(abs(a - p.sum(axis=1))/a)) < epsilon:
                    flag = False
            k += 1

    return p


def entropy_approximation_solver(alpha_v, beta_v, q):

    flag = False

    def sps_plog(p):
        return sps.csr_matrix((p.data*np.log(p.data), p.indices, p.indptr), p.shape)

    def dms_dual_grad(z, alpha, beta):

        p, q = calc_p_q(z)

        return beta.dot(p) - q.dot(alpha)

    def calc_p_q(z):

        densify = False
        if not sps.isspmatrix(z):
            z = sps.csr_matrix(z)
            densify = True
            print 'densifying'
        data = zip(*[(np.exp(z_cs), np.exp(-z_cs)) for z_cs in z.data])
        p = sps.csr_matrix((data[0], z.indices, z.indptr))
        q = sps.csr_matrix((data[1], z.indices, z.indptr))

        p_indptr = np.arange(z.shape[0]+1)
        p_indices = np.arange(z.shape[0])
        q_indptr = np.arange(z.shape[1]+1)
        q_indices = np.arange(z.shape[1])


        p = sps.csr_matrix((1/np.array(p.sum(axis=1)).squeeze(), p_indices, p_indptr)).dot(p)
        q = q.dot(sps.csr_matrix((1/np.array(q.sum(axis=0)).squeeze(), q_indices, q_indptr)))

        if densify:
            p = p.todense()
            q = q.todense()

        return p, q

    def dms_entropy(alpha, beta, p, q):

        cust_entropy = np.asscalar((beta.dot(sps_plog(p))).sum(axis=1).sum(axis=0))
        serv_entropy = np.asscalar((sps_plog(q).dot(alpha)).sum(axis=1).sum(axis=0))

        return cust_entropy + serv_entropy

    def dms_entropy_dual(z, alpha, beta):

        if not sps.isspmatrix(z):
            z = sps.csr_matrix(z)
            print 'sprsifying'
        p, q = calc_p_q(z)
        cust_entropy = np.asscalar((beta.dot(sps_plog(p))).sum(axis=1).sum(axis=0))
        serv_entropy = np.asscalar((sps_plog(q).dot(alpha)).sum(axis=1).sum(axis=0))
        penalty = (z.multiply(beta.dot(p) - q.dot(alpha))).sum()
        return cust_entropy + serv_entropy + penalty

    def gradient_descent(x, f, gf, a, eps, max_iter, check_point, add_func=None,  norm=2, prt=True):

        prev_f = f(x)
        improve_threshold = 0.1

        for k in range(max_iter):

            delta_x = a*gf(x)
            if add_func is None:
                x = x - delta_x
            else:
                x = add_func(x, delta_x)

            if k % check_point == 0:
                cur_f = f(x)
                p, q = calc_p_q(x)
                if np.max(np.abs(f.keywords['beta'].dot(p) - q.dot(f.keywords['alpha']))) < eps:
                    return x
                elif prt:
                    print 'cur_val = ', np.max(np.abs(f.keywords['beta'].dot(p) - q.dot(f.keywords['alpha'])))
                    print 'prev_val = ', prev_f
                    p, q = calc_p_q(x)
                    print 'primal_val = ', dms_entropy(p=p, q=q, **f.keywords)
                    print 'duality_gap = ', dms_entropy(p=p, q=q, **f.keywords) - cur_f
                    print 'improvement percentage = ', 100.0 * np.abs(1 - (cur_f/prev_f))
                    cur_improve = np.abs(1 - (cur_f/prev_f))
                    print 'improvement threshold%= ', 100*improve_threshold
                    if cur_improve <= improve_threshold:
                        a = a/2
                        improve_threshold = cur_improve/10.0
                        print 'reducing a'
                        print 'old a:', 2*a, 'new a:', a
                    print 'p'
                    print p
                    print 'q'
                    print q
                    print 'max feasibility gap'
                    #print f.keywords['beta'].dot(p) - q.dot(f.keywords['alpha'])
                    print np.max(f.keywords['beta'].dot(p) - q.dot(f.keywords['alpha']))
                prev_f = cur_f
        return x


    alpha_m = sps.csr_matrix((alpha_v, np.arange(len(alpha_v)), np.arange(len(alpha_v)+1)))
    beta_m = sps.csr_matrix((beta_v,  np.arange(len(alpha_v)), np.arange(len(alpha_v)+1)))

    grad_dms_ent_dual_p = partial(dms_dual_grad, alpha=beta_m, beta=alpha_m)
    dms_ent_dual_p = partial(dms_entropy_dual, alpha=beta_m, beta=alpha_m)
    z0 = q

    z = gradient_descent(x=z0, f=dms_ent_dual_p, gf=grad_dms_ent_dual_p,
                         a=0.2, eps=10**-5, max_iter=2*10**5, check_point=10**3)
    print 'now this is z'
    print z
    #z, ent, d = fmin_l_bfgs_b(dms_ent_dual_p, z.todense(), grad_dms_ent_dual_p)

    p, q = calc_p_q(z)

    r = alpha_m.dot(p)
    r_dict = dict((p, {'rcs-ent': r[p[0], p[1]]}) for p in zip(*q.nonzero()))

    return r_dict, r


def bipartite_entropy_approximation_solver(alpha_v, beta_v, q):

    flag = False

    def sps_plog(p):
        return sps.csr_matrix((p.data*np.log(p.data), p.indices, p.indptr), p.shape)

    def dms_dual_grad(z, alpha, beta):

        p, q = calc_p_q(z)

        return beta.dot(p) - q.dot(alpha)

    def calc_p_q(z):

        densify = False
        if not sps.isspmatrix(z):
            z = sps.csr_matrix(z)
            densify = True
            print 'densifying'
        data = zip(*[(np.exp(z_cs), np.exp(-z_cs)) for z_cs in z.data])
        p = sps.csr_matrix((data[0], z.indices, z.indptr))
        q = sps.csr_matrix((data[1], z.indices, z.indptr))

        p_indptr = np.arange(z.shape[0]+1)
        p_indices = np.arange(z.shape[0])
        q_indptr = np.arange(z.shape[1]+1)
        q_indices = np.arange(z.shape[1])


        p = sps.csr_matrix((1/np.array(p.sum(axis=1)).squeeze(), p_indices, p_indptr)).dot(p)
        q = q.dot(sps.csr_matrix((1/np.array(q.sum(axis=0)).squeeze(), q_indices, q_indptr)))

        if densify:
            p = p.todense()
            q = q.todense()

        return p, q

    def dms_entropy(alpha, beta, p, q):

        cust_entropy = np.asscalar((beta.dot(sps_plog(p))).sum(axis=1).sum(axis=0))
        serv_entropy = np.asscalar((sps_plog(q).dot(alpha)).sum(axis=1).sum(axis=0))

        return cust_entropy + serv_entropy

    def dms_entropy_dual(z, alpha, beta):

        if not sps.isspmatrix(z):
            z = sps.csr_matrix(z)
            print 'sprsifying'
        p, q = calc_p_q(z)
        cust_entropy = np.asscalar((beta.dot(sps_plog(p))).sum(axis=1).sum(axis=0))
        serv_entropy = np.asscalar((sps_plog(q).dot(alpha)).sum(axis=1).sum(axis=0))
        penalty = (z.multiply(beta.dot(p) - q.dot(alpha))).sum()
        return cust_entropy + serv_entropy + penalty

    def gradient_descent(x, f, gf, a, eps, max_iter, check_point, add_func=None,  norm=2, prt=True):

        prev_f = f(x)
        improve_threshold = 0.1

        for k in range(max_iter):

            delta_x = a*gf(x)
            if add_func is None:
                x = x - delta_x
            else:
                x = add_func(x, delta_x)

            if k % check_point == 0:
                cur_f = f(x)
                p, q = calc_p_q(x)
                if np.max(np.abs(f.keywords['beta'].dot(p) - q.dot(f.keywords['alpha']))) < eps:
                    return x
                elif prt:
                    print 'cur_val = ', np.max(np.abs(f.keywords['beta'].dot(p) - q.dot(f.keywords['alpha'])))
                    print 'prev_val = ', prev_f
                    p, q = calc_p_q(x)
                    print 'primal_val = ', dms_entropy(p=p, q=q, **f.keywords)
                    print 'duality_gap = ', dms_entropy(p=p, q=q, **f.keywords) - cur_f
                    print 'improvement percentage = ', 100.0 * np.abs(1 - (cur_f/prev_f))
                    cur_improve = np.abs(1 - (cur_f/prev_f))
                    print 'improvement threshold%= ', 100*improve_threshold
                    if cur_improve <= improve_threshold:
                        a = a/2
                        improve_threshold = cur_improve/10.0
                        print 'reducing a'
                        print 'old a:', 2*a, 'new a:', a
                    print 'p'
                    print p
                    print 'q'
                    print q
                    print 'max feasibility gap'
                    #print f.keywords['beta'].dot(p) - q.dot(f.keywords['alpha'])
                    print np.max(f.keywords['beta'].dot(p) - q.dot(f.keywords['alpha']))
                prev_f = cur_f
        return x


    alpha_m = sps.csr_matrix((alpha_v, np.arange(len(alpha_v)), np.arange(len(alpha_v)+1)))
    beta_m = sps.csr_matrix((beta_v,  np.arange(len(alpha_v)), np.arange(len(alpha_v)+1)))

    grad_dms_ent_dual_p = partial(dms_dual_grad, alpha=beta_m, beta=alpha_m)
    dms_ent_dual_p = partial(dms_entropy_dual, alpha=beta_m, beta=alpha_m)
    z0 = q

    z = gradient_descent(x=z0, f=dms_ent_dual_p, gf=grad_dms_ent_dual_p,
                         a=0.2, eps=10**-5, max_iter=2*10**5, check_point=10**3)
    print 'now this is z'
    print z
    #z, ent, d = fmin_l_bfgs_b(dms_ent_dual_p, z.todense(), grad_dms_ent_dual_p)

    p, q = calc_p_q(z)

    r = alpha_m.dot(p)
    r_dict = dict((p, {'rcs-ent': r[p[0], p[1]]}) for p in zip(*q.nonzero()))

    return r_dict, r


def solve_maxmin(alpha, beta, q, prt=False, prob_type='MaxMin'):

    nc = len(beta)   # nc - number of customers
    ns = len(alpha)  # ns- number of servers
    rnc = range(nc)  # 0,1,2,...,nc-1
    rns = range(ns)  # 0,1,2,...,ns-1
    qs = dict((i, set()) for i in rns)  # adjacency list for servers
    qc = dict((j, set()) for j in rnc)  # adjacency list for customers
    m = 0
    for p in zip(*q.nonzero()):
        m += 1
        qc[p[0]].add(p[1])
        qs[p[1]].add(p[0])
    types = ''
    s = time()
    if prob_type == 'MaxMin':
        tup_vals = [(qual,
                     str(qual),
                     [['c' + str(qual[0]), 's' + str(qual[1]), str(qual)],
                     [1.0, alpha[qual[0]], -1.0]]) for qual in zip(*q.nonzero())]
    elif prob_type == 'MinMax':
        tup_vals = [(qual,
                     str(qual),
                     [['c' + str(qual[0]), 's' + str(qual[1]), str(qual)],
                     [1.0, alpha[qual[0]], 1.0]]) for qual in zip(*q.nonzero())]
    else:
        tup_vals = [(qual,
                     str(qual),
                     [['c' + str(qual[0]), 's' + str(qual[1]), str(qual)+'Min', str(qual)+'Max' ],
                     [1.0, alpha[qual[0]], -1.0, 1.0]]) for qual in zip(*q.nonzero())]

    tup_vals = list(zip(*tup_vals))
    col_names = list(tup_vals[1])

    translator = dict(zip(col_names, list(tup_vals[0])))
    constraints_coeff = list(tup_vals[2])

    if prob_type == 'MaxMin':
        col_names.append('min_p')
        constraints_coeff.append([col_names[:-1], [1.0]*m])
    elif prob_type == 'MinMax':
        col_names.append('max_p')
        constraints_coeff.append([col_names[:-1], [-1.0]*m])
    else:
        col_names.append('min_p')
        constraints_coeff.append([[qual + 'Min' for qual in col_names[:-2]], [1.0]*m])
        col_names.append('max_p')
        constraints_coeff.append([[qual + 'Max' for qual in col_names[:-2]], [-1.0]*m])

    lb = [0.0]*(len(col_names))
    ub = [1.0]*(len(col_names))
    if prob_type in {'MaxMin', 'MinMax'}:
        obj = [0.0]*len(col_names[:-1]) + [1.0]
    else:
        obj = [0.0]*len(col_names[:-2]) + [-1.0] + [1.0]

    if prob_type in {'MaxMin', 'MinMax'}:
        row_names = col_names[:-1] + ['s'+str(sj) for sj in rns] + ['c' + str(ci) for ci in rnc]
    else:
        row_names = [qual + 'Min' for qual in col_names[:-2]] + [qual + 'Max' for qual in col_names[:-2]] + \
                    ['s'+str(sj) for sj in rns] + ['c' + str(ci) for ci in rnc]

    if prob_type in {'MaxMin', 'MinMax'}:
        senses = 'L'*m + 'L'*ns + 'E'*nc
        rhs = [0.0]*m + [beta[sj] for sj in rns] + [1.0]*nc
    else:
        senses = 'L'*(2*m) + 'L'*ns + 'E'*nc
        rhs = [0.0]*(2*m) + [beta[sj] for sj in rns] + [1.0]*nc

    print time()-s, 'seconds to construct model '
    s = time()
    prob = cplex.Cplex()
    if prob_type == 'MaxMin':
        prob.objective.set_sense(prob.objective.sense.maximize)
    elif prob_type == 'MinMax':
        prob.objective.set_sense(prob.objective.sense.minimize)
    else:
        prob.objective.set_sense(prob.objective.sense.minimize)



    if prt:
        print 'obj:', len(obj), obj
        print 'lb:', len(lb), lb
        print 'ub:', len(ub), ub
        print 'col_names:', len(col_names), col_names
        print 'type:', len(types), types
        print 'const_coeff:', len(constraints_coeff), constraints_coeff
        print 'row_names:', len(row_names), row_names
        print 'rhs:', len(rhs)

    prob.linear_constraints.add(rhs=rhs, senses=senses, names=row_names)
    prob.variables.add(obj=obj, lb=lb, ub=ub, names=col_names, types=types, columns=constraints_coeff)

    print time()-s, 'seconds to readin the model '
    s = time()

    if prt:
        print 'row_names:'
        print row_names
        print 'constraints:'
        print constraints_coeff

    try:
        prob.solve()
    except:
        print('not working')
        return

    numcols = prob.variables.get_num()
    numrows = prob.linear_constraints.get_num()

    slack = prob.solution.get_linear_slacks()
    x = prob.solution.get_values()

    print time()-s, 'seconds to solve model'

    s = time()
    alpha_m = sps.csr_matrix((alpha, np.arange(len(alpha)), np.arange(len(alpha)+1)))

    r = np.zeros(q.shape)
    r_dict = dict()

    for qual,res in zip(col_names, x):

        if qual in {'min_p', 'max_p'}:
            print res
        else:
            qual = translator[qual]
            r[qual[0], qual[1]] = alpha[qual[0]]*res
            if prob_type == 'MaxMin':
                r_dict[(float(qual[0]), float(qual[1]))] = {'rcs-max_min': alpha[qual[0]]*res}
            elif prob_type == 'MinMax':
                r_dict[(float(qual[0]), float(qual[1]))] = {'rcs-min_max': alpha[qual[0]]*res}
            else:
                r_dict[(float(qual[0]), float(qual[1]))] = {'rcs-min_gap': alpha[qual[0]]*res}

    return r_dict, r


def grid_workload_decomposition(lamda, q, path=None, nc=None):

    if path is None:
        path = '\\Users\\dean.grosbard\\Dropbox\\Software3.0\\fss'

    big_m = np.asscalar(lamda.sum())
    inputf = open('inputHPF.txt', 'w', 1)
    theta_max = 1
    theta_min = -100
    edges = zip(*q.nonzero())
    num_nodes = len(lamda) + 2
    num_edges = (num_nodes-2) + len(edges)/2
    if nc is None:
        rnd = range(1,1 + len(lamda)/2, 1)
        rns = range(1 + len(lamda)/2, len(lamda)+1, 1)
    else:
        rnd = range(1, 1 + nc, 1)
        rns = range(nc + 1, 1, len(lamda) + 1)

    inputf.write('p ' + str(num_nodes) +
                 ' ' + str(num_edges) +
                 ' ' + str(theta_min) +
                 ' ' + str(theta_max) +
                 ' 0'+'\n')
    inputf.write('n 0 s'+'\n')
    inputf.write('n '+str(num_nodes-1)+' t'+'\n')
    # print 'rnd', rnd
    # print 'lamda', lamda
    for ci in rnd:
        ub = lamda[ci-1]
        inputf.write('a ' + '0' + ' ' + str(ci) + ' ' + str(ub) + ' ' + '0.0' + '\n')
    for i, edge in enumerate(edges):
        ci = edge[0] + 1
        sj = edge[1] + 1
        if ci < sj:
            inputf.write('a ' + str(ci) + ' ' + str(sj) + ' ' + str(big_m) + ' ' + '0.0' + '\n')
    for sj in rns:
        ub = float(lamda[sj-1])
        coefficient = float(-1 * lamda[sj-1])
        inputf.write('a ' + str(sj) + ' ' + str(num_nodes-1) + ' ' + str(ub) + ' ' + str(coefficient) + '\n')

    inputf.close()
    m = subprocess.call('./hpf inputHPF.txt outputHPF.txt', shell=True)
    outputf = open('outputHPF.txt', 'r+', 1)

    # t_count = 0
    # bps = []
    # for line in outputf:
    #     data = line.split()
    #     if data[0] == 't':
    #         t_count = t_count+1
    #
    #     elif data[0] == 'l':
    #         rank = 1
    #         for bp in data[1:-1]:
    #             bps.append(1-float(bp))

    ##################
    workload_sets = dict()
    bps = []
    for line in outputf:
        data = line.split()

        if data[0] == 'l':
            rank = 0
            for bp in data[1:-1]:
                #print float(bp)
                workload_sets[rank] = {'workload': 1-float(bp), 'demnand_nodes': [], 'supply_nodes': []}
                bps.append(1-float(bp))
                rank += 1

        elif data[0] == 'n':
            node = int(data[1])
            if 0 < int(data[1]) < num_nodes - 1:
                singleton = True
                for i in range(3, len(data), 1):
                    if data[i] == '1': # Check if at sum point it moves to the source set if not it is a singleton
                        if node in rnd:
                            workload_sets[i-3]['demnand_nodes'].append(node-1)
                        if node in rns:
                            workload_sets[i-3]['supply_nodes'].append(node-1)
                        singleton = False
                        break
                if singleton:
                    print node, 'single'
                    return

    return workload_sets


def bipartite_workload_decomposition(Q, lamda, mu, path=None):

    m = len(lamda)
    n = len(mu)

    if path is None:
        path = '\\Users\\dean.grosbard\\Dropbox\\Software3.0\\fss'

    lamda_sum = np.asscalar(lamda.sum())
    inputf = open('inputHPF.txt', 'w', 1)
    theta_max = 1
    theta_min = -100
    edges = set(zip(*Q.nonzero()))
    num_nodes = m + n + 2
    num_edges = (num_nodes-2) + len(edges)
    rn = range(1, 1 + m, 1)
    rm = range(1 + m, m + n + 1, 1)

    inputf.write('p ' + str(num_nodes) +
                 ' ' + str(num_edges) +
                 ' ' + str(theta_min) +
                 ' ' + str(theta_max) +
                 ' 0'+'\n')
    inputf.write('n 0 s'+'\n')
    inputf.write('n ' + str(num_nodes-1) + ' t'+'\n')

    for i in rn:
        ub = lamda[i-1]
        inputf.write('a ' + '0' + ' ' + str(i) + ' ' + str(ub) + ' ' + '0.0' + '\n')

    for edge in sorted(list(edges)):

        i = edge[0] + 1
        j = edge[1] + m + 1

        inputf.write('a ' + str(i) + ' ' + str(j) + ' ' + str(lamda_sum) + ' ' + '0.0' + '\n')

    for j in rm:
        ub = float(mu[j - m - 1])
        coefficient = float(-1 * mu[j - m - 1])
        inputf.write('a ' + str(j) + ' ' + str(num_nodes-1) + ' ' + str(ub) + ' ' + str(coefficient) + '\n')

    inputf.close()

    _ = subprocess.call('./hpf inputHPF.txt outputHPF.txt', shell=True)
    outputf = open('outputHPF.txt', 'r+', 1)

    rho_m = [0.0] * m
    rho_n = [0.0] * n

    workload_sets = dict()
    bps = []

    for line in outputf:

        data = line.split()

        if data[0] == 'l':
            rank = 0
            for bp in data[1:-1]:
                workload_sets[rank] = \
                    {'rho': 1-float(bp),  'demnand_nodes': set(), 'supply_nodes': set()}
                bps.append(1-float(bp))
                rank += 1

        elif data[0] == 'n':
            node = int(data[1]) - 1
            if 0 < int(data[1]) < num_nodes - 1:
                singleton = True
                for i in range(3, len(data), 1):
                    if data[i] == '1':  # Check if at sum point it moves to the source set if not it is a singleton
                        rho = bps[i - 3]
                        if node < m:
                            rho_m[node] = rho
                        else:
                            rho_n[node - m] = rho
                        if node in rn:
                            workload_sets[i-3]['demnand_nodes'].add(node)
                        if node in rm:
                            workload_sets[i-3]['supply_nodes'].add(node - m)
                        singleton = False
                        break
                if singleton:
                    print node, 'single'
                    return None, None, None
    print '-----------------'
    print 'rho_n', rho_n
    print '-----------------'

    return workload_sets, np.array(rho_m), np.array(rho_n)


def greenkhorn_with_q(a, b, M, Q, reg, numItermax=10000, stopThr=1e-9, verbose=False, log=False):
    """
    Solve the entropic regularization optimal transport problem and return the OT matrix

    The algorithm used is based on the paper

    Near-linear time approximation algorithms for optimal transport via Sinkhorn iteration
        by Jason Altschuler, Jonathan Weed, Philippe Rigollet
        appeared at NIPS 2017

    which is a stochastic version of the Sinkhorn-Knopp algorithm [2].

    The function solves the following optimization problem:

    .. math::
        \gamma = arg\min_\gamma <\gamma,M>_F + reg\cdot\Omega(\gamma)

        s.t. \gamma 1 = a

             \gamma^T 1= b

             \gamma\geq 0
    where :

    - M is the (ns,nt) metric cost matrix
    - :math:`\Omega` is the entropic regularization term :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - a and b are source and target weights (sum to 1)



    Parameters
    ----------
    a : np.ndarray (ns,)
        samples weights in the source domain
    b : np.ndarray (nt,) or np.ndarray (nt,nbb)
        samples in the target domain, compute sinkhorn with multiple targets
        and fixed M if b is a matrix (return OT loss + dual variables in log)
    M : np.ndarray (ns,nt)
        loss matrix
    reg : float
        Regularization term >0
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshol on error (>0)
    log : bool, optional
        record log if True


    Returns
    -------
    gamma : (ns x nt) ndarray
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters

    Examples
    --------



    References
    ----------

    .. [2] M. Cuturi, Sinkhorn Distances : Lightspeed Computation of Optimal Transport, Advances in Neural Information Processing Systems (NIPS) 26, 2013
       [22] J. Altschuler, J.Weed, P. Rigollet : Near-linear time approximation algorithms for optimal transport via Sinkhorn iteration, Advances in Neural Information Processing Systems (NIPS) 31, 2017


    See Also
    --------
    ot.lp.emd : Unregularized OT
    ot.optim.cg : General regularized OT

    """
    m = a.shape[0]
    n = b.shape[0]

    if sps.isspmatrix(M):

        Ms = M
        Qs = Q

        row_sum = a.reshape((m, 1))
        col_sum = b.reshape((n, 1))

        # quivalent to K= np.exp(-M/reg), but faster to compute

        Ks = Qs.multiply(sps.csr_matrix((np.exp(-1.0 * Ms.data/reg), Ms.indices, Ms.indptr)))
        Gs = (sps.diags(np.array([1. / m] * m)).dot(Ks)).dot(sps.diags(np.array([1. / n] * n)))

        Kr = [Ks[i, :] for i in range(m)]
        Kc = [Ks[j, :] for j in range(n)]
        Gs = sps.csr_matrix(Gs)

        viol_row = (Gs.sum(1) - row_sum).T
        viol_col = (Gs.sum(0).T - col_sum).T

        stopThr_val = 1

        row_mul = np.full(m, 1. / m).reshape((m, 1))  # an m x 1 vector
        col_mul = np.full(n, 1. / n).reshape((n, 1))  # an 1 x n vector

        row_mul_mat = sps.csr_matrix(sps.diags(np.full(m, 1. / m)))  # an m x m matrix
        col_mul_mat = sps.csr_matrix(sps.diags(np.full(n, 1. / n)))  # an n x n matrix

        # if log:
        #     log['col_mul'] = col_mul
        #     log['row_mul'] = row_mul

        mat_state = 'csr'
        s = time()

        for k in range(numItermax):

            if k % 2 == 0:
                if k > 0:
                    print time() - s
                s = time()

            i = np.argmax(np.abs(viol_row))
            j = np.argmax(np.abs(viol_col))
            m_viol_row = np.abs(viol_row[0, i])
            m_viol_col = np.abs(viol_col[0, j])
            stopThr_val = np.maximum(m_viol_row, m_viol_col)

            if m_viol_row > m_viol_col:

                if mat_state == 'csc':

                    Ks = Ks.tocoo().transpose().tocsr()
                    Gs = Gs.tocoo().transpose().tocsr()
                    col_mul_mat = col_mul_mat.tocoo().tocsr()
                    row_mul_mat = row_mul_mat.tocoo().tocsr()
                    mat_state = 'csr'

                old_row_mul = row_mul[i, 0]
                row_mul[i, 0] = row_sum[i, 0] / Kr[i].dot(col_mul)
                row_mul_mat[i, i] = row_mul[i, 0]

                Gs[i, :] = row_mul[i, 0] * Kr[i].dot(col_mul_mat)

                viol_row[0, i] = row_mul[i, 0] * Ks[i, :].dot(col_mul) - row_sum[i, 0]  # should be 0 ot close to
                viol_col += Ks[i, :].dot(col_mul_mat) * (row_mul[i, 0] - old_row_mul)

            else:

                if mat_state == 'csr':

                    Ks = Ks.tocoo().transpose().tocsr()
                    Gs = Gs.tocoo().transpose().tocsr()
                    col_mul_mat = col_mul_mat.tocoo().tocsr()
                    row_mul_mat = row_mul_mat.tocoo().tocsr()
                    mat_state = 'csc'

                old_col_mul = col_mul[j, 0]

                col_mul[j, 0] = col_sum[j, 0] / Ks[j, :].dot(row_mul)
                col_mul_mat[j, j] = col_mul[j, 0]

                Gs[j, :] = col_mul[j, 0] * Ks[j, :].dot(row_mul_mat)

                viol_col[0, j] = col_mul[j, 0] * Ks[j, :].dot(row_mul) - col_sum[j, 0]  # should be 0 or close to
                viol_row += Ks[j, :].dot(row_mul_mat) * (-old_col_mul + col_mul[j, 0])

            if stopThr_val <= stopThr:
                break
        else:
            print('Warning: Algorithm did not converge')

        if log:
            log['col_mul'] = col_mul
            log['row_mul'] = row_mul

        if log:
            return Gs, log
        else:
            return Gs

    else:

        # Next 3 lines equivalent to K= np.exp(-M/reg), but faster to compute
        K = np.empty_like(M)
        np.divide(M, -reg, out=K)
        np.exp(K, out=K)
        np.multiply(K, Q, out=K)

        u = np.full(n, 1. / n)
        v = np.full(m, 1. / m)

            # a ,u are of size (n,)
            # b ,v are of size (m,)
            # viol is of size (n, 1)
            # viol2 is of size (1, m)
            # Matrices M, K, Q are of size n x m
            # K[i,:].T is of size m x 1 fits with
            #

        G = u[:, np.newaxis] * K * v[np.newaxis, :]

        viol = G.sum(1) - a  # violation of row sums
        viol_2 = G.sum(0) - b  # violation of col sums

        stopThr_val = 1
        s = time()
        if log:
            log['u'] = u
            log['v'] = v

        for i in range(numItermax):

            if i % 2 == 0:
                if i > 0:
                    print time() - s
                s = time()

            i_1 = np.argmax(np.abs(viol))
            i_2 = np.argmax(np.abs(viol_2))
            m_viol_1 = np.abs(viol[i_1])
            m_viol_2 = np.abs(viol_2[i_2])
            stopThr_val = np.maximum(m_viol_1, m_viol_2)

            if m_viol_1 > m_viol_2:
                old_u = u[i_1]
                u[i_1] = a[i_1] / (K[i_1, :].dot(v))
                G[i_1, :] = u[i_1] * K[i_1, :] * v

                viol[i_1] = u[i_1] * K[i_1, :].dot(v) - a[i_1]  # Why not just set to 0 , machine precision ?
                viol_2 += (K[i_1, :].T * (u[i_1] - old_u) * v)  # don't really need the transpose....

            else:
                old_v = v[i_2]
                v[i_2] = b[i_2] / (K[:, i_2].T.dot(u))
                G[:, i_2] = u * K[:, i_2] * v[i_2]
                viol += (-old_v + v[i_2]) * K[:, i_2] * u
                viol_2[i_2] = v[i_2] * K[:, i_2].dot(u) - b[i_2]

                #print('b',np.max(abs(aviol -viol)),np.max(abs(aviol_2 - viol_2)))

            if stopThr_val <= stopThr:
                break
        else:
            print('Warning: Algorithm did not converge')

        if log:
            log['u'] = u
            log['v'] = v

        if log:
            return G, log
        else:
            return G


def fast_primal_dual_algorithm(A, b, z, pi0=None, act_rows=None , check_every=10**3, max_iter=10**6, epsilon=10**-6, prt=False):

    m, n = A.shape
    pi_k = np.zeros((n, ))
    pi_hat = np.zeros((n, ))
    prev_pi_hat = np.zeros((n, ))
    prev_gap = np.zeros((m,))
    lamda = np.zeros((m, ))
    prev_lamda = np.zeros((m, ))
    zeta = np.zeros((m, ))
    ze = z * exp(-1.0)
    v = np.amin(z[np.where(z > 0)])

    def f(pi):

        if act_rows is None:
            res = np.divide(pi, z, out=np.zeros_like(pi), where=z!=0)
            res = pi * ma.log(res).filled(0)

            return res.sum()
        else:

            tmp_pi = pi[act_rows]
            tmp_z = z[act_rows]
            return (tmp_pi*np.log(tmp_pi/tmp_z)).sum()

    def check_optimality_gap():

            pi_eta = ze*np.exp(-1*At.dot(eta))
            f_pi_eta = f(pi_eta) + eta.dot(A.dot(pi_eta) - b)
            gap_k = f(pi_hat) - f_pi_eta
            gap_k_pct = gap_k/np.abs(f_pi_eta)
            return gap_k, gap_k_pct

    def check_feasibility_gap():

        gap_k = b - A.dot(pi_hat)

        return ((gap_k*gap_k).sum())**0.5

    def check_stop(i, prt =False):

        opt_gap, opt_gap_pct = check_optimality_gap()
        feas_gap = check_feasibility_gap()
        if prt or i % 10**5 == 0:
            print 'iteration',  i
            print 'optimality gap is:', opt_gap
            print 'optimality gap pct is: ', opt_gap_pct
            print 'feasibility gap is: ', feas_gap
        if opt_gap_pct < epsilon:
            if feas_gap < epsilon:
                return True
        return False

    if sps.isspmatrix(A):

        At = A.transpose().tocoo().tocsr()
        Aabs = sps.csr_matrix((np.abs(A.data), A.indices, A.indptr))
        L = (1.0/v) * np.amin(Aabs.sum(1))

    else:
        At = A.transpose()
        L = (1.0/v) * np.amin(np.abs(A.sum(1)))


    print 'L', L


    for i in np.arange(max_iter):

        alpha = (i + 1.0)/2.0
        tau = 2.0/(i+3.0)

        if i == 0:

            print 'starting fast primal dual gradient descent'
            s = time()

        #pi_k = pi0 if i == 0 and pi0 is not None else ze*np.exp(At.dot(lamda))
        #print i
        # print '-------------------------------'
        # # print 'prev_pi_hat'
        # # print prev_pi_hat.reshape((4, 3))
        # # print 'pi_hat'
        # # print pi_hat.reshape((4, 3))
        # # print 'prev_col_sum'
        # # print prev_pi_hat.reshape((4, 3)).sum(0)
        # # print 'cur_col_sum'
        # # print pi_hat.reshape((4, 3)).sum(0)
        # # print 'prev_row_sum'
        # # print prev_pi_hat.reshape((4, 3)).sum(1)
        # # print 'cur_row_sum'
        # # print pi_hat.reshape((4, 3)).sum(1)
        # # print 'prev col violation'
        # # print b[4:7] - prev_pi_hat.reshape((4, 3)).sum(0)
        # print 'cur col violation'
        # print b[4:7] - pi_hat.reshape((4, 3)).sum(0)
        # # print 'prev row violation'
        # # print b[:4] - prev_pi_hat.reshape((4, 3)).sum(1)
        # print 'cur row violation'
        # print b[:4] - pi_hat.reshape((4, 3)).sum(1)
        # # print 'gap'
        # # print b - A.dot(pi_hat)
        # # print 'prev_gap'
        # # print prev_gap
        # print 'lamda'
        # print lamda
        # # print 'prev_lamda'
        # # print prev_lamda
        #
        # # prev_pi_hat = pi_hat
        # # prev_gap = b - A.dot(pi_hat)
        # prev_lamda = lamda
        pi_k = ze*np.exp(-At.dot(lamda))

        #print pi_k.sum()
        pi_hat = tau * pi_k + (1.0 - tau) * pi_hat

        if i > 0 and i % check_every == 0:
            if check_stop(i, prt):
                break

        gap = b - A.dot(pi_k)

        eta = lamda - (1.0/L) * gap
        zeta = zeta - (alpha/L) * gap
        lamda = (tau * zeta) + (1.0 - tau) * eta


    print 'ended fast primal-dual algorithm after ' + str(i) + ' iterations'
    print 'run time:', time() - s, 'seconds'
    return pi_hat, lamda


if __name__ == '__main__':
#
    v = np.array(
        [[1,1,1,1,0],
         [0,1,1,1,1],
         [1,0,1,1,1],
         [1,1,0,1,1],
         [1,1,1,0,1]]).astype(float)
    v = sps.csr_matrix(v)

    ps = np.array(
        [[0.8,1,0.79,0,0],
         [0,1,0.92,0.93,0],
         [0,0,0.8,0.8,0.85],
         [0.9,0,0,0.73,0.76],
         [0.95,0.77,0,0,0.92]]).astype(float)

    ps = sps.csr_matrix(ps)
    lamda_c = np.array([2,2,1,2,3])
    mcz = np.array([1,1,1,1,1])
    msz = np.array([1,1,1,1,1])
    ms = v
    mc = v
    pc = v
    lamda_s = np.array([2,4,1,1,2])
    #
    # p = shelikhovskii(v, a, b)
    # print p
    # print p.sum(axis=0)
    # print p.sum(axis=1)
    #
    # p = strange_shelikhovskii(v, q, a, b)
    # print p
    # print p.sum(axis=0)
    # print p.sum(axis=1)

    pi_c, pi_s, pi_cz, pi_sz = fast_primal_dual_algorithm_grid(-mc, -ms, mcz, msz, ps, pc, lamda_s, lamda_c)

    print pi_c
    print pi_cz
    print (pi_c.multiply(ps).sum(axis=1).A.ravel() + pi_cz)

    print pi_s
    print pi_sz
    print (pi_s.multiply(pc).sum(axis=0).A.ravel() + pi_sz)


