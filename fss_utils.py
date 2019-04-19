import numpy as np
import operator
from itertools import permutations, combinations, chain, product
from scipy import stats as stats
from random import uniform
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy import sparse as sps
from numpy.linalg import norm
from ot import sinkhorn
import pickle
from shapely.geometry import Polygon
import heapq
import networkx as nx
from solvers import bipartite_workload_decomposition
import pandas as pd


def create_input_files(lamda, mu, q, **kwargs):

    input_i = pd.DataFrame({'i': np.arange(len(lamda)), 'lamda': lamda})
    input_j = pd.DataFrame({'j': np.arange(len(lamda)), 'mu': mu})
    if sps.isspmatrix(q):
        input_ij =


def filter_df(df, filter_dict):
    print filter_dict
    for key, value in filter_dict.iteritems():
        if not value[1]:
            df = df[df[key] == value[0]]
        else:
            df = df[np.isclose(df[key],value[0])]

    return df


def is_non_trivial_and_connected(q):

    m, n = q.shape
    if sps.isspmatrix(q):
        m_zero = sps.csr_matrix(np.zeros((m, m)))
        n_zero = sps.csr_matrix(np.zeros((n, n)))
    else:
        m_zero = np.zeros((m, m))
        n_zero = np.zeros((n, n))

    if sps.isspmatrix(q):
        row_deg = q.sum(axis=1).A
        col_deg = q.sum(axis=0).A
    else:
        row_deg = q.sum(axis=1)
        col_deg = q.sum(axis=0)

    min_deg = min(np.amin(row_deg), np.amin(col_deg))
    print min_deg

    if sps.isspmatrix(q):
        q = sps.vstack((sps.hstack((m_zero, q)), sps.hstack((q.T, n_zero))))
        g = nx.from_scipy_sparse_matrix(q)
    else:
        q = np.vstack((np.hstack((m_zero, q)), np.hstack((q.T, n_zero))))
        g = nx.from_numpy_array(q)

    return min_deg >= 2, nx.is_connected(g)


def gini_score(y):

    # " y is a heap "
    n = len(y)
    return (2*(y*np.arange(1, n+1, 1).sum())/(n*y.sum())) - (n+1/n)


def calc_area_between_curves(x1, y1, x2, y2):

    #x_y_curve1 = [(0.121,0.232),(2.898,4.554),(7.865,9.987)] #these are your points for curve 1 (I just put some random numbers)
    #x_y_curve2 = [(1.221,1.232),(3.898,5.554),(8.865,7.987)] #these are your points for curve 2 (I just put some random numbers)

    x_y_curve1 = list(zip(x1, y1))
    x_y_curve2 = list(zip(x2, y2))

    polygon_points = [] #creates a empty list where we will append the points to create the polygon

    for xyvalue in x_y_curve1:
        polygon_points.append([xyvalue[0],xyvalue[1]]) #append all xy points for curve 1

    for xyvalue in x_y_curve2[::-1]:
        polygon_points.append([xyvalue[0],xyvalue[1]]) #append all xy points for curve 2 in the reverse order (from last point to first point)

    for xyvalue in x_y_curve1[0:1]:
        polygon_points.append([xyvalue[0],xyvalue[1]]) #append the first point in curve 1 again, to it "closes" the polygon

    polygon = Polygon(polygon_points)
    area = polygon.area
    return area


def fast_choice(arr, size, shuffle=False):

    idx = set()
    while len(idx) < size:
        idx.add(np.random.randint(0, len(arr)))
    idx = np.array(list(idx))
    if shuffle:
        np.random.shuffle(idx)
    return [arr[x] for x in idx]


def transform_to_normal_form(M, W, Q, Z, row_sum, col_sum):

    if sps.isspmatrix(Q):
        print 'in sparse'
        # print Z.shape
        # print M.shape
        # print type(M)
        # print W.shape
        # print sps.csr_matrix((np.exp(-1*M.data/W.data), M.indices, M.indptr)).shape
        Z_hat = Z.multiply(W).multiply(sps.csr_matrix((np.exp(-1*M.data/W.data), M.indices, M.indptr)))
        Q_hat = sps.csr_matrix((Q.data/W.data, Q.indices, Q.indptr))
        z = Z_hat.todense().A.ravel()

    else:

        Z_hat = Z * W * np.exp(-M/W)
        Q_hat = Q/W
        z = Z_hat.ravel()

    A, b, col_set = metrize_constranits(Q_hat, row_sum, col_sum)

    return A, b, z, list(col_set)


def metrize_constranits(Q, row_sum, col_sum, eq_double_vars=False, prt=False):

    m = Q.shape[0]
    n = Q.shape[1]

    k = m + n
    l = m * n

    rows = []
    cols = []
    data = []
    col_set = set()

    for i, j in zip(*Q.nonzero()):
            if prt:
                print (i, j),'-->', (i, i * n + j), (m + j, i * n + j)
            rows.append(i)
            rows.append(m + j)
            cols.append(i * n + j)
            cols.append(i * n + j)
            col_set.add(i * n + j)
            data.append(Q[i, j])
            data.append(Q[i, j])

    rows = np.array(rows)
    cols = np.array(cols)
    data = np.array(data)
    A = sps.coo_matrix((data, (rows, cols)), shape=(k, l)).tocsr()
    b = np.concatenate((row_sum, col_sum))

    return A, b, col_set


def add_drain_nodes(q, lamda, mu):

    mu_sum = mu.sum()
    lamda_sum = lamda.sum()
    lamda = np.append(lamda, mu_sum)
    mu = np.append(mu, lamda_sum)
    if sps.issparse(q):
        q = sps.hstack((q, sps.csr_matrix(np.ones(q.shape[0], 1))))
        q = sps.vstack((q, sps.csr_matrix(np.ones(q.shape[1], 1))))
    else:
        q = np.hstack((q, sps.csr_matrix(np.ones(q.shape[0], 1))))
        q = np.vstack((q, sps.csr_matrix(np.ones(q.shape[1], 1))))

    return q, lamda, mu


def con_print(str, con):

    if con:
        print str


def sakasegawa(c,u):
    if c == 0 or (1-u) <= 0:
        return np.inf
    else:
        return u**(((2.0*(c+1.0))**0.5)-1)/(c*(1-u))


def fss_wt_approximation(q, mr, lamda, mu):
        print 'fss_wt_approximation'
        nc, ns = mr.shape
        m = np.exp((-1 * mr * np.log(mr + (np.ones(mr.shape) - np.sign(q)))).sum(axis=1))
        #m = q.sum(axis=1)
        print m.sum()
        u = np.squeeze(mr.dot((np.squeeze(mr.sum(axis=0))/mu)))/lamda
        dnum = m*(np.ones(nc) - u)
        num = u ** ((2.0 * (m + np.ones(nc)))**0.5 - np.ones(nc))
        scale = np.ones(nc)/(mr.dot(mu)/lamda)
        print scale * num/dnum
        #m = np.exp((-1 * mr * np.log(mr + (np.ones(mr.shape) - np.sign(q)))).sum(axis=1))
        m = q.sum(axis=1)
        print m.sum()
        #u = np.squeeze(mr.dot((np.squeeze(mr.sum(axis=0))/mu)))/lamda
        dnum = m*(np.ones(nc) - u)
        num = u ** ((2.0 * (m + np.ones(nc)))**0.5 - np.ones(nc))
        #scale = np.ones(nc)/(mr.dot(mu)/lamda)
        return scale * num/dnum


def gridify(vec, mapping, m, n):

    result = np.zeros((m,n))
    print vec[0].shape
    for i in range(m*n):
        result[mapping[i][0]] = vec[i]

    return result


def sps_plog(p):

    return sps.csr_matrix((p.data*np.log(p.data), p.indices, p.indptr), p.shape)


def matching_rate_calculator(alpha, beta, q, check_feasibility=True):

    nc = len(beta)
    ns = len(alpha)
    rnc = range(nc)
    rns = range(ns)
    c = set(rnc)
    s = set(rns)
    if sps.isspmatrix(q):
        q = q.todense()
    qs = dict((i, set(np.nonzero(q[:, i])[0])) for i in rns)
    qc = dict((j, set(np.nonzero(q[j, :])[0])) for j in rnc)
    B = 0

    def prod(iterable):
        return reduce(operator.mul, iterable, 1)

    def u_f(ss):
        ssn = s - ss
        return frozenset(c - set(chain.from_iterable([qs[si] for si in ssn])))

    def ak_f(ss):
        if len(ss) > 0:
            return sum(alpha[i] for i in ss)
        else:
            print 'alpha 0'
            return 0

    def bk_f(ss):
        if len(ss) > 0:
            return sum(beta[i] for i in ss)
        else:
            print 'beta 0'
            return 0

    a = dict()
    b = dict()
    u = dict()
    pip = dict()

    for k in range(1, ns+1, 1):
        for ss in combinations(rns, k):
            ss = frozenset(ss)
            u[ss] = u_f(ss)
            a[ss] = ak_f(ss)
            b[ss] = bk_f(ss)
        a[frozenset([])] = 0
        b[frozenset([])] = 0
        u[frozenset([])] = frozenset([])

    if check_feasibility:

        for k in range(1, ns, 1):
            for ss in combinations(rns, k):
                fs = frozenset(ss)
                if b[fs] < a[u[fs]]:
                    return 'Not Feasible:', ss, set(u[fs]), a[u[fs]], b[ss]


    flag = True

    for p in permutations(rns):
        # if flag:
        #     print p
        #     for k in range(1, ns, 1):
        #         print 'k',k
        #         print p[:k]
        #     flag = False
        pipp = prod(b[frozenset(p[:k])] - a[u[frozenset(p[:k])]] for k in range(1, ns, 1))**-1
        B += pipp
        pip[p] = pipp

    for p in pip.keys():
        pip[p] = pip[p]/B

    def phi_f(ss, ci):
        if u[ss] != frozenset([]):
            if ci in u[ss]:
                return 1
            else:
                return 0
        else:
            return 0

    def gma_f(ss, si):
        return sum(alpha[i] for i in u[ss]-qs[si])

    phi = dict()
    gma = dict()
    ddv = dict()
    eev = dict()
    ffv = dict()

    for ci in rnc:
        for k in range(1, ns+1, 1):
            for ss in combinations(rns, k):
                ss = frozenset(ss)
                phi[(ss, ci)] = phi_f(ss, ci)

    for si in rns:
        gma[(frozenset([]), si)] = 0
        for k in range(1, ns+1, 1):
            for ss in combinations(rns, k):
                ss = frozenset(ss)
                gma[(ss, si)] = gma_f(ss, si)

    for ci, sj in product(rnc, rns):
        if q[ci, sj] > 0:
            eev[((),sj)] = 1
            for k in range(1, ns+1, 1):
                for comb in combinations(rns, k):
                    ss = frozenset(comb)
                    ddv[(ss, sj)] = b[ss]-gma[(ss, sj)]
                    for p in permutations(ss):
                        sp = frozenset(p[:-1])
                        eev[(p, sj)] = eev[(p[:-1], sj)] * ddv[(ss, sj)]
                        if len(ss) == 1:
                            ffv[(p, sj)] = 1
                        else:
                            ffv[(p, sj)] = ffv[(p[:-1], sj)] * (b[sp]-a[u[sp]])

    r = np.zeros((nc,ns))

    for ci, sj in product(rnc, rns):
        if q[ci, sj] > 0:
            for p in permutations(rns):
                r[ci, sj] += alpha[ci]*beta[sj]*pip[p] * \
                             sum(phi[(ss, ci)]*ffv[(pss,sj)]/eev[(pss, sj)]
                                 for ss, pss in [(frozenset(p[:k+1]), p[:k+1]) for k in rns])

    r_dict = dict((p, {'rcs-aw': r[p[0], p[1]]}) for p in zip(*q.nonzero()))
    return r_dict, r


def matching_rate_calculator_n_sys(l1, l2, m1):


    m2 = 1.0 - m1
    ro = l1 + l2

    p12 = l1 / (l2 * ro)
    p12 += l1 / (l2 * m1)
    p12 += l1 * m2 / ((1.0 + ro) * (1.0 - ro) * m1)
    p12 += l1 / ((1.0 + ro) * (1.0 - ro))

    p1 = 0.0
    p1 += p12
    p1 += l1 / ((1.0 + ro) * (1.0 - ro) * m1)
    p1 += 1.0/ro
    p1 += 1.0/(m2 - l2)
    p1 += l1*m1/((1.0 + ro) * (1.0 - ro) * (1.0 - l2) * m2)

    pi12 = l1 * p12/p1


    #
    # B = 1.0 / l2
    # B += (ro + m1) / (m1 * l2)
    # B += (ro + 1.0) / (m1 * (1 - ro))
    # B += 1.0 / l1
    # B += (ro + m2) / (l1 * (m2 - l2))
    # B += (ro + 1.0) / (ro * (m2 - l2))
    #
    # p = (1.0/ l2) * (l2 / ro)
    # p += ((ro + m1)/(m1 * l2))* (l2 / (l2 + l1 + m1))
    # p += ((ro + 1.0) / (m1 * (1.0 - ro))) * (m2/(1.0 + ro)) * (l2 / ro)
    # p += ((1.0 + ro)/(m2 * (1.0 - ro))) * (m2/(1.0 + ro)) * (l2 / ro)
    #
    # q = (1.0/ l2) * (l2 / ro)
    # q += ((ro + m1)/(m1 * l2))* (l2 / (l2 + l1 + m1))
    # q += ((ro + 1.0) / (m1 * (1.0 - ro))) * ((m2/(1.0 + ro)) * (l2 / ro) + (m1/(1.0 + ro))*)

    # p = p/B

    return pi12


def random_unit_partition(size):

    partition = np.random.uniform(0, 1, size-1)
    partition.sort()
    partition = np.concatenate(([0], np.sort(partition),[1]))
    return partition[1:] - partition[:-1]


def scale_restricted_random_unit_partition(size, scale=10.):

    vals = np.random.uniform(1., 1.*scale, size)
    vals = vals/vals.sum()
    return vals


def gaussian_pdf_2d(m, n, centers, normalize=False):

    lims = (-3, 3)  # support of the PDF
    xx, yy = np.meshgrid(np.linspace(lims[0], lims[1], m), np.linspace(lims[0], lims[1], n))
    points = np.stack((xx, yy), axis=-1)
    pdf = np.zeros((m,n))
    for mean, weight in centers:  # Whatever your (i, j) is
        covariance = np.random.uniform(0, 1, (2, 2))
        covariance = np.dot(covariance, covariance.transpose()) + np.eye(2)
        covariance = covariance/covariance.sum()
        mean = (6.0*float(mean[0])/m - 3, 6.0*(float(mean[1])/n) - 3)
        pdf += weight * stats.multivariate_normal.pdf(points, mean, covariance)
    if normalize:
        pdf = pdf*(1/pdf.sum())
    return pdf


def undirected_2d_grid_adj_mat(m, n, d=1, weights= None,  r=1, l_norm=np.inf, sparse=True, bipartite=False):

    g = nx.empty_graph(0,None)
    rows = range(m)
    columns = range(n)
    g.add_nodes_from((i, j) for i in rows for j in columns)
    for i in range(m):
        for j in range(n):
            for o in range(i, min(m, i + d), 1):
                for k in range(max(0, j - d, j * (i == o)), min(j + d, n)):
                    dist = norm(np.array([o-i, k-j]), l_norm)
                    if dist <= d:
                        g.add_edge((i, j), (o, k))
                        for w in weights:
                            g[(i, j)][(o, k)][w[0]] = w[1]((i, j),(o, k))

    q = nx.adjacency_matrix(g)
    node_map = list(g.nodes())

    if bipartite:
        q = sps.vstack((sps.hstack((0*sps.eye(m*n), q)), sps.hstack((q, 0*sps.eye(m*n)))))
        node_map = dict(enumerate(list(zip(node_map, ['d']*len(node_map))) + list(zip(node_map, ['s']*len(node_map)))))

    if not sparse:
        q = q.todense()

    return q, node_map


def prep_non_bipartite_q_mat(q):

    if sps.isspmatrix(q):
        q = sps.vstack((sps.hstack((q, q + sps.eye(q.shape[0], q.shape[1]))),
                        sps.hstack((q + sps.eye(q.shape[0], q.shape[1]), q))))
    else:
        q = np.vstack((np.hstack((q, q + np.eye(q.shape[0], q.shape[1]))),
               np.hstack((q + np.eye(q.shape[0], q.shape[1]), q))))

    return q


def gaussian_pdf_2d_shitty(m, n, centers, normalize=False):

    lims = (-3, 3)  # support of the PDF
    xx, yy = np.meshgrid(np.linspace(lims[0], lims[1], m), np.linspace(lims[0], lims[1], n))
    points = np.stack((xx, yy), axis=-1)
    pdf = np.zeros((m, n))
    for mean, weight in centers:  # Whatever your (i, j) is

        #covariance = np.array([uniform(0,1) for _ in range(4)]).reshape((2,2))
        covariance = np.array([uniform(0,100000) for _ in range(4)]).reshape((2,2)).astype(int)
        covariance = np.dot(covariance, covariance.transpose())/100000.**2 + np.eye(2)
        covariance = covariance/covariance.sum()
        mean = (6.0*float(mean[0])/m - 3, 6.0*(float(mean[1])/n) - 3)
        pdf += weight * stats.multivariate_normal.pdf(points, mean, covariance)

    if normalize:
        pdf = pdf*(1/(pdf.sum()+1.))
    return pdf


if __name__ == '__main__':
    print ''
