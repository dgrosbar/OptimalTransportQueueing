from math import log, exp
import numpy as np
from scipy import sparse as sps
from scipy import stats as stats
from math import log, ceil
import networkx as nx
from fss_utils import gaussian_pdf_2d
from random import uniform
from numpy.linalg import norm
from solvers import grid_workload_decomposition, shelikhovskii, fast_primal_dual_algorithm, greenkhorn_with_q
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from fss_utils import *
from time import time
from itertools import product
import ot


def grid_graph(n, centers=None):

    if centers is None:
        centers = [((uniform(0, n),uniform(0, n)), uniform(0, 1))]
    lamda = gaussian_pdf_2d(n, n, centers)
    grid = nx.grid_2d_graph(n, n)
    range(len(n**2)),grid.nodes()


def erdos_renyi_graph(n):

    n = 100
    p = 2*log(n)/n
    lamda = np.concatenate(([0], np.sort(np.random.uniform(0, 1, n)),[1]))
    q = np.triu(np.random.choice(a=[1, 0], size=(n, n), p=[p, 1-p]))
    return q+np.tril(q.transpose(),-1)


def erdos_renyi_connected_graph(nc, ns, p=0.2, k=2):
    q = sps.csr_matrix(np.minimum(np.random.binomial(1, p, (nc, ns)) + k_chain(nc, ns, k).todense(), np.ones((nc, ns))))
    return q


def erdos_renyi_bipartite_graph(nc, ns, p=0.2, k=2):
    q = sps.csr_matrix(np.random.binomial(1, p, (nc, ns)))
    return q


def k_chain(nc, ns, k, sparse=True):

    if sparse:
        diags = [[1]*ns]*(2*k-1)
        offsets = list(range(k)) + [-nc+i for i in range(1, k, 1)]
        return sps.csr_matrix(sps.diags(diags, offsets, (nc, ns)))
    else:
        rnc = range(nc)
        base = [1]*k + [0]*(ns-k)

        return np.array([base] + [base[nc-ci:] + base[:-ci] for ci in rnc[1:]])


def undirected_grid_2d_bipartie_graph2(m, n, d=1, r=1,
                                      centers_d=None, centers_s=None,
                                      num_of_centers_supply=None, num_of_centers_demand=None,
                                      l_norm=np.inf, plot=False, alpha=1.0):

    g = nx.empty_graph(0,None)
    rows = range(m)
    columns = range(n)

    # adding all the nodes
    g.add_nodes_from((i, j) for i in rows for j in columns)

    # adding all the edges

    for k in range(d+1):
        for l in range(d+1):
            g.add_edges_from(((i, j), (i+l, j+k)) for i in rows for j in columns
                             if i + l <= m-1 and j + k <= n-1 and norm(np.array([l, k]), l_norm) <= r)

    nodes = list(g.nodes())

    if centers_d is None:
        if num_of_centers_demand is None:
            num_of_centers_demand = int(0.5*((m*n)**0.5))
        centers_d = [((uniform(0.2*m, 0.8*m), uniform(0.2*n, 0.8*n)), uniform(0, 1))
                     for _ in range(num_of_centers_demand)]
    if centers_s is None:
        if num_of_centers_supply is None:
            num_of_centers_supply = int(0.5*((m*n)**0.5))
        centers_s = [((uniform(0.2*m, 0.8*m), uniform(0.2*n, 0.8*n)), uniform(0, 1))
                     for _ in range(num_of_centers_supply)]

    lamda_d_pdf = gaussian_pdf_2d(m, n, centers_d, normalize=True)
    lamda_d = np.array([lamda_d_pdf[node] for node in nodes])

    lamda_s_pdf = alpha * gaussian_pdf_2d(m, n, centers_s, normalize=True) + (1 - alpha) * lamda_d_pdf
    lamda_s = np.array([lamda_s_pdf[node] for node in nodes])

    lamda = np.concatenate((lamda_d, lamda_s))
    grid_adj_mat = nx.adjacency_matrix(g)
    layered_grid_adj_mat = sps.vstack((sps.hstack((0*sps.eye(m*n), grid_adj_mat)),
                                       sps.hstack((grid_adj_mat, 0*sps.eye(m*n)))))
    nodes = dict(enumerate(list(zip(nodes, ['d']*len(nodes))) + list(zip(nodes, ['s']*len(nodes)))))

    workload_decomp = grid_workload_decomposition(lamda, layered_grid_adj_mat)
    max_workload = workload_decomp[0]['workload']

    lamda_s_pdf = lamda_s_pdf*max_workload
    lamda_s = lamda_s*max_workload

    for s in workload_decomp:
        workload_decomp[s]['workload'] = workload_decomp[s]['workload']/max_workload

    supply_decomp = np.zeros((m, n))
    demand_decomp = np.zeros((m, n))
    for st in workload_decomp:
        wl = workload_decomp[st]['workload']
        for d in workload_decomp[st]['demnand_nodes']:
            demand_decomp[nodes[d][0]] = wl
        for s in workload_decomp[st]['supply_nodes']:
            supply_decomp[nodes[s][0]] = wl

    p = sps.vstack((grid_adj_mat, np.ones((1, m*n))))
    d_s_gap = np.asscalar(lamda_s.sum() - (lamda_d*0.95).sum())
    print('starting shelikovskii')
    s = time()
    pi = shelikhovskii(p, np.hstack((lamda_d*0.95, np.array([d_s_gap]))), lamda_s)
    print(time() - s, ' sec to run shelikovskii')
    p = sps.csr_matrix(p)
    print('starting_primal_dual')

    s = time()
    print(time() - s, ' sec to run primal dual')

    r = pi[m*n:].todense()
    #print(np.min(r))
    pi = pi[:-1, :]
    max_ent_workload = np.zeros((m, n))
    rho_s = ((lamda_s - r)/lamda_s)
    for i in range(m*n):
        max_ent_workload[nodes[i][0]] = rho_s[0, i]

    fifo_ct = np.zeros((m,n))
    c = np.exp(-1*(sps_plog(pi.dot(sps.diags(1/lamda_s)))).dot(sps.diags(lamda_s)).sum(axis=1))
    rho_c = (sps.diags(1/(0.95*lamda_d)).dot(pi)).dot(rho_s.transpose())

    #print(c)

    sakasegawa_ct = []
    for i in range(m*n):
        sakasegawa_ct.append(sakasegawa(c[i,0],rho_c[i,0]))

    for i in range(m*n):
        fifo_ct[nodes[i][0]] = log(sakasegawa_ct[i])

    if plot:
        fig, ((ax1, ax2), (ax3, ax4),(ax5,ax6)) = plt.subplots(3, 2)
        ax1.imshow(lamda_d_pdf, interpolation='nearest')
        ax1.set_title('demand')
        ax2.imshow(lamda_s_pdf, interpolation='nearest')
        ax2.set_title('supply')
        ax3.imshow(demand_decomp, interpolation='nearest')
        ax3.set_title('demand decomposotion')
        ax4.imshow(supply_decomp, interpolation='nearest')
        ax4.set_title('supply decomposotion')
        ax5.imshow(max_ent_workload, interpolation='nearest')
        ax5.set_title('max_ent_workload')
        ax6.imshow(fifo_ct, interpolation='nearest')
        ax6.set_title('fifo_ct')
        plt.show()


    return dict(zip(
        ['lamda_s','lamda_d',
         'grid_adj_mat', 'nodes',
         'lamda_d_pdf', 'lamda_s_pdf',
         'demand_decomp', 'supply_decomp',
         'fifo_ct', 'max_ent_workload'],
        [lamda_s, lamda_d * 0.95,
         grid_adj_mat, nodes,
         lamda_d_pdf, lamda_s_pdf,
         demand_decomp, supply_decomp,
         fifo_ct, max_ent_workload]))

def undirected_grid_2d_bipartie_graph(m, n, d=1, r=1,
                                      centers_d=None, centers_s=None,
                                      num_of_centers_supply=None, num_of_centers_demand=None,
                                      l_norm=np.inf, plot=False, alpha=1.0):

    g = nx.empty_graph(0,None)
    rows = range(m)
    columns = range(n)

    # adding all the nodes
    g.add_nodes_from((i, j) for i in rows for j in columns)

    # adding all the edges

    for k in range(d+1):
        for l in range(d+1):
            g.add_edges_from(((i, j), (i+l, j+k)) for i in rows for j in columns
                             if i + l <= m-1 and j + k <= n-1 and norm(np.array([l, k]), l_norm) <= r)

    nodes = list(g.nodes())

    if centers_d is None:
        if num_of_centers_demand is None:
            num_of_centers_demand = int(0.5*((m*n)**0.5))
        centers_d = [((uniform(0.2*m, 0.8*m), uniform(0.2*n, 0.8*n)), uniform(0, 1))
                     for _ in range(num_of_centers_demand)]
    if centers_s is None:
        if num_of_centers_supply is None:
            num_of_centers_supply = int(0.5*((m*n)**0.5))
        centers_s = [((uniform(0.2*m, 0.8*m), uniform(0.2*n, 0.8*n)), uniform(0, 1))
                     for _ in range(num_of_centers_supply)]

    lamda_d_pdf = gaussian_pdf_2d_shitty(m, n, centers_d, normalize=True)
    lamda_d = np.array([lamda_d_pdf[node] for node in nodes])

    lamda_s_pdf = alpha * gaussian_pdf_2d_shitty(m, n, centers_s, normalize=True) + (1 - alpha) * lamda_d_pdf
    lamda_s = np.array([lamda_s_pdf[node] for node in nodes])

    lamda = np.concatenate((lamda_d, lamda_s))
    grid_adj_mat = nx.adjacency_matrix(g)
    layered_grid_adj_mat = sps.vstack((sps.hstack((0*sps.eye(m*n), grid_adj_mat)),
                                       sps.hstack((grid_adj_mat, 0*sps.eye(m*n)))))
    nodes = dict(enumerate(list(zip(nodes, ['d']*len(nodes))) + list(zip(nodes, ['s']*len(nodes)))))

    workload_decomp = grid_workload_decomposition(lamda, layered_grid_adj_mat)
    max_workload = workload_decomp[0]['workload']

    lamda_s_pdf = lamda_s_pdf*max_workload
    lamda_s = lamda_s*max_workload

    for s in workload_decomp:
        workload_decomp[s]['workload'] = workload_decomp[s]['workload']/max_workload

    supply_decomp = np.zeros((m, n))
    demand_decomp = np.zeros((m, n))
    for st in workload_decomp:
        wl = workload_decomp[st]['workload']
        for d in workload_decomp[st]['demnand_nodes']:
            demand_decomp[nodes[d][0]] = wl
        for s in workload_decomp[st]['supply_nodes']:
            supply_decomp[nodes[s][0]] = wl

    return dict(zip(
        ['lamda_s','lamda_d',
         'grid_adj_mat', 'nodes',
         'lamda_d_pdf', 'lamda_s_pdf',
         'demand_decomp', 'supply_decomp',
         'fifo_ct', 'max_ent_workload'],
        [lamda_s, lamda_d,
         grid_adj_mat, nodes,
         lamda_d_pdf, lamda_s_pdf,
         demand_decomp, supply_decomp]))



def supply_demand_grid(m, n, d=1, r=1,
                       centers_d=None, centers_s=None,
                       num_of_centers_supply=None, num_of_centers_demand=None,
                       l_norm=np.inf, plot=False, alpha=1.0):

    g = nx.empty_graph(0,None)

    # adding all the nodes

    g.add_nodes_from(product(range(m), range(n)))

    # adding all the edges

    def p(b, v):
        return min(max(1 - (1 - exp(-b * v)) / (1 - exp(-b * 20)), 0), 1)

    bs = log(0.2)/10
    bd = log(0.05)/10
    qds = 0.05  # probability that demand d leaves if supply s does not accept match
    qdd = 0.75  # probability that demand d leaves if demand d does not accept match
    qsd = 0.02  # probability that supply s leaves if demand d does not accept match
    qss = 0.02  # probability that supply s leaves if demand s does not accept match
    msd = 1.0   # revenue from matching s with d
    cd = 5.0  # loss from demand d canceling
    u = 0
    print('l_norem', l_norm)
    for i in range(m):
        for j in range(n):
            for o in range(i, min(m, i + d), 1):
                for k in range(max(0, j - d, j * (i == o)), min(j + d, n)):
                    dist = norm(np.array([o-i, k-j]), l_norm)
                    if dist <= d:
                        pda = 1.0  # p(bd, dist)  # probability that demand d accepts the match
                        psa = 1.0  # p(bs, dist)  # probability that supply s accepts the match
                        pd = pda * psa + (1.0 - pda) * qdd + (1.0 - psa) * qds  # probability that a unit of demand d is spent by a match with s
                        ps = pda * psa + (1.0 - pda) * qsd + (1.0 - psa) * qss  # probability that a unit of supply s is spent by a match with d
                        md = -1*(d - dist)  # * psa * pda + ((1.0 - pda) * qdd + (1.0 - psa) * qds) * cd  # Expected loss from matching s and d
                        g.add_edge((i, j), (o, k), m=md, pd=pd, ps=ps)
                        u += 1

    print('u: ', u)

    nodes = list(g.nodes())

    if centers_d is None:
        if num_of_centers_demand is None:
            num_of_centers_demand = int(0.5*((m*n)**0.5))
        centers_d = [((uniform(0.2*m, 0.8*m), uniform(0.2*n, 0.6*n)), uniform(0.0, 1.0))
                     for _ in range(num_of_centers_demand)]
    if centers_s is None:
        if num_of_centers_supply is None:
            num_of_centers_supply = int(0.5*((m*n)**0.5))
        centers_s = [((uniform(d, m-d), uniform(d, n-d)), uniform(0.0, 1.0))
                     for _ in range(num_of_centers_supply)]

    lamda_d_pdf = gaussian_pdf_2d(m, n, centers_d, normalize=True)
    lamda_d = np.array([lamda_d_pdf[node] for node in nodes])

    lamda_s_pdf = alpha * gaussian_pdf_2d(m, n, centers_s, normalize=True) + (1.0 - alpha) * lamda_d_pdf
    lamda_s = np.array([lamda_s_pdf[node] for node in nodes])

    pd = nx.adjacency_matrix(g, weight='pd')
    ps = nx.adjacency_matrix(g, weight='ps')
    m = nx.adjacency_matrix(g, weight='m')

    nodes = dict(enumerate(list(zip(nodes, ['d']*len(nodes))) + list(zip(nodes, ['s']*len(nodes)))))

    return {'lamda_d': lamda_d, 'lamda_s': lamda_s, 'pd': pd, 'ps': ps, 'm': m, 'nodes': nodes}


def undirected_full_2d_bipartie_graph(m, n, d=1, r=1,
                                      centers_d=None, centers_s=None, balance=False,
                                      num_of_centers_supply=None, num_of_centers_demand=None,
                                      l_norm=np.inf, plot=False, alpha=1.0):

    g = nx.empty_graph(0,None)
    rows = range(m)
    columns = range(n)

    # adding all the nodes
    g.add_nodes_from((i, j) for i in rows for j in columns)

    # adding all the edges
    for k in range(d+1):
        for l in range(d+1):
            g.add_edges_from(((i,j), (i+l,j+k)) for i in rows for j in columns
                             if i + l <= m-1 and j + k <= n-1 and norm(np.array([l, k]), l_norm) <= r)

    nodes = list(g.nodes())

    if centers_d is None:
        if num_of_centers_demand is None:
            num_of_centers_demand = int(0.5*((m*n)**0.5))
        centers_d = [((uniform(0.2*m, 0.8*m), uniform(0.2*n, 0.6*n)), uniform(0, 1))
                     for _ in range(num_of_centers_demand)]
    if centers_s is None:
        if num_of_centers_supply is None:
            num_of_centers_supply = int(0.5*((m*n)**0.5))
        centers_s = [((uniform(d, m-d), uniform(d, n-d)), uniform(0, 1))
                     for _ in range(num_of_centers_supply)]

    lamda_d_pdf = gaussian_pdf_2d(m, n, centers_d, normalize=True)
    lamda_d = np.array([lamda_d_pdf[node] for node in nodes])

    lamda_s_pdf = alpha * gaussian_pdf_2d(m, n, centers_s, normalize=True) + (1 - alpha) * lamda_d_pdf
    lamda_s = np.array([lamda_s_pdf[node] for node in nodes])

    lamda = np.concatenate((lamda_d, lamda_s))
    grid_adj_mat = nx.adjacency_matrix(g)
    layered_grid_adj_mat = sps.vstack((sps.hstack((0*sps.eye(m*n), grid_adj_mat)),
                                       sps.hstack((grid_adj_mat, 0*sps.eye(m*n)))))
    nodes = dict(enumerate(list(zip(nodes, ['d']*len(nodes))) + list(zip(nodes, ['s']*len(nodes)))))

    workload_decomp = grid_workload_decomposition(lamda, layered_grid_adj_mat)
    max_workload = workload_decomp[0]['workload']

    lamda_s_pdf = lamda_s_pdf*max_workload
    lamda_s = lamda_s*max_workload

    for s in workload_decomp:
        workload_decomp[s]['workload'] = workload_decomp[s]['workload']/max_workload

    supply_decomp = np.zeros((m, n))
    demand_decomp = np.zeros((m, n))
    for st in workload_decomp:
        wl = workload_decomp[st]['workload']
        for d in workload_decomp[st]['demnand_nodes']:
            demand_decomp[nodes[d][0]] = wl
        for s in workload_decomp[st]['supply_nodes']:
            supply_decomp[nodes[s][0]] = wl

    p = sps.vstack((grid_adj_mat, np.ones((1, m*n))))
    d_s_gap = np.asscalar(lamda_s.sum() - (lamda_d*0.95).sum())
    print('starting shelikovskii')
    pi = shelikhovskii(p, np.hstack((lamda_d*0.95, np.array([d_s_gap]))), lamda_s)
    r = pi[m*n:].todense()
    #print(np.min(r))
    pi = pi[:-1, :]
    max_ent_workload = np.zeros((m, n))
    rho_s = ((lamda_s - r)/lamda_s)
    for i in range(m*n):
        max_ent_workload[nodes[i][0]] = rho_s[0, i]

    fifo_ct = np.zeros((m,n))
    c = np.exp(-1*(sps_plog(pi.dot(sps.diags(1/lamda_s)))).dot(sps.diags(lamda_s)).sum(axis=1))
    rho_c = (sps.diags(1/(0.95*lamda_d)).dot(pi)).dot(rho_s.transpose())

    #print(c)

    sakasegawa_ct = []
    for i in range(m*n):
        sakasegawa_ct.append(sakasegawa(c[i,0],rho_c[i,0]))

    for i in range(m*n):
        fifo_ct[nodes[i][0]] = log(sakasegawa_ct[i])

    if plot:
        fig, ((ax1, ax2), (ax3, ax4),(ax5,ax6)) = plt.subplots(3, 2)
        ax1.imshow(lamda_d_pdf, interpolation='nearest')
        ax1.set_title('demand')
        ax2.imshow(lamda_s_pdf, interpolation='nearest')
        ax2.set_title('supply')
        ax3.imshow(demand_decomp, interpolation='nearest')
        ax3.set_title('demand decomposotion')
        ax4.imshow(supply_decomp, interpolation='nearest')
        ax4.set_title('supply decomposotion')
        ax5.imshow(max_ent_workload, interpolation='nearest')
        ax5.set_title('max_ent_workload')
        ax6.imshow(fifo_ct, interpolation='nearest')
        ax6.set_title('fifo_ct')
        plt.show()


    return dict(zip(
        ['lamda_s','lamda_d',
         'grid_adj_mat', 'nodes',
         'lamda_d_pdf', 'lamda_s_pdf',
         'demand_decomp', 'supply_decomp',
         'fifo_ct', 'max_ent_workload'],
        [lamda_s, lamda_d * 0.95,
         grid_adj_mat, nodes,
         lamda_d_pdf, lamda_s_pdf,
         demand_decomp, supply_decomp,
         fifo_ct, max_ent_workload]))


def generate_fss(m, n, rho, sparse=True, mu_rand=False, lamda_rand=True):

    if lamda_rand:
        lamda = random_unit_partition(m) * rho
    else:
        lamda = np.ones(m)*(1.0/m)
    if mu_rand:
        mu = random_unit_partition(m)
    else:
        mu = np.ones(n)*(1.0/n)

    if m >= n:
        rows = range(m) + range(m)
        cols = [i % n for i in range(m)] + [(i + 1) % n for i in range(m)]
        data = [1.0] * (2 * m)
    else:
        rows = [(i - 1) % m for i in range(n)] + [i % m for i in range(n)]
        cols = range(n) + range(n)
        data = [1.0] * (2 * n)

    for i in range(m):

        min_deg = max(ceil(n * lamda[i]), 2)
        max_deg = min(min_deg * 3, n)
        k = np.random.randint(min_deg, max_deg) - 2
        qual = fast_choice(range(n-2), k)
        for j in qual:
            rows.append(i)
            cols.append((i + 2 + j) % n)
            data.append(1.0)

    if sparse:
        Q = sps.coo_matrix((data, (rows, cols)), shape=(m, n))
        Q = Q.tocsr()
    else:
        Q = np.zeros((m, n))
        Q[rows, cols] = data

    return Q, lamda, mu


if __name__ == '__main__':

    m = 10
    n = 10
    res = undirected_grid_2d_bipartie_graph(m, n, d=3, r=3,
                                          centers_d=None, centers_s=None,
                                          num_of_centers_supply=None, num_of_centers_demand=None,
                                          l_norm=np.inf, plot=False, alpha=1.0)
    for key, val in res.iteritems():
        print(key)
        print(val)

# p = sps.vstack((g['grid_adj_mat'], np.ones((1, m*n))))
# d_s_gap = np.asscalar(g['lamda_s'].sum() - (g['lamda_d']*0.95).sum())
# lamda_d = np.hstack((g['lamda_d']*0.95, np.array([d_s_gap])))
# lamda_s = g['lamda_s']
# v = shelikhovskii(p, lamda_d, lamda_s)

