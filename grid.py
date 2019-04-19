from numpy.linalg import norm
import networkx as nx
from random import uniform
from scipy import stats as stats
import numpy as np
from solvers import grid_workload_decomposition
from scipy import sparse as sps


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


def undirected_grid_2d_bipartie_graph(m, n, d=1, r=1,
                                      centers_d=None, centers_s=None,
                                      num_of_centers_supply=None, num_of_centers_demand=None,
                                      l_norm=np.inf, plot=False, alpha=1.0):

    g = nx.empty_graph(0,None)
    rows = range(m)
    columns = range(n)

    # adding all the nodes
    g.add_nodes_from((i, j) for i in rows for j in columns)
    #
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


if __name__ == '__main__':

    m = 10
    n = 10
    res_dict = undirected_grid_2d_bipartie_graph(m, n, d=3, r=3,
                                                 centers_d=None, centers_s=None,
                                                 num_of_centers_supply=None, num_of_centers_demand=None,
                                                 l_norm=np.inf, plot=False, alpha=1.0)
    print res_dict['lamda_d_pdf']
