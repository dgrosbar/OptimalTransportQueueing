from simulators2 import online_matching_simulator, MultiSim, SimExperiment
from generators import *
from solvers import *
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from fss_utils import *
from scipy import sparse as sps
from math import ceil, exp, log
from time import time
import pandas as pd
from scipy.stats import entropy
from figures import get_file_data
from memory_profiler import profile


def k_chains(nc, ns, k, u, sparse=True, noise=0):

    q = k_chain(nc,  ns, k, sparse=True)
    mu = np.array([1.0]*ns)
    noise = np.random.uniform(low=-noise, high=noise, size=nc)
    lamda = np.array([1.0]*nc) + noise
    lamda = u * nc * lamda/lamda.sum()
    for policy in ['fifo-alis', 'max-ent']:

        matching_rates, waiting_times, idle_times = 1#queueing_simulator(lamda, mu, q, warm_up=None, sim_len=None, sims=1, policy=policy)
        matching_rates = matching_rates.observations[0]
        waiting_times = waiting_times.observations[0]
        waiting_times = waiting_times/matching_rates.sum(axis=1)
        matching_rates = lamda.sum() * matching_rates/matching_rates.sum()
        rows, cols = matching_rates.nonzero()
        edges = [(row, col) for row,col in zip(rows,cols)]
        plt.plot([str(edge) for edge in edges], [matching_rates[row, col] for row, col in edges], label='simulation' + policy)

    q = sps.csr_matrix(np.vstack((q.todense(), np.ones((1, len(mu))))))
    p = shelikhovskii(sps.csr_matrix(q), np.append(lamda, [(mu.sum()-lamda.sum())]), mu)
    plt.plot([str(edge) for edge in edges], [p[row, col] for row, col in edges], label='max entropy')
    plt.legend()
    #
    # print q[:nc, :]
    # print 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
    # print 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
    # print 'matching_rates'
    # print matching_rates
    # print '--------------------------------------'
    # print p[:nc, :].todense()
    # print 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
    # print 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
    # print 'lamda'
    # print lamda
    # print 'sum axis = 1'
    # print np.squeeze(matching_rates.sum(axis=1))
    # print '--------------------------------------'
    # print p[:nc, :].todense().sum(axis=1).T
    # print 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
    # print 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
    # print 'mu'
    # print mu
    # print 'sum axis = 0'
    # print np.squeeze(matching_rates.sum(axis=0))
    # print '--------------------------------------'
    # print p.todense().sum(axis=0)
    # print 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
    # print 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
    # print 'total sum'
    # print matching_rates.sum()
    # print '--------------------------------------'
    # print p.todense().sum()
    # #print np.abs(matching_rates - p[:nc,:]).sum()
    # print 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
    # print 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
    # print 'wating times'
    # print [str(x)[:5] for x in waiting_times]
    # print '-----------------------------------------'
    plt.show()
    # a_wt = fss_wt_approximation(q[:nc, :], p.todense().A[:nc, :], lamda, mu)
    # print [str(x)[:5] for x in a_wt]
    # print '-------------------------------'
    # print [x[0] for x in sorted(enumerate(waiting_times), key=lambda x: x[1])]
    # print [x[0] for x in sorted(enumerate(a_wt), key=lambda x: x[1])]
    # print 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
    # print 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'


def matching_market():

    n = 30
    m = 30
    d = 6
    g = supply_demand_grid(m, n, d, d, l_norm=1, alpha=0.9)
    ms = g['m'] * 0.5
    mc = g['m'] * 0.5
    mcz = np.array([0.0] * (m*n))
    msz = np.array([0.0] * (m*n))

    #
    pi_c_hat1, pi_s_hat1, pi_cz_hat1, pi_sz_hat1 = \
        fast_primal_dual_algorithm(mc, ms, mcz, msz, g['pd'], g['ps'], g['lamda_s'], g['lamda_d'], check_every=100)
    #
    dist_traveled1 = (ms + mc).multiply(pi_c_hat1).sum()
    customers_served1 = pi_c_hat1.sum()

    mcz = np.array([float(d)] * (m*n))
    msz = np.array([0.0] * (m*n))

    pi_c_hat2, pi_s_hat2, pi_cz_hat2, pi_sz_hat2 = \
        fast_primal_dual_algorithm(mc, ms, mcz, msz, g['pd'], g['ps'], g['lamda_s'], g['lamda_d'], check_every=100)

    dist_traveled2 = (ms + mc).multiply(pi_c_hat2).sum()
    customers_served2 = pi_c_hat2.sum()

    lamda = np.concatenate((g['lamda_d'], g['lamda_s']))

    q = sps.vstack((sps.hstack((sps.csr_matrix(np.zeros((m*n, m*n))), g['pd'])),
                    sps.hstack((g['pd'], sps.csr_matrix(np.zeros((m*n, m*n)))))))
    q = sps.csr_matrix(q)

    eta = np.ones(len(lamda))*(sum(lamda)/len(lamda))

    w = sps.vstack((sps.hstack((sps.csr_matrix(np.zeros((m*n, m*n))), g['pd'])),
                    sps.hstack((g['pd'], sps.csr_matrix(np.zeros((m*n, m*n)))))))

    w = sps.csr_matrix(w)

    sim_res1 = online_matching_simulator(lamda=lamda, q=q, eta=eta, abandonments=True, w=w, policy='closest',
                                         sims=1, prt=False, sim_len=10**6)

    # sim_res2 = online_matching_simulator(lamda=lamda, q=q, eta=eta, abandonments=True, w=w, policy='closest',
    #                                 sims=1, prt=False, sim_len=10**6)

    match_rates = sim_res1['matching_rates_node'].get_mean()
    match_rates_demand = match_rates[:m*n]
    match_rates_supply = match_rates[m*n:]

    abandonment_rates = sim_res1['abandonment_rates'].get_mean()
    abandonment_rates_demand = abandonment_rates[:m*n]
    abandonment_rates_supply = abandonment_rates[m*n:]

    demand_utilization_sim = match_rates_demand/(match_rates_demand + abandonment_rates_demand)
    supply_utilization_sim = match_rates_supply/(match_rates_supply + abandonment_rates_supply)

    demand_utilization_sim[np.where(match_rates_demand + abandonment_rates_demand < 10)] = np.nan
    supply_utilization_sim[np.where(match_rates_supply + abandonment_rates_supply < 10)] = np.nan

    supply_matched1 = gridify(pi_s_hat1.sum(axis=0).T, g['nodes'], m, n)
    supply_lost1 = gridify(pi_sz_hat1, g['nodes'], m, n)
    supply_utilization1 = supply_matched1/(supply_matched1 + supply_lost1)

    demand_matched1 = gridify(pi_c_hat1.sum(axis=1), g['nodes'], m, n)
    demand_lost1 = gridify(pi_cz_hat1, g['nodes'], m, n)
    demand_utilization1 = demand_matched1/(demand_matched1 + demand_lost1)

    supply_matched2 = gridify(pi_s_hat2.sum(axis=0).T, g['nodes'], m, n)
    supply_lost2 = gridify(pi_sz_hat2, g['nodes'], m, n)
    supply_utilization2 = supply_matched2/(supply_matched2 + supply_lost2)

    demand_matched2 = gridify(pi_c_hat2.sum(axis=1), g['nodes'], m, n)
    demand_lost2 = gridify(pi_cz_hat2, g['nodes'], m, n)
    demand_utilization2 = demand_matched2/(demand_matched2 + demand_lost2)

    # ax7.imshow(customer_utilization3, interpolation='nearest')
    # ax7.set_title('dt: ' + str(dist_traveled3)[:4] + ' cs: ' + str(customers_served3)[:4])
    # ax8.imshow(suppy_utilization3, interpolation='nearest')
    # ax8.set_title('supply utilization')

    fig, ((ax11, ax12, ax13, ax14),(ax21, ax22, ax23, ax24)) = plt.subplots(2, 4)

    ax11.imshow(gridify(g['lamda_d'], g['nodes'], m, n), interpolation='nearest')
    ax11.set_title('demand')
    ax21.imshow(gridify(g['lamda_s'], g['nodes'], m, n), interpolation='nearest')
    ax21.set_title('supply')

    ax12.imshow(demand_utilization1, interpolation='nearest')
    ax12.set_title('dt: ' + str(dist_traveled1)[:4] + ' cs: ' + str(customers_served1)[:4])
    ax22.imshow(supply_utilization1, interpolation='nearest')
    ax22.set_title('supply utilization')

    ax13.imshow(demand_utilization2, interpolation='nearest')
    ax13.set_title('dt: ' + str(dist_traveled2)[:4] + ' cs: ' + str(customers_served2)[:4])
    ax23.imshow(supply_utilization2, interpolation='nearest')
    ax23.set_title('supply utilization')

    ax14.imshow(gridify(demand_utilization_sim, g['nodes'], m, n), interpolation='nearest')
    ax14.set_title('demand utilization')
    ax24.imshow(gridify(supply_utilization_sim, g['nodes'], m, n), interpolation='nearest')
    ax24.set_title('supply utilization')

    plt.show()


def fifo_flaw():

    fifo_flow_exp = SimExperiment('fifo_flaw')

    for n in range(2,10, 1) + range(10, 110, 10):

        rho1 = 0.8
        rho2 = 0.8
        lamda = np.array([rho1] + n*[rho2])
        mu = np.ones(n+1)
        Q = np.diag(np.ones(n+1), 0)
        Q[1:n+1, 0] = np.ones(n)
        print Q
        rho1_dic = {'res': ['i', 'j', 'ij'], 'val': rho1, 'matrix': False}
        rho2_dic = {'res': ['i', 'j', 'ij'], 'val': rho2, 'matrix': False}
        N_dic = {'res': ['i', 'j', 'ij'], 'val': n, 'matrix': False}
        fifo_flow_exp.simulate(lamda=lamda, mu=mu, q=Q, rho1=rho1_dic, rho2=rho2_dic, sim_len=n*10**5, sims=30, N=N_dic)


def fifo_flaw2():

    fifo_flow_exp = SimExperiment('fifo_flaw13')
    sim_len = 10**5

    for n in [10, 20, 50, 100]:

        N_dic = {'res': ['i', 'j', 'ij'], 'val': n, 'matrix': False}

        for rho1 in [0.7, 0.8, 0.9, 0.95]:
            for rho2 in [0.7, 0.8, 0.9, 0.95]:

                rho1_dic = {'res': ['i', 'j', 'ij'], 'val': rho1, 'matrix': False}
                rho2_dic = {'res': ['i', 'j', 'ij'], 'val': rho2, 'matrix': False}

                lamda = np.array(n*[rho1] + n*[rho2])
                mu = np.ones(2*n)

                Q = np.vstack((np.hstack((np.diag(np.ones(n), 0), np.diag(np.ones(n), 0))),
                              np.hstack((np.zeros((n,n)), np.ones((n,n))))))
                Qp = np.vstack((Q, np.ones(Q.shape[1])))

                lamda_p = np.append(lamda, mu.sum() - lamda.sum())

                Wp_dic = dict()

                Wp_dic['plain'] = copy(Qp)

                # Wp = np.diag(np.ones(2*n) - lamda, 0).dot(Q)
                # Wp = np.diag(1.0/np.log(Q.sum(axis=1)), 0).dot(Wp)
                # Wp = np.vstack((Wp, (lamda).reshape((1, 2*n))))
                #
                # #Wp_dic['non_squared_log_div'] = copy(Wp)
                #
                # Wp = np.diag(np.ones(2*n) - lamda**2, 0).dot(Q)
                # Wp = np.diag(1.0/np.log(Q.sum(axis=1)), 0).dot(Wp)
                # Wp = np.vstack((Wp, (lamda**2).reshape((1, 2*n))))
                #
                # #Wp_dic['squared_log_div'] = copy(Wp)
                #
                # Wp = np.diag(np.ones(2*n) - lamda, 0).dot(Q)
                # Wp = np.vstack((Wp, (lamda).reshape((1, 2*n))))
                # Wp = np.diag(1.0/np.log(Qp.sum(axis=1)), 0).dot(Wp)
                #
                # #Wp_dic['non_squared_log_div_all'] = copy(Wp)
                #
                # Wp = np.diag(np.ones(2*n) - lamda**2, 0).dot(Q)
                # Wp = np.vstack((Wp, (lamda**2).reshape((1, 2*n))))
                # Wp = np.diag(1.0/np.log(Qp.sum(axis=1)), 0).dot(Wp)
                #
                # #Wp_dic['squared_log_div_all'] = copy(Wp)

                Wp = np.diag(np.ones(2*n) - lamda**2, 0).dot(Q)
                Wp = np.vstack((Wp, (lamda**2).reshape((1, 2*n))))

                Wp_dic['squared'] = copy(Wp)

                Wp = np.diag(np.ones(2*n) - lamda, 0).dot(Q)
                Wp = np.vstack((Wp, lamda.reshape((1, 2*n))))

                Wp_dic['non_squared'] = copy(Wp)

                Wp_dic['prio_0'] = np.vstack((np.hstack((np.diag(np.ones(n), 0), np.diag(2*np.ones(n), 0))),
                                              np.hstack((np.zeros((n, n)), np.ones((n, n))))))

                Wp_dic['prio_1'] = np.vstack((np.hstack((np.diag(np.ones(n), 0), np.diag(np.ones(n), 0))),
                                              np.hstack((np.zeros((n, n)), 2*np.ones((n, n))))))

                pi_ent = shelikhovskii(Qp, lamda_p, mu)
                pi_ent = pi_ent/pi_ent[:2*n, :].sum()

                pi_ent_dic = {'res': ['ij'], 'val': pi_ent[:2*n, :], 'matrix': True}

                for sim_name, Wp in Wp_dic.iteritems():

                    if sim_name == 'prio_0' or sim_name == 'prio_1':

                        fifo_flow_exp.simulate(lamda=lamda, mu=mu, w=Wp, q=Q,
                                               rho1=rho1_dic, rho2=rho2_dic, N=N_dic, pi_ent=pi_ent_dic,
                                               sim_len=n*sim_len, sims=1, sim_name=sim_name)
                    else:

                        Qp = sps.csr_matrix(Qp)
                        Wps = sps.csr_matrix(Wp)
                        Mps = 0*Wps
                        if sim_name != 'plain':

                            A,b,z, cols = transform_to_normal_form(Mps, Wps, Qp, Qp, lamda_p, mu)

                            pi_hat, duals = fast_primal_dual_algorithm(A,b,z)

                            pi_hat = pi_hat.reshape((2*n+1, 2*n))
                            pi_hat = np.divide(pi_hat, Wp, out=np.zeros_like(pi_hat), where=Wp != 0)
                            pi_hat = pi_hat/pi_hat[:2*n, :].sum()
                            pi_hat_dic = {'res': ['ij'], 'val': pi_hat[:2*n, :], 'matrix': True}
                            w = np.divide(pi_hat[:2*n, :], pi_ent[:2*n, :],
                                          out=np.zeros_like(pi_hat[:2*n, :]), where=pi_ent[:2*n, :] != 0)
                        else:
                            pi_hat_dic = pi_ent_dic
                            w = Qp

                        print '----------------------------------------------------------------------------------'
                        print n, rho1, rho2, sim_name, 'fifo'
                        print '----------------------------------------------------------------------------------'

                        fifo_flow_exp.simulate(lamda=lamda, mu=mu, w=w, q=Q,
                                               rho1=rho1_dic, rho2=rho2_dic,
                                               N=N_dic, pi_ent=pi_ent_dic, pi_hat=pi_hat_dic,
                                               sim_len=n*sim_len, sims=1, sim_name=sim_name)

                        print '----------------------------------------------------------------------------------'
                        print n, rho1, rho2, sim_name, 'max_weight'
                        print '----------------------------------------------------------------------------------'

                        fifo_flow_exp.simulate(lamda=lamda, mu=mu, w=w, q=Q,
                                               rho1=rho1_dic, rho2=rho2_dic,
                                               N=N_dic, pi_ent=pi_ent_dic, pi_hat=pi_hat_dic,
                                               sim_len=n*sim_len, sims=1, sim_name=sim_name, j_policy='max_weight')

                Q = np.vstack((np.hstack((np.diag(np.ones(n), 0), np.zeros((n, n)))),
                              np.hstack((np.zeros((n,n)), np.ones((n, n))))))

                print '----------------------------------------------------------------------------------'
                print n, rho1, rho2, 'balanced', 'fifo'
                print '----------------------------------------------------------------------------------'

                fifo_flow_exp.simulate(lamda=lamda, mu=mu, q=Q,
                                       rho1=rho1_dic, rho2=rho2_dic, N=N_dic,
                                       sim_len=n*sim_len, sims=1, sim_name='balanced')

                print '----------------------------------------------------------------------------------'
                print n, rho1, rho2, 'balanced', 'max_weight'
                print '----------------------------------------------------------------------------------'

                fifo_flow_exp.simulate(lamda=lamda, mu=mu, q=Q,
                                       rho1=rho1_dic, rho2=rho2_dic, N=N_dic,
                                       sim_len=n*sim_len, sims=1, sim_name='balanced', j_policy='max_weight')


def power_of_an_arc():

    fifo_flow_exp = SimExperiment('power_of_an_arc3')
    sim_len = 10**5

    for n in [1, 2, 5, 10, 20, 50]:

        N_dic = {'res': ['i', 'j', 'ij'], 'val': n, 'matrix': False}

        for rho1 in [0.99, .95, .9, .8, .6]:
            for rho2 in [0.99, .95, .9, .8, .6]:

                rho1_dic = {'res': ['i', 'j', 'ij'], 'val': rho1, 'matrix': False}
                rho2_dic = {'res': ['i', 'j', 'ij'], 'val': rho2, 'matrix': False}

                lamda = np.array([rho1] + n*[rho2])
                mu = np.ones(n+1)
                rho = lamda.sum()/mu.sum()

                Q = np.ones((n,n))
                Q = np.pad(Q, ((1, 0), (1, 0)), mode='constant', constant_values=(0, 0))

                Q[0, 0] += 1
                Q[0, 1] += 1

                wls, rho_m, rho_n = bipartite_workload_decomposition(Q, lamda, mu)

                Qp = sps.vstack((Q, np.ones((1, n))), format='csr')
                lamda_p = np.append(lamda, mu.sum() - lamda.sum())

                pi_ent = shelikhovskii(Qp, lamda_p, mu)
                pi_ent = pi_ent.todense().A
                rho_ent = pi_ent[:n, :].sum(axis=0)
                q_ent = pi_ent[:n, :]/(1.0 - rho_ent)
                q_ent = (q_ent.T/q_ent.sum(axis=1)).T

                pi_ent_dic = {'res': ['ij'], 'val': pi_ent, 'matrix': True}
                rho_dic = {'res': ['i', 'j', 'ij'], 'val': rho, 'matrix': False}

                Wp = np.diag(np.ones(n) - rho_n, 0).dot(Q)
                Wp = np.vstack((Wp, np.ones((1, n)) * rho_n))
                Wps = sps.csr_matrix(Wp)
                Mps = 0*sps.csr_matrix(Qp)
                Zps = Qp.dot(sps.diags(mu))
                A,b,z, cols = transform_to_normal_form(Mps, Wps, Qp, Zps, lamda_p, mu)
                pi_hat, duals = fast_primal_dual_algorithm(A,b,z)
                pi_hat = pi_hat.reshape((n+1, n))
                pi_hat = np.divide(pi_hat, Wp, out=np.zeros_like(pi_hat), where=Wp != 0)
                pi_hat = pi_hat/pi_hat[:n, :].sum()
                pi_hat = pi_hat*lamda.sum()
                rho_hat = pi_hat[:n, :].sum(axis=0)
                q_hat = pi_hat[:n, :]/(1.0 - rho_hat)
                q_hat = (q_hat.T/q_hat.sum(axis=1)).T
                pi_hat_dic = {'res': ['ij'], 'val': pi_hat[:n, :], 'matrix': True}
                w = np.divide(q_hat, q_ent, out=np.zeros_like(q_hat), where=q_ent != 0)

                w_dic = {'res': ['ij'], 'val': w, 'matrix': True}
                q_ent_dic = {'res': ['ij'], 'val': q_ent, 'matrix': True}
                q_hat_dic = {'res': ['ij'], 'val': q_hat, 'matrix': True}

                if n <= 5:
                    print Q

                print '----------------------------------------------------------------------------------'
                print n, rho1, rho2,  'fifo'
                print '----------------------------------------------------------------------------------'

                fifo_flow_exp.simulate(lamda=lamda, mu=mu, q=Q,
                                       rho1=rho1_dic, rho2=rho2_dic, rho=rho_dic,
                                       pi_ent=pi_ent_dic, pi_hat=pi_hat_dic, q_ent=q_ent_dic, q_hat=q_hat_dic, w=w_dic,
                                       N=N_dic,sim_len=n*sim_len, sims=30, sim_name=str(n))

                print '----------------------------------------------------------------------------------'
                print n, rho1, rho2, 'max_weight'
                print '----------------------------------------------------------------------------------'

                fifo_flow_exp.simulate(lamda=lamda, mu=mu,q=Q,
                                       rho1=rho1_dic, rho2=rho2_dic, rho=rho_dic,
                                       pi_ent=pi_ent_dic, pi_hat=pi_hat_dic, q_ent=q_ent_dic, q_hat=q_hat_dic, w=w_dic,
                                       N=N_dic, sim_len=n*sim_len, sims=30, sim_name=str(n), j_policy='max_weight')


def flexible_queueing_system_with_utility(m, n):

    max_rho = 2.0
    mu = None
    lamda = None
    Q = None
    sparse = None
    rho_n = None

    while max_rho > 1.000000000001:

        Q, lamda, mu = generate_fss(m, n, 1.0)

        sparse = sps.isspmatrix(Q)

        wls, rho_m, rho_n = bipartite_workload_decomposition(Q, lamda, mu)

        max_rho = np.amax(rho_n)
        print 'max_rho', max_rho

    simulator = SimExperiment('optimal_transport_queueing')

    M = Q.multiply(np.random.uniform(0,1, Q.shape))

    for rho in [0.95, 0.99]:  # np.array([0.5 + 0.1*i for i in range(5)] + [0.95, 0.97, 0.99]):

        lamda_p = np.append(lamda*rho, 1.0 - rho)
        lamda_p = lamda_p/lamda_p.sum()
        Qp = sps.vstack((Q, np.ones(n)), format='csr') if sparse else np.vstack((Q, np.ones(n)))
        Mp = sps.vstack((M, np.ones(n)), format='csr') if sparse else np.vstack((M, np.ones(n)))
        print len(Mp.data)
        #Mp = sps.diags(np.array([1.]*(m) +[0.]), format='csr').dot(Mp) if sparse else Mp
        print len(Mp.data)

        wm = np.ones(n)*rho**2

        #Z = sps.vstack((Q.dot(sps.diags(mu)), zm), format='csr') if sparse else np.vstack((Q*mu, zm))

        Z = Qp.dot(sps.diags(mu, format='csr')) if sparse else Qp.dot(mu)

        W_fifo = Qp

        W_rho_sq = sps.vstack((Q*(1.0 - rho**2), wm), format='csr') if sparse else np.vstack((Q*(1.0 - rho**2), wm))

        # W_rho_s_sq = sps.vstack((Q.dot(sps.diags(np.ones(n) - rho_n**2)), rho_n**2), format='csr') if sparse else \
        #     np.vstack((Q*(np.ones(n) - rho_n**2), rho_n**2))

        W_rho = sps.vstack((Q*(1.0 - rho), wm), format='csr') if sparse else np.vstack((Q*(1.0 - rho), wm))

        # W_rho_s = sps.vstack((Q.dot(sps.diags(np.ones(n) - rho_n)), rho_n**2), format='csr') if sparse else \
        #      np.vstack((Q*(np.ones(n) - rho_n), rho_n))

        #Qp = sps.vstack((Q.T, np.ones(m)), format='csr') if sparse else np.vstack((Q.T, np.ones(m)))

        pi_ent = shelikhovskii(Qp, lamda_p, mu)
        r_ent = pi_ent[m, :]
        pi_ent = pi_ent[:m, :]
        pi_ent = pi_ent/pi_ent.sum()
        pi_ent = pi_ent.todense().A

        pi_ent_dic = {'res': ['ij'], 'val': pi_ent[:m, :], 'matrix': True}
        m_dic = {'res': ['ij'], 'val': M.todense().A[:m, :], 'matrix': True}
        rho_dic = {'res': ['i', 'j', 'ij'], 'val': rho, 'matrix': False}

        for W, name in zip([W_fifo, W_rho_sq, W_rho], ['W_fifo', 'W_rho_sq', 'W_rho']):
            for phi in [0.1, 0.5, 1., 2., 5.]:
                phi_dic = {'res': ['i', 'j', 'ij'], 'val': phi, 'matrix': False}
                A, b, z, act_rows = transform_to_normal_form(phi*Mp, W, Qp, Z, lamda_p, mu)
                pi_opt, duals = fast_primal_dual_algorithm(A, b, z, act_rows=act_rows)
                pi_opt = pi_opt.reshape((m+1, n))[:m, :]
                w = np.divide(pi_opt, pi_ent, out=np.zeros_like(pi_opt), where=pi_ent != 0)
                print name
                print sps.csr_matrix(w)
                pi_opt_dic = {'res': ['ij'], 'val': w[:m, :], 'matrix': True}
                simulator.simulate(lamda=lamda*rho, mu=mu, q=Q, w=w, i_policy='rand', j_policy='rand', sims=1,
                                   sim_name=name, rho=rho_dic, pi_ent=pi_ent_dic, pi_opt=pi_opt_dic, phi=phi_dic, M=m_dic)
                simulator.simulate(lamda=lamda*rho, mu=mu, q=Q, w=w,i_policy='alis', j_policy='fifo', sims=1,
                                   sim_name=name, rho=rho_dic, pi_ent=pi_ent_dic, pi_opt=pi_opt_dic, phi=phi_dic, M=m_dic)
                simulator.simulate(lamda=lamda*rho, mu=mu, q=Q, w=w, i_policy='alis', j_policy='max_weight', sims=1,
                                   sim_name=name, rho=rho_dic, pi_ent=pi_ent_dic, pi_opt=pi_opt_dic, phi=phi_dic, M=m_dic)
                simulator.simulate(lamda=lamda*rho, mu=mu, q=Q, w=Q, i_policy='rand', j_policy='rand', sims=1,
                                   sim_name=name, rho=rho_dic, pi_ent=pi_ent_dic, pi_opt=pi_opt_dic, phi=phi_dic, M=m_dic)
                simulator.simulate(lamda=lamda*rho, mu=mu, q=Q, w=Q, i_policy='alis', j_policy='fifo', sims=1,
                                   sim_name=name, rho=rho_dic, pi_ent=pi_ent_dic, pi_opt=pi_opt_dic, phi=phi_dic, M=m_dic)
                simulator.simulate(lamda=lamda*rho, mu=mu, q=Q, w=Q, i_policy='alis', j_policy='max_weight', sims=1,
                                   sim_name=name, rho=rho_dic, pi_ent=pi_ent_dic, pi_opt=pi_opt_dic, phi=phi_dic, M=m_dic)
    return



    # def experiment(Me, We, Qpe, Ze, lamda_e, mu_e, policy):
    #
    #     A, b, z, act_rows = transform_to_normal_form(Me, We, Qpe, Ze, lamda_e, mu_e)
    #     p, duals = fast_primal_dual_algorithm(A, b, z, act_rows=act_rows)
    #     sim_reses = queueing_simulator(lamda, mu, Qp, sims=1, policy=policy, w=p)
    #     return p, sim_reses, duals


    # M = sps.vstack((Q, np.ones(n)), format='csr') if sparse else np.vstack((Q, np.ones(n)))
    #
    A, b, z, act_rows = transform_to_normal_form(0*Qp, Qp, Qp, Qp, lamda_p, mu)
    #
    # p0, duals = fast_primal_dual_algorithm(A, b, z, act_rows=act_rows)
    #
    # A, b, z, act_rows = transform_to_normal_form(0*Qp, W10, Qp, Z, lamda_p, mu)
    #
    # p1, duals = fast_primal_dual_algorithm(A, b, z, act_rows=act_rows)
    #
    # A, b, z, act_rows = transform_to_normal_form(0*Qp, W20, Qp, Z, lamda_p, mu)
    #
    # p2, duals = fast_primal_dual_algorithm(A, b, z, act_rows=act_rows)
    #
    # A, b, z, act_rows = transform_to_normal_form(0*Qp, W11, Qp, Z, lamda_p, mu)
    #
    # p3, duals = fast_primal_dual_algorithm(A, b, z, act_rows=act_rows)
    #
    # A, b, z, act_rows = transform_to_normal_form(0*Qp, W21, Qp, Z, lamda_p, mu)
    #
    # p4, duals = fast_primal_dual_algorithm(A, b, z, act_rows=act_rows)

    # p0 = p0.reshape((m+1, n))
    # p1 = p1.reshape((m+1, n))
    # p2 = p2.reshape((m+1, n))
    # p3 = p3.reshape((m+1, n))
    # p4 = p4.reshape((m+1, n))

    plt.plot(p0[m, :], label='0')
    plt.plot(p1[m, :], label='1')
    plt.plot(p2[m, :], label='2')
    plt.plot(p3[m, :], label='3')
    plt.plot(p4[m, :], label='4')
    plt.legend()
    plt.show()

    # print 'p0'
    # print p0.reshape((m+1, n))
    # print 'p1'
    # print p1.reshape((m+1, n))
    # print 'p2'
    # print p2.reshape((m+1, n))


    fifo_alis_u = queueing_simulator(lamda, mu, Qp, warm_up=None, sim_len=None, sims=1, policy='fifo_alis', w=p)
    max_weight_u = queueing_simulator(lamda, mu, Qp, warm_up=None, sim_len=None, sims=1, policy='max_weight', w=p)



    A, b, z = transform_to_normal_form(Qp, W, Qp, Z, lamda_p, mu)
    p, duals = fast_primal_dual_algorithm(A, b, z).reshape((m+1, n))
    fifo_alis_u = queueing_simulator(lamda, mu, Qp, warm_up=None, sim_len=None, sims=1, policy='fifo_alis', w=p)
    max_weight_u = queueing_simulator(lamda, mu, Qp, warm_up=None, sim_len=None, sims=1, policy='max_weight', w=p)

    # matching_rates = fifo_alis['matching_rates'].observations[0]
    # print matching_rates
    # waiting_times = fifo_alis['waiting_times'].observations[0]
    # waiting_times = waiting_times/matching_rates.sum(axis=1)
    # matching_rates = matching_rates/matching_rates.sum()
    # rows, cols = matching_rates.nonzero()
    # edges = [(row, col) for row,col in zip(rows, cols)]
    # plt.plot([str(edge) for edge in edges], [matching_rates[row, col] for row, col in edges], label='simulation')
    # plt.plot([str(edge) for edge in edges], [p[row, col]*(1.0/rho) for row, col in edges], label='max entropy')
    # plt.legend()
    # plt.show()


def n_system():

    qual_mat = sps.csr_matrix([[1.0, 1.0], [0.0, 1.0]])
    qual_mat_p = sps.csr_matrix([[1.0, 1.0], [0.0, 1.0], [1.0, 1.0]])
    mu = np.array([1.0, 1.0])
    u_equiv = []
    # lam2 = []
    # u2 = []
    # pi_12 = []
    # pi_11 = []
    # p_ent_12 = []
    # p_fifo_12 = []
    # o_shli = []
    for lam1 in np.arange(0.4, 2.0, 0.25):
        p12_sim = []
        p12_ent = []
        p12_aw = []
        p12_aw_mrc = []
        lam2_range = []
        for lam2 in np.arange(0.3, min(2.0-lam1,0.95), 0.1):
            lam2_range.append(lam2)
            lamda = np.array([lam1, lam2])
            p12_ent.append((lam1 * (1.0 - lam2)/(2.0 - lam2))/(lam1 + lam2))
            p12_aw.append((2.0/(lam1 + lam2)) * matching_rate_calculator_n_sys(lam1*0.5, lam2*0.5, 0.5))
            _ , r = matching_rate_calculator(lamda, mu, qual_mat, check_feasibility=False)
            p12_aw_mrc.append(r[0, 1]/r.sum())
            matching_rates, waiting_times, idle_times = 1 #queueing_simulator(lamda, mu, qual_mat, warm_up=None, sim_len=None, sims=1)
            p = shelikhovskii(qual_mat_p,
                              np.append(lamda,[(mu.sum()-lamda.sum())]), mu)
            matching_rates = matching_rates.observations[0]
            waiting_times = waiting_times.observations[0]
            waiting_times = waiting_times/matching_rates.sum(axis=1)
            matching_rates = matching_rates/matching_rates.sum()
            p12_sim.append(matching_rates[0, 1])
            # lam2.append(i)
            #
            # print 'waiting times'
            # print waiting_times
            # print waiting_times[1]/(1.0 + waiting_times[1])
            print '--------------------------------------------------------'
        # plt.plot(lam2, u_equiv, label='u_euiv')
        plt.plot(lam2_range, p12_ent, label='ent', linewidth=3.0)
        plt.plot(lam2_range, p12_sim, label='sim')
        plt.plot(lam2_range, p12_aw_mrc, label='aw_mrc')
        plt.plot(lam2_range, p12_aw, label='aw')
        plt.legend()
        plt.show()


def increasing_system():

    sim_len = 10**5
    simulator = SimExperiment('increasing_N_q2')
    for rho in np.arange(.90, .55, -0.05):
        for k in [30, 2, 3, 5, 10,  50, 100]:

            N_dic = {'res': ['i', 'j', 'ij'], 'val': k, 'matrix': False}

            Q = np.tril(np.ones((k,k)))
            lamda = np.ones(k)
            mu = np.ones(k)

            lamda_p = np.append(lamda*rho, k*(1.0 - rho))
            # lamda_p = lamda_p/lamda_p.sum()
            # mu = mu/mu.sum()
            Qp = sps.vstack((Q, np.ones((1, k))), format='csr')

            pi_ent = shelikhovskii(Qp, lamda_p, mu)
            pi_ent = pi_ent.todense().A
            rho_ent = pi_ent[:k, :].sum(axis=0)
            q_ent = pi_ent[:k, :]/(1.0 - rho_ent)
            q_ent = (q_ent.T/q_ent.sum(axis=1)).T


            #print pi_ent

            pi_ent_dic = {'res': ['ij'], 'val': pi_ent, 'matrix': True}
            rho_dic = {'res':['i', 'j', 'ij'], 'val': rho, 'matrix': False}

            Wp = np.diag(np.ones(k) - rho, 0).dot(Q)
            Wp = np.vstack((Wp, np.ones((1,k))*rho))
            Wps = sps.csr_matrix(Wp)
            Mps = 0*sps.csr_matrix(Qp)
            A,b,z, cols = transform_to_normal_form(Mps, Wps, Qp, Qp, lamda_p, mu)
            pi_hat, duals = fast_primal_dual_algorithm(A,b,z)
            pi_hat = pi_hat.reshape((k+1, k))
            pi_hat = np.divide(pi_hat, Wp, out=np.zeros_like(pi_hat), where=Wp != 0)
            pi_hat = pi_hat/pi_hat[:k, :].sum()
            pi_hat = pi_hat*rho*k
            rho_hat = pi_hat[:k, :].sum(axis=0)
            q_hat = pi_hat[:k, :]/(1.0 - rho_hat)
            q_hat = (q_hat.T/q_hat.sum(axis=1)).T
            pi_hat_dic = {'res': ['ij'], 'val': pi_hat[:k, :], 'matrix': True}
            w = np.divide(q_hat, q_ent, out=np.zeros_like(q_hat), where=q_ent != 0)
            w_prio = np.flipud(np.fliplr(np.triu(np.arange(k**2).reshape((k,k)),0)))

            print '----------------------------------------------------------------------------------'
            print k, rho, 'fifo', 'rho_weighted'
            print '----------------------------------------------------------------------------------'

            simulator.simulate(lamda=lamda*rho, mu=mu, w=w, q=Q, rho=rho_dic, N=N_dic, pi_ent=pi_ent_dic,pi_hat=pi_hat_dic,
                               sim_len=k*sim_len, sims=1, sim_name='rho_weighted_FIFO')

            print '----------------------------------------------------------------------------------'
            print k,rho, 'max_weight', 'rho_weighted'
            print '----------------------------------------------------------------------------------'

            simulator.simulate(lamda=lamda*rho, mu=mu, w=w, q=Q, rho=rho_dic, N=N_dic, pi_ent=pi_ent_dic,pi_hat=pi_hat_dic,
                               sim_len=k*sim_len, sims=1, sim_name='rho_weighted_MW',
                               j_policy='max_weight')

            print '----------------------------------------------------------------------------------'
            print k, rho, 'fifo', 'plain'
            print '----------------------------------------------------------------------------------'

            simulator.simulate(lamda=lamda*rho, mu=mu,  q=Q, rho=rho_dic, N=N_dic, pi_ent=pi_ent_dic,
                               pi_hat=pi_hat_dic, sim_len=k*sim_len, sims=1, sim_name='plain_FIFO')

            print '----------------------------------------------------------------------------------'
            print k,rho, 'max_weight', 'plain'
            print '----------------------------------------------------------------------------------'

            simulator.simulate(lamda=lamda*rho, mu=mu, q=Q, rho=rho_dic, N=N_dic, pi_ent=pi_ent_dic, pi_hat=pi_hat_dic,
                               sim_len=k*sim_len, sims=1, sim_name='plain_MW', j_policy='max_weight')

            print '----------------------------------------------------------------------------------'
            print k,rho, 'prio', 'plain'
            print '----------------------------------------------------------------------------------'

            simulator.simulate(lamda=lamda*rho, w=w_prio, mu=mu, q=Q, rho=rho_dic, N=N_dic,
                               pi_ent=pi_ent_dic, pi_hat=pi_hat_dic, sim_len=k*sim_len, sims=1, sim_name='plain_prio',
                               j_policy='prio', i_policy='prio')


def increasing_system_weight_calibration():

    sim_len = 10**5
    simulator = SimExperiment('increasing_N_q2')
    for rho in np.arange(.90, .55, -0.05):
        for k in [30, 2, 3, 5, 10,  50, 100]:

            N_dic = {'res': ['i', 'j', 'ij'], 'val': k, 'matrix': False}

            Q = np.tril(np.ones((k,k)))
            lamda = np.ones(k)
            mu = np.ones(k)

            lamda_p = np.append(lamda*rho, k*(1.0 - rho))
            # lamda_p = lamda_p/lamda_p.sum()
            # mu = mu/mu.sum()
            Qp = sps.vstack((Q, np.ones((1, k))), format='csr')

            pi_ent = shelikhovskii(Qp, lamda_p, mu)
            pi_ent = pi_ent.todense().A
            rho_ent = pi_ent[:k, :].sum(axis=0)
            q_ent = pi_ent[:k, :]/(1.0 - rho_ent)
            q_ent = (q_ent.T/q_ent.sum(axis=1)).T


            #print pi_ent

            pi_ent_dic = {'res': ['ij'], 'val': pi_ent, 'matrix': True}
            rho_dic = {'res':['i', 'j', 'ij'], 'val': rho, 'matrix': False}

            Wp = np.diag(np.ones(k) - rho, 0).dot(Q)
            Wp = np.vstack((Wp, np.ones((1,k))*rho))
            Wps = sps.csr_matrix(Wp)
            Mps = 0*sps.csr_matrix(Qp)
            Zps = Qp.dot(sps.diags(mu))
            A,b,z, cols = transform_to_normal_form(Mps, Wps, Qp, Zps, lamda_p, mu)
            pi_hat, duals = fast_primal_dual_algorithm(A,b,z)
            pi_hat = pi_hat.reshape((k+1, k))
            pi_hat = np.divide(pi_hat, Wp, out=np.zeros_like(pi_hat), where=Wp != 0)
            pi_hat = pi_hat/pi_hat[:k, :].sum()
            pi_hat = pi_hat*rho*k
            rho_hat = pi_hat[:k, :].sum(axis=0)
            q_hat = pi_hat[:k, :]/(1.0 - rho_hat)
            q_hat = (q_hat.T/q_hat.sum(axis=1)).T
            pi_hat_dic = {'res': ['ij'], 'val': pi_hat[:k, :], 'matrix': True}
            w = np.divide(q_hat, q_ent, out=np.zeros_like(q_hat), where=q_ent != 0)
            w_prio = np.flipud(np.fliplr(np.triu(np.arange(k**2).reshape((k,k)),0)))

            print '----------------------------------------------------------------------------------'
            print k, rho, 'fifo', 'rho_weighted'
            print '----------------------------------------------------------------------------------'

            simulator.simulate(lamda=lamda*rho, mu=mu, w=w, q=Q, rho=rho_dic, N=N_dic, pi_ent=pi_ent_dic,pi_hat=pi_hat_dic,
                               sim_len=k*sim_len, sims=1, sim_name='rho_weighted_FIFO')

            print '----------------------------------------------------------------------------------'
            print k,rho, 'max_weight', 'rho_weighted'
            print '----------------------------------------------------------------------------------'

            simulator.simulate(lamda=lamda*rho, mu=mu, w=w, q=Q, rho=rho_dic, N=N_dic, pi_ent=pi_ent_dic,pi_hat=pi_hat_dic,
                               sim_len=k*sim_len, sims=1, sim_name='rho_weighted_MW',
                               j_policy='max_weight')

            print '----------------------------------------------------------------------------------'
            print k, rho, 'fifo', 'plain'
            print '----------------------------------------------------------------------------------'

            simulator.simulate(lamda=lamda*rho, mu=mu,  q=Q, rho=rho_dic, N=N_dic, pi_ent=pi_ent_dic,
                               pi_hat=pi_hat_dic, sim_len=k*sim_len, sims=1, sim_name='plain_FIFO')

            print '----------------------------------------------------------------------------------'
            print k,rho, 'max_weight', 'plain'
            print '----------------------------------------------------------------------------------'

            simulator.simulate(lamda=lamda*rho, mu=mu, q=Q, rho=rho_dic, N=N_dic, pi_ent=pi_ent_dic, pi_hat=pi_hat_dic,
                               sim_len=k*sim_len, sims=1, sim_name='plain_MW', j_policy='max_weight')

            print '----------------------------------------------------------------------------------'
            print k,rho, 'prio', 'plain'
            print '----------------------------------------------------------------------------------'

            simulator.simulate(lamda=lamda*rho, w=w_prio, mu=mu, q=Q, rho=rho_dic, N=N_dic,
                               pi_ent=pi_ent_dic, pi_hat=pi_hat_dic, sim_len=k*sim_len, sims=1, sim_name='plain_prio',
                               j_policy='prio', i_policy='prio')


def k_chains_exp():

    crp = False

    m = 50
    n = 50
    sim_len = 10**5
    num = 0
    simulator = SimExperiment('k_chains_id_servers3')

    for exp_num in range(6, 30, 1):

        exp_dic = {'res': ['i', 'j', 'ij'], 'val': exp_num, 'matrix': False}

        for k, max_rho in zip([3, 5, 7, 10], [1.5, 1.2, 1.1, 1.1]):

            chain_len_dic = {'res': ['i', 'j', 'ij'], 'val': k, 'matrix': False}
            m_dic = {'res': ['i', 'j', 'ij'], 'val': m, 'matrix': False}
            n_dic = {'res': ['i', 'j', 'ij'], 'val': n, 'matrix': False}
            w_dic = {'res': ['i', 'j', 'ij'], 'val': 'weighted', 'matrix': False}
            nw_dic = {'res': ['i', 'j', 'ij'], 'val': 'not_weighted', 'matrix': False}

            Q = k_chain(m,  n, k, sparse=True)
            Q = Q.todense().A
            cur_max_rho = max_rho + 1.
            lamda = []
            mu = []

            while cur_max_rho >= max_rho:

                lamda = random_unit_partition(m)
                mu = np.ones(n)*(1./n)

                wls, rho_m, rho_n = bipartite_workload_decomposition(Q, lamda, mu)
                cur_max_rho = np.amax(rho_n)
                print cur_max_rho
                lamda = (1./cur_max_rho) * lamda
                wls, rho_m, rho_n = bipartite_workload_decomposition(Q, lamda, mu)

            for rho in [.99, .95, .9, .8, .6]:

                lamda_p = np.append(lamda*rho, mu.sum() - rho*lamda.sum())
                Qp = sps.vstack((Q, np.ones((1, n))), format='csr')
                print 'starting shelikhovskii'

                pi_ent = shelikhovskii(Qp, lamda_p, mu)
                pi_ent = pi_ent.todense().A
                print 'pi_ent_sums'
                print pi_ent[:m,:].sum(axis=1) - lamda*rho
                print pi_ent.sum(axis=0) - mu
                q_ent = pi_ent[:m, :]/pi_ent[m, :]
                q_ent_i = (q_ent.T/q_ent.sum(axis=1)).T
                q_ent_j = q_ent/q_ent.sum(axis=0)

                pi_ent_dic = {'res': ['ij'], 'val': pi_ent, 'matrix': True}
                rho_dic = {'res': ['i', 'j', 'ij'], 'val': rho, 'matrix': False}

                Wp = np.diag(np.ones(n) - rho_n*rho, 0).dot(Q)
                Wp = np.vstack((Wp, rho_n*rho))
                Wps = sps.csr_matrix(Wp)
                Mps = 0*sps.csr_matrix(Qp)
                Zps = Qp.dot(sps.diags(mu))

                A,b,z, cols = transform_to_normal_form(Mps, Wps, Qp, Zps, lamda_p, mu)
                pi_hat, duals = fast_primal_dual_algorithm(A, b, z, max_iter=10**9)
                pi_hat = pi_hat.reshape((m+1, n))
                pi_hat = np.divide(pi_hat, Wp, out=np.zeros_like(pi_hat), where=Wp != 0)
                print 'pi_hat_sums'
                print pi_hat[:m, :].sum(axis=1) - lamda*rho
                print pi_hat.sum(axis=0) - mu

                q_hat = pi_hat[:m, :]/pi_hat[m, :]
                q_hat_i = (q_hat.T/q_hat.sum(axis=1)).T
                q_hat_j = q_hat/q_hat.sum(axis=0)
                pi_hat_dic = {'res': ['ij'], 'val': pi_hat[:m, :], 'matrix': True}
                w_i = np.divide(q_hat_i, q_ent_i, out=np.zeros_like(q_hat_i), where=q_ent_i != 0)
                w_j = np.divide(q_hat_j, q_ent_j, out=np.zeros_like(q_hat_j), where=q_ent_i != 0)

                print '----------------------------------------------------------------------------------'
                print 1, exp_num, k, rho, 'fifo', 'rho_weighted_i'
                print '----------------------------------------------------------------------------------'

                simulator.simulate(lamda=lamda*rho, mu=mu, w_i=w_i, w_j=w_i, q=Q,
                                   rho=rho_dic, n=n_dic, m=m_dic, pi_ent=pi_ent_dic, pi_hat=pi_hat_dic, exp_num=exp_dic,
                                   chain_len=chain_len_dic, weighted=w_dic,
                                   sim_len=m*sim_len, sims=1, sim_name='rho_weighted_i_FIFO')

                print '----------------------------------------------------------------------------------'
                print 2, exp_num, k, rho, 'fifo', 'rho_weighted_j'
                print '----------------------------------------------------------------------------------'

                simulator.simulate(lamda=lamda*rho, mu=mu, w_i=w_j, w_j=w_j, q=Q,
                                   rho=rho_dic, n=n_dic, m=m_dic, pi_ent=pi_ent_dic, pi_hat=pi_hat_dic, exp_num=exp_dic,
                                   chain_len=chain_len_dic, weighted=w_dic,
                                   sim_len=m*sim_len, sims=1, sim_name='rho_weighted_j_FIFO')

                print '----------------------------------------------------------------------------------'
                print 3, exp_num, k, rho, 'fifo', 'rho_weighted_ij'
                print '----------------------------------------------------------------------------------'

                simulator.simulate(lamda=lamda*rho, mu=mu, w_i=w_i, w_j=w_j, q=Q,
                                   rho=rho_dic, n=n_dic, m=m_dic, pi_ent=pi_ent_dic, pi_hat=pi_hat_dic, exp_num=exp_dic,
                                   chain_len=chain_len_dic, weighted=w_dic,
                                   sim_len=m*sim_len, sims=1, sim_name='rho_weighted_ij_FIFO')

                print '----------------------------------------------------------------------------------'
                print 4, exp_num, k, rho, 'fifo', 'plain'
                print '----------------------------------------------------------------------------------'

                simulator.simulate(lamda=lamda*rho, mu=mu,  q=Q,
                                   rho=rho_dic, n=n_dic, m=m_dic, pi_ent=pi_ent_dic, pi_hat=pi_hat_dic, exp_num=exp_dic,
                                   chain_len=chain_len_dic, weighted=nw_dic,
                                   sim_len=m*sim_len, sims=1, sim_name='plain_FIFO')

                print '----------------------------------------------------------------------------------'
                print 5, exp_num, k, rho, 'max_weight', 'rho_weighted_i'
                print '----------------------------------------------------------------------------------'

                simulator.simulate(lamda=lamda*rho, mu=mu, w_i=w_i, w_j=w_i, q=Q,
                                   rho=rho_dic, n=n_dic, m=m_dic, pi_ent=pi_ent_dic, pi_hat=pi_hat_dic, exp_num=exp_dic,
                                   chain_len=chain_len_dic, weighted=w_dic,
                                   sim_len=m*sim_len, sims=1, sim_name='rho_weighted_i_MW',
                                   j_policy='max_weight')

                print '----------------------------------------------------------------------------------'
                print 6, exp_num, k, rho, 'max_weight', 'rho_weighted_i'
                print '----------------------------------------------------------------------------------'

                simulator.simulate(lamda=lamda*rho, mu=mu, w_i=w_j, w_j=w_j, q=Q,
                                   rho=rho_dic, n=n_dic, m=m_dic, pi_ent=pi_ent_dic, pi_hat=pi_hat_dic, exp_num=exp_dic,
                                   chain_len=chain_len_dic, weighted=w_dic,
                                   sim_len=m*sim_len, sims=1, sim_name='rho_weighted_j_MW',
                                   j_policy='max_weight')

                print '----------------------------------------------------------------------------------'
                print 7, exp_num, k, rho, 'max_weight', 'rho_weighted_ij'
                print '----------------------------------------------------------------------------------'

                simulator.simulate(lamda=lamda*rho, mu=mu, w_i=w_i, w_j=w_j, q=Q,
                                   rho=rho_dic, n=n_dic, m=m_dic, pi_ent=pi_ent_dic, pi_hat=pi_hat_dic, exp_num=exp_dic,
                                   chain_len=chain_len_dic, weighted=w_dic,
                                   sim_len=m*sim_len, sims=1, sim_name='rho_weighted_ij_MW',
                                   j_policy='max_weight')

                print '----------------------------------------------------------------------------------'
                print 8, exp_num, k, rho, 'max_weight', 'plain'
                print '----------------------------------------------------------------------------------'

                simulator.simulate(lamda=lamda*rho, mu=mu, q=Q,
                                   rho=rho_dic, n=n_dic, m=m_dic, pi_ent=pi_ent_dic, pi_hat=pi_hat_dic, exp_num=exp_dic,
                                   chain_len=chain_len_dic, weighted=nw_dic,
                                   sims=1, sim_name='plain_MW', j_policy='max_weight', sim_len=m*sim_len)


def k_chains_exp_sim_len():

    crp = False

    m = 50
    n = 50
    sim_len = 10**5
    num = 0
    simulator = SimExperiment('sim_len_exp2')

    for exp_num in range(10):

        exp_dic = {'res': ['i', 'j', 'ij'], 'val': exp_num, 'matrix': False}

        for k, max_rho in zip([10], [1.1]):

            chain_len_dic = {'res': ['i', 'j', 'ij'], 'val': k, 'matrix': False}
            m_dic = {'res': ['i', 'j', 'ij'], 'val': m, 'matrix': False}
            n_dic = {'res': ['i', 'j', 'ij'], 'val': n, 'matrix': False}
            w_dic = {'res': ['i', 'j', 'ij'], 'val': 'weighted', 'matrix': False}
            nw_dic = {'res': ['i', 'j', 'ij'], 'val': 'not_weighted', 'matrix': False}

            Q = k_chain(m,  n, k, sparse=True)
            Q = Q.todense().A
            cur_max_rho = max_rho + 1.
            lamda = []
            mu = []

            while cur_max_rho >= max_rho:

                lamda = random_unit_partition(m)
                mu = (1./n)*np.ones(n)

                wls, rho_m, rho_n = bipartite_workload_decomposition(Q, lamda, mu)
                cur_max_rho = np.amax(rho_n)
                print cur_max_rho
                lamda = (1./cur_max_rho) * lamda
                wls, rho_m, rho_n = bipartite_workload_decomposition(Q, lamda, mu)

            for rho in [.99]:

                lamda_p = np.append(lamda*rho, mu.sum() - rho*lamda.sum())
                Qp = sps.vstack((Q, np.ones((1, n))), format='csr')
                print 'starting shelikhovskii'

                pi_ent = shelikhovskii(Qp, lamda_p, mu)
                pi_ent = pi_ent.todense().A
                print 'pi_ent_sums'
                print pi_ent[:m,:].sum(axis=1) - lamda*rho
                print pi_ent.sum(axis=0) - mu
                q_ent = pi_ent[:m, :]/pi_ent[m, :]
                q_ent_i = (q_ent.T/q_ent.sum(axis=1)).T
                q_ent_j = q_ent/q_ent.sum(axis=0)

                pi_ent_dic = {'res': ['ij'], 'val': pi_ent, 'matrix': True}
                rho_dic = {'res': ['i', 'j', 'ij'], 'val': rho, 'matrix': False}

                Wp = np.diag(np.ones(n) - rho_n*rho, 0).dot(Q)
                Wp = np.vstack((Wp, rho_n*rho))
                Wps = sps.csr_matrix(Wp)
                Mps = 0*sps.csr_matrix(Qp)
                Zps = Qp.dot(sps.diags(mu))

                A,b,z, cols = transform_to_normal_form(Mps, Wps, Qp, Zps, lamda_p, mu)
                pi_hat, duals = fast_primal_dual_algorithm(A, b, z, max_iter=10**9)
                pi_hat = pi_hat.reshape((m+1, n))
                pi_hat = np.divide(pi_hat, Wp, out=np.zeros_like(pi_hat), where=Wp != 0)
                print 'pi_hat_sums'
                print pi_hat[:m, :].sum(axis=1) - lamda*rho
                print pi_hat.sum(axis=0) - mu

                q_hat = pi_hat[:m, :]/pi_hat[m, :]
                q_hat_i = (q_hat.T/q_hat.sum(axis=1)).T
                q_hat_j = q_hat/q_hat.sum(axis=0)
                pi_hat_dic = {'res': ['ij'], 'val': pi_hat[:m, :], 'matrix': True}
                w_i = np.divide(q_hat_i, q_ent_i, out=np.zeros_like(q_hat_i), where=q_ent_i != 0)
                w_j = np.divide(q_hat_j, q_ent_j, out=np.zeros_like(q_hat_j), where=q_ent_i != 0)

                print '----------------------------------------------------------------------------------'
                print 1, exp_num, k, rho, 'fifo', 'rho_weighted_i'
                print '----------------------------------------------------------------------------------'

                simulator.simulate(lamda=lamda*rho, mu=mu, w_i=w_i, w_j=w_i, q=Q,
                                   rho=rho_dic, n=n_dic, m=m_dic, pi_ent=pi_ent_dic, pi_hat=pi_hat_dic, exp_num=exp_dic,
                                   chain_len=chain_len_dic, weighted=w_dic,
                                   sim_len=m*10**5, sims=1, sim_name='5,000,000')

                print '----------------------------------------------------------------------------------'
                print 1, exp_num, k, rho, 'fifo', 'rho_weighted_i'
                print '----------------------------------------------------------------------------------'

                simulator.simulate(lamda=lamda*rho, mu=mu, w_i=w_i, w_j=w_i, q=Q,
                                   rho=rho_dic, n=n_dic, m=m_dic, pi_ent=pi_ent_dic, pi_hat=pi_hat_dic, exp_num=exp_dic,
                                   chain_len=chain_len_dic, weighted=w_dic,
                                   sim_len=10**6, sims=1, sim_name='1,000,000')


def k_grid_exp():

    crp = False

    m = 50
    n = 50
    sim_len = 10**5

    simulator = SimExperiment('k_grids_exp')

    grid = undirected_grid_2d_bipartie_graph()

    for exp in range(30):

        exp_dic = {'res': ['i', 'j', 'ij'], 'val': exp, 'matrix': False}

        for rho in [.99, .95, .9, .8, .6]:

            for k, max_rho in zip([1, 2, 5, 10], [1.8, 1.5, 1.5, 1.1]):

                chain_len_dic = {'res': ['i', 'j', 'ij'], 'val': k, 'matrix': False}
                m_dic = {'res': ['i', 'j', 'ij'], 'val': m, 'matrix': False}
                n_dic = {'res': ['i', 'j', 'ij'], 'val': n, 'matrix': False}
                w_dic = {'res': ['i', 'j', 'ij'], 'val': 'weighted', 'matrix': False}
                nw_dic = {'res': ['i', 'j', 'ij'], 'val': 'not_weighted', 'matrix': False}

                Q = k_chain(m,  n, k, sparse=True)
                Q = Q.todense().A
                cur_max_rho = max_rho + 1.
                lamda = []
                mu = []

                if not((rho == .95 and k == 3) or (rho == .85 and k == 3) or (rho == .95 and k == 5) or (rho == .95 and k == 7)):

                    while cur_max_rho >= max_rho:

                        lamda = random_unit_partition(m)
                        mu = np.ones(n)*(1./n)

                        wls, rho_m, rho_n = bipartite_workload_decomposition(Q, lamda, mu)
                        cur_max_rho = np.amax(rho_n)
                        print cur_max_rho
                        lamda = (1./cur_max_rho) * lamda
                        wls, rho_m, rho_n = bipartite_workload_decomposition(Q, lamda, mu)

                    lamda_p = np.append(lamda*rho, mu.sum() - rho*lamda.sum())
                    Qp = sps.vstack((Q, np.ones((1, n))), format='csr')
                    print 'starting shelikhovskii'

                    pi_ent = shelikhovskii(Qp, lamda_p, mu)
                    pi_ent = pi_ent.todense().A

                    rho_ent = pi_ent[:m, :].sum(axis=0)

                    q_ent = pi_ent[:m, :]/(1.0 - rho_ent)
                    q_ent_i = (q_ent.T/q_ent.sum(axis=1)).T
                    q_ent_j = q_ent/q_ent.sum(axis=0)

                    pi_ent_dic = {'res': ['ij'], 'val': pi_ent, 'matrix': True}
                    rho_dic = {'res': ['i', 'j', 'ij'], 'val': rho, 'matrix': False}

                    Wp = np.diag(np.ones(n) - rho_n*rho, 0).dot(Q)
                    Wp = np.vstack((Wp, rho_n*rho))

                    Wps = sps.csr_matrix(Wp)

                    Mps = 0*sps.csr_matrix(Qp)
                    A,b,z, cols = transform_to_normal_form(Mps, Wps, Qp, Qp, lamda_p, mu)
                    pi_hat, duals = fast_primal_dual_algorithm(A, b, z)
                    pi_hat = pi_hat.reshape((m+1, n))
                    pi_hat = np.divide(pi_hat, Wp, out=np.zeros_like(pi_hat), where=Wp != 0)
                    pi_hat = pi_hat/pi_hat[:m, :].sum()
                    pi_hat = pi_hat*rho
                    rho_hat = pi_hat[:m, :].sum(axis=0)/mu
                    q_hat = pi_hat[:m, :]/(1.0 - rho_hat)
                    q_hat_i = (q_hat.T/q_hat.sum(axis=1)).T
                    q_hat_j = q_hat/q_hat.sum(axis=0)
                    pi_hat_dic = {'res': ['ij'], 'val': pi_hat[:m, :], 'matrix': True}
                    w_i = np.divide(q_hat_i, q_ent_i, out=np.zeros_like(q_hat_i), where=q_ent_i != 0)
                    w_j = np.divide(q_hat_j, q_ent_j, out=np.zeros_like(q_hat_j), where=q_ent_i != 0)

                    print '----------------------------------------------------------------------------------'
                    print 1, k, rho, 'fifo', 'rho_weighted_i'
                    print '----------------------------------------------------------------------------------'

                    simulator.simulate(lamda=lamda*rho, mu=mu, w_i=w_i, w_j=w_i, q=Q,
                                       rho=rho_dic, n=n_dic, m=m_dic, pi_ent=pi_ent_dic, pi_hat=pi_hat_dic, exp_num=exp_dic,
                                       chain_len=chain_len_dic, weighted=w_dic,
                                       sim_len=m*sim_len, sims=1, sim_name='rho_weighted_i_FIFO')

                    print '----------------------------------------------------------------------------------'
                    print 2, k, rho, 'fifo', 'rho_weighted_j'
                    print '----------------------------------------------------------------------------------'

                    simulator.simulate(lamda=lamda*rho, mu=mu, w_i=w_j, w_j=w_j, q=Q,
                                       rho=rho_dic, n=n_dic, m=m_dic, pi_ent=pi_ent_dic, pi_hat=pi_hat_dic, exp_num=exp_dic,
                                       chain_len=chain_len_dic, weighted=w_dic,
                                       sim_len=m*sim_len, sims=1, sim_name='rho_weighted_j_FIFO')

                    print '----------------------------------------------------------------------------------'
                    print 3, k, rho, 'fifo', 'rho_weighted_ij'
                    print '----------------------------------------------------------------------------------'

                    simulator.simulate(lamda=lamda*rho, mu=mu, w_i=w_i, w_j=w_j, q=Q,
                                       rho=rho_dic, n=n_dic, m=m_dic, pi_ent=pi_ent_dic, pi_hat=pi_hat_dic, exp_num=exp_dic,
                                       chain_len=chain_len_dic, weighted=w_dic,
                                       sim_len=m*sim_len, sims=1, sim_name='rho_weighted_ij_FIFO')

                    print '----------------------------------------------------------------------------------'
                    print 4, k, rho, 'max_weight', 'rho_weighted_i'
                    print '----------------------------------------------------------------------------------'

                    simulator.simulate(lamda=lamda*rho, mu=mu, w_i=w_i, w_j=w_i, q=Q,
                                       rho=rho_dic, n=n_dic, m=m_dic, pi_ent=pi_ent_dic, pi_hat=pi_hat_dic, exp_num=exp_dic,
                                       chain_len=chain_len_dic, weighted=w_dic,
                                       sim_len=m*sim_len, sims=1, sim_name='rho_weighted_i_MW',
                                       j_policy='max_weight')

                    print '----------------------------------------------------------------------------------'
                    print 5, k, rho, 'max_weight', 'rho_weighted_i'
                    print '----------------------------------------------------------------------------------'

                    simulator.simulate(lamda=lamda*rho, mu=mu, w_i=w_j, w_j=w_j, q=Q,
                                       rho=rho_dic, n=n_dic, m=m_dic, pi_ent=pi_ent_dic, pi_hat=pi_hat_dic, exp_num=exp_dic,
                                       chain_len=chain_len_dic, weighted=w_dic,
                                       sim_len=m*sim_len, sims=1, sim_name='rho_weighted_j_MW',
                                       j_policy='max_weight')

                    print '----------------------------------------------------------------------------------'
                    print 6, k, rho, 'max_weight', 'rho_weighted_ij'
                    print '----------------------------------------------------------------------------------'

                    simulator.simulate(lamda=lamda*rho, mu=mu, w_i=w_i, w_j=w_j, q=Q,
                                       rho=rho_dic, n=n_dic, m=m_dic, pi_ent=pi_ent_dic, pi_hat=pi_hat_dic, exp_num=exp_dic,
                                       chain_len=chain_len_dic, weighted=w_dic,
                                       sim_len=m*sim_len, sims=1, sim_name='rho_weighted_ij_MW',
                                       j_policy='max_weight')

                    print '----------------------------------------------------------------------------------'
                    print 7, k, rho, 'fifo', 'plain'
                    print '----------------------------------------------------------------------------------'

                    simulator.simulate(lamda=lamda*rho, mu=mu,  q=Q,
                                       rho=rho_dic, n=n_dic, m=m_dic, pi_ent=pi_ent_dic, pi_hat=pi_hat_dic, exp_num=exp_dic,
                                       chain_len=chain_len_dic, weighted=nw_dic,
                                       sim_len=m*sim_len, sims=1, sim_name='plain_FIFO')

                    print '----------------------------------------------------------------------------------'
                    print 8, k, rho, 'max_weight', 'plain'
                    print '----------------------------------------------------------------------------------'

                    simulator.simulate(lamda=lamda*rho, mu=mu, q=Q,
                                       rho=rho_dic, n=n_dic, m=m_dic, pi_ent=pi_ent_dic, pi_hat=pi_hat_dic, exp_num=exp_dic,
                                       chain_len=chain_len_dic, weighted=nw_dic,
                                       sims=1, sim_name='plain_MW', j_policy='max_weight', sim_len=m*sim_len)


def erdos_renyi_exp():

    crp = False

    sim_len = 10**5

    simulator = SimExperiment('Erdos_Renyi')


    max_rho = 1.5

    lamda = None
    mu = None
    rho_n = None
    Q = None

    for exp in range(30):

        exp_dic = {'res': ['i', 'j', 'ij'], 'val': exp, 'matrix': False}

        for n in [50, 100, 500, 1000]:

            m = n
            p = 2.*log(float(n))/float(n)

            for rho in np.arange(.95, .55, -0.1):

                p_dic = {'res': ['i', 'j', 'ij'], 'val': p, 'matrix': False}
                m_dic = {'res': ['i', 'j', 'ij'], 'val': m, 'matrix': False}
                n_dic = {'res': ['i', 'j', 'ij'], 'val': n, 'matrix': False}
                w_dic = {'res': ['i', 'j', 'ij'], 'val': 'weighted', 'matrix': False}
                nw_dic = {'res': ['i', 'j', 'ij'], 'val': 'not_weighted', 'matrix': False}

                cur_max_rho = max_rho + 1.
                connected = False
                non_trivial = False
                if n!=50 or exp !=0:
                    while cur_max_rho >= max_rho or not connected or not non_trivial:

                        Q = erdos_renyi_connected_graph(n, n, p)
                        Q = Q.todense().A

                        non_trivial, connected = is_non_trivial_and_connected(Q)

                        lamda = random_unit_partition(m)
                        mu = random_unit_partition(n)

                        wls, rho_m, rho_n = bipartite_workload_decomposition(Q, lamda, mu)
                        cur_max_rho = np.amax(rho_n)
                        print cur_max_rho
                        lamda = (1./cur_max_rho) * lamda
                        wls, rho_m, rho_n = bipartite_workload_decomposition(Q, lamda, mu)

                    lamda_p = np.append(lamda*rho, mu.sum() - rho*lamda.sum())
                    Qp = sps.vstack((Q, np.ones((1, n))), format='csr')
                    print 'starting shelikhovskii'

                    pi_ent = shelikhovskii(Qp, lamda_p, mu)
                    pi_ent = pi_ent.todense().A

                    rho_ent = pi_ent[:m, :].sum(axis=0)

                    q_ent = pi_ent[:m, :]/(1.0 - rho_ent)
                    q_ent_i = (q_ent.T/q_ent.sum(axis=1)).T
                    q_ent_j = q_ent/q_ent.sum(axis=0)

                    pi_ent_dic = {'res': ['ij'], 'val': pi_ent, 'matrix': True}
                    rho_dic = {'res': ['i', 'j', 'ij'], 'val': rho, 'matrix': False}

                    Wp = np.diag(np.ones(n) - rho_n*rho, 0).dot(Q)
                    Wp = np.vstack((Wp, rho_n*rho))

                    Wps = sps.csr_matrix(Wp)

                    Mps = 0*sps.csr_matrix(Qp)
                    A,b,z, cols = transform_to_normal_form(Mps, Wps, Qp, Qp, lamda_p, mu)
                    pi_hat, duals = fast_primal_dual_algorithm(A, b, z)
                    pi_hat = pi_hat.reshape((m+1, n))
                    pi_hat = np.divide(pi_hat, Wp, out=np.zeros_like(pi_hat), where=Wp != 0)
                    pi_hat = pi_hat/pi_hat[:m, :].sum()
                    pi_hat = pi_hat*rho
                    rho_hat = pi_hat[:m, :].sum(axis=0)/mu
                    q_hat = pi_hat[:m, :]/(1.0 - rho_hat)
                    q_hat_i = (q_hat.T/q_hat.sum(axis=1)).T
                    q_hat_j = q_hat/q_hat.sum(axis=0)
                    pi_hat_dic = {'res': ['ij'], 'val': pi_hat[:m, :], 'matrix': True}
                    w_i = np.divide(q_hat_i, q_ent_i, out=np.zeros_like(q_hat_i), where=q_ent_i != 0)
                    w_j = np.divide(q_hat_j, q_ent_j, out=np.zeros_like(q_hat_j), where=q_ent_i != 0)

                    print '----------------------------------------------------------------------------------'
                    print 1, rho, 'fifo', 'rho_weighted_i'
                    print '----------------------------------------------------------------------------------'

                    simulator.simulate(lamda=lamda*rho, mu=mu, w_i=w_i, w_j=w_i, q=Q,
                                       rho=rho_dic, n=n_dic, m=m_dic, pi_ent=pi_ent_dic, pi_hat=pi_hat_dic, exp_num=exp_dic,
                                       p=p_dic, weighted=w_dic,
                                       sim_len=m*sim_len, sims=1, sim_name='rho_weighted_i_FIFO')

                    print '----------------------------------------------------------------------------------'
                    print 2, rho, 'fifo', 'rho_weighted_j'
                    print '----------------------------------------------------------------------------------'

                    simulator.simulate(lamda=lamda*rho, mu=mu, w_i=w_j, w_j=w_j, q=Q,
                                       rho=rho_dic, n=n_dic, m=m_dic, pi_ent=pi_ent_dic, pi_hat=pi_hat_dic, exp_num=exp_dic,
                                       p=p_dic, weighted=w_dic,
                                       sim_len=m*sim_len, sims=1, sim_name='rho_weighted_j_FIFO')

                    print '----------------------------------------------------------------------------------'
                    print 3, rho, 'fifo', 'rho_weighted_ij'
                    print '----------------------------------------------------------------------------------'

                    simulator.simulate(lamda=lamda*rho, mu=mu, w_i=w_i, w_j=w_j, q=Q,
                                       rho=rho_dic, n=n_dic, m=m_dic, pi_ent=pi_ent_dic, pi_hat=pi_hat_dic, exp_num=exp_dic,
                                       p=p_dic, weighted=w_dic,
                                       sim_len=m*sim_len, sims=1, sim_name='rho_weighted_ij_FIFO')

                    print '----------------------------------------------------------------------------------'
                    print 4, rho, 'max_weight', 'rho_weighted_i'
                    print '----------------------------------------------------------------------------------'

                    simulator.simulate(lamda=lamda*rho, mu=mu, w_i=w_i, w_j=w_i, q=Q,
                                       rho=rho_dic, n=n_dic, m=m_dic, pi_ent=pi_ent_dic, pi_hat=pi_hat_dic, exp_num=exp_dic,
                                       p=p_dic, weighted=w_dic,
                                       sim_len=m*sim_len, sims=1, sim_name='rho_weighted_i_MW',
                                       j_policy='max_weight')

                    print '----------------------------------------------------------------------------------'
                    print 5, rho, 'max_weight', 'rho_weighted_i'
                    print '----------------------------------------------------------------------------------'

                    simulator.simulate(lamda=lamda*rho, mu=mu, w_i=w_j, w_j=w_j, q=Q,
                                       rho=rho_dic, n=n_dic, m=m_dic, pi_ent=pi_ent_dic, pi_hat=pi_hat_dic, exp_num=exp_dic,
                                       p=p_dic, weighted=w_dic,
                                       sim_len=m*sim_len, sims=1, sim_name='rho_weighted_j_MW',
                                       j_policy='max_weight')

                    print '----------------------------------------------------------------------------------'
                    print 6, rho, 'max_weight', 'rho_weighted_ij'
                    print '----------------------------------------------------------------------------------'

                    simulator.simulate(lamda=lamda*rho, mu=mu, w_i=w_i, w_j=w_j, q=Q,
                                       rho=rho_dic, n=n_dic, m=m_dic, pi_ent=pi_ent_dic, pi_hat=pi_hat_dic, exp_num=exp_dic,
                                       p=p_dic, weighted=w_dic,
                                       sim_len=m*sim_len, sims=1, sim_name='rho_weighted_ij_MW',
                                       j_policy='max_weight')

                    print '----------------------------------------------------------------------------------'
                    print 7, rho, 'fifo', 'plain'
                    print '----------------------------------------------------------------------------------'

                    simulator.simulate(lamda=lamda*rho, mu=mu,  q=Q,
                                       rho=rho_dic, n=n_dic, m=m_dic, pi_ent=pi_ent_dic, pi_hat=pi_hat_dic, exp_num=exp_dic,
                                       p=p_dic, weighted=nw_dic,
                                       sim_len=m*sim_len, sims=1, sim_name='plain_FIFO')

                    print '----------------------------------------------------------------------------------'
                    print 8, rho, 'max_weight', 'plain'
                    print '----------------------------------------------------------------------------------'

                    simulator.simulate(lamda=lamda*rho, mu=mu, q=Q,
                                       rho=rho_dic, n=n_dic, m=m_dic, pi_ent=pi_ent_dic, pi_hat=pi_hat_dic, exp_num=exp_dic,
                                       p=p_dic, weighted=nw_dic,
                                       sims=1, sim_name='plain_MW', j_policy='max_weight', sim_len=m*sim_len)


def test_w():

    n=4
    m=5
    rho = 1.
    lamda = np.array([0.35,0.35, 0.3, 0.7, 0.7])
    mu = np.array([.5,.5, 1., 1.])
    lamda_p = np.append(lamda*rho, mu.sum() - rho*lamda.sum())

    Q = sps.csr_matrix(np.array([[1, 1, 0, 0],
                                 [1, 1, 0, 0],
                                 [1, 1, 1, 1],
                                 [0, 0, 1, 1],
                                 [0, 0, 1, 1]]))
    Qp = sps.vstack((Q, np.ones((1, n))), format='csr')
    Mps = 0*Qp
    Wps = np.array([[.3, .3, 0., 0.],
                    [.3, .3, 0., 0.],
                    [.3, .3, .1, .1],
                    [.0, .0, .1, .1],
                    [.0, .0, .1, .1],
                    [.7, .7, .9, .9]])
    Wps = sps.csr_matrix(Wps)

    Z = Qp.dot(sps.csr_matrix(np.array([[.5, 0., 0., 0.],
                                        [0., .5, 0., 0.],
                                        [0., 0., 1., 0.],
                                        [0., 0., 0., 1.]])))
    print Z
    #transform_to_normal_form(M, W, Q, Z, row_sum, col_sum)
    A,b,z, cols = transform_to_normal_form(Mps, Wps, Qp, Z, lamda_p, mu)
    print A.todense().A
    print b
    print z
    pi_hat1, duals1 = fast_primal_dual_algorithm(A, b, z)
    pi_hat1 = pi_hat1.reshape((m+1, n))
    print pi_hat1
    A,b,z, cols = transform_to_normal_form(Mps, Wps, Qp, Qp, lamda_p, mu)
    pi_hat2, duals2 = fast_primal_dual_algorithm(A, b, z)
    pi_hat2 = pi_hat2.reshape((m+1, n))
    print pi_hat2
    # tester = SimExperiment('W3')
    # tester.simulate(lamda=lamda, mu=mu, q=Q, sim_len=3*10**5, sims=3)


def service_time_dependency_experiment(m, n):

    max_rho = 2.0

    while max_rho > 1.000000000001:

        Q, lamda, mu = generate_fss(m, n, 1.0)

        sparse = sps.isspmatrix(Q)

        wls, rho_m, rho_n = bipartite_workload_decomposition(Q, lamda, mu)

        max_rho = np.amax(rho_n)
        print 'max_rho', max_rho

    simulator = SimExperiment('k_chains')

    for rho in [0.9]:  # np.array([0.5 + 0.1*i for i in range(5)] + [0.95, 0.97, 0.99]):

        lamda_p = np.append(lamda*rho, 1.0 - rho)
        lamda_p = lamda_p/lamda_p.sum()
        Qp = sps.vstack((Q, np.ones(n)), format='csr') if sparse else np.vstack((Q, np.ones(n)))
        #Qp = sps.vstack((Q.T, np.ones(m)), format='csr') if sparse else np.vstack((Q.T, np.ones(m)))
        pi_ent = shelikhovskii(Qp, lamda_p, mu)
        r_ent = pi_ent[m, :]
        pi_ent = pi_ent[:m, :]
        pi_ent = pi_ent/pi_ent.sum()

        pi_ent = {'res': ['ij'], 'val': pi_ent.todense().A[:m, :], 'matrix': True}
        rho_dic = {'res':['i', 'j', 'ij'], 'val': rho, 'matrix': False}
        simulator.simulate(lamda=lamda*rho, s=np.ones(n), mu=mu, q=Q,  sims=1,sim_len=10**5, rho=rho_dic, sim_name='',  pi_ent=pi_ent, i_policy='rand', j_policy='rand')
        simulator.simulate(lamda=lamda*rho, s=np.ones(n), mu=mu, q=Q,  sims=1,sim_len=10**5, rho=rho_dic, sim_name='',  pi_ent=pi_ent, i_policy='alis', j_policy='fifo')
        simulator.simulate(lamda=lamda*rho, s=np.ones(n), mu=mu, q=Q,  sims=1,sim_len=10**5, rho=rho_dic, sim_name='',  pi_ent=pi_ent, i_policy='alis', j_policy='max_weight')
        # simulator.simulate(lamda=np.ones(m)*(1./float(m)), s=lamda*rho, q=Q,mu=mu, sims=2, sim_len=10**6, rho=rho_dic, sim_name='customer', pi_ent=pi_ent)
        # simulator.simulate(lamda=lamda**0.5, s=lamda**0.5, mu=mu, sims=2,q=Q,  sim_len=10**6, rho=rho_dic, sim_name='split_customer',pi_ent=pi_ent)

    # simulator = SimExperiment('service_time_dependency_30_x_30_28_sims')
    #
    # for rho in np.array([0.5 + 0.1*i for i in range(5)] + [0.95, 0.97, 0.99]):
    #     print rho
    #     lamda_p = np.append(lamda*rho, 1.0 - rho)
    #     pi_ent = shelikhovskii(Qp, lamda_p, mu)
    #
    #     pi_ent = {'res': ['ij'], 'val': pi_ent.todense().A[:m, :], 'matrix': True}
    #     rho_dic ={'res':['i', 'j', 'ij'], 'val':rho, 'matrix':False}
    #
    #     simulator.simulate(lamda=lamda*rho, s=np.ones(n), mu=mu, q=Q,  sims=28,sim_len=10**6, rho=rho_dic, sim_name='homogeneous',  pi_ent=pi_ent)
    #     simulator.simulate(lamda=np.ones(m)*(1./float(m)), s=lamda*rho, q=Q,mu=mu, sims=28, sim_len=10**6, rho=rho_dic, sim_name='customer', pi_ent=pi_ent)
    #     simulator.simulate(lamda=lamda**0.5, s=lamda**0.5, mu=mu, sims=28,q=Q,  sim_len=10**6, rho=rho_dic, sim_name='split_customer',pi_ent=pi_ent)


def service_time_dependency_experiment2(m, n):

    max_rho = 2.0

    while max_rho > 1.000000000001:

        Q, lamda, mu = generate_fss(m, n, 1.0)

        sparse = sps.isspmatrix(Q)

        wls, rho_m, rho_n = bipartite_workload_decomposition(Q, lamda, mu)

        max_rho = np.amax(rho_n)
        print 'max_rho', max_rho

    simulator = SimExperiment('k_chains')

    for rho in [0.9]:  # np.array([0.5 + 0.1*i for i in range(5)] + [0.95, 0.97, 0.99]):

        lamda_p = np.append(lamda*rho, 1.0 - rho)
        lamda_p = lamda_p/lamda_p.sum()
        Qp = sps.vstack((Q, np.ones(n)), format='csr') if sparse else np.vstack((Q, np.ones(n)))
        #Qp = sps.vstack((Q.T, np.ones(m)), format='csr') if sparse else np.vstack((Q.T, np.ones(m)))
        pi_ent = shelikhovskii(Qp, lamda_p, mu)
        r_ent = pi_ent[m, :]
        pi_ent = pi_ent[:m, :]
        pi_ent = pi_ent/pi_ent.sum()

        pi_ent = {'res': ['ij'], 'val': pi_ent.todense().A[:m, :], 'matrix': True}
        rho_dic = {'res': ['i', 'j', 'ij'], 'val': rho, 'matrix': False}
        simulator.simulate(lamda=lamda*rho, s=np.ones(n), mu=mu, q=Q,  sims=1,sim_len=10**5, rho=rho_dic, sim_name='',  pi_ent=pi_ent, i_policy='rand', j_policy='rand')
        simulator.simulate(lamda=lamda*rho, s=np.ones(n), mu=mu, q=Q,  sims=1,sim_len=10**5, rho=rho_dic, sim_name='',  pi_ent=pi_ent, i_policy='alis', j_policy='fifo')
        simulator.simulate(lamda=lamda*rho, s=np.ones(n), mu=mu, q=Q,  sims=1,sim_len=10**5, rho=rho_dic, sim_name='',  pi_ent=pi_ent, i_policy='alis', j_policy='max_weight')
        # simulator.simulate(lamda=np.ones(m)*(1./float(m)), s=lamda*rho, q=Q,mu=mu, sims=2, sim_len=10**6, rho=rho_dic, sim_name='customer', pi_ent=pi_ent)
        # simulator.simulate(lamda=lamda**0.5, s=lamda**0.5, mu=mu, sims=2,q=Q,  sim_len=10**6, rho=rho_dic, sim_name='split_customer',pi_ent=pi_ent)

    # simulator = SimExperiment('service_time_dependency_30_x_30_28_sims')
    #
    # for rho in np.array([0.5 + 0.1*i for i in range(5)] + [0.95, 0.97, 0.99]):
    #     print rho
    #     lamda_p = np.append(lamda*rho, 1.0 - rho)
    #     pi_ent = shelikhovskii(Qp, lamda_p, mu)
    #
    #     pi_ent = {'res': ['ij'], 'val': pi_ent.todense().A[:m, :], 'matrix': True}
    #     rho_dic ={'res':['i', 'j', 'ij'], 'val':rho, 'matrix':False}
    #
    #     simulator.simulate(lamda=lamda*rho, s=np.ones(n), mu=mu, q=Q,  sims=28,sim_len=10**6, rho=rho_dic, sim_name='homogeneous',  pi_ent=pi_ent)
    #     simulator.simulate(lamda=np.ones(m)*(1./float(m)), s=lamda*rho, q=Q,mu=mu, sims=28, sim_len=10**6, rho=rho_dic, sim_name='customer', pi_ent=pi_ent)
    #     simulator.simulate(lamda=lamda**0.5, s=lamda**0.5, mu=mu, sims=28,q=Q,  sim_len=10**6, rho=rho_dic, sim_name='split_customer',pi_ent=pi_ent)


def k_chain_noise_experiment(m, n, k):

    max_rho = 2.0

    Q = k_chain(m, n, k)

    sparse = sps.isspmatrix(Q)

    simulator = SimExperiment('k_chains_noise2')

    for rho in [0.9]:  # np.array([0.5 + 0.1*i for i in range(5)] + [0.95, 0.97, 0.99]):

        for noise in np.arange(0, 1.1, 0.1):

            for k in range(30):

                max_rho = 2.0

                while max_rho > 1.000000000001:

                    if 'lamda':
                        lamda = np.ones(m) + np.random.uniform(-noise, noise, m)
                        lamda = lamda/lamda.sum()
                        mu = np.ones(n)
                    if 'mu':
                        lamda = np.ones(m)
                        mu = np.ones(n) + np.random.uniform(-noise, noise, n)
                        mu = mu/mu.sum()
                    if 'both':
                        lamda = np.ones(m) + np.random.uniform(-noise, noise, m)
                        lamda = lamda/lamda.sum()
                        mu = np.ones(n) + np.random.uniform(-noise, noise, n)
                        mu = mu/mu.sum()

                    wls, rho_m, rho_n = bipartite_workload_decomposition(Q, lamda, mu)
                    if rho_n is not None:
                        max_rho = np.amax(rho_n)
                    print 'max_rho', max_rho

                lamda_p = np.append(lamda*rho, 1.0 - rho)
                lamda_p = lamda_p/lamda_p.sum()
                Qp = sps.vstack((Q, np.ones(n)), format='csr') if sparse else np.vstack((Q, np.ones(n)))
                pi_ent = shelikhovskii(Qp, lamda_p, mu)
                r_ent = pi_ent[m, :]
                pi_ent = pi_ent[:m, :]
                pi_ent = pi_ent/pi_ent.sum()
                pi_ent = {'res': ['ij'], 'val': pi_ent.todense().A[:m, :], 'matrix': True}
                rho_dic = {'res': ['i', 'j', 'ij'], 'val': rho, 'matrix': False}
                noise_dic = {'res': ['i', 'j', 'ij'], 'val': noise, 'matrix': False}
                rep = {'res': ['i', 'j', 'ij'], 'val': k, 'matrix': False}
                simulator.simulate(lamda=lamda*rho, s=np.ones(n), mu=mu, q=Q,  sims=1,sim_len=10**5, rep=rep, noise=noise_dic, rho=rho_dic, sim_name='',  pi_ent=pi_ent, i_policy='rand', j_policy='rand')
                simulator.simulate(lamda=lamda*rho, s=np.ones(n), mu=mu, q=Q,  sims=1,sim_len=10**5, rep=rep, noise=noise_dic, rho=rho_dic, sim_name='',  pi_ent=pi_ent, i_policy='alis', j_policy='fifo')
                simulator.simulate(lamda=lamda*rho, s=np.ones(n), mu=mu, q=Q,  sims=1,sim_len=10**5, rep=rep, noise=noise_dic, rho=rho_dic, sim_name='',  pi_ent=pi_ent, i_policy='alis', j_policy='max_weight')
                if noise == 0.0:
                    break


def two_chain_predictions():

    max_rho = 2.0

    simulator = SimExperiment('2_chain_2')

    noise_type = 'lamda'

    for n in [20, 50, 100, 1000]:

        Q = k_chain(n, n, 2)
        Qm = k_chain(n, n, 2)
        Qm[n-1, 0] = 0.
        if n == 5:
            print 'Q'
            print Q
            print 'Qm'
            print Qm
        sparse = sps.isspmatrix(Q)

        for k in range(5):

            for rho in [0.7, 0.8, 0.9, 0.95]:  # np.array([0.5 + 0.1*i for i in range(5)] + [0.95, 0.97, 0.99]):

                for noise in np.arange(0, 0.5, 0.1):

                    max_rho = 2.0

                    while max_rho > 1.000000000001:

                        if noise_type == 'lamda':
                            print noise_type
                            lamda = np.ones(n) + np.random.uniform(-noise, noise, n)
                            lamda = lamda/lamda.sum()
                            mu = np.ones(n)/float(n)
                        if noise_type == 'mu':
                            print noise_type
                            lamda = np.ones(n)
                            mu = np.ones(n) + np.random.uniform(-noise, noise, n)
                            mu = mu/mu.sum()
                        if noise_type == 'both':
                            print noise_type
                            lamda = np.ones(n) + np.random.uniform(-noise, noise, n)
                            lamda = lamda/lamda.sum()
                            mu = np.ones(n) + np.random.uniform(-noise, noise, n)
                            mu = mu/mu.sum()

                        wls, rho_m, rho_n = bipartite_workload_decomposition(Qm, lamda, mu)
                        if rho_n is not None:
                            max_rho = np.amax(rho_n)
                        print 'max_rho', max_rho

                    lamda_p = np.append(lamda*rho, 1.0 - rho)
                    print lamda_p.sum()
                    lamda_p = lamda_p/lamda_p.sum()
                    Qp = sps.vstack((Q, np.ones(n)), format='csr') if sparse else np.vstack((Q, np.ones(n)))
                    Qpm = sps.vstack((Qm, np.ones(n)), format='csr') if sparse else np.vstack((Qm, np.ones(n)))
                    print 'start shelikhovskii'
                    pi_ent = shelikhovskii(Qp, lamda_p, mu)
                    print 'end shelikhovskii'
                    r_ent = pi_ent[n, :]
                    pi_ent = pi_ent[:n, :]
                    pi_ent = pi_ent/pi_ent.sum()
                    pi_ent = {'res': ['ij'], 'val': pi_ent.todense().A[:n, :], 'matrix': True}
                    pi_ent_m = shelikhovskii(Qpm, lamda_p, mu)
                    r_ent_m = pi_ent_m[n, :]
                    pi_ent_m = pi_ent_m[:n, :]
                    pi_ent_m = pi_ent_m/pi_ent_m.sum()
                    pi_ent_m = {'res': ['ij'], 'val': pi_ent_m.todense().A[:n, :], 'matrix': True}
                    rho_dic = {'res': ['i', 'j', 'ij'], 'val': rho, 'matrix': False}
                    noise_dic = {'res': ['i', 'j', 'ij'], 'val': noise, 'matrix': False}
                    chain_dic = {'res': ['i', 'j', 'ij'], 'val': 'closed', 'matrix': False}
                    size_dic = {'res': ['i', 'j', 'ij'], 'val': n, 'matrix': False}
                    simulator.simulate(lamda=lamda*rho, s=np.ones(n), mu=mu, q=Q,  sims=1, sim_len=n*10**5,
                                       noise=noise_dic, rho=rho_dic, sim_name=str(k),
                                       pi_ent=pi_ent,
                                       i_policy='alis', j_policy='fifo', sys_size=size_dic, chain=chain_dic)
                    chain_dic = {'res': ['i', 'j', 'ij'], 'val': 'open', 'matrix': False}
                    simulator.simulate(lamda=lamda*rho, s=np.ones(n), mu=mu, q=Qm,  sims=1, sim_len=n*10**5,
                                       noise=noise_dic, rho=rho_dic, sim_name=str(k),
                                       pi_ent=pi_ent_m,
                                       i_policy='alis', j_policy='fifo', sys_size=size_dic, chain=chain_dic)


def greenkhorn_test():

    ns = 1000
    nc = 1000
    noise = 0.2
    u = 1.0
    Q = erdos_renyi_connected_graph(nc, ns, 2.0*log(nc + ns)/(nc + ns), k=2).astype(float)
    Qd = Q.todense().A
    mu = np.array([1.0]*ns)
    #np.random.seed(0)
    noise = np.random.uniform(low=-noise, high=noise, size=nc)
    lamda = np.array([1.0]*nc) + noise
    lamda = u * nc * lamda/lamda.sum()
    Md = np.zeros((nc, ns))
    for i in range(nc):
        for j in range(ns):
            if Q[i,j] > 0.0:
                Md[i,j] = float(abs(i-j)) + 1.0
    M = sps.csr_matrix(Md)
    s = time()
    print 'starting'
    Gd = greenkhorn_with_q(lamda, mu, Md, Qd, 1.0, numItermax=10)
    print '-----------------------------------'
    print 'dense', time() - s, 'sec'
    # print Gd
    # print Gd.sum(0)
    # print Gd.sum(1)
    s = time()
    Gs = greenkhorn_with_q(lamda, mu, M, Q, 1.0, numItermax=10)
    # print 'sparse', time() - s, 'sec'
    # print Gs
    # print Gs.sum(0)
    # print Gs.sum(1)


def weighted_entropy():

    q = k_chain(10, 10, 4)
    lamda = np.ones(10)
    lamda = lamda/lamda.sum()
    mu = np.ones(10)*0.1
    print lamda
    print mu
    wls, rho_m, rho_n = bipartite_workload_decomposition(q, lamda, mu)
    w = k_chain(10, 10, 4).multiply(np.random.uniform(0,1, (10, 10)))
    w = w.todense().A
    w = w/w.sum()
    print type(w)
    A, b, z, cols = transform_to_normal_form(0*q, sps.csr_matrix(w), q, q, lamda, mu)
    pi_hat1, duals = fast_primal_dual_algorithm(A, b, z, max_iter=10**9)
    pi_hat1 = pi_hat1.reshape((10, 10))
    pi_hat1 = np.divide(pi_hat1, w, out=np.zeros_like(pi_hat1), where=w != 0)
    pi_ent = shelikhovskii(q, lamda, mu)
    w[0,0] = w[0,0] + 0.01
    A, b, z, cols = transform_to_normal_form(0*q, sps.csr_matrix(w), q, q, lamda, mu)
    pi_hat2, duals = fast_primal_dual_algorithm(A, b, z, max_iter=10**9)
    pi_hat2 = pi_hat2.reshape((10, 10))
    pi_hat2 = np.divide(pi_hat2, w, out=np.zeros_like(pi_hat1), where=w != 0)
    for i,j in sorted(zip(*w.nonzero()), key=lambda x: (x[0], x[1])):
         print (i,j),w[i,j]-0.01*((i == 0) * (j == 0)), w[i,j], pi_ent[i,j],pi_hat1[i,j], pi_hat2[i,j],pi_hat1[i,j] > pi_hat2[i,j]


def check_shli():

    q = k_chain(10, 10, 4)
    lamda = np.ones(10)
    lamda = lamda/lamda.sum()
    mu = np.ones(10)*0.1
    pi_ent = shelikhovskii(q, lamda, mu)
    print pi_ent


if __name__ == '__main__':

    #fifo_flaw2()
    #flexible_queueing_system(30, 30, 0.8)
    #k_chain_experiment(30, 30, 7)
    # print sum([14.,15.,16.,17.,8.,20.])
    # print sum([4., 25., 6., 27., 8., 20.])
    # lamda = np.array([14.,15.,16.,17.,8.,20.])
    # mu = np.array([4., 25., 6., 27., 8., 20.])
    # q = np.array([[1,1,1,0,0,0],
    #               [0,1,1,1,0,0],
    #               [0,0,1,1,1,0],
    #               [0,0,0,1,1,1],
    #               [1,0,0,0,1,1],
    #               [1,1,0,0,0,1]])
    #
    # p_dic,p = matching_rate_calculator(lamda, mu, q)
    # tot = 0.0
    # for key, item in p_dic.iteritems():
    #     print key, item['rcs-aw']
    #     tot += item['rcs-aw']
    # print p
    #
    # adan_weiss = SimExperiment('adan_wiess')
    # adan_weiss.simulate(lamda=lamda, mu=mu, q=q, sims=30, sim_len=10**6)
    #
    # reses_i = get_file_data('adan_wiess', 'reses_i')
    # reses_ij = get_file_data('adan_wiess', 'reses_ij')
    # reses_ij[['MR_ij_sim']] = 90.*reses_ij['MR_ij_sim']
    # print reses_ij[['i','j','MR_ij_sim']]
    #increasing_system()
    k_chains_exp()
    #power_of_an_arc()
    #k_chains_exp_sim_len()
    #erdos_renyi_exp()
    #test_w()
    # fifo_flow_exp.simulate(lamda=lamda, mu=mu, w=w, q=Q,
    #                        rho1=rho1_dic, rho2=rho2_dic,
    #                        N=N_dic, pi_ent=pi_ent_dic, pi_hat=pi_hat_dic,
    #                        sim_len=n*sim_len, sims=1, sim_name=sim_name)
    # Q = np.array([[1.0, 1.0, 0.0],[0.0, 1.0, 1.0],[1.0, 0.0, 1.0]])
    # parse_sim_data('dgdfgfg', Q)

