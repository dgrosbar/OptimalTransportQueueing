import cplex
from cplex.exceptions import CplexError
import numpy as np
import pandas as pd
from time import time
import random
from scipy import sparse as sps
from collections import deque
import heapq
import simpy
from math import ceil, log
from functools import partial
from generators import undirected_grid_2d_bipartie_graph

from generators import erdos_renyi_graph
from fss_utils import random_unit_partition, con_print


def matching_simulator(alpha, beta, q, prt=False, sims=500, sim_len=10**5, seed=None, side='server'):

    if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    nc = len(beta)   # nc - number of customers
    ns = len(alpha)  # ns- number of servers
    rnc = range(nc)  # 0,1,2,...,nc-1
    rns = range(ns)  # 0,1,2,...,ns-1
    matching_rates = []
    qs = dict((i, set(np.nonzero(q[:, i])[0])) for i in rns)  # adjacency list for servers
    qc = dict((j, set(np.nonzero(q[j, :])[0])) for j in rnc)  # adjacency list for customers

    if side == 'server':

        for k in range(sims):
            print 'starting sim ', k
            start_time = time()
            c_stream = np.random.choice(range(len(alpha)), size=sim_len, replace=True, p=alpha*(1/(alpha.sum())))
            s_stream = np.random.choice(range(len(beta)), size=sim_len, replace=True, p=beta*(1/(beta.sum())))
            matching_rates.append(np.zeros(q.shape))
            j = 0
            for sj in s_stream:
                if j % 1000 == 0:
                    print j, 'servers assigned time elapsed:', time()-start_time
                for i in range(len(c_stream)):
                    ci = c_stream[i]
                    if ci in qs[sj]:
                        matching_rates[k][ci, sj] += 1
                        del c_stream[i]
                        j = +1
                        break

            matching_rates[k] = beta.sum() * (matching_rates[k]/matching_rates[k].sum())
            print 'ending sim ', k, 'duration:', time() - start_time

        matching_rates_mean = sum(matching_rates)*(1.0/sims)
        matching_rates_stdev = (sum((matching_rates[k]-matching_rates_mean)**2 for k in range(sims))*(1.0/(sims-1)))**0.5

        return matching_rates_mean, matching_rates_stdev

    else:

        for k in range(sims):
            print 'starting sim ', k
            start_time = time()
            c_stream = list(np.random.choice(range(len(alpha)),size=sim_len, replace=True, p=alpha*(1/(alpha.sum()))))
            s_stream = list(np.random.choice(range(len(beta)),size=sim_len, replace=True, p=beta*(1/(beta.sum()))))
            matching_rates.append(np.zeros(q.shape))
            for ci in c_stream:
                for i in range(len(s_stream)):
                    sj = s_stream[i]
                    if sj in qc[ci]:
                        matching_rates[k][ci, sj] += 1
                        del s_stream[i]
                        break

            matching_rates[k] = beta.sum() * (matching_rates[k]/matching_rates[k].sum())
            print 'ending sim ', k, 'duration:', time() - start_time

        matching_rates_mean = sum(matching_rates)*(1.0/sims)
        matching_rates_stdev = (sum((matching_rates[k]-matching_rates_mean)**2 for k in range(sims))*(1.0/(sims-1)))**0.5

        return matching_rates_mean, matching_rates_stdev


def advanced_matching_simulator(alpha, beta, q, prt=False, block_size=10**5,
                                sims=100, warm_up=10**5, sim_len=10**6, seed=None, report=10**5):

    if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    nc = len(beta)   # nc - number of customers
    ns = len(alpha)  # ns - number of servers
    rnc = range(nc)  # 0,1,2,...,nc-1
    rns = range(ns)  # 0,1,2,...,ns-1
    matching_rates = tuple(dict((p, 0) for p in zip(*q.nonzero())) for _ in range(sims))
    assigned_counts = [0]*sims
    c_queues = tuple(deque() for c in rnc)
    qs = dict((i, set()) for i in rns)  # adjacency list for servers
    qc = dict((j, set()) for j in rnc)  # adjacency list for customers
    for p in zip(*q.nonzero()):
        qc[p[0]].add(p[1])
        qs[p[1]].add(p[0])

    for k in range(sims):

        print 'building sim ', k

        start_time = time()
        c_len = int(ceil(sim_len * 3))
        c_queues = tuple(deque() for _ in rnc)
        c_stream = np.random.choice(range(len(alpha)), size=c_len + warm_up, replace=True, p=alpha*(1/(alpha.sum())))
        max_c = len(c_stream)
        for ci in enumerate(c_stream):
            c_queues[ci[1]].appendleft(ci[0])

        s_stream = np.random.choice(range(len(beta)), size=sim_len + warm_up, replace=True, p=beta*(1/(beta.sum())))

        print 'starting sim ', k, 'time_elpased:', time()-start_time

        for sj in s_stream:
            if assigned_counts[k] > 0 and assigned_counts[k] % report == 0:
                print assigned_counts[k], 'servers assigned time elapsed:', time()-start_time
            customers = []
            if prt:
                print 'server', sj
            for ci in qs[sj]:
                if len(c_queues[ci]) > 0:
                    if len(c_queues[ci]) == 1:
                        c_stream = \
                            np.random.choice(range(len(alpha)), size=c_len + warm_up, replace=True,
                                             p=alpha*(1/(alpha.sum())))
                        for new_ci in enumerate(c_stream):
                            c_queues[new_ci[1]].appendleft(max_c + new_ci[0])
                        max_c += len(c_stream)
                    heapq.heappush(customers, (c_queues[ci][-1], ci))
                else:
                    print 'out of', ci
            if len(customers) > 0:
                if prt:
                    print 'the available HOL customers are', customers

                ci = heapq.heappop(customers)[1]

                if prt:
                    print 'the customer cosen is', ci
                    print 'this is queue of ', ci, 'before'
                    print c_queues[ci]

                c_queues[ci].pop()

                if prt:
                    print 'this is queue of ', ci, 'after'
                    print c_queues[ci]

                if assigned_counts[k] >= warm_up:
                    matching_rates[k][(ci, sj)] += 1
                assigned_counts[k] += 1
        for key in sorted(matching_rates[k].keys()):
            matching_rates[k][key] = beta.sum()*float(matching_rates[k][key])/sim_len

        print 'ending sim ', k, 'duration:', time() - start_time

    r_dict = dict((p, {'rcs-sim': 0, 'stdev-sim': 0}) for p in zip(*q.nonzero()))
    r = np.zeros(q.shape)

    for p in zip(*q.nonzero()):
        for sim in matching_rates:
            r_dict[p]['rcs-sim'] += sim[p]/sims
            r_dict[p]['stdev-sim'] += (sim[p])**2/sims
            r[p[0], p[1]] += sim[p]/sims
        r_dict[p]['V-sim'] = (r_dict[p]['stdev-sim'] - r_dict[p]['stdev-sim']**2)**0.5

    return r_dict, r


def queueing_simulator(lamda, mu, q, eta=None, phi=None, prt=False, sims=500, warm_up=10**5, sim_len=10**6,
                       report_every=10**5, seed=None):

    # lambda is a saved word in Python hence we use the abbreviation lamda

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    nc = len(lamda)   # nc - number of customers
    ns = len(mu)  # ns- number of servers
    rnc = range(nc)  # 0,1,2,...,nc-1
    rns = range(ns)  # 0,1,2,...,ns-1
    matching_rates = []
    qs = dict((i, set(np.nonzero(q[:, i])[0])) for i in rns)  # adjacency list for servers
    qc = dict((j, set(np.nonzero(q[j, :])[0])) for j in rnc)  # adjacency list for customers

    def arrival():

        arrival_count = 0
        for customer_arrival in c_stream:

            arrival_count += 1
            if arrival_count > 0 and arrival_count % report_every == 0:
                print arrival_count, 'customers arrived - time_elapsed:', time() - start_time[0]

            ci = customer_arrival[0]
            yield env.timeout(customer_arrival[1])
            arrival_time = env.now
            if len(c_queues[ci]) > 0:
                c_queues[ci].appendleft(arrival_time)
            else:
                available_servers = []
                for sj in qc[ci]:
                    if s_state[sj] == 0:
                        heapq.heappush(available_servers, (s_idled_at[sj], sj))
                if len(available_servers) > 0:
                    #print 'these are the available servers:',available_servers
                    sj = heapq.heappop(available_servers)
                    #print 'server choosen is :',sj
                    env.process(assignment(ci, sj[1]))
                else:
                    c_queues[ci].appendleft(arrival_time)

    def assignment(ci, sj):

        assignment_count[0] += 1
        if assignment_count[0] > warm_up:
            matching_rates[k][ci, sj] += 1.0
        s_state[sj] = 1
        service_time = random.expovariate(mu[sj])
        yield env.timeout(service_time)
        available_customers = []
        available_customers2 = []
        for ci in qs[sj]:
            if len(c_queues[ci]) > 0:
                heapq.heappush(available_customers, (c_queues[ci][-1], ci))
        if len(available_customers) > 0:
            ci = heapq.heappop(available_customers)
            c_queues[ci[1]].pop()
            env.process(assignment(ci[1], sj))
        else:
            s_state[sj] = 0
            s_idled_at[sj] = env.now

    for k in range(sims):
        print 'starting sim ', k+1
        start_time = [time()]
        env = simpy.Environment()
        matching_rates.append(np.zeros(q.shape))
        c_queues = tuple(deque() for c in rnc)
        s_idled_at = [0 for _ in rns]
        s_state = [0 for _ in rns]
        total_rate = np.asscalar(lamda.sum())
        arrival_ratios = lamda*(1/total_rate)
        customer_arrivals = np.random.choice(a=rnc, size=sim_len, p=arrival_ratios)
        interarrival_times = np.random.exponential(scale=1/total_rate, size=sim_len)
        c_stream = zip(customer_arrivals,interarrival_times)
        assignment_count = [0]
        sim_time = interarrival_times.sum()
        env.process(arrival())
        env.run(until=sim_time)
        matching_rates[k] = (1.0/assignment_count[0]) * (matching_rates[k])
        print matching_rates[k]
        print 'ending queueing sim ', k+1, 'duration:', time() - start_time[0]

    matching_rates_mean = sum(matching_rates)*(1.0/sims)
    if sims > 1:
        matching_rates_stdev = (sum((matching_rates[k]-matching_rates_mean)**2 for k in range(sims))*(1.0/(sims-1)))**0.5
    else:
        matching_rates_stdev = []

    return matching_rates_mean, matching_rates_stdev


def online_matching_simulator(lamda, q, w=None, eta=None, prt=False, sims=500, abandonments=False, fifo=True,
                              warm_up=10**5, self_match=True, sim_len=10**6,report_every=10**5, seed=None):

    def simulate():

        in_system = set()

        print 'total_customers', len(arrival_stream)

        for event in simulation_stream:

            # event = (0: serial #, 1: arrival_time, 2: abandonment_time,  3: arrival_type, 4: ['a'])
            # event = (0: serial #, 1: abandonment_time, 2:arrival_time ,  3: arrival_type, 4: ['a'])

            serial = event[0]
            cur_time = event[1]
            i = event[3]

            if event[4] == 'a':
                # print event
                # print type(event[0])

                if event[0] > 0 and event[0] % report_every == 0:
                    print arrival_count[0], 'customer # arrived - time_elapsed:', time() - start_time[0]
                    print arrival_count[0], 'customer # arrived - sim time:', cur_time

                arrival_count[0] += 1.0

                if len(queues[i]) > 0:
                    if self_match:
                        queues[i].appendleft(cur_time)
                        in_system.add(serial)
                    else:
                        waiting_time = cur_time - queues[i][-1]
                        queues[i].pop()
                        matching_rates[k][i, i] += 2.0
                        assignment_count[0] += 1.0
                        waiting_times_edge[k][i, i] += waiting_time
                        waiting_times_node[k][i] += waiting_time

                else:
                    available_matches = []
                    for j in adj_list[i]:
                        # print ql[i]
                        # print 'i', i
                        # print 'j', j
                        if len(queues[j]) > 0:
                            if fifo:
                                heapq.heappush(available_matches, (-1*(cur_time - queues[j][-1])*w[i, j], j))
                            else:
                                heapq.heappush(available_matches, (-1*(len(queues[j]))*w[i, j], j))
                    if len(available_matches) > 0:
                        if fifo:
                            waiting_time, j = heapq.heappop(available_matches)
                        else:
                            queue_len, j = heapq.heappop(available_matches)
                            waiting_time = cur_time - queues[j][-1]
                        queues[j].pop()
                        matching_rates[k][min(i, j), max(i, j)] += 2.0
                        assignment_count[0] += 1.0
                        waiting_times_edge[k][j, i] += waiting_time
                        waiting_times_node[k][j] += waiting_time
                    else:
                        queues[i].appendleft(cur_time)
                        in_system.add(serial)

            elif event[4] == 'd':

                arrival_time = event[2]

                if event[0] in in_system:
                    try:
                        queues[i].remove(arrival_time)
                    except:
                        continue
                    in_system.discard(serial)
                    abandonment_rates[k][i] += 1.0
                    # print 'cur_time', cur_time
                    # print 'arrival_time', arrival_time
                    # print cur_time - arrival_time
                    # print 'k', k
                    # print 'i', i
                    # print 'abandonment_times', abandonment_times
                    abandonment_wait_times[k][i] += cur_time - arrival_time

        for i in range(len(queues)):
            print i, len(queues[i])


    if w is None:
        w = q

    if abandonments:
        non_abandoning_types = set(np.where(eta == 0)[0])
        if len(non_abandoning_types):
            eta[np.where(eta == 0)] = 1.0

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # nc - number of customers
    n = len(lamda)  # ns- number of servers
    rn = range(len(lamda))
    print 'n',n
    matching_rates = []
    waiting_times_edge = []
    waiting_times_node = []
    abandonment_rates = []
    abandonment_wait_times = []

    adj_list = dict((i, set(np.nonzero(q[:, i])[0])) for i in rn)  # adjacency list for nodes


    for k in range(sims):

        print 'starting sim ', k+1

        start_time = [time()]

        matching_rates.append(np.zeros(q.shape))
        waiting_times_node.append(np.array([0.0]*n))
        waiting_times_edge.append(np.zeros(q.shape))
        abandonment_rates.append(np.array([0.0]*n))
        abandonment_wait_times.append(np.array([0.0]*n))

        queues = tuple(deque() for i in rn)

        total_rate = np.asscalar(lamda.sum())
        arrival_ratios = lamda*(1/total_rate)
        arrival_types = np.random.choice(a=rn, size=sim_len, p=arrival_ratios)
        interarrival_times = np.random.exponential(scale=1/total_rate, size=sim_len)
        arrival_times = np.cumsum(interarrival_times)
        patience_times = np.random.exponential(eta[arrival_types])
        abandonment_times = arrival_times + patience_times
        abandonment_stream = zip(range(sim_len), abandonment_times, arrival_times, arrival_types, ['d']*sim_len)
        abandonment_stream = filter(lambda x: x[3] not in non_abandoning_types, abandonment_stream)
        arrival_stream = zip(range(sim_len), arrival_times, abandonment_times, arrival_types, ['a']*sim_len)
        simulation_stream = sorted(arrival_stream + abandonment_stream, key=lambda x: x[1])
        assignment_count = [0]
        arrival_count = [0]
        sim_time = interarrival_times.sum()
        simulate()
        # matching_rates[k] = (1.0/arrival_count[0]) * (matching_rates[k])
        # abandonment_rates[k] = (1.0/arrival_count[0]) * (abandonment_rates[k])
        print matching_rates[k]
        print 'ending queueing sim ', k+1, 'duration:', time() - start_time[0]

    matching_rates_mean = sum(matching_rates)*(1.0/sims)
    waiting_times_mean_edge = sum(waiting_times_edge)*(1.0/sims)
    waiting_times_mean_node = sum(waiting_times_node)*(1.0/sims)
    abandonment_rates_mean = sum(abandonment_rates)*(1.0/sims)
    abandonment_wait_times_mean = sum(abandonment_wait_times)*(1.0/sims)

    if sims > 1:

        matching_rates_stdev = (sum((matching_rates[n]-matching_rates_mean)**2
                                    for n in range(sims))*(1.0/(sims-1)))**0.5
        waiting_times_stdev_edge = (sum((waiting_times_edge[n]-waiting_times_mean_edge)**2
                                        for n in range(sims))*(1.0/(sims-1)))**0.5
        waiting_times_stdev_node = (sum((waiting_times_node[n]-waiting_times_mean_node)**2
                                        for n in range(sims))*(1.0/(sims-1)))**0.5
        abandonment_rates_stdev = (sum((abandonment_rates[n]-abandonment_rates_mean)**2
                                       for n in range(sims))*(1.0/(sims-1)))**0.5
        abandonment_wait_times_stdev = (sum((abandonment_wait_times[n]-abandonment_wait_times_mean)**2
                                            for n in range(sims))*(1.0/(sims-1)))**0.5
    else:
        matching_rates_stdev = []
        waiting_times_stdev_edge = []
        waiting_times_stdev_node = []
        abandonment_rates_stdev = []
        abandonment_wait_times_stdev = []

    return dict(zip(
        ['matching_rates_mean','matching_rates_stdev', 'waiting_times_mean_edge', 'waiting_times_stdev_edge',
         'waiting_times_mean_node', 'waiting_times_stdev_node','abandonment_rates_mean', 'abandonment_rates_stdev',
         'abandonment_times_mean', 'abandonment_times_stdev'],
        [matching_rates_mean,matching_rates_stdev, waiting_times_mean_edge, waiting_times_stdev_edge,
         waiting_times_mean_node, waiting_times_stdev_node,abandonment_rates_mean,
         abandonment_rates_stdev,abandonment_wait_times_mean, abandonment_wait_times_stdev]))


if __name__ == '__main__':

    m = 20
    n = 20
    d = 4
    plot = True
    g = undirected_grid_2d_bipartie_graph(m, n, d, d, plot=plot, alpha=0.95, l_norm=1)
    grid_adj_mat = g['grid_adj_amt']
    lamda = np.concatenate((g['lamda_d'], g['lamda_s']))
    q = sps.vstack((sps.hstack((0*sps.eye(m*n), grid_adj_mat)),
                                       sps.hstack((grid_adj_mat, 0*sps.eye(m*n)))))
    eta = np.ones(len(lamda))
    online_matching_simulator(lamda=lamda, q=q, eta=eta)

    # eta = np.array([1.0, 1.0, 1.0])
    # reses = online_matching_simulator(lamda, q, w=None, eta=eta, sims=1, abandonments=True, fifo=True, sim_len=100, report_every=10)
    # for key, val in reses.items():
    #     print key
    #     print val

    # pre = pd.DataFrame.from_csv('C:\Users\deang\Dropbox\Research\Flexible_Service_Systems\Software2.0\upTo101wMaxMinMinMax.csv')
    # cases = pre[['size', 'serial']].drop_duplicates()
    # full_df = pd.DataFrame(columns={'size', 'serial', 'ci', 'sj', 'rcs-min_gap'})
    #
    # for index, row in cases.iterrows():
    #
    #     q = np.zeros((row[0], row[0]))
    #     alpha = [0]*row[0]
    #     beta = [0]*row[0]
    #     case = pre[(pre['size'] == row[0]) & (pre['serial'] == row[1])]
    #     print case
    #     for i,r in case[['ci','sj','rcs-ent']].iterrows():
    #
    #         alpha[int(r[0])] += r[2]
    #         beta[int(r[1])] += r[2]
    #         q[int(r[0]), int(r[1])] = 1.0
    #
    #     r_min_max_dict, r_max_min = solve_maxmin(alpha, beta, q, prob_type='MinGap')
    #     r_min_max_df = pd.DataFrame.from_dict(r_min_max_dict, orient='index')
    #     r_min_max_df = r_min_max_df.reset_index(level=[0, 1])
    #     r_min_max_df = r_min_max_df.rename(columns={'level_0': 'ci', 'level_1': 'sj'})
    #
    #     r_min_max_df.loc[:,'size'] = row[0]
    #     r_min_max_df.loc[:,'serial'] = row[1]
    #
    #     full_df = pd.concat([full_df, r_min_max_df])
    #
    #
    # post = pd.merge(left=pre, right=full_df, on=['size', 'serial', 'ci', 'sj'], how='left')
    # post.to_csv('upTo101wMinGap.csv')

    # q = k_chain(6, 6, 3)
    # alphaII = np.array([0.5/(1+k) for k in range(10)])
    # betaII = np.array([0.5/(1+k) for k in range(10)])
    #
    # q = [[1]*max(k,2) + (10-max(k,2))*[0] for k in range(10,0,-1)]
    # q = sps.csr_matrix(np.array(q))
    # print q
    # # max_min_r_dict, max_min_r = solve_maxmin(alphaII, betaII, q)
    # aw_r_dict, aw_r = advanced_matching_simulator(alphaII, betaII, q, sim_len=10**5)
    # ent_r_dict, ent_r = entropy_approximation_solver(alphaII, betaII, q)
    # for key in aw_r_dict:
    #     print key, aw_r_dict[key], ent_r_dict[key]


    # print 'min_max'
    # print max_min_r
    # print 'aw rates'
    # print aw_r


    # full_df = pd.DataFrame(columns=
    #                        {'size', 'serial', 'ci', 'sj',
    #                         'rcs-ent','rcs-aw', 'rcs-sim', 'stdev-sim',
    #                         'solve time ent','calc time','sim_time'})
    # print full_df
    # for size in [8, 10, 30, 50, 101]:
    #     for k in range(10):
    #         feasible = False
    #         while not feasible:
    #             alpha_v = random_unit_partition(size)
    #             beta_v = random_unit_partition(size)
    #             q = erdos_renyi_connected_graph(size, size, (2*log(size)-2)/size)
    #             #q = k_chain(size, size, 2)
    #             feasible, util = feasibility_check(alpha_v, beta_v, q)
    #         s = time()
    #         ent_r_dict, ent_r = entropy_approximation_solver(alpha_v, beta_v, q)
    #         solve_time = time() - s
    #         print 'done solving:'
    #         s = time()
    #         sim_r_dict, sim_r = advanced_matching_simulator(alpha_v, beta_v, q, sim_len=(10**4)*size, sims=100)
    #         print 'done simulating'
    #         sim_time = time() - s
    #         if size <= 8:
    #             s = time()
    #             aw_r_dict, aw_r = matching_rate_calculator(alpha_v, beta_v, q)
    #             aw_calc_time = time() - s
    #         else:
    #             aw_r_dict = dict((p, {'rcs-aw': '-'}) for p in zip(*q.nonzero()))
    #             aw_calc_time = '-'
    #
    #         ent_r_df = pd.DataFrame.from_dict(ent_r_dict, orient='index')
    #         sim_r_df = pd.DataFrame.from_dict(sim_r_dict, orient='index')
    #         awc_r_df = pd.DataFrame.from_dict(sim_r_dict, orient='index')
    #         ent_r_df = ent_r_df.reset_index(level=[0, 1])
    #         sim_r_df = sim_r_df.reset_index(level=[0, 1])
    #         ent_r_df = ent_r_df.rename(columns={'level_0': 'ci', 'level_1': 'sj'})
    #         sim_r_df = sim_r_df.rename(columns={'level_0': 'ci', 'level_1': 'sj'})
    #         data_df = pd.merge(left=ent_r_df, right=sim_r_df, on=['ci', 'sj'], how='left')
    #         data_df.loc[:, 'size'] = size
    #         data_df.loc[:, 'serial'] = k
    #         data_df.loc[:, 'solve time'] = solve_time
    #         data_df.loc[:, 'sim time'] = sim_time
    #         data_df.loc[:, 'calc time'] = aw_calc_time
    #         full_df = pd.concat([full_df, data_df])
    #     full_df = full_df[['size', 'serial', 'ci', 'sj',
    #                        'rcs-ent','rcs-aw', 'rcs-sim', 'stdev-sim',
    #                        'solve time', 'calc time', 'sim time']]
    #     full_df.to_csv('upTo' + str(size) + '.csv')




    # print 'from sim:'
    # print sps.csr_matrix(r)
    # print 'from ent approx:'
    # print ent_approx
    # print np.abs(ent_approx.todense() - r)
    # print np.linalg.norm(ent_approx.todense() - r, 1)

    # for size in [10000, 5000, 1000, 500, 50, 5]:
    #     q = k_chain(size, size, 3)
    #     print 'built'
    #     betaI = np.array([0.15]*size)
    #     alphaI = np.array([0.15]*size)
    #     if size <= 100:
    #         r_m = advanced_matching_simulator(alphaI, betaI, q, sim_len=10**6)
    #     else:
    #         r_m = advanced_matching_simulator(alphaI, betaI, q, sim_len=(10**6)*(size/100))
    #
    #     pr_m = pd.DataFrame(r_m).transpose()
    #     print pr_m

    #
        #pr_m.to_csv('matching_rates_table1_' + str(size) + '.csv')
        #pr_v.to_csv('matching_rate_stdev_table1_' + str(size) + '.csv')

    # r_m, r_v = matching_simulator(alphaI, betaI, q, sims=100, side='customer')
    # pr_m = pd.DataFrame(r_m)
    # pr_v = pd.DataFrame(r_v)
    #
    # pr_m.to_csv('matching_rates_table1_customer.csv')
    # pr_v.to_csv('matching_rate_stdev_table1_customer.csv')

    # r_mc = matching_rate_calculator(alphaI, betaI, q)

    # print r_mc

    #r_m, r_v = queueing_simulator(alphaI, betaI, q, prt=False, sims=100, warm_up=10**5, sim_len=10**6, seed=None)

    # print r_m
    #
    # pr_m = pd.DataFrame(r_m)
    # pr_v = pd.DataFrame(r_v)
    #
    # pr_m.to_csv('matching_rates_table1_q.csv')
    # pr_v.to_csv('matching_rate_stdev_table1_q.csv')
