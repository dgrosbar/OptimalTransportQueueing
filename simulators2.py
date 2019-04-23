from time import time
import numpy as np
from math import log
import heapq
from collections import deque
import random as random
from generators import undirected_grid_2d_bipartie_graph
from scipy import sparse as sps
from fss_utils import con_print
from functools import partial
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import simpy
import pandas as pd
import os
import os.path
import shutil
from scipy import sparse as sps
from memory_profiler import profile
import gc


def online_matching_simulator(lamda, q, w_i=None, w_j=None, eta=None, prt=False, abandonments=False,
                              policy='fifo', tie_break=False, self_match=False,
                              sims=500, warm_up=10**5, sim_len=10**6,report_every=10**5, seed=None):
    sim_print = partial(con_print, con=prt)

    def simulate():

        in_system = set()

        sim_print(('total_customers', len(arrival_stream)))

        for event in simulation_stream:

            # event = (0: serial #, 1: arrival_time, 2: abandonment_time,  3: arrival_type, 4: ['a'])
            # event = (0: serial #, 1: abandonment_time, 2:arrival_time ,  3: arrival_type, 4: ['a'])

            serial = event[0]
            cur_time = event[1]
            i = event[3]

            if event[4] == 'a':
                sim_print('arrival ' + str(event[0]))
                sim_print(event)
                if event[0] > 0 and event[0] % report_every == 0:
                    print(str(arrival_count[0]) + ' customer # arrived - time_elapsed: ' + str(time() - start_time[0]))
                    print(str(arrival_count[0]) + ' customer # arrived - sim time: ' + str(cur_time))

                arrival_count[0] += 1.0
                if len(queues[i]) > 0:
                    sim_print('arrival ' + str(serial) + 'is joining queue ' + str(i))
                    if not self_match:
                        queues[i].appendleft(cur_time)
                        in_system.add(serial)
                    else:
                        waiting_time = cur_time - queues[i][-1]
                        queues[i].pop()
                        matching_rates_edge[k][i, i] += 2.0
                        matching_rates_node[k][i] += 2.0
                        assignment_count[0] += 1.0
                        waiting_times_edge[k][i, i] += waiting_time
                        waiting_times_node[k][i] += waiting_time

                else:
                    sim_print('looking for a match for arrival ' + str(serial) + ' of type ' + str(i))
                    available_matches = []
                    sim_print('adj list for arrival type ' + str(i) + ':')
                    sim_print(adj_list[i])
                    for j in adj_list[i]:
                        if len(queues[j]) > 0:
                            if policy == 'fifo':
                                heapq.heappush(available_matches,
                                               (-1*(cur_time - queues[j][-1])*w[i, j],-1*(cur_time - queues[j][-1]), j))
                            elif policy == 'max_weight':
                                heapq.heappush(available_matches,
                                               (-1*(len(queues[j]))*w[i, j], -1*(cur_time - queues[j][-1]), j))
                            else:
                                heapq.heappush(available_matches,
                                               (-1*w[i, j], -1*(cur_time - queues[j][-1]), j))

                    if len(available_matches) > 0:

                        if not tie_break or len(available_matches) == 1:
                            weight, waiting_time, j = heapq.heappop(available_matches)
                        else:
                            min_weight_matches = []
                            min_weight = None
                            while len(available_matches) > 1:
                                weight, waiting_time, j = heapq.heappop(available_matches)
                                if min_weight is None:
                                    min_weight = weight
                                if weight == min_weight:
                                    heapq.heappush(min_weight_matches, (waiting_time, weight, j))
                                else:
                                    break
                            waiting_time, weight, j = heapq.heappop(min_weight_matches)

                        sim_print(('customer of type: ', j, 'chosen. waiting time:', -1 * waiting_time))
                        queues[j].pop()

                        matching_rates_edge[k][min(i, j), max(i, j)] += 2.0
                        matching_rates_node[k][i] += 1.0
                        matching_rates_node[k][j] += 1.0
                        sim_print(('making a match of', min(i,j),' to', max(i, j)))
                        sim_print(matching_rates_edge[k][min(i, j), max(i, j)])
                        assignment_count[0] += 1.0
                        waiting_times_edge[k][j, i] += -1 * waiting_time
                        waiting_times_node[k][j] += -1 * waiting_time

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
                    abandonment_wait_times[k][i] += cur_time - arrival_time

        waiting_times_node[k] = waiting_times_node[k]/matching_rates_node[k]

    if w is None:

        w = q

    if abandonments:
        non_abandoning_types = set(np.where(eta == 0)[0])
        if len(non_abandoning_types):
            eta[np.where(eta == 0)] = 1.0

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    n = len(lamda)
    rn = range(len(lamda))

    matching_rates_node = MultiSimStat(n)
    matching_rates_edge = MultiSimStat((n, n))
    waiting_times_edge = MultiSimStat((n, n))
    waiting_times_node = MultiSimStat(n)
    abandonment_rates = MultiSimStat(n)
    abandonment_wait_times = MultiSimStat(n)

    adj_list = dict((i, set(np.nonzero(q[:, i])[0])) for i in rn)  # adjacency list for nodes

    for k in range(sims):

        print('starting sim ', k+1)

        start_time = [time()]

        queues = tuple(deque() for i in rn)

        total_rate = np.asscalar(lamda.sum())
        arrival_ratios = lamda*(1/total_rate)
        arrival_types = np.random.choice(a=rn, size=sim_len, p=arrival_ratios)
        interarrival_times = np.random.exponential(scale=1/total_rate, size=sim_len)
        arrival_times = np.cumsum(interarrival_times)

        if abandonments:

            patience_times = np.random.exponential(1/eta[arrival_types])
            abandonment_times = arrival_times + patience_times
            abandonment_stream = zip(range(sim_len), abandonment_times, arrival_times, arrival_types, ['d']*sim_len)
            abandonment_stream = filter(lambda x: x[3] not in non_abandoning_types, abandonment_stream)
            arrival_stream = zip(range(sim_len), arrival_times, abandonment_times, arrival_types, ['a']*sim_len)
            simulation_stream = sorted(arrival_stream + abandonment_stream, key=lambda x: x[1])
        else:
            arrival_stream = zip(range(sim_len), arrival_times, [0]*sim_len, arrival_types, ['a']*sim_len)
            simulation_stream = sorted(arrival_stream, key=lambda x: x[1])

        assignment_count = [0]
        arrival_count = [0]
        sim_time = interarrival_times.sum()
        simulate()

        print('matching rates', sps.csr_matrix(matching_rates_node[k]))
        print('ending queueing sim ', k+1, 'duration:', time() - start_time[0])

    return dict(zip(
        ['matching_rates_node',
         'matching_rates_edge',
         'waiting_times_edge',
         'waiting_times_node',
         'abandonment_rates',
         'abandonment_wait_times'],
        [matching_rates_node,
         matching_rates_edge,
         waiting_times_edge,
         waiting_times_node,
         abandonment_rates,
         abandonment_wait_times]))


class SimExperiment(object):

    # Multi sim manitinas a connection to a directory that stores the result files

    def __init__(self, dir):

        self.dir = dir

    def simulate(self, save_exp=True, del_old=False,  **kwargs):

        sim = MultiSim(**kwargs)
        sim.run_queueing_simulation()
        reses_i, reses_j, reses_ij = sim.parse_sim_data()

        if save_exp:
            if not os.path.exists(self.dir):
                    os.mkdir(self.dir)
            elif del_old:
                shutil.rmtree(self.dir, ignore_errors=True)
                os.mkdir(self.dir)

            for name, res in zip(['reses_ij', 'reses_i', 'reses_j'], [reses_ij, reses_i, reses_j]):
                # print(name)
                # print(res)
                filename = self.dir + '/' + name + '.csv'
                # print('filename', filename)
                if os.path.exists(filename):
                    tmp_res = pd.DataFrame.from_csv(filename)
                    tmp_res = pd.concat((tmp_res, res))
                    tmp_res.to_csv(filename)
                else:
                    res.to_csv(filename)
        gc.collect()

    def get_file(self, res_name):

        filename = self.dir + '/' + res_name + '.csv'
        res = pd.read_csv(filename)
        return res

    def save_res(self, res, res_name):

        filename = self.dir + '/' + res_name + '.csv'
        res.to_csv(filename)
        return True


class MultiSim(object):

    """ MultiSim is a single single Q, lamda, mu, s, and can support experiments with various policies and save to the
        same file every time"""

    def __init__(self, lamda, mu, q, eta=None, s=None,
                 interarrival_dist='exponential', service_dist='exponential', abandonment_dist='exponential',
                 sim_name='_', i_policy='alis',j_policy='fifo', w_i=None, w_j=None, m=None,
                 prt=False, report_every=10**5, warm_up=None, sim_len=None, seed=None, sims=30, full_gini=False,
                 **kwargs):

        self.added_cols = kwargs

        self.lamda = lamda
        self.mu = mu
        self.eta = eta
        self.Q = q
        self.S = np.ones(len(lamda)) if s is None else s
        self.Wi = q if w_i is None else w_i
        self.Wj = q if w_j is None else w_j
        self.M = m

        self.m = self.Q.shape[0]
        self.n = self.Q.shape[1]
        self.rm = range(self.m)
        self.rn = range(self.n)

        self.interarrival_dist = interarrival_dist
        self.service_dist = service_dist
        self.abandonment_dist = abandonment_dist

        self.log_periods = np.zeros((2, sims))

        if sps.isspmatrix(self.Q):
            self.qs = dict((i, set(np.nonzero(self.Q[:, i])[0])) for i in self.rn)  # adjacency list for servers
            self.qc = dict((j, set(np.nonzero(self.Q[j, :])[1])) for j in self.rm)  # adjacency list for customers
        else:
            self.qs = dict((i, set(np.nonzero(self.Q[:, i])[0])) for i in self.rn)  # adjacency list for servers
            self.qc = dict((j, set(np.nonzero(self.Q[j, :])[0])) for j in self.rm)  # adjacency list for customers
        if sps.isspmatrix(self.Q):
            self.service_rate = (sps.diags(self.S, format='csr').dot(self.Q)).dot(sps.diags(1.0/self.mu, format='csr'))
        else:
            self.service_rate = self.S.reshape((self.m, 1)).dot((1.0/self.mu).reshape((1, self.n)))

        self.sims = sims

        if sim_len is None:
            self.sim_len = 10**5 + 10**4  #len(self.lamda) * (10**5 + 10**4)
        elif warm_up is not None:
            self.sim_len = sim_len + warm_up
        else:
            self.sim_len = sim_len + int(0.1 * sim_len)
        if warm_up is None:
            self.warm_up = self.sim_len - int((10.0/11.0)*self.sim_len)
        if report_every is None:
            self.report_every = self.sim_len - (10.0/11.0)*self.sim_len
        else:
            self.report_every = report_every

        self.sim_name = sim_name
        self.base_seed = seed

        self.total_rate = np.asscalar(self.lamda.sum())
        self.arrival_ratios = self.lamda*(1/self.total_rate)

        self.matching_rates = np.zeros((3, sims, self.m, self.n))
        self.waiting_times = np.zeros((2, sims, self.m, self.n))
        self.service_times = np.zeros((1, sims, self.m, self.n))
        self.idle_periods = np.zeros((3, sims, self.n))
        self.queue_lengths = np.zeros((2, sims, self.m))

        self.i_policy = i_policy
        self.j_policy = j_policy

        self.arrival_count = None
        self.assignment_count = None
        self.abandonment_count = None
        self.logging = None
        self.log_start = None
        self.start_time = None
        self.c_queues = None
        self.s_servers = None
        self.k = None
        self.sim_time = None
        self.c_stream = None
        self.env = None

        self.full_gini = full_gini
        self.gini = [[]*sims]

    def run_queueing_simulation(self, matching=False):

        for k in range(self.sims):

            self.assignment_count = 0
            self.abandonment_count = 0
            self.arrival_count = 0
            self.start_time = time()
            self.logging = False
            self.log_start = 0.0
            self.env = simpy.Environment()
            self.c_queues = [tuple(deque() for c in self.rm), tuple(deque() for c in self.rm), [0 for _ in self.rm]]
            self.s_servers = [[0 for _ in self.rn], [0 for _ in self.rn]]
            self.k = k

            print('sim_len', self.sim_len, 'arrivals')

            if self.base_seed is not None:
                np.random.seed(self.base_seed + k*10*self.sim_len)
            customer_arrivals = np.random.choice(a=self.rm, size=self.sim_len, p=self.arrival_ratios)
            interarrival_times = np.random.exponential(scale=1/self.total_rate, size=self.sim_len)

            customer_service_time_pct = np.random.uniform(0.0, 1.0, size=self.sim_len)

            self.c_stream = zip(customer_arrivals, interarrival_times, customer_service_time_pct)

            print('total work load arrival rate is:', (self.lamda*self.S).sum())

            self.sim_time = int(interarrival_times.sum())
            self.log_periods[1, k] = self.sim_time

            self.env.process(self.event_stream())
            self.env.run(until=self.sim_time)

            print('ending queueing sim ', k+1, 'duration:', time() - self.start_time)

    def event_stream(self):

        for customer_arrival in self.c_stream:

            self.arrival_count += 1

            if self.arrival_count > 0 and self.arrival_count % self.report_every == 0:
                print(self.arrival_count, 'customers arrived - time_elapsed:', time() - self.start_time)
                print('assignment_count is:', self.assignment_count)
                print('logging is:', self.logging)

            i, time_to_arrival, service_time_pct = customer_arrival

            yield self.env.timeout(time_to_arrival)

            arrival_time = self.env.now
            c_queue_len = len(self.c_queues[0][i])

            if c_queue_len > 0:

                self.c_queues[0][i].appendleft(arrival_time)
                self.c_queues[1][i].appendleft(service_time_pct)
                time_in_state = self.env.now - max(self.c_queues[2][i], self.log_start)
                self.c_queues[2][i] = self.env.now

                if self.logging:
                    self.queue_lengths[:, self.k, i] += time_in_state * np.array([c_queue_len, c_queue_len**2])

            else:

                available_servers = []

                for j in self.qc[i]:

                    if self.s_servers[0][j] == 0:

                        if self.i_policy == 'alis':
                            heapq.heappush(available_servers,
                                           (-self.Wi[i, j] * (self.env.now - self.s_servers[1][j]), j))
                        elif self.i_policy == 'rand':
                            heapq.heappush(available_servers,
                                           (-self.Wi[i, j] * random.uniform(0,1), j))
                        elif self.i_policy == 'prio':
                            heapq.heappush(available_servers,
                                           (-self.Wi[i, j], j))
                        else:
                            heapq.heappush(available_servers,
                                           (-self.Wi[i, j] * (self.env.now - self.s_servers[1][j]), j))

                if len(available_servers) > 0:

                    weight, j = heapq.heappop(available_servers)

                    if self.logging:

                        idle_time = self.env.now - max(self.s_servers[1][j], self.log_start)
                        self.idle_periods[:, self.k, j] += np.array([1.0, idle_time, idle_time**2])
                        if self.full_gini:
                            heapq.heappush(self.gini[self.k], 0)

                    self.s_servers[1][j] = self.env.now
                    self.s_servers[0][j] = 1
                    self.env.process(self.assignment(i, j, 0, service_time_pct))

                else:
                    self.c_queues[0][i].appendleft(arrival_time)
                    self.c_queues[1][i].appendleft(service_time_pct)
                    self.c_queues[2][i] = self.env.now

    def assignment(self, i, j, s_c, service_time_pct):

        self.assignment_count += 1

        if not self.logging:
            if self.assignment_count > self.warm_up:
                print('assignment_count', self.assignment_count)
                print('warm_up', self.warm_up)
                self.logging = True
                self.log_periods[0, self.k] = self.env.now
                self.log_start = self.env.now

        #service_time = random.expovariate(self.mu[j]/self.S[i])
        service_time = -log(service_time_pct)*self.service_rate[i, j]

        if self.logging:
            self.matching_rates[:, self.k, i, j] += [1.0, s_c, 1.0 - s_c]
            self.service_times[:, self.k, i, j] += min(service_time, self.sim_time - self.env.now)

        yield self.env.timeout(service_time)

        available_customers = []

        for i in self.qs[j]:

            if len(self.c_queues[0][i]) > 0:

                if self.j_policy == 'fifo':
                    heapq.heappush(available_customers,
                                   (-self.Wj[i, j]*(self.env.now - self.c_queues[0][i][-1]), i))
                elif self.j_policy == 'max_weight':
                    heapq.heappush(available_customers,
                                   (-self.Wj[i, j]*len(self.c_queues[0][i]), i))
                elif self.j_policy == 'util':
                    heapq.heappush(available_customers,(-self.Wj[i, j], i))
                elif self.j_policy == 'rand':
                    for _ in range(len(self.c_queues[0][i])):
                        heapq.heappush(available_customers,
                                       (-self.Wj[i,j]*random.uniform(0, 1), i))
                elif self.j_policy == 'prio':
                    for _ in range(len(self.c_queues[0][i])):
                        heapq.heappush(available_customers,
                                       (-self.Wj[i, j], i))

        if len(available_customers) > 0:

            weight, i = heapq.heappop(available_customers)

            if self.logging:

                wait_time = self.env.now - max(self.c_queues[0][i][-1], self.log_start)

                queue_len = len(self.c_queues[0][i])
                time_in_state = self.env.now - max(self.c_queues[2][i], self.log_start)
                self.waiting_times[:, self.k, i, j] += np.array([wait_time, wait_time**2.0])
                if self.full_gini:
                    heapq.heappush(self.gini[self.k], wait_time)
                self.queue_lengths[:, self.k, i] += time_in_state*np.array([queue_len, queue_len**2.0])

            self.c_queues[0][i].pop()
            service_time = self.c_queues[1][i].pop()
            self.c_queues[2][i] = self.env.now
            self.env.process(self.assignment(i, j, 1.0, service_time))

        else:

            self.s_servers[0][j] = 0
            self.s_servers[1][j] = self.env.now

    def parse_sim_data(self):

        quals = self.Q.nonzero()

        reses_ij = pd.DataFrame.from_records(zip(*quals), columns=['i', 'j'])
        reses_ij.loc[:, 'ij'] = reses_ij['i'].apply(str) + ',' + reses_ij['j'].apply(str)
        reses_i = pd.DataFrame({'i': self.rm})
        reses_j = pd.DataFrame({'j': self.rn})

        if self.full_gini:
            full_ginis = []
            for y in self.gini:
                n = len(y)
                y = np.array([heapq.heappop(y) for _ in range(n)])
                print(y)
                print(y*np.arange(1, n+1, 1))
                print(2*(y*np.arange(1, n+1, 1)).sum())
                print(y.sum())
                print(n*y.sum())
                print (((2*y*np.arange(1, n+1, 1)).sum())/(n*y.sum())) - ((n+1)/n)
                full_ginis.append((((2*y*np.arange(1, n+1, 1)).sum())/(n*y.sum())) - ((n+1)/n))
                print(sum(full_ginis)/float(len(full_ginis)))
            reses_i.loc[:, 'full_gini'] = sum(full_ginis)/float(len(full_ginis))

        sim_attr = ['interarrival_dist','service_dist', 'abandonment_dist', 'i_policy', 'j_policy', 'sim_name']

        for attr_name in sim_attr:
            attr_val = getattr(self, attr_name)
            for res in [reses_ij, reses_i, reses_j]:
                res.loc[:, attr_name] = attr_val

        for key in self.added_cols:

            for res_name, res in zip(['i', 'j', 'ij'], [reses_i, reses_j, reses_ij]):
                if res_name in self.added_cols[key]['res']:
                    if res_name == 'ij' and self.added_cols[key]['matrix']:
                        res.loc[:, key] = self.added_cols[key]['val'][quals]
                    else:
                        res.loc[:, key] = self.added_cols[key]['val']

        if sps.isspmatrix(self.Wi):
            reses_ij.loc[:, 'w_ij_i'] = self.Wi.todense().A[quals]
        else:
            reses_ij.loc[:, 'w_ij_i'] = self.Wi[quals]

        if sps.isspmatrix(self.Wj):
            reses_ij.loc[:, 'w_ij_j'] = self.Wj.todense().A[quals]
        else:
            reses_ij.loc[:, 'w_ij_j'] = self.Wj[quals]

        reses_i.loc[:, 'S_i'] = self.S
        reses_i.loc[:, 'lamda'] = self.lamda

        reses_j.loc[:, 'mu'] = self.mu

        if self.eta is None:
            reses_i.loc[:, 'eta'] = 0.0
        else:
            reses_i.loc[:, 'eta'] = self.eta

        matching_rates_ij = self.matching_rates/(self.matching_rates[0].sum((1, 2)).reshape((1, self.sims, 1, 1)))
        matching_rates_ij = {'mean': matching_rates_ij.mean(axis=1),
                             'std': matching_rates_ij.std(axis=1),
                             'count': self.matching_rates.mean(axis=1)}

        # print(matching_rates_ij['count'].shape)

        for loc, name in zip([0, 1, 2], ['MR_ij', 'MR_i_c_j', 'MR_j_c_i']):
            for stat in ['mean', 'count', 'std']:
                if stat == 'mean':
                    reses_ij.loc[:, name + '_sim'] = matching_rates_ij[stat][loc][quals]
                elif stat == 'count':
                     reses_ij.loc[:, name + '_count'] = matching_rates_ij[stat][loc][quals]
                else:
                    if self.sims > 1:
                        reses_ij.loc[:, name + '_sim_std'] = matching_rates_ij[stat][loc][quals]

        waiting_time_ij = self.waiting_times

        # print('total_waiting_time:', waiting_time_ij[0].sum())

        waiting_time_ij = np.divide(waiting_time_ij, self.matching_rates[0],
                                    out=np.zeros_like(waiting_time_ij), where=waiting_time_ij != 0)

        reses_ij.loc[:, 'WT_ij_sim'] = waiting_time_ij[0].mean(axis=0)[quals]
        if self.sims > 1:
            reses_ij.loc[:, 'WT_ij_sim_std'] = waiting_time_ij[0].std(axis=0)[quals]

        waiting_time_ij_std = (waiting_time_ij[1] - waiting_time_ij[0]**2)**0.5

        reses_ij.loc[:, 'WT_ij_std_sim'] = waiting_time_ij_std.mean(axis=0)[quals]
        if self.sims > 1:
            reses_ij.loc[:, 'WT_ij_std_sim_std'] = waiting_time_ij_std.std(axis=0)[quals]

        waiting_time_i = self.waiting_times.sum(3)/self.matching_rates[0].sum(2)

        reses_i.loc[:, 'WT_i_sim'] = waiting_time_i[0].mean(axis=0)
        if self.sims > 1:
            reses_i.loc[:, 'WT_i_sim_std'] = waiting_time_i[0].std(axis=0)

        waiting_time_i_std = (waiting_time_i[1] - waiting_time_i[0]**2)**0.5

        reses_i.loc[:, 'WT_i_std_sim'] = waiting_time_i_std.mean(axis=0)
        if self.sims > 1:
            reses_i.loc[:, 'WT_i_std_sim_std'] = waiting_time_i_std.std(axis=0)

        waiting_time_j = self.waiting_times.sum(2)/self.matching_rates[0].sum(1)

        reses_j.loc[:, 'WT_j_sim'] = waiting_time_j[0].mean(axis=0)
        if self.sims > 1:
            reses_j.loc[:, 'WT_j_sim_std'] = waiting_time_j[0].std(axis=0)

        waiting_time_j_std = (waiting_time_j[1] - waiting_time_j[0]**2)**0.5

        reses_j.loc[:, 'WT_j_std_sim'] = waiting_time_j_std.mean(axis=0)
        if self.sims > 1:
            reses_j.loc[:, 'WT_j_std_sim_std'] = waiting_time_j_std.std(axis=0)

        sim_times = self.log_periods
        sim_times = sim_times[1, :] - sim_times[0, :]

        idle_periods_j = self.idle_periods
        idleness_j = idle_periods_j[1, :, :]/sim_times.reshape((self.sims, 1))

        reses_j.loc[:, 'r_j_sim'] = idleness_j.mean(axis=0)

        if self.sims > 1:
            reses_j.loc[:, 'r_j_sim_std'] = idleness_j.std(axis=0)

        reses_j.loc[:, 'u_j_sim'] = 1.0 - idleness_j.mean(axis=0)
        if self.sims > 1:
            reses_j.loc[:, 'u_j_sim_std'] = idleness_j.std(axis=0)

        service_time_ij = self.service_times
        service_time_ij = service_time_ij/sim_times.reshape((1, self.sims, 1, 1))

        reses_ij.loc[:, 'u_ij_sim'] = service_time_ij[0].mean(axis=0)[quals]
        if self.sims > 1:
            reses_ij.loc[:, 'u_ij_sim_std'] = service_time_ij[0].std(axis=0)[quals]

        idle_periods_j = self.idle_periods[1:, :, :]/self.idle_periods[0, :, :]

        reses_j.loc[:, 'IP_j_sim'] = idle_periods_j[0].mean(axis=0)
        if self.sims > 1:
            reses_j.loc[:, 'IP_j_sim_std'] = idle_periods_j[0].std(axis=0)

        idle_periods_std = (idle_periods_j[1] - idle_periods_j[0]**2)**0.5

        reses_j.loc[:, 'IP_j_std_sim'] = idle_periods_std.mean(axis=0)
        if self.sims > 1:
            reses_j.loc[:, 'IP_j_std_sim_std'] = idle_periods_std.std(axis=0)

        queue_lengths_i = self.queue_lengths

        # print('total_queue_lengths:', queue_lengths_i[0].sum())

        queue_lengths_i = queue_lengths_i/sim_times.reshape((1, self.sims, 1))

        reses_i.loc[:, 'LQ_i_sim'] = queue_lengths_i[0].mean(axis=0)
        if self.sims > 1:
            reses_i.loc[:, 'LQ_i_sim_std'] = queue_lengths_i[0].std(axis=0)

        queue_lengths_std = (queue_lengths_i[1] - queue_lengths_i[0]**2)**0.5

        reses_i.loc[:, 'LQ_i_std_sim'] = queue_lengths_std.mean(axis=0)
        if self.sims > 1:
            reses_i.loc[:, 'LQ_i_std_sim_std'] = queue_lengths_std.std(axis=0)

        return reses_i, reses_j, reses_ij


#     S_SERVERS = None
#     ENV = None
#     K = None
#     ASSIGNMENT_COUNT = None
#
#     nc = len(lamda)  # nc - number of customers
#     ns = len(mu)     # ns- number of servers
#     rnc = range(nc)  # 0,1,2,...,nc-1
#     rns = range(ns)  # 0,1,2,...,ns-1
#
#     if sps.isspmatrix(Q):
#         qs = dict((i, set(np.nonzero(Q[:, i])[0])) for i in rns)  # adjacency list for servers
#         qc = dict((j, set(np.nonzero(Q[j, :])[1])) for j in rnc)  # adjacency list for customers
#     else:
#         qs = dict((i, set(np.nonzero(Q[:, i])[0])) for i in rns)  # adjacency list for servers
#         qc = dict((j, set(np.nonzero(Q[j, :])[0])) for j in rnc)  # adjacency list for customers
#
#     matching_rates = np.zeros((3, sims, nc, ns))
#     waiting_times = np.zeros((2, sims, nc, ns))
#     service_times = np.zeros((1, sims, nc, ns))
#     idle_periods = np.zeros((3, sims, ns))
#     queue_lengths = np.zeros((2, sims, nc))
#     log_periods = np.zeros((2, sims))
#
#     if w is None:
#         w = Q
#     if S is None:
#         S = np.ones(len(lamda))
#
#     if sim_len is None:
#
#         sim_len = len(lamda) * (10**5 + 10**4)
#
#     if warm_up is None:
#
#         warm_up = sim_len - (10.0/11.0)*sim_len
#
#     def arrival(c_stream):
#
#         arrival_count = 0
#
#         for customer_arrival in c_stream:
#
#             arrival_count += 1
#
#             if arrival_count > 0 and arrival_count % report_every == 0:
#                 print(arrival_count, 'customers arrived - time_elapsed:', time() - start_time[0])
#                 print('assignment_count is:', ASSIGNMENT_COUNT)
#                 print('log[0] is:', log[0])
#
#             i = customer_arrival[0]
#
#             yield ENV.timeout(customer_arrival[1])
#
#             arrival_time = ENV.now
#             c_queue_len = len(C_QUEUES[0][i])
#
#             if c_queue_len > 0:
#
#                 C_QUEUES[0][i].appendleft(arrival_time)
#                 time_in_state = ENV.now - C_QUEUES[1][i]
#                 C_QUEUES[1][i] = ENV.now
#
#                 if log[0]:
#                     queue_lengths[:, K, i] += time_in_state * np.array([c_queue_len, c_queue_len**2])
#
#             else:
#
#                 available_servers = []
#
#                 for j in qc[i]:
#
#                     if S_SERVERS[0][j] == 0:
#
#                         if policy == 'fifo_alis':
#                             heapq.heappush(available_servers,
#                                            (-w[i, j] * (ENV.now - S_SERVERS[1][j]), j))
#                         elif policy == 'max_weight':
#                             heapq.heappush(available_servers,
#                                            (-w[i, j] * (ENV.now - S_SERVERS[1][j]), j))
#
#                 if len(available_servers) > 0:
#
#                     weight, j = heapq.heappop(available_servers)
#
#                     if ASSIGNMENT_COUNT > warm_up:
#
#                         idle_time = ENV.now - S_SERVERS[1][j]
#
#                         if log[0]:
#                             idle_periods[:, K, j] += np.array([1.0, idle_time, idle_time**2])
#
#                     ENV.process(assignment(i, j, 0))
#
#                 else:
#                     C_QUEUES[0][i].appendleft(arrival_time)
#                     C_QUEUES[1][i] = ENV.now
#
#     def assignment(i, j, s_c):
#
#         ASSIGNMENT_COUNT += 1
#
#         if not log[0]:
#             if ASSIGNMENT_COUNT > warm_up:
#                 print('ASSIGNMENT_COUNT', ASSIGNMENT_COUNT)
#                 print('warm_up', warm_up)
#                 log[0] = True
#                 log_periods[0, K] = ENV.now
#
#         S_SERVERS[0][j] = 1
#         service_time = random.expovariate(mu[j]/S[i])
#
#         if log[0]:
#             matching_rates[:, K, i, j] += [1.0, s_c, 1.0 - s_c]
#             service_times[:, K, i, j] += service_time
#
#         yield ENV.timeout(service_time)
#
#         available_customers = []
#
#         for i in qs[j]:
#
#             if len(C_QUEUES[0][i]) > 0:
#                 if policy == 'fifo_alis':
#                     heapq.heappush(available_customers,
#                                    (-w[i, j]*(ENV.now - C_QUEUES[0][i][-1]), i))
#                 elif policy == 'max_weight':
#                     heapq.heappush(available_customers,
#                                    (-w[i, j]*len(C_QUEUES[0][i]), i))
#
#         if len(available_customers) > 0:
#
#             weight, i = heapq.heappop(available_customers)
#
#             if log[0]:
#
#                 wait_time = ENV.now - C_QUEUES[0][i][-1]
#                 queue_len = len(C_QUEUES[0][i])
#
#                 waiting_times[:, K, i, j] += np.array([wait_time, wait_time**2.0])
#                 queue_lengths[:, K, i] += np.array([queue_len, queue_len**2.0])
#
#             C_QUEUES[0][i].pop()
#
#             ENV.process(assignment(i, j, 1))
#
#         else:
#
#             S_SERVERS[0][j] = 0
#             S_SERVERS[1][j] = ENV.now
#
#         # lambda is a saved word in Python hence we use the abbreviation lamda
#
#     def run_sim(k):
#
#         print('starting sim ', k+1)
#
#         start_time = [time()]
#         ENV = simpy.Environment()
#         C_QUEUES = [tuple(deque() for c in rnc), [0 for _ in rnc]]
#         S_SERVERS = [[0 for _ in rns], [0 for _ in rns]]
#         K = k
#         ASSIGNMENT_COUNT = 0
#         log = [False]
#
#         print('sim_len', sim_len, type(sim_len))
#
#         total_rate = np.asscalar(lamda.sum())
#         arrival_ratios = lamda*(1/total_rate)
#         customer_arrivals = np.random.choice(a=rnc, size=sim_len, p=arrival_ratios)
#         interarrival_times = np.random.exponential(scale=1/total_rate, size=sim_len)
#         c_stream = zip(customer_arrivals, interarrival_times)
#
#         print('total work load arrival rate is:', lamda*S.sum())
#
#         sim_time = int(interarrival_times.sum())
#         log_periods[1, k] = sim_time
#
#         ENV.process(arrival())
#         ENV.run(until=sim_time)
#
#         print('ending queueing sim ', k+1, 'duration:', time() - start_time[0])
#
#     sim_reses = {'lambda': lamda, 'mu': mu, 'S': S, 'W': w, 'eta': eta, 'policy': policy,
#                  'matching_rates': matching_rates,'waiting_times': waiting_times,
#                  'idle_times': idle_periods,'queue_lengths': queue_lengths, 'log_periods': log_periods,
#                  'service_times': service_times}
#
#     reses_i, reses_j, reses_ij = parse_sim_data(sim_reses, Q)
#  if __name__ == '__main__':
#
#     m = 100
#     n = 100
#     d = 10
#     plot = False
#     wait_time_grid_demand, wait_time_grid_supply = np.zeros((m, n)), np.zeros((m, n))
#
#     g = undirected_grid_2d_bipartie_graph(m, n, d, d, plot=plot, alpha=0.95, l_norm=1)
#
#     grid_adj_mat = g['grid_adj_mat']
#     nodes = g['nodes']
#     lamda = np.concatenate((g['lamda_d'], g['lamda_s']))
#     q = sps.vstack((sps.hstack((0*sps.eye(m*n), grid_adj_mat)),
#                                        sps.hstack((grid_adj_mat, 0*sps.eye(m*n)))))
#     q = sps.csr_matrix(q)
#     eta = np.ones(len(lamda))*(sum(lamda)/len(lamda))
#     sim_res = online_matching_simulator(lamda=lamda, q=q, eta=eta, abandonments=True, sims=1,prt=False, sim_len=10**7)
#     wait_times = sim_res['waiting_times_mean_node']
#     match_rates = sim_res['matching_rates_mean_node']
#
#     for i in range(m*n):
#         if match_rates[i] > 10 and wait_times[i] > 0:
#             print(wait_times[i])
#             wait_time_grid_demand[nodes[i][0]] = log(wait_times[i])
#         if match_rates[m*n + i] > 10 and wait_times[m*n + i] > 0:
#             print(wait_times[m*n + i])
#             wait_time_grid_supply[nodes[i][0]] = log(wait_times[m*n + i])
#
#     print(sps.csr_matrix(wait_time_grid_demand))
#     print(sps.csr_matrix(wait_time_grid_supply))
#
#     fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4)
#     ax1.imshow(g['lamda_d_pdf'], interpolation='nearest')
#     ax1.set_title('demand')
#     ax2.imshow(g['lamda_s_pdf'], interpolation='nearest')
#     ax2.set_title('supply')
#     ax3.imshow(g['demand_decomp'], interpolation='nearest')
#     ax3.set_title('demand decomposotion')
#     ax4.imshow(g['supply_decomp'], interpolation='nearest')
#     ax4.set_title('supply decomposotion')
#     ax5.imshow(g['max_ent_workload'], interpolation='nearest')
#     ax5.set_title('max_ent_workload')
#     ax6.imshow(g['fifo_ct'], interpolation='nearest')
#     ax6.set_title('fifo_ct_estimate')
#     ax7.imshow(wait_time_grid_demand, interpolation='nearest')
#     ax7.set_title('wait_time_demand')
#     ax8.imshow(wait_time_grid_supply, interpolation='nearest')
#     ax8.set_title('wait_time_supply')
#     plt.show()
