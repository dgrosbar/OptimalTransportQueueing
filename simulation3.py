import numpy as np
import pandas as pd
from scipy import sparse as sps
import simpy
import pandas as pd
import os
import os.path
import shutil
from scipy import sparse as sps
import gc
from time import time
from collections import deque
from math import log

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

    '''
        MultiSim is a single single Q, lamda, mu, s, and can support experiments with various policies and save to the
        same file every time
    '''

    def __init__(self, lamda, mu, q, eta=None, s=None, 
        interarrival_dist='exponential', service_dist='exponential', abandonment_dist='exponential', sim_name='_', i_policy='alis',j_policy='fifo',
        w_i=None, w_j=None, m=None, prt=False, report_every=10**5, warm_up=None, sim_len=None, seed=None, sims=30, full_gini=False, **kwargs):

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

        self.matching_rates_sim = dict(((i, j), np.zeros(3)) for i in self.rm for j in self.rn)
        self.waiting_times_sim = dict(((i, j),np.zeros(2)) for i in self.rm for j in self.rn)
        self.service_times_sim = dict(((i, j),np.zeros(1)) for i in self.rm for j in self.rn)
        self.idle_periods_sim = dict((j, np.zeros(3)) for j in self.rn)
        self.queue_lengths_sim = dict((i, np.zeros(2)) for i in self.rm)


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

            print( 'im_len', self.sim_len, 'arrivals')

            if self.base_seed is not None:
                np.random.seed(self.base_seed + k*10*self.sim_len)
            customer_arrivals = np.random.choice(a=self.rm, size=self.sim_len, p=self.arrival_ratios)
            interarrival_times = np.random.exponential(scale=1/self.total_rate, size=self.sim_len)

            customer_service_time_pct = np.random.uniform(0.0, 1.0, size=self.sim_len)

            self.c_stream = zip(customer_arrivals, interarrival_times, customer_service_time_pct)

            print( 'otal work load arrival rate is:', (self.lamda*self.S).sum())

            self.sim_time = int(interarrival_times.sum())
            self.log_periods[1, k] = self.sim_time

            self.env.process(self.event_stream())
            self.env.run(until=self.sim_time)

            self.store_sim_data(k)
            self.clear_sim_data()

            print( 'nding queueing sim ', k+1, 'duration:', time() - self.start_time)

    def clear_sim_data(self):

        self.matching_rates_sim = dict(((i, j),np.zeros(3)) for i in range(self.n) for j in range(m))
        self.waiting_times_sim = dict(((i, j),np.zeros(2)) for i in range(self.n) for j in range(m))
        self.service_times_sim = dict(((i, j),np.zeros(1)) for i in range(self.n) for j in range(m))
        self.idle_periods_sim = dict((j, np.zeros(3)) for j in range(self.m))
        self.queue_lengths_sim = dict((i, np.zeros(2)) for j in range(self.n))

    def store_sim_data(self, k):

        for i in self.rm:

            self.queue_lengths[:, k, i] += self.queue_lengths_sim[i]

            for j in self.qc[i]:
                self.matching_rates[:, k, i, j] += self.matching_rates_sim[i, j]
                self.waiting_times[:, k, i, j] += self.waiting_times_sim[i, j]
                self.service_times[:, k, i, j] += self.service_times_sim[i, j]

        for j in self.rn:

            self.idle_periods[:, k, j] += self.idle_periods_sim[j]

    def event_stream(self):

        for customer_arrival in self.c_stream:

            self.arrival_count += 1

            if self.arrival_count > 0 and self.arrival_count % self.report_every == 0:
                print( self.arrival_count, 'customers arrived - time_elapsed:', time() - self.start_time)
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
                    self.queue_lengths_sim[i] += time_in_state * np.array([c_queue_len, c_queue_len**2])

            else:

                # available_servers = []

                # for j in self.qc[i]:

                #     if self.s_servers[0][j] == 0:

                if self.i_policy == 'alis':
                    available_servers = [(-self.Wi[i, j] * (self.env.now - self.s_servers[1][j]), j) for j in self.qc[i] if self.s_servers[0][j] == 0]
                    # heapq.heappush(available_servers,
                    #                (-self.Wi[i, j] * (self.env.now - self.s_servers[1][j]), j))
                elif self.i_policy == 'rand':
                    available_servers = [(-self.Wi[i, j] * random.uniform(0,1), j) for j in self.qc[i] if self.s_servers[0][j] == 0]
                    # heapq.heappush(available_servers,
                    #                (-self.Wi[i, j] * random.uniform(0,1), j))
                elif self.i_policy == 'prio':
                    available_servers = [(-self.Wi[i, j] , j) for j in self.qc[i] if self.s_servers[0][j] == 0]
                    # heapq.heappush(available_servers,
                    #                (-self.Wi[i, j], j))
                else:
                    available_servers = [(-self.Wi[i, j] * (self.env.now - self.s_servers[1][j]), j) for j in self.qc[i] if self.s_servers[0][j] == 0]

                if len(available_servers) > 0:

                    weight, j = min(available_servers)

                    if self.logging:

                        idle_time = self.env.now - max(self.s_servers[1][j], self.log_start)
                        self.idle_periods_sim[j] += np.array([1.0, idle_time, idle_time**2])
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
                print( 'ssignment_count', self.assignment_count)
                print( 'arm_up', self.warm_up)
                self.logging = True
                self.log_periods[0, self.k] = self.env.now
                self.log_start = self.env.now

        #service_time = random.expovariate(self.mu[j]/self.S[i])
        service_time = -log(service_time_pct)*self.service_rate[i, j]

        if self.logging:
            self.matching_rates_sim[i, j] += [1.0, s_c, 1.0 - s_c]
            self.service_times_sim[i, j] += min(service_time, self.sim_time - self.env.now)

        yield self.env.timeout(service_time)

        # available_customers = []

        # for i in self.qs[j]:

        #     if len(self.c_queues[0][i]) > 0:

        if self.j_policy == 'fifo':
            available_customers = [(-self.Wj[i, j]*(self.env.now - self.c_queues[0][i][-1]), i) for i in self.qs[j] if len(self.c_queues[0][i])]
            # heapq.heappush(available_customers,
            #                (-self.Wj[i, j]*(self.env.now - self.c_queues[0][i][-1]), i))
        elif self.j_policy == 'max_weight':
            available_customers = [(-self.Wj[i, j]*len(self.c_queues[0][i]), i) for i  in self.qs[j] if len(self.c_queues[0][i])]                   
            # heapq.heappush(available_customers,
            #                (-self.Wj[i, j]*len(self.c_queues[0][i]), i))
        elif self.j_policy == 'util':
            available_customers = [(-self.Wj[i, j], i) for i  in self.qs[j] if len(self.c_queues[0][i])]
            # heapq.heappush(available_customers,(-self.Wj[i, j], i)) 
        elif self.j_policy == 'rand':
            available_customers = [(-self.Wj[i,j]*random.unifrom(0,1), i) for i in self.qs[j] if len(self.c_queues[0][i]) for _ in range(len(self.c_queues[0][i]))]
            # for _ in range(len(self.c_queues[0][i])):
            #     heapq.heappush(available_customers,
            #                    (-self.Wj[i,j]*random.uniform(0, 1), i))
        elif self.j_policy == 'prio':
            available_customers = [(-self.Wj[i, j], i) for i in self.qs[j] for _ in range(len(self.c_queues[0][i]))]
            # for _ in range(len(self.c_queues[0][i])):
            #     heapq.heappush(available_customers,
            #                        (-self.Wj[i, j], i))

        if len(available_customers) > 0:

            weight, i = min(available_customers)

            if self.logging:

                wait_time = self.env.now - max(self.c_queues[0][i][-1], self.log_start)

                queue_len = len(self.c_queues[0][i])
                time_in_state = self.env.now - max(self.c_queues[2][i], self.log_start)
                self.waiting_times_sim[i, j] += np.array([wait_time, wait_time**2.0])
                if self.full_gini:
                    heapq.heappush(self.gini[self.k], wait_time)
                self.queue_lengths_sim[i] += time_in_state * np.array([queue_len, queue_len**2.0])

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
                print(y, np.arange(1, n+1, 1))
                print(2(y*np.arange(1, n+1, 1)).sum())
                print(y, sum())
                print(n, y.sum())
                print(((2*y*np.arange(1, n+1, 1)).sum())/((n*y.sum()) - ((n+1)/n)))
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

