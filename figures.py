import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import sparse as sps
import matplotlib.ticker as ticker
from mpl_toolkits.axisartist.parasite_axes import SubplotHost
from solvers import fast_primal_dual_algorithm
from scipy.stats import entropy
from itertools import product
from fss_utils import calc_area_between_curves, sakasegawa
from collections import OrderedDict
from matplotlib import rc
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from fss_utils import is_non_trivial_and_connected, filter_df, transform_to_normal_form
from solvers import bipartite_workload_decomposition, fast_primal_dual_algorithm
import networkx as nx
from generators import erdos_renyi_bipartite_graph
from math import log
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})



LINE_STYLES = OrderedDict(
    [('solid',               (0, ())),
     ('loosely dotted',      (0, (1, 10))),
     ('dotted',              (0, (1, 5))),
     ('densely dotted',      (0, (1, 1))),

     ('loosely dashed',      (0, (5, 10))),
     ('dashed',              (0, (5, 5))),
     ('densely dashed',      (0, (5, 1))),

     ('loosely dashdotted',  (0, (3, 10, 1, 10))),
     ('dashdotted',          (0, (3, 5, 1, 5))),
     ('densely dashdotted',  (0, (3, 1, 1, 1))),

     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))])

MARKERS2 = Line2D.filled_markers
MARKERS = ["o", "v", "*", "x", "+", "*", "s"]

prop_cycle = plt.rcParams['axes.prop_cycle']
COLORS = prop_cycle.by_key()['color']


def contur(func):

    fig, ax = plt.subplots()
    c = np.arange(0.8, 10.2, 0.2)
    u = np.arange(0.5, 0.999, 0.005)
    X, Y = np.meshgrid(u, c)
    ct = np.array([func(c,u) for c,u in zip(np.ravel(Y), np.ravel(X))])
    Z = ct.reshape(X.shape)
    # logct=np.array([log_sakasegawa(c,u) for c,u in zip(np.ravel(Y), np.ravel(X))])
    # logZ = logct.reshape(X.shape)
    cmap = plt.cm.get_cmap("viridis")
    cmap.set_under("magenta")
    cmap.set_over("yellow")



    #CV=ax.contourf(X, Y, logZ, cmap=cmap, levels=np.array([log_sakasegawa(6,float(x)/100) for x in range(75,100,1)]), extend='max') # negative contours will be dashed by default)
    #CV=ax.pcolor(X, Y, Z, cmap=cmap, norm=colors.LogNorm(vmin=0.3, vmax=33)) # negative contours will be dashed by default)
    #CV=ax.contourf(X, Y, logZ, cmap=cmap, levels=np.arange(-5,log_sakasegawa(1,0.99),0.5), extend='max') # negative contours will be dashed by default)
    #CS  = ax.contour(X, Y, Z, levels=[x for x in np.arange(0,int(sakasegawa(1,0.99)),5)],colors='k')#, levels=[sakasegawa(128,x) for x in np.arange(0.9,0.99,0.005) ],colors='w')
    CNFB = ax.contour(X, Y, Z, levels=[sakasegawa(10,x) for x in np.arange(0.5, 1.0,0.01)],colors='grey',linewidths=3)
    #print([sakasegawa(10,x) for x in np.arange(0.75,1.0,0.01)])
    #CNF = ax.contour(X, Y, Z, levels=[sakasegawa(10,x) for x in np.arange(0.866,1,0.2)],colors='w',linewidths=3)
    #ax.contour(X, Y, Z, levels=[sakasegawa(6,x) for x in [0.865,0.867]],colors='b',linewidths=5)
    #ax.scatter(x=[0.866 for i in range(0,60,1)],y=[float(i)/10 for i in range(0,60,1)],color='k',s=1)
    # for k in [1.06]+range(7):
    #      ax.scatter(x=[float(i)/1000 for i in range(750,1000,1)],y=[k for i in range(750,1000,1)],color='white',s=1)
    #ax.scatter(x=[0.9,0.85,0.8],y=[3,2,1.06],color='b', s=300)

    ax.xaxis.tick_top()
    ax.set_ylim(1, 10)
    ygridlines = ax.get_ygridlines()
    for line in ygridlines:
        line.set_linewidth(1)
        line.set_linestyle('-.')
    ax.set_xlim(0.5,0.99)
    ax.set_yticks(np.sort([1, 10]+[x for x in range(2,10,1)]))
    ax.set_xticks(np.arange(0.5,1,0.01))
    font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 18}
    plt.xlabel('Utilization',fontsize=30)
    plt.ylabel('# of tools',fontsize=30)
    mpl.rc('font', **font)
    #plt.clabel(CNFB, inline=1, fontsize=17,colors='k')#manual=[(x,1+20*(1-x)) for x in np.arange(0.75, 0.97, 0.013)], colors='k')
    plt.clabel(CNFB, inline=1, fontsize=17,manual=[(x,1+20*(1-x)) for x in np.arange(0.75, 0.97, 0.013)], colors='k')
    #cbar=fig.colorbar(CV, ax=ax, extend='max',ticks=[0.25,0.5,1,2,4,8,16])
    #cbar.set_ticklabels([0.25,0.5,1,2,4,8,16])
    #cbar.set_label('Cycle Time',fontsize=30)
    plt.show()


def get_fss(reses_i, reses_j, reses_ij):

    cords = reses_ij[['i', 'j']].drop_duplicates()
    i_cords, j_cords = cords['i'], cords['j']
    q_shape = np.max(cords, axis=0) + 1
    q = np.zeros((q_shape))
    q[i_cords, j_cords] = 1.0

    lamda = reses_i.sort_values(by='i')['lamda'].drop_duplicates().values
    s = reses_i.sort_values(by='i')['S'].values
    mu = reses_j.sort_values(by='j')['mu'].values


def get_experiment_data(view_exp, filters):

    reses_ij = filter_df(get_file_data(view_exp, 'reses_ij'), filters)
    reses_i = filter_df(get_file_data(view_exp, 'reses_i'), filters)
    reses_j = filter_df(get_file_data(view_exp, 'reses_j'), filters)
    return reses_i, reses_j, reses_ij


def get_file_data(dir, filename):

    filename = dir + '/' + filename +'.csv'
    res = pd.read_csv(filename)
    return res


def service_time_dependency_experiment():

    reses_ij = get_file_data('optimal_transport_queueing', 'reses_ij')
    fig1 = plt.figure()
    ax1 = SubplotHost(fig1, 111)
    fig1.add_subplot(ax1)
    print reses_ij['phi']
    # data = reses_ij[(reses_ij['rho'] > 0.99) &
    #             (reses_ij['j_policy'] == 'max_weight') ][['ij','i','j', 'MR_ij_sim', 'pi_ent', 'j_policy', 'rho']]
    # print data
    #print list(reses_ij.columns.values)

    # First X-axis

    for rho, linestyle in [(0.9,'-')]:
        for j_policy, color in [('fifo', 'blue'), ('max_weight', 'red'), ('rand', 'green')]:
            data = reses_ij[(reses_ij['rho'] >= rho-0.001) & (reses_ij['rho'] <= rho + 0.001) & (reses_ij['phi'] <= .1) &
                            (reses_ij['j_policy'] == j_policy)][['ij','i','j', 'MR_ij_sim', 'pi_ent', 'j_policy']]
            print data
            data = data.sort_values(by=['i','j'])
            #data.loc[:, 'ij'] = data['i'].apply(str) + ',' + data['j'].apply(str)
            ax1.plot(data['ij'], data['MR_ij_sim']*rho, label=j_policy + ' ' + str(rho), linestyle=linestyle, color=color)
            if j_policy == 'fifo':
                ax1.plot(data['ij'], data['pi_ent'], label='pi_ent', color='black', linestyle=linestyle)
    ax1.set_xticklabels(data['ij'])
    #ax1.xaxis.set_label_text('First X-axis') # Uncomment to label axis
    ax1.yaxis.set_label_text("$\pi_{ij}$")
    plt.legend()
    plt.show()


def optimal_transport():

    reses_ij = get_file_data('optimal_transport_queueing', 'reses_ij')
    fig1 = plt.figure()
    ax1 = SubplotHost(fig1, 111)
    fig1.add_subplot(ax1)
    print reses_ij['phi']
    # data = reses_ij[(reses_ij['rho'] > 0.99) &
    #             (reses_ij['j_policy'] == 'max_weight') ][['ij','i','j', 'MR_ij_sim', 'pi_ent', 'j_policy', 'rho']]
    # print data
    #print list(reses_ij.columns.values)

    # First X-axis

    for rho, linestyle in [(0.9,'-')]:
        for j_policy, color in [('fifo', 'blue'), ('max_weight', 'red'), ('rand', 'green')]:
            data = reses_ij[(reses_ij['rho'] >= rho-0.001) & (reses_ij['rho'] <= rho + 0.001) & (reses_ij['phi'] <= .1) &
                            (reses_ij['j_policy'] == j_policy)][['ij','i','j', 'MR_ij_sim', 'pi_ent', 'j_policy']]
            print data
            data = data.sort_values(by=['i','j'])
            ax1.plot(data['ij'], data['MR_ij_sim']*rho, label=j_policy + ' ' + str(rho), linestyle=linestyle, color=color)
            if j_policy == 'fifo':
                ax1.plot(data['ij'], data['pi_ent'], label='pi_ent', color='black', linestyle=linestyle)
    ax1.set_xticklabels(data['ij'])
    #ax1.xaxis.set_label_text('First X-axis') # Uncomment to label axis
    ax1.yaxis.set_label_text("$\pi_{ij}$")
    plt.legend()
    plt.show()


def fifo_flaw():
    exp = 'fifo_flaw13'
    reses_i = get_file_data(exp, 'reses_i')
    reses_ij = get_file_data(exp, 'reses_ij')
    # data = reses_ij[(reses_ij['i'] % reses_ij['N'] == 0) & (reses_ij['j'] % reses_ij['N'] == 0)][['N', 'i','j','sim_name', 'WT_ij_sim', 'pi_hat', 'pi_ent', 'MR_ij_sim']]
    data = reses_i[(reses_i['i'] == 0) | (reses_i['i'] == reses_i['N'])][['N', 'i','j_policy','sim_name', 'WT_i_sim']]
    cols = ['i', 'N', 'j', 'WT_ij_sim', 'pi_hat', 'pi_ent', 'MR_ij_sim','sim_name', 'j_policy']
    rename_cols = ['j', 'WT_ij_sim', 'pi_hat', 'pi_ent', 'MR_ij_sim']
    rename_dict = dict((key, val) for key, val in zip(rename_cols, [name + '_0' for name in rename_cols]))
    data_ij = pd.merge(left=data,
                       right=reses_ij[((reses_ij['i'] == 0) &
                                       (reses_ij['j'] == 0)) |
                                      ((reses_ij['i'] == reses_ij['N']) &
                                       (reses_ij['j'] == reses_ij['N']))][cols].rename(columns=rename_dict),
                       on=['i', 'N', 'sim_name', 'j_policy'], how='left')
    rename_dict = dict((key, val) for key, val in zip(rename_cols, [name + '_n' for name in rename_cols]))
    data_ij = pd.merge(left=data_ij,
                       right=reses_ij[((reses_ij['i'] == 0) & (reses_ij['j'] == reses_ij['N'])) |
                                      ((reses_ij['i'] == reses_ij['N']) & (reses_ij['j'] == 2*reses_ij['N'] - 1))]
                       [cols].rename(columns=rename_dict),
                            on=['i', 'N', 'sim_name', 'j_policy'], how='left')
    pd.options.display.max_columns = 15
    pd.options.display.max_rows = 1000
    pd.set_option('display.width', 1000)
    data_ij.loc[:,'pi_hat_n/pi_ent_n'] = data_ij['pi_hat_n']/data_ij['pi_ent_n']
    for col in ['pi_ent_0', 'pi_ent_n', 'pi_hat_n', 'pi_hat_0']:
        data_ij[col] = data_ij[col]*data_ij['N']*0.85*2
    print data_ij[(data_ij['i'] % data_ij['N'] == 0) & (data_ij['N'] == 10)]\
        [['i','j_0','j_n','j_policy','sim_name','N',
          'pi_ent_0', 'pi_hat_0','pi_ent_n', 'pi_hat_n','pi_hat_n/pi_ent_n','MR_ij_sim_0', 'MR_ij_sim_n',
          'WT_i_sim']].sort_values(by=['j_policy','N','i','sim_name',])


def fifo_flaw_2():

    pd.options.display.max_columns = 100
    pd.options.display.max_rows = 1000
    pd.set_option('display.width', 1000)

    exp = 'fifo_flaw13'
    reses_i = get_file_data(exp, 'reses_i')
    reses_ij = get_file_data(exp, 'reses_ij')
    reses_i.loc[:, 'cust_type'] = 1*(reses_i['i'] >= reses_i['N'])

    #print reses_ij[['sim_name', 'j_policy', 'rho1', 'rho2']].drop_duplicates()
    #print reses_ij[['i', 'N','sim_name', 'j_policy', 'MR_ij_sim', 'rho1', 'rho2']].groupby(['i', 'N', 'sim_name','rho1', 'rho2','j_policy'], as_index=False).sum()
    reses_i = pd.merge(left=reses_i,
                       right=reses_ij[['i', 'N','sim_name', 'j_policy', 'MR_ij_sim', 'rho1', 'rho2']]
                       .groupby(['i', 'N', 'sim_name','j_policy','rho1', 'rho2'], as_index=False).sum()
                       .rename(columns={'MR_ij_sim': 'MR_i_sim'}),
                       on=['i', 'N', 'sim_name','j_policy','rho1', 'rho2'], how='left')

    reses_i['MRxWT_i_sim'] = reses_i['MR_i_sim'] * reses_i['WT_i_sim']

    type_reses_i = reses_i[['cust_type', 'N','sim_name', 'j_policy', 'rho1', 'rho2','MR_i_sim', 'MRxWT_i_sim']]\
        .groupby(['cust_type', 'N','sim_name', 'j_policy', 'rho1', 'rho2'], as_index=False).sum()
    type_reses_i.loc[:, 'WT_type_sim'] = type_reses_i['MRxWT_i_sim']/type_reses_i['MR_i_sim']
    base_i = type_reses_i[['N','sim_name', 'j_policy', 'rho1', 'rho2']].drop_duplicates()

    for k in [0,1]:
        base_i = pd.merge(left=base_i,
                          right=type_reses_i[type_reses_i['cust_type'] == k][
                              ['N','sim_name', 'j_policy', 'WT_type_sim', 'rho1', 'rho2']]
                          .rename(columns={'WT_type_sim': 'WT_' + str(k) + '_sim'}),
                          on=['N','sim_name', 'j_policy', 'rho1', 'rho2'], how='left')

    reses_ij['edge_type'] = np.where((reses_ij['i'] < reses_ij['N']) & (reses_ij['i'] == reses_ij['j']), '00', '01')
    reses_ij['edge_type'] = np.where(reses_ij['i'] >= reses_ij['N'], '11', reses_ij['edge_type'])
    type_reses_ij = reses_ij[['edge_type','N','sim_name', 'j_policy', 'MR_ij_sim', 'rho1', 'rho2']]\
        .groupby(['edge_type','N','sim_name', 'j_policy', 'rho1', 'rho2'], as_index=False).sum()

    base_ij = type_reses_ij[['N','sim_name', 'j_policy', 'rho1', 'rho2']].drop_duplicates()
    for k in ['00','01','11']:
        base_ij = pd.merge(left=base_ij,
                           right=type_reses_ij[type_reses_ij['edge_type'] == k][[
                               'N','sim_name', 'j_policy', 'MR_ij_sim','rho1', 'rho2']]
                           .rename(columns={'MR_ij_sim': 'MR_' + k + '_sim'}),
                           on=['N','sim_name', 'j_policy','rho1', 'rho2'], how='left')

    base_ij = base_ij.fillna(0)
    base_ij.loc[:,'lamda_sys'] = base_ij['rho1'] + base_ij['rho2']
    for col in ['MR_00_sim','MR_01_sim','MR_11_sim']:
        base_ij[col] = base_ij[col] * base_ij['lamda_sys']
    type_reses_i = pd.merge(left=base_i,
                            right=base_ij,
                            on=['N','sim_name', 'j_policy','rho1', 'rho2'], how='left')

    type_reses_i.loc[:,'WT_sim'] = np.where(type_reses_i['sim_name'] != 'balanced',
                                            ((type_reses_i['WT_0_sim'] * (type_reses_i['MR_00_sim'] +
                                                                          type_reses_i['MR_01_sim']) +
                                              type_reses_i['WT_1_sim'] * type_reses_i['MR_11_sim'])/
                                             (type_reses_i['MR_00_sim'] + type_reses_i['MR_01_sim']
                                              + type_reses_i['MR_11_sim'])),
                                            (type_reses_i['WT_0_sim']*type_reses_i['MR_00_sim'] +
                                             type_reses_i['WT_1_sim']*type_reses_i['MR_11_sim'])/
                                            (type_reses_i['MR_00_sim'] + type_reses_i['MR_11_sim']))
    type_reses_i = type_reses_i.fillna(0)
    type_reses_i.loc[:, 'u_0_sim'] = type_reses_i['MR_00_sim']
    type_reses_i.loc[:, 'u_1_sim'] = type_reses_i['MR_01_sim'] + type_reses_i['MR_11_sim']

    for N in [2, 3, 5, 7, 10, 20]:
        for j_p in ['max_weight', 'fifo']:
            for rho1 in [0.7, 0.8, 0.9, 0.95]:
                for rho2 in [0.7, 0.8, 0.9, 0.95]:
                # print type_reses_i[(type_reses_i['N'] == N) &
                #                    (type_reses_i['rho1'] == rho1) & (type_reses_i['rho2'] == rho2)
                # ].sort_values(by=['N', 'WT_sim'])
                    print type_reses_i[(type_reses_i['j_policy'] == j_p) & (type_reses_i['N'] == N) &
                                       ((type_reses_i['sim_name'] == 'balanced') |
                                        (type_reses_i['sim_name'] == 'non_squared') |
                                        (type_reses_i['sim_name'] == 'prio_1')| (type_reses_i['sim_name'] == 'prio_0') |
                                        (type_reses_i['sim_name'] == 'plain')) &
                                       (type_reses_i['rho1'] == rho1) & (type_reses_i['rho2'] == rho2)
                    ].sort_values(by=['N','j_policy', 'WT_sim'])
                    print '--------------------------------------------------------------------------------------------------------------------------------'
        # reses_ij = get_file_data(exp, 'reses_ij')
    # # data = reses_ij[(reses_ij['i'] % reses_ij['N'] == 0) & (reses_ij['j'] % reses_ij['N'] == 0)][['N', 'i','j','sim_name', 'WT_ij_sim', 'pi_hat', 'pi_ent', 'MR_ij_sim']]
    # data = reses_i[(reses_i['i'] == 0) | (reses_i['i'] == reses_i['N'])][['N', 'i','j_policy','sim_name', 'WT_i_sim']]
    # cols = ['i', 'N', 'j', 'WT_ij_sim', 'pi_hat', 'pi_ent', 'MR_ij_sim','sim_name', 'j_policy']
    # rename_cols = ['j', 'WT_ij_sim', 'pi_hat', 'pi_ent', 'MR_ij_sim']
    # rename_dict = dict((key, val) for key, val in zip(rename_cols, [name + '_0' for name in rename_cols]))
    # data_ij = pd.merge(left=data,
    #                    right=reses_ij[((reses_ij['i'] == 0) &
    #                                    (reses_ij['j'] == 0)) |
    #                                   ((reses_ij['i'] == reses_ij['N']) &
    #                                    (reses_ij['j'] == reses_ij['N']))][cols].rename(columns=rename_dict),
    #                    on=['i', 'N', 'sim_name', 'j_policy'], how='left')
    # rename_dict = dict((key, val) for key, val in zip(rename_cols, [name + '_n' for name in rename_cols]))
    # data_ij = pd.merge(left=data_ij,
    #                         right=reses_ij[((reses_ij['i'] == 0) &
    #                                         (reses_ij['j'] == reses_ij['N'])) |
    #                                        ((reses_ij['i'] == reses_ij['N']) &
    #                                         (reses_ij['j'] == 2*reses_ij['N'] -1))][cols].rename(columns=rename_dict),
    #                         on=['i', 'N', 'sim_name', 'j_policy'], how='left')
    # pd.options.display.max_columns = 15
    # pd.options.display.max_rows = 1000
    # pd.set_option('display.width', 1000)
    # data_ij.loc[:,'pi_hat_n/pi_ent_n'] = data_ij['pi_hat_n']/data_ij['pi_ent_n']
    # for col in ['pi_ent_0', 'pi_ent_n', 'pi_hat_n', 'pi_hat_0']:
    #     data_ij[col] = data_ij[col]*data_ij['N']*0.85*2
    # print data_ij[(data_ij['i'] % data_ij['N'] == 0) & (data_ij['N'] == 10)]\
    #     [['i','j_0','j_n','j_policy','sim_name','N',
    #       'pi_ent_0', 'pi_hat_0','pi_ent_n', 'pi_hat_n','pi_hat_n/pi_ent_n','MR_ij_sim_0', 'MR_ij_sim_n',
    #       'WT_i_sim']].sort_values(by=['j_policy','N','i','sim_name',])
    #


def k_chain_noise():

    reses_ij = get_file_data('k_chains_noise2', 'reses_ij')
    reses_i = get_file_data('k_chains_noise2', 'reses_i')
    fig1 = plt.figure()
    ax1 = SubplotHost(fig1, 111)
    fig1.add_subplot(ax1)
    ent = reses_i[['rep', 'noise', 'i_policy', 'j_policy', 'lamda']]\
        .groupby(['rep', 'noise', 'i_policy', 'j_policy'])\
        .apply(lambda x: pd.Series({'lamda_entropy': entropy(x['lamda'])})).reset_index()
    kl_div = reses_ij[['rep', 'noise', 'i_policy', 'j_policy', 'MR_ij_sim', 'pi_ent']]\
        .groupby(['rep', 'noise', 'i_policy', 'j_policy'])\
        .apply(lambda x: pd.Series({'KL(MR_ij_Sim, pi_ent)': entropy(x['MR_ij_sim'], x['pi_ent'])})).reset_index()
    kl_div = pd.merge(left=kl_div,
                      right=reses_ij[['rep', 'noise', 'i_policy', 'j_policy', 'MR_ij_sim']]
                      .groupby(['rep', 'noise', 'i_policy', 'j_policy'])
                      .apply(lambda x: pd.Series({'H(MR_ij_Sim)': entropy(x['MR_ij_sim'])})).reset_index(),
                      on=['rep', 'noise', 'i_policy', 'j_policy'], how='left')

    reses = pd.merge(left=ent, right=kl_div, on=['rep', 'noise', 'i_policy', 'j_policy'], how='left')
    # ax1.plot(reses_ij[(reses_ij['j_policy'] == 'rand') & (reses_ij['rep'] == 0) & (reses_ij['noise'] == 0.0)]['ij'],
    #     reses_ij[(reses_ij['j_policy'] == 'rand') & (reses_ij['rep'] == 0) & (reses_ij['noise'] == 0.0)]['MR_ij_sim'], color='red', linestyle='-')
    # ax1.plot(reses_ij[(reses_ij['j_policy'] == 'fifo') & (reses_ij['rep'] == 7) & (reses_ij['noise'] == 1.0)]['ij'],
    #     reses_ij[(reses_ij['j_policy'] == 'fifo') & (reses_ij['rep'] == 7) & (reses_ij['noise'] == 1.0)]['MR_ij_sim'], color='blue', linestyle='-')
    # ax1.plot(reses_ij[(reses_ij['j_policy'] == 'fifo') & (reses_ij['rep'] == 0) & (reses_ij['noise'] == 0.0)]['ij'],
    #     reses_ij[(reses_ij['j_policy'] == 'rand') & (reses_ij['rep'] == 0) & (reses_ij['noise'] == 0.0)]['pi_ent'], color='red', linestyle=':')
    # ax1.plot(reses_ij[(reses_ij['j_policy'] == 'rand') & (reses_ij['rep'] == 7) & (reses_ij['noise'] == 1.0)]['ij'],
    #     reses_ij[(reses_ij['j_policy'] == 'fifo') & (reses_ij['rep'] == 7) & (reses_ij['noise'] == 1.0)]['pi_ent'], color='blue', linestyle=':')

    # ax1.plot(reses_ij[(reses_ij['j_policy'] == 'rand') & (reses_ij['rep'] == 0) & (reses_ij['noise'] == 0.0)]['ij'],
    #     reses_ij[(reses_ij['j_policy'] == 'rand') & (reses_ij['rep'] == 0) & (reses_ij['noise'] == 0.0)]['MR_ij_sim'], color='red')
    ax1.bar(reses_ij[(reses_ij['j_policy'] == 'fifo') & (reses_ij['rep'] == 7) & (reses_ij['noise'] == 1.0)]['ij'],
        reses_ij[(reses_ij['j_policy'] == 'fifo') & (reses_ij['rep'] == 7) & (reses_ij['noise'] == 1.0)]['MR_ij_sim'], color='blue')
    # ax1.plot(reses_ij[(reses_ij['j_policy'] == 'fifo') & (reses_ij['rep'] == 0) & (reses_ij['noise'] == 0.0)]['ij'],
    #     reses_ij[(reses_ij['j_policy'] == 'rand') & (reses_ij['rep'] == 0) & (reses_ij['noise'] == 0.0)]['pi_ent'], color='red')
    ax1.bar(reses_ij[(reses_ij['j_policy'] == 'rand') & (reses_ij['rep'] == 7) & (reses_ij['noise'] == 1.0)]['ij'],
        reses_ij[(reses_ij['j_policy'] == 'fifo') & (reses_ij['rep'] == 7) & (reses_ij['noise'] == 1.0)]['pi_ent'] - reses_ij[(reses_ij['j_policy'] == 'fifo') & (reses_ij['rep'] == 7) & (reses_ij['noise'] == 1.0)]['MR_ij_sim'], color='red')
    # for j_policy, color in [('fifo', 'blue'), ('max_weight', 'red'), ('rand', 'green')]:
    #
    #     data = reses[reses['j_policy'] == j_policy]
    #     ax1.scatter(data['H(MR_ij_Sim)'], data['KL(MR_ij_Sim, pi_ent)'], color=color, label=j_policy)


    #ax1.set_xticklabels(data['ij'])
    #ax1.xaxis.set_label_text('First X-axis') # Uncomment to label axis
    ax1.yaxis.set_label_text("$\pi_{ij}$")
    plt.legend()
    plt.show()
    # Second X-axis
    # ax2 = ax1.twiny()
    # offset = 0, -25 # Position of the second axis
    # new_axisline = ax2.get_grid_helper().new_fixed_axis
    # ax2.axis["bottom"] = new_axisline(loc="bottom", axes=ax2, offset=offset)
    # ax2.axis["top"].set_visible(False)
    #
    # len(data[j])
    #
    # ax2.set_xticks([0.0, 0.6, 1.0])
    # ax2.xaxis.set_major_formatter(ticker.NullFormatter())
    # ax2.xaxis.set_minor_locator(ticker.FixedLocator([0.3, 0.8]))
    # ax2.xaxis.set_minor_formatter(ticker.FixedFormatter(['mammal', 'reptiles']))
    #
    # # Third X-axis
    # ax3 = ax1.twiny()
    # offset = 0, -50
    # new_axisline = ax3.get_grid_helper().new_fixed_axis
    # ax3.axis["bottom"] = new_axisline(loc="bottom", axes=ax3, offset=offset)
    # ax3.axis["top"].set_visible(False)
    #
    # ax3.set_xticks([0.0, 1.0])
    # ax3.xaxis.set_major_formatter(ticker.NullFormatter())
    # ax3.xaxis.set_minor_locator(ticker.FixedLocator([0.5]))
    # ax3.xaxis.set_minor_formatter(ticker.FixedFormatter(['vertebrates']))
    #
    # ax1.grid(1)
    # plt.show()


def view_matching_rates(exp_dir):

    reses_i = get_file_data(exp_dir, 'reses_i')
    reses_ij = get_file_data(exp_dir, 'reses_ij')
    reses_ij[['MR_ij_sim']] = 90.*reses_ij['MR_ij_sim']
    print reses_ij[['i','j','MR_ij_sim']]


def view_2_chain():

    reses_ij = get_file_data('2_chain_2', 'reses_ij')
    reses_i = get_file_data('2_chain_2', 'reses_i')
    # print reses_ij[reses_ij['i'] == reses_ij['j']][
    #                      ['i', 'j', 'noise', 'sim_name', 'MR_ij_sim', 'pi_ent','j_policy', 'sys_size','chain','rho']].rename(columns={'j': 'i^'})
    reses = pd.merge(left=reses_i[['i','noise', 'sim_name', 'lamda', 'j_policy', 'sys_size', 'chain','rho']],
                     right=reses_ij[reses_ij['i'] == reses_ij['j']][
                         ['i', 'j', 'noise', 'sim_name', 'MR_ij_sim', 'pi_ent','j_policy', 'sys_size','chain','rho']].rename(columns={'j': 'i^'}),
                     on=['i','noise', 'sim_name','j_policy','chain','sys_size','rho'], how='left')
    reses = pd.merge(left=reses,
                     right=reses_ij[reses_ij['j'].astype(int) == (reses_ij['i'].astype(int) + 1) % reses_ij['sys_size']][
                         ['i', 'j', 'noise', 'sim_name', 'MR_ij_sim', 'pi_ent','j_policy', 'sys_size','chain','rho']].rename(columns={'j': 'i^+1'}),
                     on=['i', 'noise', 'sim_name','j_policy','chain','sys_size','rho'], how='left')

    reses = reses.fillna(0)

    reses.loc[:, 'abs_error'] = reses['rho']*(np.abs(reses['MR_ij_sim_x'] - reses['pi_ent_x']) + np.abs(reses['MR_ij_sim_y'] - reses['pi_ent_y']))

    print reses.sort_values(by=['noise', 'rho', 'sim_name', 'sys_size','chain', 'i'])

    reses = reses[['noise', 'sim_name', 'j_policy', 'lamda','chain','sys_size','rho', 'abs_error']].groupby(['noise', 'sim_name', 'j_policy','chain','sys_size','rho'],
                                                                          as_index=False).sum()


    reses.loc[:, 'rel_error'] = reses['abs_error']/reses['lamda']

    reses = reses[['noise', 'j_policy', 'rel_error','chain','sys_size','rho']].groupby(['noise', 'j_policy','chain','sys_size','rho'], as_index=False).mean()

    print reses[reses['j_policy'] == 'fifo'].sort_values(by=['sys_size','noise', 'rho', 'chain'])


def view_increasing_systems():

    reses_ij = get_file_data('increasing_N_q2', 'reses_ij')
    reses_i = get_file_data('increasing_N_q2', 'reses_i')
    reses_j = get_file_data('increasing_N_q2', 'reses_j')

    reses_ij.loc[:,'pi_ent/pi_hat'] = reses_ij['pi_hat']/reses_ij['pi_ent']

    reses_ij = pd.merge(left=reses_ij,
                        right=reses_ij[['j', 'N', 'sim_name', 'pi_hat', 'pi_ent']]
                        .groupby(['j', 'N', 'sim_name'], as_index=False).sum()
                        .rename(columns={'pi_ent': 'rho_ent', 'pi_hat': 'rho_hat'}),
                        on=['j', 'N', 'sim_name'], how='left')

    reses_ij.loc[:,'Q_ij_ent'] = reses_ij['pi_ent']/(1.0-reses_ij['rho_ent'])
    reses_ij.loc[:,'Q_ij_hat'] = reses_ij['pi_hat']/(1.0-reses_ij['rho_hat'])

    reses_j.loc[:, 'WT_j_scv_sim'] = (reses_j['WT_j_std_sim']**2)/(reses_j['WT_j_sim'])**2
    reses_i.loc[:, 'WT_i_scv_sim'] = (reses_i['WT_i_std_sim']**2)/(reses_i['WT_i_sim'])**2

    merge_cols = ['i', 'N', 'sim_name', 'rho']

    reses_i = pd.merge(left=reses_i,
                       right=reses_ij[merge_cols + ['MR_ij_sim']].groupby(by=merge_cols, as_index=False).sum()
                       .rename(columns={'MR_ij_sim': 'MR_i_sim'}), on=merge_cols, how='left')

    reses_i = pd.merge(left=reses_i,
                       right=reses_ij[['N', 'sim_name', 'rho', 'MR_ij_sim']]
                       .groupby(by=['N', 'sim_name', 'rho'], as_index=False).sum()
                       .rename(columns={'MR_ij_sim': 'MR_sim'}), on=['N', 'sim_name', 'rho'], how='left')

    reses_i.loc[:, 'WTxMR_i_sim'] = reses_i['WT_i_sim']*reses_i['MR_i_sim']

    reses_i = reses_i.join(reses_i.sort_values(by=['N', 'rho', 'sim_name', 'WT_i_sim'])
                           .groupby(['N', 'rho', 'sim_name'])[['MR_i_sim','WTxMR_i_sim']].cumsum(axis=0)
                           .rename(columns={'WTxMR_i_sim': 'cum_WT_i_sim', 'MR_i_sim': 'cum_MR_i_sim'}))


    view_rho = 0.85

    reses_i = reses_i[(reses_i['rho'] == view_rho)]
    reses_j = reses_j[(reses_j['rho'] == view_rho)]



    reses_j.loc[:,'norm_j'] = (reses_j['j'] + 1.)/reses_j['N']
    #print reses_i

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    #print colors
    fig, ax = plt.subplots(1,2)
    i = -1
    j = -1
    k = None
    lines = [[],[]]
    color = colors[0]
    util=True
    if util:
        for key, grp in reses_j.groupby(['N', 'sim_name', 'rho']):

            if (key[1] == 'plain_FIFO' or key[1] == 'plain_MW') and key[0] in [2, 5, 10, 50, 100]:

                norm_j = grp['norm_j']
                rho_j = 1. - grp['r_j_sim']

                if 'MW' in key[1]:
                    k = 0
                    i += 1
                    h = i
                elif 'FIFO' in key[1]:
                    k = 1
                    j += 1
                    h = j


                #ax[k].fill_between(cum_mr_i_sim, cum_mr_i_sim*max_wt, cum_wt_i_sim, color=color, alpha=0.2)
                #ax[k].fill_between(cum_mr_i_sim, 0, cum_wt_i_sim,  color=color, label=key[0])
                # ax[k].plot(cum_mr_i_sim, cum_mr_i_sim*max_wt, color='black', linewidth=.5)
                # ax[k].plot(cum_mr_i_sim, cum_wt_i_sim, color='black')
                # ax[k].plot(cum_mr_i_sim, wt_i_sim, color='black', linewidth=.5,
                #            linestyle='-', marker=MARKERS[h],markersize=4,label='n=' + str(key[0]))
                ax[k].plot(norm_j, rho_j, color='black', linewidth=.5,
                       linestyle='-', marker=MARKERS[h],markersize=4,label='n=' + str(key[0]))
                # ax[k].plot(cum_mr_i_sim, cum_wt_i_sim, color='black')
                ax[k].set_xlabel('Server', fontsize=16)
                ax[k].set_ylabel('Utilization', fontsize=16)
                ax[k].set_xticks([0, 0.5, 1.])
                ax[k].set_xticklabels(['1','n/2', 'n'], fontsize=16)
                ax[k].set_yticks([0.1*t for t in range(11)])
                ax[k].set_ylim((.2, 1.05))

                ax[k].legend(title='')


            # ax[1].bar(left=[v for v in list((grp['norm_cum_customer_n_by_wt']-grp['norm_customer_n']))],
            #           width=[v for v in list(grp['norm_customer_n'])],
            #           height=[v for v in list(grp['customer_wt'])],
            #           label=key,
            #           align='edge', alpha=-1)
            # ax[1].set_xlim((0, 1))
            # ax[1].legend()
        ax[0].set_title(r"LQF-ALIS," r"$\quad \rho=.85$", fontsize=16, color='black')
        ax[1].set_title(r"FIFO-ALIS," r"$\quad\rho=.85$", fontsize=16, color='black')
        plt.show()

    else:

        for key, grp in reses_i.groupby(['N', 'sim_name', 'rho']):

            if (key[1] == 'plain_FIFO' or key[1] == 'plain_MW') and key[0] in [2, 5, 10, 50, 100]:

                cum_wt_i_sim = np.append(grp['cum_WT_i_sim'], np.array([0]))

                cum_mr_i_sim = np.append(grp['cum_MR_i_sim'], np.array([0]))
                wt_i_sim = np.append(grp['WT_i_sim'], np.array([0]))
                wt_sim = cum_wt_i_sim.mean()
                max_wt = np.amax(cum_wt_i_sim)
                area_1 = calc_area_between_curves(cum_mr_i_sim, cum_mr_i_sim*max_wt, cum_mr_i_sim, cum_mr_i_sim*0)
                area_2 = calc_area_between_curves(cum_mr_i_sim, cum_wt_i_sim, cum_mr_i_sim, cum_mr_i_sim*0)
                print key
                #print grp[['i', 'cum_WT_i_sim', 'WT_i_sim']]
                print max_wt
                #print "{:.0%}".format((area_1 - area_2)/area_1)

                sort_wt_i_sim = np.sort(wt_i_sim)
                nn = sort_wt_i_sim.shape[0]
                gini_score = (2*((np.arange(1, nn + 1, 1)*sort_wt_i_sim).sum()))/(nn * sort_wt_i_sim.sum()) - ((nn+1)/nn)
                #print "{:.0%}".format(gini_score)

                #print cum_mr_i_sim
                if 'MW' in key[1]:
                    k = 0
                    i += 1
                    h = i
                elif 'FIFO' in key[1]:
                    k = 1
                    j += 1
                    h = j


                #ax[k].fill_between(cum_mr_i_sim, cum_mr_i_sim*max_wt, cum_wt_i_sim, color=color, alpha=0.2)
                #ax[k].fill_between(cum_mr_i_sim, 0, cum_wt_i_sim,  color=color, label=key[0])
                # ax[k].plot(cum_mr_i_sim, cum_mr_i_sim*max_wt, color='black', linewidth=.5)
                # ax[k].plot(cum_mr_i_sim, cum_wt_i_sim, color='black')
                # ax[k].plot(cum_mr_i_sim, wt_i_sim, color='black', linewidth=.5,
                #            linestyle='-', marker=MARKERS[h],markersize=4,label='n=' + str(key[0]))
                lines[k] += ax[k].plot(1.-cum_mr_i_sim, wt_i_sim, color='black', linewidth=.5,
                       linestyle='-', marker=MARKERS[h],markersize=4,label='n=' + str(key[0]))

                lines[k] += ax[k].plot(np.arange(0,1.5,0.5), max_wt * np.ones(3) , color='black', linewidth=.5,
                       linestyle=LINE_STYLES['dashed'], marker=MARKERS[h],markersize=4)

        lines[0] += ax[0].plot(np.arange(0,1.5,0.5),  5.66667 * np.ones(3), color='black', linewidth=.5,
               linestyle=LINE_STYLES['dotted'],markersize=4)
        lines[1] += ax[1].plot(np.arange(0,1.5,0.5),  5.66667 * np.ones(3), color='black', linewidth=.5,
               linestyle=LINE_STYLES['dotted'],markersize=4)


        for k in [0,1]:

            leg = Legend(ax[k], [lines[k][g] for g in [1,3,5,7,9]], [","]*5, loc='upper left',
                         bbox_to_anchor=(.62,.95), frameon=False, title='System \n Avg.', handlelength=3)
            leg._legend_box.align = "left"
            plt.setp(leg.get_title(), multialignment='center')
            ax[k].add_artist(leg)

            # ax[k].plot(cum_mr_i_sim, cum_wt_i_sim, color='black')
            ax[k].set_xlabel('Customer Class', fontsize=14)
            ax[k].set_ylabel('Waiting Time', fontsize=14)
            ax[k].set_xticks([0, 0.5, 1.])
            ax[k].set_yticks([2*g for g in range(9)])
            ax[k].set_yticklabels([2*g for g in range(9)], fontsize=14)
            ax[k].set_xticklabels(['1','n/2', 'n'], fontsize=14)
            ax[k].set_ylim((0, 18))
            ax[k].text(.642, 12.5, '                                      \n'
                                 '                                      \n'
                                 '                                      \n'
                                 '                                      \n'  
                                 '                                      \n'   
                                 '                                      \n'                            
                                 '                                      \n'                                  '                                      \n',
                       color='black', bbox=dict(facecolor='none', edgecolor='grey',pad=1))

            leg2 = ax[k].legend(title='Class \n Avg.', frameon=False, bbox_to_anchor=(.97,.95), )
            leg2._legend_box.align = "left"
            plt.setp(leg2.get_title(), multialignment='center')
            # ax[1].bar(left=[v for v in list((grp['norm_cum_customer_n_by_wt']-grp['norm_customer_n']))],
            #           width=[v for v in list(grp['norm_customer_n'])],
            #           height=[v for v in list(grp['customer_wt'])],
            #           label=key,
            #           align='edge', alpha=-1)
            # ax[1].set_xlim((0, 1))
            # ax[1].legend()
        ax[0].set_title(r"LQF-ALIS," r"$\quad \rho=.85$", fontsize=16, color='black')
        ax[1].set_title(r"FIFO-ALIS," r"$\quad\rho=.85$", fontsize=16, color='black')
        ax[0].text(0.5, 6., 'M/M/1 Waiting Time', fontsize=10,
         horizontalalignment='center',
         verticalalignment='center',
         multialignment='center')
        ax[1].text(0.5, 6., 'M/M/1 Waiting Time', fontsize=10,
         horizontalalignment='center',
         verticalalignment='center',
         multialignment='center')
        plt.show()


def gini_coefficent():

    fig, ax = plt.subplots()
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    mpl.rcParams['text.latex.preamble'] = [r'\usepackage{xfrac}']
    mpl.rcParams['hatch.linewidth'] = 0.05

    lines = []
    lines += ax.plot(np.arange(0,1.1,0.1), np.arange(0,1.1,0.1), color='black')
    lorenz_curve = np.array([sum([float(k) + (0.01 * i == 0) for k in range(i)])/45. for i in range(11)])
    lines += ax.plot(np.arange(0,1.1,0.1), lorenz_curve, color='black')
    lines += ax.plot(np.arange(0,1.1,0.1), [0]*11, color='black')
    lines += ax.plot([1]*11, np.arange(0,1.1,0.1), color='black')
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticks([-1, -1, 0.25, 0.5, 0.75, 1])
    ylabels = [r'$\frac{-1}{-1}'
               r'$0$',
               r'$\frac{Y}{4}}$',
               r'$\displaystyle\frac{WT}{4}$',
               r'$\displaystyle\frac{WT}{2}$',
               r'$\displaystyle\frac{3WT}{4}$',
               r'$WT$']

    ax.set_yticklabels(ylabels)
    ax.set_xticks(np.arange(-0.4, 1.1, 0.2))
    ax.set_xticklabels(["{:.0%}".format(x) for x in np.arange(-0.4, 1.1, 0.2)])
    ax.fill_between(np.arange(0,1.1,0.1), lorenz_curve, np.arange(0,1.1,0.1),  facecolor='grey', alpha=0.5)
    ax.fill_between(np.arange(0,1.1,0.1), 0, lorenz_curve,  facecolor='grey', alpha=0.2, hatch='....')
    ax.set_ylim((0, 1))
    ax.set_xlim((0, 1))
    ax.set_xlabel(r'% of customers', fontsize=16)
    ax.set_ylabel(r'Contribution to Avg. WT', fontsize=16)

    ax.text(0.8, .25, 'II', fontsize=50,
            horizontalalignment='center',
            verticalalignment='center',
            multialignment='center')

    ax.text(0.45, .32, 'I', fontsize=50,
            horizontalalignment='center',
            verticalalignment='center',
            multialignment='center')

    angle1 = plt.gca().transData.transform_angles(np.array((45,)), np.array((0.5,0.55)).reshape((1, 2)))[0]
    angle2 = plt.gca().transData.transform_angles(np.array((43,)), np.array((0.5,0.55)).reshape((1, 2)))[0]

    ax.text(0.5, .54, 'Line of Equality',
            fontsize=14, rotation=angle1, rotation_mode='anchor', horizontalalignment='center')

    ax.text(0.52, .2, 'Lorenz Curve',
            fontsize=14, rotation=angle2, rotation_mode='anchor', horizontalalignment='center', verticalalignment='top')

    plt.show()


def gini_graphs(view_exp, filters, base_cols, n_name='N', show=True):


    #base_cols = ['chain_len', 'n', 'exp_num', 'sim_name', 'rho', 'i_policy', 'j_policy']

    reses_ij = filter_df(get_file_data(view_exp, 'reses_ij'), filters)
    reses_i = filter_df(get_file_data(view_exp, 'reses_i'), filters)
    reses_j = filter_df(get_file_data(view_exp, 'reses_j'), filters)

    reses_ij.loc[:,'pi_ent/pi_hat'] = reses_ij['pi_hat']/reses_ij['pi_ent']

    reses_ij = pd.merge(left=reses_ij,
                        right=reses_ij[base_cols + ['j', 'pi_hat', 'pi_ent']]
                        .groupby(base_cols + ['j'], as_index=False).sum()
                        .rename(columns={'pi_ent': 'rho_ent', 'pi_hat': 'rho_hat'}),
                        on=base_cols + ['j'], how='left')

    reses_ij.loc[:,'Q_ij_ent'] = reses_ij['pi_ent']/(1.0-reses_ij['rho_ent'])
    reses_ij.loc[:,'Q_ij_hat'] = reses_ij['pi_hat']/(1.0-reses_ij['rho_hat'])

    reses_j.loc[:, 'WT_j_scv_sim'] = (reses_j['WT_j_std_sim']**2)/(reses_j['WT_j_sim'])**2
    reses_i.loc[:, 'WT_i_scv_sim'] = (reses_i['WT_i_std_sim']**2)/(reses_i['WT_i_sim'])**2

    reses_i = pd.merge(left=reses_i,
                       right=reses_ij[base_cols + ['i', 'MR_ij_sim']].groupby(by=base_cols+['i'], as_index=False).sum()
                       .rename(columns={'MR_ij_sim': 'MR_i_sim'}), on=base_cols + ['i'], how='left')

    #print reses_i[base_cols + ['i', 'MR_i_sim']]

    reses_i = pd.merge(left=reses_i,
                       right=reses_ij[base_cols + ['MR_ij_sim']]
                       .groupby(by=base_cols, as_index=False).sum()
                       .rename(columns={'MR_ij_sim': 'MR_sim'}), on=base_cols, how='left')

    reses_i.loc[:, 'WTxMR_i_sim'] = reses_i['WT_i_sim']*reses_i['MR_i_sim']

    reses_i = reses_i.join(reses_i.sort_values(by=base_cols + ['WT_i_sim'])
                           .groupby(base_cols)[['MR_i_sim','WTxMR_i_sim']].cumsum(axis=0)
                           .rename(columns={'WTxMR_i_sim': 'cum_WT_i_sim', 'MR_i_sim': 'cum_MR_i_sim'}))


    reses_j.loc[:, 'norm_j'] = (reses_j['j'] + 1.)/reses_j[n_name]

    fig, ax = plt.subplots(1, 2)
    i = -1
    j = -1
    k = None
    lines = [[], []]
    color = COLORS[0]
    util = False
    fill_between = False
    gini_curve = True

    #base_cols = [0'chain_lem', 1'm', 2'exp_num', 3'sim_name', 4'rho']
    res_dic = dict()
    for key, grp in reses_i.groupby(base_cols):

        grp = grp.sort_values(by=['cum_MR_i_sim'])
        key_dict = dict(zip(base_cols, key))

        if ('FIFO' in key_dict['sim_name'] or 'MW' in key_dict['sim_name'] or 'prio' in key_dict['sim_name']) \
                and key_dict[n_name] in [5, 10, 50, 100, 500]:

            if 'MW' in key_dict['sim_name']:
                k = 0
                i += 1
                h = i
            elif 'FIFO' in key_dict['sim_name'] or 'prio' in key_dict['sim_name']:
                k = 1
                j += 1
                h = j

            if util:
                norm_j = grp['norm_j']
                rho_j = 1. - grp['r_j_sim']
                ax[k].plot(norm_j, rho_j, color='black', linewidth=.5,
                       linestyle='-', marker=MARKERS[h],markersize=4,label='n=' + str(key[1]))
                # ax[k].plot(cum_mr_i_sim, cum_wt_i_sim, color='black')
                ax[k].set_xlabel('Server', fontsize=16)
                ax[k].set_ylabel('Utilization', fontsize=16)
                ax[k].set_xticks([0, 0.5, 1.])
                ax[k].set_xticklabels(['1','n/2', 'n'], fontsize=16)
                ax[k].set_ylim((0, 1))

                ax[k].legend(title='')
            else:

                cum_wt_i_sim = np.append(np.array([0]), grp['cum_WT_i_sim'])
                cum_mr_i_sim = np.append(np.array([0]), grp['cum_MR_i_sim'])
                #print grp[['i', 'WT_i_sim', 'cum_MR_i_sim']]
                #print cum_mr_i_sim
                wt_i_sim = np.append(np.array([0]), grp['WT_i_sim'])
                max_cum_wt = np.amax(cum_wt_i_sim)
                max_wt = np.amax(wt_i_sim)
                area_1 = calc_area_between_curves(cum_mr_i_sim, cum_mr_i_sim*max_cum_wt, cum_mr_i_sim, cum_mr_i_sim*0)
                area_2 = calc_area_between_curves(cum_mr_i_sim, cum_wt_i_sim, cum_mr_i_sim, cum_mr_i_sim*0)
                area_3 = calc_area_between_curves(cum_mr_i_sim, cum_mr_i_sim, cum_mr_i_sim, cum_mr_i_sim*0)
                area_4 = calc_area_between_curves(cum_mr_i_sim, cum_wt_i_sim/max_cum_wt, cum_mr_i_sim, cum_mr_i_sim*0)
                gini1 = (area_1 - area_2)/area_1
                gini2 = (area_3 - area_4)/area_3
                print key
                res_dic[key_dict['sim_name']] = {'avg. WT': max_cum_wt, 'gini1': gini1,'gini2': gini2, 'worst':max_wt }
                print max_cum_wt
                print "{:.0%}".format((area_1 - area_2)/area_1)
                # sort_wt_i_sim = np.sort(wt_i_sim)
                # nn = sort_wt_i_sim.shape[0]
                # gini_score = (2*((np.arange(1, nn + 1, 1)*sort_wt_i_sim).sum()))/(nn * sort_wt_i_sim.sum()) - ((nn+1)/nn)
                # print "{:.0%}".format(gini_score)

                if fill_between:
                    ax[k].fill_between(cum_mr_i_sim, cum_mr_i_sim*max_cum_wt, cum_wt_i_sim, color=color, alpha=0.2)
                    ax[k].fill_between(cum_mr_i_sim, 0, cum_wt_i_sim,  color=color, label=key)
                if gini_curve:
                    ax[k].plot(cum_mr_i_sim, cum_mr_i_sim*max_cum_wt, color='black', linewidth=.1)
                    ax[k].plot(cum_mr_i_sim, cum_wt_i_sim, color='black',linewidth=.5,
                               linestyle='-', marker=MARKERS[h%len(MARKERS)], markersize=3,
                               label=key_dict['sim_name'] + '-' + str(key_dict['rho']))
                    if 'prio' in key_dict['sim_name']:
                        ax[0].plot(cum_mr_i_sim, cum_mr_i_sim*max_cum_wt, color='black', linewidth=.1)
                        ax[0].plot(cum_mr_i_sim, cum_wt_i_sim, color='black',linewidth=.5,
                                   linestyle='-', marker=MARKERS[(h)%len(MARKERS)], markersize=3,
                                   label=key_dict['sim_name'] + '-' + str(key_dict['rho']))

                # ax[k].plot(cum_mr_i_sim, wt_i_sim, color='black', linewidth=.5,
                #            linestyle='-', marker=MARKERS2[h],markersize=4,label='n=' + str(key[1]))
                #print 1.-cum_mr_i_sim
                else:
                    lines[k] += ax[k].plot(1.-cum_mr_i_sim, wt_i_sim, color='black', linewidth=.5,
                           linestyle='-', marker=MARKERS[h%len(MARKERS)], markersize=4,
                                           label=key_dict['sim_name'] + '-' + str(key_dict['rho']))

                # lines[k] += ax[k].plot(np.arange(0,1.2,0.2), wt_sim * np.ones(6), color='black', linewidth=.5,
                #                        linestyle=LINE_STYLES['dashed'], marker=MARKERS2[h], markersize=4)

                for k in [0, 1]:

                    # leg = Legend(ax[k], [lines[k][g] for g in range(len(lines[k])) if g%2 == 1],[" ,",", ",", ",", "],
                    #              loc='upper left', bbox_to_anchor=(.7, .95), frameon=True, title='Total')
                    # leg._legend_box.align = "left"
                    # ax[k].add_artist(leg)

                    # ax[k].plot(cum_mr_i_sim, cum_wt_i_sim, color='black')
                    ax[k].set_xlabel('Customer Class', fontsize=16)
                    ax[k].set_ylabel('Avg. Waiting Time', fontsize=16)
                    ax[k].set_xticks([0, 0.5, 1.])
                    #ax[k].set_yticks([2*g for g in range(9)])
                    ax[k].set_xticklabels(['1','n/2', 'n'], fontsize=16)
                    #ax[k].set_ylim((0, 18))
                    ax[k].legend()

                    # leg2 = ax[k].legend(title='Class', frameon=True, bbox_to_anchor=(1.,.95), )
                    # leg2._legend_box.align = "left"

                    # ax[0].set_title(r"LQF-ALIS," r"$\quad \rho=.95$", fontsize=16, color='black')
                    # ax[1].set_title(r"FIFO-ALIS," r"$\quad\rho=.95$", fontsize=16, color='black')
    for key, val in sorted(res_dic.iteritems()):
        if 'FIFO' in key or 'prio' in key:
            print key, [(key2, ':', val2) for key2, val2 in val.iteritems()]
    print '-------------------------------------------------'
    print '-------------------------------------------------'
    for key, val in sorted(res_dic.iteritems()):
        if 'MW' in key or 'prio' in key:
            print key, [(key2, ':', val2) for key2, val2 in val.iteritems()]
    if show:
        plt.show()


    # print reses_ij[((reses_ij['sim_name'] == 'rho_weighted_FIFO')| (reses_ij['sim_name'] == 'plain_FIFO') )&
    #                (reses_ij['N'] == 30) & (reses_ij['j'] == 0)][['sim_name', 'i','j','pi_ent','pi_hat','rho_ent','rho_hat','Q_ij_ent','Q_ij_hat', 'MR_ij_sim', 'MR_ij_count', 'WT_ij_sim', 'w_ij']]
    # print reses_ij[reses_ij['i'] == reses_ij['j']][
    #                      ['i', 'j', 'noise', 'sim_name', 'MR_ij_sim', 'pi_ent','j_policy', 'sys_size','chain','rho']].rename(columns={'j': 'i^'})
    # print reses_i
    # res = dict()
    # cols = ['i', 'sim_name','rho', 'N','WT_i_sim', 'WT_i_std_sim', 'WT_i_scv_sim']
    # for sim_name in ['rho_weighted_FIFO', 'plain_FIFO', 'plain_prio', 'rho_weighted_MW', 'plain_MW']:
    #         rename_dic = dict((colname, colname + '_' + sim_name) for colname in cols[2:])
    #         res[sim_name] =\
    #            reses_i[(reses_j['sim_name'] == sim_name)][cols].rename(columns=rename_dic)

    # res = reses_i[['i', 'sim_name', 'rho', 'N','WT_i_sim', 'WT_i_scv_sim']]
    # res = pd.merge(left=res,
    #                right=reses_ij[['i', 'sim_name', 'rho', 'N', 'MR_ij_sim']].groupby(['i', 'sim_name', 'rho', 'N'],
    #                                       as_index=False).sum().rename(columns={'MR_ij_sim': 'MR_i_sim'}),
    #                how='left', on=['i', 'sim_name', 'rho', 'N'])
    #
    # res = pd.merge(left=res,
    #                right=res[['i', 'sim_name', 'rho', 'N','WT_i_sim']].sort_values(by='WT_i_sim').groupby(['i', 'sim_name', 'rho', 'N'], as_index=False).cumsum())
    #
    # res = res.sort_values(by=['sim_name', 'rho', 'N','WT_i_sim'])
    # #res.loc[:, 'cum_WT_i'] = res.groupby(['sim_name', 'rho', 'N','WT_i_sim'])['WT_i_sim'].apply(lambda x: x.cumsum())
    # print res

    # print reses_ij[((reses_ij['sim_name'] == 'rho_weighted_FIFO') |
    #                (reses_ij['sim_name'] == 'plain_FIFO')) & (reses_ij['N'] == 50) & (reses_ij['i'] == 49)][['i','j','N','MR_ij_sim','i_policy','j_policy','sim_name','pi_ent','pi_hat','pi_ent/pi_hat','WT_ij_sim','WT_ij_std_sim']]
    # print reses_ij[((reses_ij['sim_name'] == 'rho_weighted_FIFO') |
    #                 (reses_ij['sim_name'] == 'plain_FIFO')) & (reses_ij['N'] == 50)][
    #     ['j','pi_ent','pi_hat']].groupby(['j']).sum()


def view_approximation(view_exp, base_cols, filters):

    reses_ij = filter_df(get_file_data(view_exp, 'reses_ij'), filters)
    reses_i = filter_df(get_file_data(view_exp, 'reses_i'), filters)
    reses_j = filter_df(get_file_data(view_exp, 'reses_j'), filters)

    #print reses_ij

    reses_ij = pd.merge(left=reses_ij,
                        right=reses_i[base_cols + ['lamda']].groupby(base_cols, as_index=False).sum(),
                        how='left', on=base_cols)

    reses_ij.loc[:, 'lamda_ij_sim'] = reses_ij['lamda'] * reses_ij['MR_ij_sim']

    reses_ij[base_cols + ['pi_ent', 'pi_hat', 'lamda_ij_sim']].groupby(base_cols, as_index=False).sum()

    reses_j = pd.merge(left=reses_j,
                       right=reses_ij[base_cols + ['j', 'lamda_ij_sim', 'pi_ent', 'pi_hat']]
                       .groupby(base_cols + ['j'], as_index=False).sum()
                       .rename(columns={'lamda_ij_sim': 'lamda_j', 'pi_ent': 'lamda_ent', 'pi_hat': 'lamda_hat'}),
                       on=base_cols + ['j'], how='left')

    reses_j.loc[:, 'rho_j_sim'] = reses_j['lamda_j']/reses_j['mu']
    reses_j.loc[:, 'rho_ent'] = reses_j['lamda_ent']/reses_j['mu']
    reses_j.loc[:, 'rho_hat'] = reses_j['lamda_hat']/reses_j['mu']

    reses_ij = pd.merge(left=reses_ij,
                        right=reses_j[base_cols + ['j','rho_j_sim', 'rho_ent', 'rho_hat', 'mu']],
                        on=base_cols + ['j'], how='left')

    print reses_ij

    fig, ax = plt.subplots(2, 2)

    v = 0
    for key, grp in reses_ij.groupby(base_cols):
        key_dict = dict(zip(base_cols, key))
        print key_dict['sim_name']
        k = 0 if 'FIFO' in key_dict['sim_name'] else 1
        if 'plain' in key_dict['sim_name'] and 'prio' not in key_dict['sim_name']:
            print 'k', k, key_dict
            ax[0, k].plot(grp['ij'], grp['pi_ent'], label='pi_ent')
            ax[0, k].plot(grp['ij'], grp['pi_hat'], label='pi_hat')
        if key_dict['sim_name'] == 'plain_prio':
            ax[0, 0].plot(grp['ij'], grp['lamda_ij_sim'], label=key_dict['sim_name'])
        ax[0, k].plot(grp['ij'], grp['lamda_ij_sim'], label=key_dict['sim_name'])
    ax[0, 0].legend()
    for key, grp in reses_j.groupby(base_cols):
        key_dict = dict(zip(base_cols, key))
        k = 0 if 'FIFO' in key_dict['sim_name'] else 1
        if 'plain' in key_dict['sim_name'] and 'prio' not in key_dict['sim_name']:
            print 'k', k, key_dict
            ax[1, k].plot(grp['j'], grp['rho_ent'], label='rho_ent')
            ax[1, k].plot(grp['j'], grp['rho_hat'], label='rho_hat')
        if key_dict['sim_name'] == 'plain_prio':
            ax[1, 0].plot(grp['j'], grp['rho_j_sim'], label=key_dict['sim_name'])
        ax[1, k].plot(grp['j'], grp['rho_j_sim'], label=key_dict['sim_name'])
    ax[0, 1].legend()
    #plt.legend()

    plt.show()


def get_pi_hat(view_exp, filters, base_cols):

    reses_ij = filter_df(get_file_data(view_exp, 'reses_ij'), filters)
    reses_i = filter_df(get_file_data(view_exp, 'reses_i'), filters)
    reses_j = filter_df(get_file_data(view_exp, 'reses_j'), filters)

    reses_ij = pd.merge(left=reses_ij,
                        right=reses_i[base_cols + ['lamda']].groupby(base_cols, as_index=False).sum(),
                        how='left', on=base_cols)

    reses_ij.loc[:, 'lamda_ij_sim'] = reses_ij['lamda'] * reses_ij['MR_ij_sim']

    cords = reses_ij[['i', 'j']].drop_duplicates()
    i_cords, j_cords = cords['i'], cords['j']
    q_shape = np.max(cords, axis=0) + 1
    Q = np.zeros((q_shape))
    Q[i_cords, j_cords] = 1.0

    lamda = reses_i.sort_values(by='i')['lamda'].drop_duplicates().values
    mu = reses_j.sort_values(by='j')['mu'].values
    n = len(mu)
    m = len(lamda)
    lamda_p = np.append(lamda, mu.sum() - lamda.sum())
    Qp = sps.vstack((Q, np.ones((1, n))), format='csr')
    wls, rho_m, rho_n = bipartite_workload_decomposition(Q, lamda, mu)
    Wp = np.diag(np.ones(n) - rho_n, 0).dot(Q)
    Wp = np.vstack((Wp, rho_n))
    Wps = sps.csr_matrix(Wp)
    Mps = 0*sps.csr_matrix(Qp)
    A,b,z, cols = transform_to_normal_form(Mps, Wps, Qp, Qp, lamda_p, mu)
    pi_hat, duals = fast_primal_dual_algorithm(A, b, z)

    pi_hat = pi_hat.reshape((m+1, n))
    pi_hat_wo_mu = np.divide(pi_hat, Wp, out=np.zeros_like(pi_hat), where=Wp != 0)
    pi_hat_wo_mu = sps.csr_matrix(pi_hat_wo_mu)
    pi_hat_wo_mu = pd.DataFrame({'i': pi_hat_wo_mu.nonzero()[0],
                                 'j': pi_hat_wo_mu.nonzero()[1],
                                 'pi_hat_wo_mu': pi_hat_wo_mu.data})

    Z = Qp.dot(sps.diags(mu))

    A,b,z, cols = transform_to_normal_form(Mps, Wps, Qp, Z, lamda_p, mu)

    pi_hat, duals = fast_primal_dual_algorithm(A, b, z)
    pi_hat_w_mu = pi_hat.reshape((m+1, n))
    pi_hat_w_mu = np.divide(pi_hat_w_mu, Wp, out=np.zeros_like(pi_hat_w_mu), where=Wp != 0)
    pi_hat_w_mu = sps.csr_matrix(pi_hat_w_mu)
    pi_hat_w_mu = pd.DataFrame({'i': pi_hat_w_mu.nonzero()[0],
                                 'j': pi_hat_w_mu.nonzero()[1],
                                 'pi_hat_w_mu': pi_hat_w_mu.data})
    reses_ij = pd.merge(left=reses_ij, right=pi_hat_wo_mu, on=['i','j'], how='left')
    reses_ij = pd.merge(left=reses_ij, right=pi_hat_w_mu, on=['i','j'], how='left')

    res = reses_ij[['i','j','pi_ent', 'pi_hat','pi_hat_wo_mu', 'pi_hat_w_mu', 'lamda_ij_sim']]
    print res
    res_u = res[['j','pi_ent', 'pi_hat','pi_hat_wo_mu', 'pi_hat_w_mu', 'lamda_ij_sim']].groupby(['j']).sum()
    res_u = pd.merge(left=res_u, right=reses_j[['j', 'mu']], on='j',how='left')
    res_u.loc[:, 'u_ent'] = res_u['pi_ent']/res_u['mu']
    res_u.loc[:, 'u_pi_hat'] = res_u['pi_hat']/res_u['mu']
    res_u.loc[:, 'u_pi_hat_wo_mu'] = res_u['pi_hat_wo_mu']/res_u['mu']
    res_u.loc[:, 'u_pi_hat_w_mu'] = res_u['pi_hat_w_mu']/res_u['mu']
    res_u.loc[:, 'u_sim'] = res_u['lamda_ij_sim']/res_u['mu']
    res_u.loc[:, 'gap'] = np.abs(res_u['u_sim'] - res_u['u_pi_hat_wo_mu'])
    print res_u.sort_values(by='mu')


if __name__ == '__main__':

    pd.options.display.max_columns = 100
    pd.options.display.max_rows = 1000
    pd.set_option('display.width', 1000)

    erid = 'Erdos_Renyi_id_servers'
    er ='Erdos_Renyi'
    kc = 'k_chains_exp6'
    kcid = 'k_chains_id_servers3'
    simlen = 'sim_len_exp2'
    inc_n = 'increasing_N_q2'
    softer = 'power_of_an_arc3'

    base_cols_chains = ['chain_len', 'n', 'exp_num', 'sim_name', 'rho', 'i_policy', 'j_policy']
    base_cols_erdos = ['n', 'exp_num', 'sim_name', 'rho', 'i_policy', 'j_policy']
    base_cols_inc_n = ['N','rho','sim_name']
    filters_dict_chains = {'rho': [.95, True], 'chain_len': [10, False], 'exp_num': [7, False]}
    filters_dict_sim_len = {'rho': [.99, True], 'chain_len': [10, False]}
    filters_dict_inc_n = {'N': [50, False], 'rho': [0.95, True]}

    # filters_dict_erdos = {'rho': .95, 'n': 50, 'exp_num': 0}
    # filters_dict_chains2 = {'rho': .95, 'chain_len': 3, 'exp_num': 2, 'sim_name': 'plain_FIFO'}
    # view_approximation(inc_n, base_cols_inc_n, filters_dict_inc_n)
    # #get_pi_hat(kc,filters_dict_chains2, base_cols_chains)

    #gini_graphs(inc_n, filters_dict_inc_n, base_cols_inc_n)

    #gini_graphs(kcid, filters_dict_chains, base_cols_chains, n_name='n',show=False)
    #reses_i = filter_df(get_file_data(kcid, 'reses_i'), dict())
    #print reses_i[['rho', 'chain_len', 'exp_num', 'sim_name']].drop_duplicates()
    #
    reses_i = filter_df(get_file_data(softer, 'reses_i'), dict())
    reses_j = filter_df(get_file_data(softer, 'reses_j'), dict())
    reses_ij = filter_df(get_file_data(softer, 'reses_ij'), dict())
    reses_ij.loc[:, 'lamda_ij_sim'] = reses_ij['MR_ij_sim'] * (reses_ij['sim_name'].astype(int) * 0.99 + 0.95)
    print reses_i[((reses_i['i'] == 0) | (reses_i['i'] == 1)) & (reses_i['j_policy'] == 'fifo')]
    print reses_j[((reses_j['j'] == 0) | (reses_j['j'] == 3)) & (reses_j['j_policy'] == 'fifo')]
    print reses_ij[((reses_ij['ij'] == '0,0') | (reses_ij['ij'] == '0,0')) & (reses_ij['j_policy'] == 'fifo')]

    # print reses_ij[(reses_ij['N'] == 10) &
    #                (reses_ij['sim_name'] == 'plain_FIFO') &
    #                np.isclose(reses_ij['rho'], 0.65)][['i', 'j', 'N','sim_name','rho','MR_ij_sim', 'pi_ent']]


    # print reses_ij[['i','j','sim_name',
    #                 'MR_ij_sim', 'MR_ij_sim_std', 'MR_ij_sim_scv',
    #                 'WT_ij_sim', 'WT_ij_sim_std', 'WT_ij_sim_scv']].pivot_table(
    #     index=['i','j'],
    #     columns='sim_name',
    #     values=['MR_ij_sim', 'MR_ij_sim_std', 'MR_ij_sim_scv',
    #                 'WT_ij_sim', 'WT_ij_sim_std', 'WT_ij_sim_scv'])

    # reses_i = reses_i[['i','exp_num','sim_name', 'WT_i_sim']].pivot_table(index=['exp_num', 'i'],
    #                                                                columns=['sim_name'],
    #                                                                values=['WT_i_sim'])

    # print reses_i


    # reses_wt = reses_i['WT_i_sim'][['5,000,000', '1,000,000']]
    # reses_wt.loc[:, 'gap'] = reses_wt['5,000,000'] - reses_wt['1,000,000']
    # reses_wt = reses_wt[['5,000,000', '1,000,000', 'gap']]
    # reses_wt_std = reses_i['WT_i_sim_std'][['5,000,000']].rename(columns={'5,000,000':'std'})
    # reses_wt = pd.concat([reses_wt, reses_wt_std], axis=1)
    # reses_wt.loc[:, 'std_gap'] = np.abs(reses_wt['gap']/reses_wt['std'])
    # print reses_wt
