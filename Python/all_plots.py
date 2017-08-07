'''
Plot Vec-BT, Vec-NBT, DSD-RW, DSD
8/4/2017, Anuththari Gamage

'''

import matplotlib.pyplot as plt
import numpy as np
import pdb

# SBM parameter changed (out of N, K, c, lambda)
x_array =[2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0] 

# Load data 
res = {}
algos = ['vec-bt', 'vec-nbt', 'dsd-rw', 'dsd-inf']
metrics = ['ccr', 'nmi']
rand_tests = 1
N = 2000
data_const = 'N{}'.format(N)
data_varied = 'C'

for r in range(rand_tests):
    res[r] = {}
    for name in algos:
        res[r][name] = {}
        for met in metrics:
            res[r][name][met] = np.empty((len(x_array)))

f = open('VEC/results/vec_data/{}vs{}.txt'.format(data_const, data_varied), 'r')

for r in range(rand_tests):
    for a in algos:
        if a == 'dsd-rw':
            f =  open('DSD/results/dsd_data/{}vs{}.txt'.format(data_const,
            data_varied), 'r')
        for m in metrics:
            line = f.readline().strip()
            res[r][a][m] = np.array(line.split(', '), dtype=float)
            if len(res[r][a][m]) != len(x_array):
                res[r][a][m] = np.concatenate((np.zeros((3,)), res[r][a][m]))


# Get nmi and ccr with mean and std. deviation
nmi_mean = {}
nmi_std = {}
ccr_mean = {}
ccr_std = {}

for a in algos:
    nm = np.zeros((len(x_array),))
    nstd = np.zeros((len(x_array),))
    cm = np.zeros((len(x_array),))
    cstd = np.zeros((len(x_array),))
    arr_ccr = np.zeros(rand_tests)
    arr_nmi = np.zeros(rand_tests)

    for x in range(len(x_array)):
        for r in range(rand_tests):
            arr_ccr[r] = res[r][a]['ccr'][x]
            arr_nmi[r] = res[r][a]['nmi'][x]
        cm[x] = np.mean(arr_ccr)
        cstd[x] = np.std(arr_ccr)
        nm[x] = np.mean(arr_nmi)
        nstd[x] = np.std(arr_nmi)

    nmi_mean[a] = nm
    nmi_std[a] = nstd
    ccr_mean[a] = cm
    ccr_std[a] = cstd
     
# Plot metrics
fig = plt.figure(figsize=(10, 6))
cmap = plt.get_cmap('Accent')
legend = []
#linestyles = ['solid', 'dashed', 'dashdot', 'dotted']

algos = ['vec-nbt', 'dsd-rw']
ls = {}
colors = {}
for i, a in enumerate(algos):
    #ls[a] = linestyles[i]
    colors[a] = cmap(float(i)/len(algos))

# Plot CCR
for a in algos:
    plt.errorbar(x_array, ccr_mean[a], yerr=ccr_std[a], color=colors[a],
            marker='.', linestyle='solid')
    legend = legend + ["ccr: %s" % a]

# Plot NMI
for a in algos:
    plt.errorbar(x_array, nmi_mean[a], yerr=nmi_std[a], color=colors[a],
            marker='.', linestyle='dashed')
    legend = legend + ["nmi: %s" % a]

plt.legend(legend, loc=0)
plt.xlabel('c')
plt.ylim(-0.05, 1.05)
plt.xticks(np.arange(np.max(x_array)+1))
plt.title('N = {} vs {}'.format(N, data_varied))
plt.savefig('figs/{}vs{}.png'.format(data_const, data_varied))
plt.show()
