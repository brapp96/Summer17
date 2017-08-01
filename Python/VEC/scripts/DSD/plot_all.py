'''
Plot Vec-BT, Vec-NBT, DSD-RW, DSD
8/1/2017, Anuththari Gamage

'''

import matplotlib.pyplot as plt
import numpy as np
import pdb

# SBM parameter changed (out of N, K, c, lambda(
x_array =[2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0] 

# Load data 
res = {}
algos = ['deep', 'nbt']
metrics = ['ccr', 'nmi']
rand_tests = 5

for r in range(rand_tests):
    res[r] = {}
    for name in algos:
        res[r][name] = {}
        for met in metrics:
            res[r][name][met] = np.empty((len(x_array)))

f = open('../N1000vsC.txt', 'r')
for r in range(rand_tests):
    for a in algos:
        for m in metrics:
            line = f.readline().strip()
            res[r][a][m] = np.array(line.split(', '), dtype=float)

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
     
# Plot CCR
fig = plt.figure(1, figsize=(10, 6))
cmap = plt.get_cmap('jet')
i = 0
for a in algos:
    color = cmap(float(i)/len(algos))
    plt.errorbar(x_array, ccr_mean[a], yerr=ccr_std[a],
                 color=color, marker='o', markersize=8)
    i += 1
legend = ["ccr-%s" % a for a in algos]
plt.legend(legend, loc=0)
plt.xlabel('c')
plt.ylim(-0.05, 1.05)
plt.xticks(np.arange(np.max(x_array)+1))
plt.title('CCR vs c')
plt.show()

# Plot NMI
fig = plt.figure(2, figsize=(10, 6))
i = 0
for a in algos:
    color = cmap(float(i)/len(algos))
    plt.errorbar(x_array, nmi_mean[a], yerr=nmi_std[a],
                 color=color, marker='o', markersize=8)
    i += 1
legend = ["nmi-%s" % a for a in algos]
plt.legend(legend, loc=0)
plt.xlabel('c')
plt.ylim(-0.05, 1.05)
plt.xticks(np.arange(np.max(x_array)+1))
plt.title('NMI vs c')
plt.show()



