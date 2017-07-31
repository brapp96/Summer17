# Plot Vec vs. DSD

import matplotlib.pyplot as plt
import numpy as np

#SBM parameters: 
N=[1000]
k=[2]
x_array =[2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0]  # Variable c
ll=[0.9]

res = {}
algos = ['deep', 'nbt','dsd-rw']
metrics = ['ccr', 'nmi']
for name in algos:
    res[name] = {}
    for met in metrics:
        res[name][met] = np.empty((len(x_array)))

##Backtracking RW:
#res['deep']['ccr'] = np.array([0.531,0.502,0.525,0.877,0.891,0.959,0.97,0.988,0.993, 0.995])
#
#res['deep']['nmi'] = np.array( [  2.66785315e-03,   2.00742472e-07,
#    1.45328689e-03, 4.63451769e-01,   5.14319075e-01,   7.56011499e-01,
#    8.11635011e-01,   9.07178521e-01,   9.39852616e-01,   9.55930328e-01])
#
##Non-backtracking RW:
#res['nbt']['ccr'] = np.array( [ 0.533,  0.505 , 0.505,  0.865,  0.892,  0.968,
#    0.962,    0.986 , 0.991,  0.994])
#
#res['nbt']['nmi'] = np.array([  3.96194610e-03 ,  2.00022219e-04,
#    5.90991598e-04 ,  4.55005502e-01,   5.23129245e-01,   7.95963354e-01 ,
#    7.69363875e-01 ,  8.93920377e-01,   9.26528437e-01 ,  9.47528692e-01])
#
## DSD with RW
#res['dsd-rw']['ccr'] = np.array([ 0.504,  0.501,  0.522,  0.893 , 0.923 , 0.976,  0.97,
#        0.989 , 0.993 , 0.995])
#
#res['dsd-rw']['nmi'] =  np.array([ 0.02020976,  0.02050361,  0.02473462,
#    0.51620573 , 0.60874932 , 0.83674586,  0.80778708,  0.91270631,  0.93985262,
#    0.95593033])
#
## DSD closed form
#res['dsd']['ccr'] = np.array([0,0, 0.514,  0.505,  0.589,  0.974,  0.972,
#    0.989,  0.993,    0.995])
#
#res['dsd']['nmi'] = np.array([0,0, 0.01667668,  0.01711541,  0.02494643,
#    0.82604847,  0.81811145,  0.91270631,  0.93985262,  0.95593033])

f = open('results', 'r')
for a in algos:
    for m in metrics:
        line = f.readline().strip()
        line = np.asarray(line.split(), dtype = float)        
        res[a][m] = line

# get nmi and ccr for all algos
nmi_mean = {}
nmi_std = {}
ccr_mean = {}
ccr_std = {}


for a in algos:
    nm = np.zeros((len(x_array),))
    nstd = np.zeros((len(x_array),))
    cm = np.zeros((len(x_array),))
    cstd = np.zeros((len(x_array),))
       
    for x in range(len(x_array)):
        nm[x] = np.mean(res[a]['nmi'][x])
        nstd[x] = np.std(res[a]['nmi'][x])
        cm[x] = np.mean(res[a]['ccr'][x])
        cstd[x] = np.std(res[a]['ccr'][x])
    nmi_mean[a] = nm
    nmi_std[a] = nstd
    ccr_mean[a] = cm
    ccr_std[a] = cstd
     

# Plot NMI
fig = plt.figure(1, figsize=(10, 6))
cmap = plt.get_cmap('jet')
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

# Plot CCR
fig = plt.figure(2, figsize=(10, 6))
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



