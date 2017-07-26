#!/usr/bin/env python2.7
"""
Main DSD run file

"""

import PPIparser
import calcDSD
import mygraph
import collections
import sys
import numpy as np
import argparse

temp = "parses PPIs from infile and calculates DSD"
parser = argparse.ArgumentParser(description=temp)

parser.add_argument("infile", help="read PPIs from infile, either "
                    + " a .csv or .tab file that contains a tab/comma/space"
                    + " delimited table with both IDs at first row and"
                    + " first column, or a .list file that contains for"
                    + " each line one interacting pair")
parser.add_argument("-c", "--converge",
                    default=False, help="calculate converged DSD",
                    action="store_true")
parser.add_argument("-n", "--nRW", default=5, help="length of random walks,"
                    + " 5 by default", type=int)
parser.add_argument("-o", "--outfile", help="output DSD file name,"
                    + " tab delimited tables, stdout by default")
parser.add_argument("-q", "--quiet",
                    default=False, help="turn off status message",
                    action="store_true")
parser.add_argument("-f", "--force",
                    default=False, help="calculate DSD for the whole graph"
                    + " despite it is not connected if it is turned on;"
                    + " otherwise, calculate DSD for the largest component ",
                    action="store_true")
parser.add_argument("-m", "--outFMT", default="1",
                    help="the format of output"
                    + " DSD file: type 1 for matrix; type 2 for pairs at each"
                    + " line; type 3 for top K proteins with lowest DSD."
                    + " Type 1 by default", type=int, choices=[1, 2, 3])
parser.add_argument("--outformat", default="matrix",
                    help="the format of output"
                    + " DSD file: 'matrix' for matrix, type 1; 'list' for"
                    + " pairs at each line, type 2; 'top' for top K proteins"
                    + " with lowest DSD, type 3."
                    + " 'matrix' by default",
                    choices=['matrix', 'list', 'top'])
parser.add_argument("-k", "--nTop", default=10, help="if chosen to output"
                    + " lowest DSD nodes, output at most K nodes with lowest"
                    + " DSD, 10 by default", type=int)
parser.add_argument("-t", "--threshold", help="threshold for PPIs' confidence"
                    + " score, if applied", type=float)

#args = ['-n', "4", 'testfiles//small.tab', '-k', '5', '-o', 'haha.test', '-f',
#        '-m', '1', '-t', '342', '--outformat', 'list', '-c', '-h']
#options = parser.parse_args(args)
options = parser.parse_args()

if options.outformat is not None:
    if options.outformat == "matrix":
        options.outFMT = 1
    elif options.outformat == "list":
        options.outFMT = 2
    elif options.outformat == "top":
        options.outFMT = 3

if options.converge:
    options.nRW = -1
else:
    if options.nRW < 0:
        temp = 'THE LENGTH OF RANDOM WALKS SPECIFIED IS NOT VALID!\n'
        print >> sys.stderr, temp

if not options.quiet:
    print '********************************************************'
    print 'Start parsing PPI file: ', options.infile
    if options.converge:
        print 'calculate the converged DSD'
    else:
        print 'the length of random walks used to calculate DSD is', options.nRW
    temp = 'the output format is chosen as No.' + str(options.outFMT)
    if options.outFMT == 1:
        print temp, ":\n        (DSD matrix)"
        if options.outfile is not None:
            options.outfile = options.outfile + '.DSD1'
    elif options.outFMT == 2:
        print temp, ":\n        (interacting pair list)"
        if options.outfile is not None:
            options.outfile = options.outfile + '.DSD2'
    elif options.outFMT == 3:
        print temp, ":\n        (top K nodes with lowest DSD)"
        if options.outfile is not None:
            options.outfile = options.outfile + '.DSD3'
    if options.threshold is not None:
        print 'the threshold for PPIs is specified as', options.threshold
    else:
        options.threshold = -1
    print '********************************************************'


### Parse input file
### get the adjacency matrix and names of files
#(ppbAdj, names) = PPIparser.GetAdj(options.infile,
#                                   options.threshold)

ppbAdj = np.loadtxt('graph')
M = int(sum(sum(ppbAdj))/2)
N = np.size(ppbAdj[0])

names = {}
for i in xrange(1, N+1):
    names[('Pro%04d' % i)] = i-1

names = collections.OrderedDict(sorted(names.items(),
                                           key=lambda x: x[1]))

if not options.quiet:
    print 'Done with parsing, there are', N, 'different nodes'
    print '    and', M, 'different PPIs originally'
### check graph connectivity
sStar = '********************************************************'
if not mygraph.CheckConnect(ppbAdj) and options.force:
    print >> sys.stderr, sStar
    temp = "!!!!!!! Warnning: the network is not connected, !!!!!!!!"
    print >> sys.stderr, temp
    temp = "! calculating all pairs of DSD might not be meaningful !"
    print >> sys.stderr, temp
    print >> sys.stderr, sStar
if not mygraph.CheckConnect(ppbAdj) and not options.force:
    print >> sys.stderr, sStar
    temp = "******* Warnning: the network is not connected, ********"
    print >> sys.stderr, temp
    temp = "****** calculate for the largest component instead *****"
    print >> sys.stderr, temp
    print >> sys.stderr, sStar
    (ppbAdj, names, nc) = mygraph.CalcLargestComponent(ppbAdj, names)
    M = int(sum(sum(ppbAdj))/2)
    N = np.size(ppbAdj[0])
    if not options.quiet:
        print "There are", nc, "components and the largest connected"
        print "component has", N, "different nodes and", M, "edges"

#print names
#print ppbAdj
if N < 3:
    temp = "Error: can't run DSD module on this PPI network, too small"
    print >> sys.stderr, temp
    exit(1)
if M < N*0.5:
    temp = "Error: can't run DSD module on this PPI network, too sparse"
    print >> sys.stderr, temp
    exit(1)
if not options.quiet:
    print 'Start calculating DSD...'

DSD = calcDSD.calculator(ppbAdj,50 , options.quiet)
import sklearn.cluster as sc
# compute Gaussian kernel     
sigma = 5 
data = np.exp(-DSD**2 / (2.*(sigma**2)))
#data = DSD
for i in range(1000):
    for j in range(1000):
            if data[i,j] == -1:
                data[i,j] = 12 
 
import pdb

# apply spectral clustering and reorder labels
spectral = sc.SpectralClustering(n_clusters = 2, affinity='precomputed')
spectral.fit(data)
labels = spectral.fit_predict(data) 

gtlabels = np.loadtxt('graph_labels')
pdb.set_trace()
