#-----------------------------------------------
# This file was created by LDT on 4 Feb 2022
#-----------------------------------------------
# This file contains the variables that are most often changed to test the fitting code 
# with different settings. 
# If deemed redundant (i.e. if code is always run with same settings) this script can be incorporated in run_parallel.py
# or in perform_fitting.py
#-----------------------------------------------

import multiprocessing

measure_shift_EDonly = False # do you want to measure ED shift only at ED?

# Set the following weights:
# RB set of weights: [10, 1e6, 500]
# LL set of weights: [200, 500, 500] or [40,1e5,0.01]

# the following lines contain the set pof weights used by the
# MultiThreadSmoothingED() and SolveProblemCVXOPT() functions
weight_GP = 100 #100 #200
low_smoothing_weight = 1e6 #1e3 #1e4
transmural_weight =  0.01 #1e3 #0.01
 
#set the sampling to be used by the GPDataset() module
sampling = 1 # sampling = 1 means all guide points are used

workers = 1 # = num of processes to be opened in parallel = number of CPUS that one wants to use

if workers> multiprocessing.cpu_count():
    print('The number of workers exceeds the number of available CPUs')
    raise ValueError


