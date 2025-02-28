import os
import sys
from potentials import four_wells_symmetric
from potentials import four_wells_symmetric_func
from KNN2 import run_trial

#Run trials
n_trials = int(sys.argv[2])
epochs = 200
traj_len = 500
num_spawn = 6
potential_func = four_wells_symmetric_func
initial_positions=[[0.8, 1, 0], [1.2, 1, 0],
                    [0.8, 1, 0], [1.2, 1, 0],
                    [0.8, 1, 0], [1.2, 1, 0]]
k=5
threshold = -55
xlim = (-0.5, 2.5, 0.05)

output_dir = 'run_trials5' #Change as desired
reset = int(sys.argv[1]) #Only used for repeated trials

run_trial(four_wells_symmetric,
            potential_func,
            initial_positions, 
            k,
            epochs,
            num_spawn,
            traj_len,
            n_trials,
            threshold,
            xlim,
            reset=reset,
            output_dir=output_dir)
