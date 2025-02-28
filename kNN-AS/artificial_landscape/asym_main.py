import os
import sys
from potentials import four_wells_asymmetric
from potentials import four_wells_asymmetric_func
from KNN2 import run_trial

############################## Src Kinase example  ###############################


#Run trials
n_trials = int(sys.argv[2])
epochs = 400
traj_len = 500
num_spawn = 6
potential_func = four_wells_asymmetric_func
initial_positions=[[0.2, 1 , 0], [1.8, 1, 0],
                    [0.2, 1 , 0], [1.8, 1, 0],
                    [0.2, 1 , 0], [1.8, 1, 0],
                    [0.2, 1 , 0], [1.8, 1, 0],
                    [0.2, 1 , 0], [1.8, 1, 0],
                    [0.2, 1 , 0], [1.8, 1, 0]]

k=5
threshold = -58
xlim = (-0.5, 2.5, 0.05)

output_dir = 'run_trials_asymmetric5' #Change as desired
reset = int(sys.argv[1]) #Only used for repeated trials

run_trial(four_wells_asymmetric,
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
