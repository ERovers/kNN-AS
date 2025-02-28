import os
import sys
from potentials import potential_with_smooth_minima
from potentials import potential_with_smooth_minima_func
from KNN2 import run_trial

############################## Src Kinase example  ###############################


#Run trials
n_trials = int(sys.argv[2])
epochs = 400
traj_len = 500
num_spawn = 6
potential_func = potential_with_smooth_minima_func
initial_positions=[[1.6, 0.4 , 0], [-1.2, -1.2, 0],
                    [1.6, 0.4 , 0], [-1.2, -1.2, 0],
                    [1.6, 0.4 , 0], [-1.2, -1.2, 0]]
k=5
threshold = -40
xlim = (-3, 3, 0.1)

output_dir = 'run_trials_smooth_minima_epoch400' #Change as desired
reset = int(sys.argv[1]) #Only used for repeated trials

run_trial(potential_with_smooth_minima,
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

