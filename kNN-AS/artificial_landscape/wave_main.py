import os
import sys
from potentials import wave_split_potential 
from potentials import wave_split_potential_func 
from KNN2 import run_trial

############################## Src Kinase example  ###############################


#Run trials
n_trials = int(sys.argv[2])
epochs = 400
traj_len = 500
num_spawn = 6
potential_func = wave_split_potential_func
initial_positions=[[-1.25, 0.5, 0], [1.25, -0.5, 0],
                    [-1.25, 0.5, 0], [1.25, -0.5, 0],
                    [-1.25, 0.5, 0], [1.25, -0.5, 0],
                    [-1.25, 0.5, 0], [1.25, -0.5, 0],
                    [-1.25, 0.5, 0], [1.25, -0.5, 0],
                    [-1.25, 0.5, 0], [1.25, -0.5, 0]]

k=5
threshold= -75
xlim = (-2.5,2.5,0.05)

output_dir = 'run_trials_wave5' #Change as desired
reset = int(sys.argv[1]) #Only used for repeated trials

run_trial(wave_split_potential,
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
