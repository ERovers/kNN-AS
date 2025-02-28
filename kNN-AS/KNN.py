import os
import numpy as np
import openmm as mm
from utils import area_explored
from potentials import four_wells_symmetric_func
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

def setup_simulation(potential='', platform='CPU'):
    '''
    Setup Langevin dynamics simulation with custom potential.
    '''
    system = mm.System()
    system.addParticle(100)  # Add particle with mass of 1 amu
    force = mm.CustomExternalForce(potential)  # Defines the potential
    force.addParticle(0, [])
    system.addForce(force)
    integrator = mm.LangevinIntegrator(300, 1,
                                       0.002)  # Langevin integrator with 300 temperature, gamma=1, step size = 0.002
    platform = mm.Platform.getPlatformByName(platform)

    return system, integrator, platform


def run_trajectory(n_steps=0, potential='', initial_position=[0, 0, 0]):
    '''
    Run a simulation of a single particle under Langevin dynamics for n_steps.
    '''
    system, integrator, platform = setup_simulation(potential=potential)
    context = mm.Context(system, integrator, platform)
    context.setPositions([initial_position])
    context.setVelocitiesToTemperature(300)
    x = np.zeros((n_steps, 3))
    for i in range(n_steps):
        x[i] = context.getState(getPositions=True).getPositions(asNumpy=True)._value
        integrator.step(1)
    return x[::25]

def KNN_sampling(Z, k, n):
    knn = NearestNeighbors(n_neighbors=k).fit(Z)
    distances, indices = knn.kneighbors(Z)
    scores = []

    size = Z.shape[-1]
    for i in range(indices.shape[0]):
        c = np.zeros((1,size))
        
        for ind in indices[i,1:]:
            c += Z[ind,:] - Z[i,:]

        scores.append(np.sqrt(np.sum(c**2)))
    return np.argsort(scores)[-n:]

def run_trial(potential, potential_func, initial_positions, k, epochs, num_spawn, traj_len, num_trials, threshold, xlim, reset=0, output_dir=''):
    '''
    Runs a trial of MA REAP with standard Euclidean distance rewards.
    
     Args
    -------------
    potential (str): potential on which to run the trial (currently only two-dimensional potentials are used).
    potential_func (callable): necessary to compute explored area.
    initial_positions (list[np.ndarray]): starting points for simulations. Lenght of list must match number of agents.
    k (int): number of neighbours to consider
    epochs (int): specifies for how many epochs to run a trial.
    num_spawn (int): number of total trajectories to spawn per epoch.
    traj_len (int): length of each trajectory ran.
    num_trials (int): number of trials.
    output_dir (str): folder where to store the results (it will be created in current working directory if it does not exist).
    output_prefix (str): common prefix for all log files.
    
    Returns
    -------------
    None. Results are saved to output_dir.
    '''

    main_dir = os.getcwd()
    path = os.path.join(main_dir, output_dir)
    if not os.path.isdir(path):
        os.mkdir(path)
    os.chdir(path)
    main_dir = os.getcwd()

    xlim = xlim
    ylim = xlim    

    start_positions = initial_positions.copy()

    for i in range(reset,num_trials):
        directory = "trial_{}".format(i)
        print("trial {}".format(i))
        path = os.path.join(main_dir, directory)
        if not os.path.isdir(path):
            os.mkdir(path)
        os.chdir(path)

        X_total = []
        area_expl = []
        initial_positions = start_positions
        for i in range(epochs+1): # number of epochs

            for j in range(num_spawn): # number of trajectories per epoch
                x = run_trajectory(n_steps=traj_len, potential=potential, initial_position=initial_positions[j])
                X_total.append(x)
            
            a,b,c = np.array(X_total).shape
            X_tmp = np.array(X_total).reshape(a*b,c)
            idx = np.random.choice(len(X_tmp), int(0.1*len(X_tmp)), replace=False)
            X_tmp2 = X_tmp[idx]
            selection = KNN_sampling(X_tmp2, k, num_spawn)
            selection = idx[selection]
            initial_positions = [X_tmp[sel] for sel in selection]
            area_expl.append(area_explored(potential_func, xlim, ylim, X_tmp[:,:2], threshold))
            
            if ((i%10)==0):
                np.save('coordinates_epoch{}.npy'.format(i), X_total)
                x_plot = np.arange(*xlim)
                y_plot = np.arange(*ylim)
                X_plot, Y_plot = np.meshgrid(x_plot, y_plot)
                Z_plot = potential_func(X_plot, Y_plot)
                #plt.pcolormesh(X_plot, Y_plot, Z_plot, cmap=plt.cm.jet, shading='auto')
                im = plt.imshow(np.flip(Z_plot,0), cmap=plt.cm.jet, extent=[xlim[0], xlim[1], ylim[0], ylim[1]])
                plt.colorbar(im)
                plt.scatter(X_tmp[:,0], X_tmp[:,1])
                plt.xlim([xlim[0], xlim[1]])
                plt.ylim([ylim[0], ylim[1]])
                plt.savefig('landscape_cross_epoch{}.png'.format(i))
                plt.clf()
                print(area_expl)
        np.save('area_explored_log.npy', area_expl)
        os.chdir(main_dir)
