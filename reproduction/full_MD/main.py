from sklearn.neighbors import NearestNeighbors
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import matplotlib as mplt
import openmm as mm
import openmm.app as app
import numpy as np
from simtk.unit import *
import mdtraj as md
import warnings
import sys
import time
import glob, os
from sys import stdout
import scipy
import itertools

def AdK_system(top_file):
    pdb = app.PDBFile(top_file)
    forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
    system = forcefield.createSystem(pdb.topology,
                                     nonbondedMethod=app.PME,
                                     nonbondedCutoff=1 * nanometer,
                                     constraints=app.HBonds)
    integrator = mm.LangevinIntegrator(300 * kelvin, 1 / picosecond, 2.0 * femtosecond)
    system.addForce(mm.MonteCarloBarostat(1 * bar, 300 * kelvin))

    return system, pdb.topology, integrator, pdb.positions

def project_angles(traj_name, top_file):
    traj = md.load(traj_name, top=top_file)

    #   NMP-CORE angle
    id_C = traj.topology.select('resid 115 to 125 and (backbone or name CB)')
    C = np.mean(traj.xyz[:, id_C, :], axis=1)
    id_B = traj.topology.select('resid 90 to 99 and (backbone or name CB)')
    B = np.mean(traj.xyz[:, id_B, :], axis=1)
    id_A = traj.topology.select('resid 35 to 55 and (backbone or name CB)')
    A = np.mean(traj.xyz[:, id_A, :], axis=1)
    BA = A - B
    BC = C - B
    NMP_angle = np.array([np.rad2deg(np.arccos(np.dot(BA[i], BC[i])/(norm(BA[i])*norm(BC[i])))) for i in range(len(BA))]).T

    # LID-CORE angle
    id_C = traj.topology.select('resid 179 to 185 and (backbone or name CB)')
    C = np.mean(traj.xyz[:, id_C, :], axis=1)
    id_B = traj.topology.select('resid 115 to 125 and (backbone or name CB)')
    B = np.mean(traj.xyz[:, id_B, :], axis=1)
    id_A = traj.topology.select('resid 125 to 153 and (backbone or name CB)')
    A = np.mean(traj.xyz[:, id_A, :], axis=1)
    BA = A - B
    BC = C - B
    LID_angle = np.array([np.rad2deg(np.arccos(np.dot(BA[i], BC[i])/(norm(BA[i])*norm(BC[i])))) for i in range(len(BA))]).T

    NMP = NMP_angle.reshape(len(traj),-1)
    LID = LID_angle.reshape(len(traj),-1)

    data = np.hstack((NMP, LID))
    
    assert (NMP.shape[0] == data.shape[0])
    assert (data.shape[1] == 2)

    return data

def frame_to_openmm(xyz_positions):
    '''
    Converts a numpy array containing (n_atoms, 3) positions (single frame) into an openmm Quantity[Vec3] object for a simulation.
    '''
    return Quantity(value=[mm.vec3.Vec3(*atom_pos) for atom_pos in xyz_positions],
                    unit=nanometer)

def run_trajectory(system, topology, integrator, positions, output_file, n_steps=1000, platform='CUDA'):
    '''
    Run a trajectory of the specified system starting from the specified position.

    Args
    -------------
    system (simtk.openmm.System): system object.
    topology (simtk.openmm.app.Topology): alanine dipeptide topology.
    positions (simtk.unit.Quantity): initial atom positions.
    output_file (str): name of .dcd trajectory that will be saved.
    n_steps (int): number of steps to run.
    platform (str): name of platform where OpenMM will run the simulation.

    Returns
    -------------
    None
    '''
    mm.Platform.getPlatformByName(platform)
    simulation = app.Simulation(topology, system, integrator)
    simulation.context.setPositions(positions)
    print("minimize energy", flush=True)
    simulation.minimizeEnergy()
    simulation.context.setVelocitiesToTemperature(300)
    simulation.reporters.append(app.DCDReporter(output_file, 3500))
    simulation.reporters.append(app.StateDataReporter(stdout, 3500, step=True, potentialEnergy=True))
    print("start simulation", flush=True)
    simulation.step(n_steps)

def main():
    n=2
    n_steps = 60000000 
    cmap = plt.get_cmap(name='hsv')
    
    main_dir = os.getcwd()
    example_dir = main_dir+'/example_data/'
    
    top_file = example_dir+'4AKE_water.pdb'
    initial_structures = [example_dir+'npt_1AKE.pdb',example_dir+'npt_4AKE.pdb']

    directory = "full_MD3"
    path = os.path.join(main_dir, directory)
    os.makedirs(path)
    os.chdir(path)
        
    angle_total = []
    positions = []
    initial_positions = []
    for fname in initial_structures:
        initial_positions.append(md.load(fname).xyz[0])
    print(initial_positions)        

    fig, ax = plt.subplots(1,3, figsize=(15,5))            
    for j in range(n): # number of trajectories per epoch
        output_file = "traj_"+str(j)+".dcd"
        system, topology, integrator, _ = AdK_system(top_file)
        run_trajectory(system, topology, integrator, frame_to_openmm(initial_positions[j]), output_file, n_steps=n_steps)
        x = project_angles(output_file, top_file)
        angle_total.append(x)
        
        ax[j].scatter(x[:,0], x[:,1], alpha=0.2)
        ax[j].set_ylim(90, 170)
        ax[j].set_xlim(35, 80)
        
        plt.savefig("Full_MD_simulation.png")
    
    angle_total = np.array(angle_total).reshape(-1,2)
    ax[2].scatter(angle_total[:,0], angle_total[:,1], alpha=0.2)
    ax[2].set_ylim(90, 170)
    ax[2].set_xlim(35, 80)
    plt.savefig("Full_MD_simulation.png")

    a,b,c = np.array(angle_total).shape
    X_tmp = np.array(angle_total).reshape(a*b,c)
    np.save('angels.npy', X_tmp)
    os.chdir(main_dir) 

if __name__ == "__main__":
    main()
