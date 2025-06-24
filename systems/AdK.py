from sklearn.neighbors import NearestNeighbors
import numpy as np
from numpy.linalg import norm
import openmm as mm
import openmm.app as app
import numpy as np
from simtk.unit import *
import mdtraj as md
from sys import stdout
import random

class ADK():
    def __init__(self, i, j, initial_positions, top_file, n_steps):
        self.initial_positions = initial_positions
        self.i = i
        self.j = j
        self.top_file = top_file
        self.n_steps = n_steps

    def AdK_system(self):
        pdb = app.PDBFile(self.top_file)
        forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
        system = forcefield.createSystem(pdb.topology,
                                     nonbondedMethod=app.PME,
                                     nonbondedCutoff=1 * nanometer,
                                     constraints=app.HBonds)
        integrator = mm.LangevinIntegrator(300 * kelvin, 1 / picosecond, 2.0 * femtosecond)
        system.addForce(mm.MonteCarloBarostat(1 * bar, 300 * kelvin))

        return system, pdb.topology, integrator, pdb.positions

    def project_angles(self, traj_name):
        traj = md.load(traj_name, top=self.top_file)

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

    def run_trajectory(self, system, topology, integrator, positions, output_file, platform='CUDA'):
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
        print("minimize energy\n", flush=True)
        simulation.minimizeEnergy()
        simulation.context.setVelocitiesToTemperature(300)
        simulation.reporters.append(app.DCDReporter(output_file, 5000))
        simulation.reporters.append(app.StateDataReporter(stdout, 5000, step=True, potentialEnergy=True))
        print("start simulation\n", flush=True)
        simulation.step(self.n_steps)

    def frame_to_openmm(self, xyz_positions):
        '''
        Converts a numpy array containing (n_atoms, 3) positions (single frame) into an openmm Quantity[Vec3] object for a simulation.
        '''
        return Quantity(value=[mm.vec3.Vec3(*atom_pos) for atom_pos in xyz_positions],unit=nanometer)

    def run(self):
        output_file = "traj_"+str(self.i)+"_"+str(self.j)+".dcd"
        k=self.j
        while True:
            try:
                system, topology, integrator, _ = self.AdK_system()
                sys.stdout.write("Starting simulation for {output_file}\n")
                self.run_trajectory(system, topology, integrator, self.frame_to_openmm(self.initial_positions[k]), output_file)
            except:
                sys.stdout.write("Restarting simulation for {output_file}\n")
                k = random.randint(0, len(self.initial_positions))
                continue
            else:
                break
        traj = md.load(output_file, top=self.top_file)
        protein_atoms = traj.topology.select('protein')
        protein_traj = traj.atom_slice(protein_atoms)
        protein_traj.save_dcd("traj_"+str(self.i)+"_"+str(self.j)+"_protein.dcd")
        traj = md.load(output_file, top=self.top_file)
        reduced_traj = traj[::10]
        reduced_traj.save_dcd(output_file)
        angles = self.project_angles(output_file)
        return angles, output_file

