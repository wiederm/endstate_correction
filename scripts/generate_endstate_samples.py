# --------------------------------------------- #
# This script generates samples at the reference and/or target level of theory
# --------------------------------------------- #
# define equilibirum sampling control parameters
smiles = "Cn1cc(Cl)c(/C=N/O)n1" # SMILES of the molecule
run_id = 1
n_samples = 1_000 # how many samples to generate
n_steps_per_sample = 1_000 # how many timesteps between samples
system_name = "test1" # name of the system
base = f"{system_name}/equilibrium_samples/run{run_id:0>2d}" # path where samples should be stored (will be created if it doesn't exist)
# --------------------------------------------- #

from openmm import LangevinIntegrator
from endstate_correction.constant import (
    temperature,
)
import numpy as np
from openmm.app import (
    Simulation,
    DCDReporter,
    StateDataReporter,
    PDBFile,
)
import os
from sys import stdout
from openmmml import MLPotential
from openmm import Platform
from openff.toolkit.topology import Molecule
from openff.toolkit.typing.engines.smirnoff import ForceField
from endstate_correction.constant import collision_rate, stepsize
from openmmml import MLPotential
from openmm import unit

########################################################
########################################################
# ------------------- set up system -------------------#
forcefield = ForceField("openff_unconstrained-2.0.0.offxml")
env = "vacuum"

potential = MLPotential("ani2x")
molecule = Molecule.from_smiles(smiles, hydrogens_are_explicit=False)
molecule.generate_conformers(n_conformers=1)

topology = molecule.to_topology()
system = forcefield.create_openmm_system(topology)
# define region that should be treated with the qml
ml_atoms = [atom.molecule_particle_index for atom in topology.atoms]
integrator = LangevinIntegrator(temperature, collision_rate, stepsize)
platform = Platform.getPlatformByName("CUDA")
topology = topology.to_openmm()
ml_system = potential.createMixedSystem(topology, system, ml_atoms, interpolate=True)
sim = Simulation(topology, ml_system, integrator, platform=platform)
##############################################################
# ------------------ Start equilibrium sampling ---------------
with open(f"{base}/{system_name}.pdb", "w") as outfile:
    PDBFile.writeFile(
        topology, molecule.conformers[0].magnitude * unit.angstrom, outfile
    )
os.makedirs(base, exist_ok=True)

nr_lambda_states = 2  # samples equilibrium distribution at both endstates
lambs = np.linspace(0, 1, nr_lambda_states)
# perform sampling for each lambda state

for lamb in lambs:
    print(f"{lamb=}")
    # define where to store samples
    trajectory_file = f"{base}/{system_name}_samples_{n_samples}_steps_{n_steps_per_sample}_lamb_{lamb:.4f}_{env}.dcd"
    print(f"Trajectory saved to: {trajectory_file}")
    # set lambda
    sim.context.setParameter("lambda_interpolate", lamb)
    # set coordinates
    sim.context.setPositions(molecule.conformers[0].magnitude * unit.angstrom)
    # try to set velocities using openMM, fall back to manual velocity seeding if it fails
    sim.context.setVelocitiesToTemperature(temperature)

    # define DCDReporter
    sim.reporters.append(
        DCDReporter(
            trajectory_file,
            n_steps_per_sample,
        )
    )
    sim.reporters.append(
        StateDataReporter(
            stdout,
            n_steps_per_sample,
            step=True,
            potentialEnergy=True,
            temperature=True,
            speed=True,
            separator="\t",
        ),
    )
    # perform sampling
    sim.step(n_samples * n_steps_per_sample)
    sim.reporters.clear()
