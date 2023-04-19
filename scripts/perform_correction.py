# --------------------------------------------- #
# This script performs endstate corrections
# --------------------------------------------- #
system_name = "test1"
n_samples = 1_000
n_steps_per_sample = 1_000
run_id = 1
traj_base = f"{system_name}/equilibrium_samples/run{run_id:0>2d}" # define directory containing MM and QML sampling data
output_base = f"{system_name}/switching"
# --------------------------------------------- #

from openmm.app import (
    Simulation,
)
from openmm import Platform
from endstate_correction.analysis import plot_endstate_correction_results
from endstate_correction.protocol import perform_endstate_correction, Protocol
import mdtraj
from openmmml import MLPotential
import pickle, sys, os
from openff.toolkit.typing.engines.smirnoff import ForceField
from openff.toolkit.topology import Molecule
from openmm import LangevinIntegrator
from endstate_correction.constant import collision_rate, stepsize, temperature

########################################################
# ------------ set up the system ----------------------
smiles = "Cn1cc(Cl)c(/C=N/O)n1"
forcefield = ForceField("openff_unconstrained-2.0.0.offxml")
env = "vacuum"

potential = MLPotential("ani2x")
molecule = Molecule.from_smiles(smiles, hydrogens_are_explicit=False)
molecule.generate_conformers(n_conformers=1)

topology = molecule.to_topology()
system = forcefield.create_openmm_system(topology)
# define region that should be treated with the qml
ml_atoms = [atom.molecule_particle_index for atom in topology.atoms]
print(ml_atoms)
integrator = LangevinIntegrator(temperature, collision_rate, stepsize)
platform = Platform.getPlatformByName("CUDA")
topology = topology.to_openmm()
ml_system = potential.createMixedSystem(topology, system, ml_atoms, interpolate=True)
sim = Simulation(topology, ml_system, integrator, platform=platform)
########################################################
########################################################
# ------------------- load samples ---------------------#
os.makedirs(f"{output_base}", exist_ok=True)
# --------------------------------------------- #
# load MM samples
mm_samples = []
base = f"{traj_base}/{system_name}_samples_{n_samples}_steps_{n_steps_per_sample}_lamb_0.0000_{env}"
mm_samples = mdtraj.load_dcd(
    f"{base}.dcd",
    top=f"{traj_base}/{system_name}.pdb",
)[
    int((1_000 / 100) * 20) :
]  # discart first 20% of the trajectory
print(f"Initializing switch from {len(mm_samples)} MM samples")
# --------------------------------------------- #
# load QML samples
qml_samples = []
base = f"{traj_base}/{system_name}_samples_{n_samples}_steps_{n_steps_per_sample}_lamb_0.0000_{env}"
qml_samples = mdtraj.load_dcd(
    f"{base}.dcd",
    top=f"{traj_base}/{system_name}.pdb",
)[
    int((1_000 / 100) * 20) :
]  # discart first 20% of the trajectory
print(f"Initializing switch from {len(qml_samples)} QML samples")
# --------------------------------------------- #
# ---------------- FEP protocol ---------------
# --------------------------------------------- #
fep_protocol = Protocol(
    method="FEP",
    sim=sim,
    reference_samples=mm_samples,
    target_samples=qml_samples,
    nr_of_switches=1_000,
)
# --------------------------------------------- #
# ----------------- NEQ protocol --------------
# --------------------------------------------- #
neq_protocol = Protocol(
    method="NEQ",
    sim=sim,
    reference_samples=mm_samples,
    #target_samples=qml_samples,
    nr_of_switches=100,
    neq_switching_length=1_000,
    save_endstates=True,
    save_trajs=True,
)

# perform correction
r_fep = perform_endstate_correction(fep_protocol)
r_neq = perform_endstate_correction(neq_protocol)

# save fep and neq results in a pickle file
pickle.dump((r_fep, r_neq), open(f"{output_base}/results.pickle", "wb"))

# plot results
plot_endstate_correction_results(system_name, r_fep, f"{output_base}/results_neq.png")
plot_endstate_correction_results(system_name, r_neq, f"{output_base}/results_neq.png")