# --------------------------------------------- #
# This script performs endstate corrections
# --------------------------------------------- #
system_name = "test1"
n_samples = 1_000
n_steps_per_sample = 1_000
run_id = 1
traj_base = f"{system_name}/equilibrium_samples/run{run_id:0>2d}" # define directory containing MM and NNP sampling data
output_base = f"{system_name}/switching"
# --------------------------------------------- #

from openmm.app import (
    Simulation,
)
from openmm import Platform
from endstate_correction.analysis import plot_endstate_correction_results
from endstate_correction.protocol import perform_endstate_correction, FEPProtocol, NEQProtocol, SMCProtocol, AllProtocol
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
# define region that should be treated with the nnp
ml_atoms = [atom.molecule_particle_index for atom in topology.atoms]
print(f"{ml_atoms=}")
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
# load NNP samples
nnp_samples = []
base = f"{traj_base}/{system_name}_samples_{n_samples}_steps_{n_steps_per_sample}_lamb_1.0000_{env}"
nnp_samples = mdtraj.load_dcd(
    f"{base}.dcd",
    top=f"{traj_base}/{system_name}.pdb",
)[
    int((1_000 / 100) * 20) :
]  # discart first 20% of the trajectory
print(f"Initializing switch from {len(nnp_samples)} NNP samples")

# define protocols
# --------------------------------------------- #
# ---------------- FEP protocol ---------------
# --------------------------------------------- #
# bidirectional
fep_protocol = FEPProtocol(
    sim=sim,
    reference_samples=mm_samples,
    target_samples=nnp_samples,
    nr_of_switches=1_000,
)
# --------------------------------------------- #
# ----------------- NEQ protocol --------------
# --------------------------------------------- #
# bidirectional
neq_protocol = NEQProtocol(
    sim=sim,
    reference_samples=mm_samples,
    target_samples=nnp_samples,
    nr_of_switches=100,
    switching_length=1_000,
    save_endstates=True,
    save_trajs=True,
)
# --------------------------------------------- #
# ----------------- SMC protocol --------------
# --------------------------------------------- #
# unidirectional (from reference to target)
smc_protocol = SMCProtocol(
    sim=sim,
    reference_samples=mm_samples,
    nr_of_walkers=100,
    nr_of_resampling_steps=1_000,
)

# combine all protocols to one and perform correction
all_protocol = AllProtocol(
    fep_protocol=fep_protocol, 
    neq_protocol=neq_protocol, 
    smc_protocol=smc_protocol
)

r = perform_endstate_correction(all_protocol)

# save results in a pickle file
pickle.dump((r), open(f"{output_base}/results.pickle", "wb"))

# plot fep and neq results
plot_endstate_correction_results(system_name, r, f"{output_base}/results.png")

# print SMC result
print(r.smc_results.logZ)