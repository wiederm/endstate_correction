# general imports
from openmm import LangevinIntegrator
from openmm.app import (
    PME,
    CharmmParameterSet,
    CharmmPsfFile,
    CharmmCrdFile,
    PDBFile,
    Simulation,
)
from endstate_correction.constant import zinc_systems, blacklist
from endstate_correction.analysis import plot_endstate_correction_results
import endstate_correction
from endstate_correction.protocol import perform_endstate_correction, Protocol
import mdtraj
import pickle, sys, os
from openmmml import MLPotential
from openmmtools.utils import get_fastest_platform

# ------------------- set up system -------------------#

# load the charmm specific files (psf, rtf, prm and str files)
psf_file = "test.psf"
crd_file = "test.crd"
psf = CharmmPsfFile(psf_file)
crd = CharmmCrdFile(crd_file)
params = CharmmParameterSet(
    f"top_all36_cgenff.rtf",
    f"par_all36_cgenff.prm",
    f"parameter.str",
)
# write pdb file
with open("temp.pdb", "w") as outfile:
    PDBFile.writeFile(psf.topology, crd.positions, outfile)

# define region that should be treated with the qml
chains = list(psf.topology.chains())
ml_atoms = [atom.index for atom in chains[0].atoms()]
print(f"{ml_atoms=}")
# generate MM system
mm_system = psf.createSystem(params=params)
# generate MM-QML mixed system
potential = MLPotential("ani2x")
ml_system = potential.createMixedSystem(
    psf.topology, mm_system, ml_atoms, interpolate=True
)
#set up simulation object
sim = Simulation(
    psf.topology,
    ml_system,
    LangevinIntegrator(300, 1, 0.001),
    platform=get_fastest_platform(minimum_precision="mixed"),
)

# ------------------- load samples --------------------#

# define directory containing MM and QML sampling data
# load MM samples
trajs = []
for i in range(1, 4): # load multiple trajectories
    traj = mdtraj.load_dcd(
        f"traj_{i}.dcd",
        top=psf_file,
    )
    traj = traj[1000:]  # remove equilibration
    trajs.append(traj)
mm_samples = mdtraj.join(trajs) # join multiple trajectories
print(f"Initializing switch from {len(mm_samples)} MM samples")

# load QML samples
qml_samples = []
for i in range(1, 4):
    traj = mdtraj.load_dcd(
        f"traj_{i}.dcd",
        top=psf_file,
    )
    traj = traj[1000:]  # remove equilibration
    trajs.append(traj)
qml_samples = mdtraj.join(trajs) # join multiple trajectories
print(f"Initializing switch from {len(qml_samples)} QML samples")

# ----------------- perform correction ----------------#

# define the output directory
output_base = f"switching/"
os.makedirs(output_base, exist_ok=True)

# ----------------------- FEP -------------------------#

fep_protocol = Protocol(
    method="FEP",
    sim=sim,
    reference_samples=mm_samples,
    target_samples=qml_samples,
    nr_of_switches=2_000,
)

# ----------------------- NEQ -------------------------#

neq_protocol = Protocol(
    method="NEQ",
    sim=sim,
    reference_samples=mm_samples,
    target_samples=qml_samples,
    nr_of_switches=500,
    neq_switching_length=5_000,
    save_endstates=True,
    save_trajs=True,
)

# perform correction
r_fep = perform_endstate_correction(fep_protocol)
r_neq = perform_endstate_correction(neq_protocol)

# save fep and neq results in a pickle file
pickle.dump((r_fep, r_neq), open(f"results.pickle", "wb"))

# plot results
plot_endstate_correction_results('mol', r_fep, f"{output_base}/results_neq.png")
plot_endstate_correction_results('mol', r_neq, f"{output_base}/results_neq.png")
