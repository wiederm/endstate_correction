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
from endstate_correction.protocol import perform_endstate_correction, FEPProtocol, NEQProtocol
import mdtraj
from openmm import unit
import pickle, sys, os
from openmmml import MLPotential

########################################################
########################################################
# ------------------- set up system -------------------#
package_path = endstate_correction.__path__[0]
system_idx = int(sys.argv[1])
system_name = zinc_systems[system_idx][0]

if system_name in blacklist:
    print("System in blacklist. Aborting.")
    sys.exit()

env = "vacuum"

print(f"Setting up system {system_name} in {env}")

# define directory containing parameters
parameter_base = f"{package_path}/data/hipen_data"
# load the charmm specific files (psf, rtf, prm and str files)
psf_file = f"{parameter_base}/{system_name}/{system_name}.psf"
crd_file = f"{parameter_base}/{system_name}/{system_name}.crd"
psf = CharmmPsfFile(psf_file)
crd = CharmmCrdFile(crd_file)
params = CharmmParameterSet(
    f"{parameter_base}/top_all36_cgenff.rtf",
    f"{parameter_base}/par_all36_cgenff.prm",
    f"{parameter_base}/{system_name}/{system_name}.str",
)
# write pdb file
with open("temp.pdb", "w") as outfile:
    PDBFile.writeFile(psf.topology, crd.positions, outfile)

# define region that should be treated with the nnp
chains = list(psf.topology.chains())
ml_atoms = [atom.index for atom in chains[0].atoms()]
print(f"{ml_atoms=}")
print("Creating mm system...")
mm_system = psf.createSystem(params=params)
# define system
potential = MLPotential("ani2x")
print("Creating mixed system...")
ml_system = potential.createMixedSystem(
    psf.topology, mm_system, ml_atoms, interpolate=True
)
sim = Simulation(psf.topology, ml_system, LangevinIntegrator(300, 1, 0.001))
print("Creating simulation object...")
########################################################
########################################################
# ------------------- load samples --------------------#
n_samples = 5_000
n_steps_per_sample = 1_000

# define directory containing MM and NNP sampling data
traj_base = f"/data/shared/projects/endstate_rew/{system_name}/sampling_charmmff/"

# load MM samples
mm_samples_list = []
for i in range(1, 4):
    base = f"{traj_base}/run0{i}/{system_name}_samples_{n_samples}_steps_{n_steps_per_sample}_lamb_0.0000"
    # if needed, convert pickle file to dcd
    # convert_pickle_to_dcd_file(f"{base}.pickle",psf_file, crd_file, f"{base}.dcd", "temp.pdb")
    traj = mdtraj.load_dcd(
        f"{base}.dcd",
        top=psf_file, # also possible to use the tmp.pdb
    )[int((n_samples / 100) * 20):]
    mm_samples_list.append(traj)
    
mm_samples = mdtraj.join(mm_samples_list) #* unit.nanometer
assert isinstance(mm_samples, mdtraj.Trajectory)
print(f"Initializing switch from {len(mm_samples)} MM samples")

# load NNP samples
nnp_samples_list = []
for i in range(1, 4):
    base = f"{traj_base}/run0{i}/{system_name}_samples_{n_samples}_steps_{n_steps_per_sample}_lamb_1.0000"
    # if needed, convert pickle file to dcd
    # convert_pickle_to_dcd_file(f"{base}.pickle",psf_file, crd_file, f"{base}.dcd", "temp.pdb")
    traj = mdtraj.load_dcd(
        f"{base}.dcd",
        top=psf_file, # also possible to use the tmp.pdb
    )[int((n_samples / 100) * 20):]
    nnp_samples_list.append(traj)
    
nnp_samples = mdtraj.join(nnp_samples_list) #* unit.nanometer
assert isinstance(nnp_samples, mdtraj.Trajectory)
print(f"Initializing switch from {len(nnp_samples)} NNP samples")

########################################################
########################################################
# ----------------- perform correction ----------------#

# define the output directory
output_base = f"/data/shared/projects/endstate_rew/{system_name}/FEP_v1/"
os.makedirs(output_base, exist_ok=True)

####################################################
# ---------------- FEP protocol --------------------
####################################################
fep_protocol = FEPProtocol(
    sim=sim,
    reference_samples=mm_samples,
    target_samples=nnp_samples,
    nr_of_switches=2_000,  # if not provided, the protocol will use all provided equilibrium samples
)

####################################################
# ----------------- NEQ protocol -------------------
####################################################
# neq_protocol = Protocol(
    # sim=sim,
    # reference_samples=mm_samples,
    # target_samples=nnp_samples,
    # nr_of_switches=3,  # 500,
    # switching_length=5,  # _000,
    # save_endstates=False,
    # save_trajs=False,
# )

# perform correction
r_fep = perform_endstate_correction(fep_protocol)
#r_neq = perform_endstate_correction(neq_protocol)

# save fep and neq results in a pickle file
#pickle.dump((r_fep, r_neq), open(f"{output_base}/results.pickle", "wb"))
pickle.dump((r_fep), open(f"{output_base}/fep_results.pickle", "wb"))

# plot results
plot_endstate_correction_results(system_name, r_fep, f"{output_base}/results_fep.png")
#plot_endstate_correction_results(system_name, r_neq, f"{output_base}/results_neq.png")
