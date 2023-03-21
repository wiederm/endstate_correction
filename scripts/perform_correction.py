# general imports
from openmm.app import (
    PME,
    CharmmParameterSet,
    CharmmPsfFile,
    CharmmCrdFile,
    PDBFile,
)
from endstate_correction.constant import zinc_systems, blacklist
from endstate_correction.analysis import plot_endstate_correction_results
import endstate_correction
from endstate_correction.protocol import perform_endstate_correction, Protocol
import mdtraj
from openmm import unit
import pickle, sys, os
from endstate_correction.utils import convert_pickle_to_dcd_file
########################################################
########################################################
# ------------ set up the waterbox system --------------
# we use a system that is shipped with the repo
package_path = endstate_correction.__path__[0]
system_idx = int(sys.argv[1])

system_name = zinc_systems[system_idx][0]
if system_name in blacklist:
    print('System in blacklist. Aborting.')
    sys.exit()
env = "vacuum"
print(system_name)
print(env)
# define the output directory
parameter_base = f"{package_path}/data/hipen_data"
# load the charmm specific files (psf, pdb, rtf, prm and str files)
psf_file = f"{parameter_base}/{system_name}/{system_name}.psf"
crd_file = f"{parameter_base}/{system_name}/{system_name}.crd"
psf = CharmmPsfFile(psf_file)
crd = CharmmCrdFile(crd_file)
params = CharmmParameterSet(
    f"{parameter_base}/top_all36_cgenff.rtf",
    f"{parameter_base}/par_all36_cgenff.prm",
    f"{parameter_base}/{system_name}/{system_name}.str",
)

# define region that should be treated with the qml
chains = list(psf.topology.chains())
ml_atoms = [atom.index for atom in chains[0].atoms()]
# define system

sim = create_charmm_system(psf=psf, parameters=params, env=env, ml_atoms=ml_atoms)

########################################################
########################################################
# ------------------- load samples ---------------------#
n_samples = 5_000
n_steps_per_sample = 1_000
traj_base = output_base = f"/data/shared/projects/endstate_rew/{system_name}/"
os.makedirs(f'{traj_base}/switching', exist_ok=True)
mm_samples = []
for i in range(1,4):
    base = f"{traj_base}/sampling_charmmff/run0{i}/{system_name}_samples_{n_samples}_steps_{n_steps_per_sample}_lamb_0.0000" 
    convert_pickle_to_dcd_file(f"{base}.pickle",psf_file, crd_file, f"{base}.dcd", "test.pdb" )
    traj = mdtraj.load_dcd(
            f"{base}.dcd",
            top=psf_file,
    )

    mm_samples.extend(traj[1000:].xyz * unit.nanometer)  # NOTE: this is in nanometer!

print(f'Initializing switch from {len(mm_samples)}')
###############################
qml_samples = []
for i in range(1,4):
    base = f"{traj_base}/sampling_charmmff/run0{i}/{system_name}_samples_{n_samples}_steps_{n_steps_per_sample}_lamb_1.0000"
    convert_pickle_to_dcd_file(f"{base}.pickle",psf_file, crd_file, f"{base}.dcd", "test.pdb" )
    traj = mdtraj.load_dcd(
            f"{base}.dcd",
            top=psf_file,
    )

    qml_samples.extend(traj[1000:].xyz * unit.nanometer)  # NOTE: this is in nanometer!
print(f'Initializing switch from {len(mm_samples)}')
####################################################
# ----------------------- Correction ----------------------
####################################################

switching_length = 50_000
print(f'{switching_length=}')
neq_protocol = Protocol(
    method="NEQ",
    direction="bidirectional",
    sim=sim,
    trajectories=[mm_samples, qml_samples],
    nr_of_switches=500,
    neq_switching_length=switching_length,
)

fep_protocol = Protocol(
    method="FEP",
    direction="bidirectional",
    sim=sim,
    trajectories=[mm_samples, qml_samples],
    nr_of_switches=2_000,
)


r_fep = perform_endstate_correction(fep_protocol)
r_neq = perform_endstate_correction(neq_protocol, save_trajs=False)
pickle.dump((r_neq, r_fep), open(f"{output_base}/switching/results_{switching_length}.pickle", 'wb'))
plot_endstate_correction_results(system_name, r_neq, f"{output_base}/switching/results_bidirectional.png")


