import pickle
import mdtraj as md
from openmm import unit
from openmm.app import CharmmPsfFile, CharmmCrdFile, PDBFile


def convert_pickle_to_dcd_file(
    pickle_file_path: str,
    path_to_psf: str,
    path_to_crd: str,
    dcd_output_path: str,
    pdb_output_path: str,
):
    """Convert pickle file trajectory to dcd file.

    Args:
        pickle_file_path (str): path where pickle file is stored
        path_to_psf (str): path where psf file is stored
        path_to_crd (str): path where crd file is stored
        dcd_output_path (str): path to save dcd file
        pdb_output_path (str): path to save pdb file
    """
    # helper function that converts pickle trajectory file to dcd file

    f = pickle.load(open(pickle_file_path, "rb"))
    traj = [frame.value_in_unit(unit.nanometer) for frame in f]
    topology = CharmmPsfFile(path_to_psf).topology
    positions = CharmmCrdFile(path_to_crd)

    PDBFile.writeFile(topology, positions.positions, file=open(pdb_output_path, "w"))
    traj = md.Trajectory(traj, topology=topology)
    traj.save_dcd(dcd_output_path)
