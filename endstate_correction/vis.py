import nglview as ng
from endstate_correction.constant import zinc_systems
import mdtraj as md
from openff.toolkit.topology import Molecule


def visualize_mol(
    smiles: str,
    traj_dir: str,
):
    """Inspect conformations generated by sampling or NEQ switching.

    Args:
        smiles (str): smiles string
        traj_dir (str): path where dcd trajecotry file is stored

    Returns:
        _type_: molecule visualization
    """

    molecule = Molecule.from_smiles(smiles, hydrogens_are_explicit=False)
    molecule.to_file("mol.pdb", file_format="pdb")
    # load trajectory and topology
    f = md.load_dcd(traj_dir, top = "mol.pdb")
    # NOTE: pdb file is needed for mdtraj, which reads the topology
    # this is not very elegant # FIXME: try to load topology directly
    top = md.load("mol.pdb").topology
    # generate trajectory instance
    traj = md.Trajectory(f.xyz, topology=top)
    # align traj
    traj.superpose(traj)
    view = ng.show_mdtraj(traj)
    return view