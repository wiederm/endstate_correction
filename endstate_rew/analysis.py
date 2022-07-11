import glob
import pickle
import os
import numpy as np
from collections import namedtuple
from typing import NamedTuple, Tuple

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.ticker import FormatStrFormatter
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
import seaborn as sns
from openmm import unit
from pymbar import BAR, EXP
from tqdm import tqdm
import mdtraj as md
from scipy.stats import wasserstein_distance
from itertools import chain
from endstate_rew.system import generate_molecule
from endstate_rew.constant import kBT, check_implementation, zinc_systems


def _collect_equ_samples(
    path: str,
    name: str,
    lambda_scheme: list,
    every_nth_frame: int = 2,
    only_endstates: bool = False,
) -> Tuple[list, np.array]:

    """
    Collect equilibrium samples

    Args:
        path (str): path to the location where the samples are stored
        name (str): name of the system (used in the sample files)
        lambda_scheme (list): list of lambda states as floats
        every_nth_frame (int, optional): prune the samples further by taking only every nth sample. Defaults to 2.

    Raises:
        RuntimeError: if multuple sample files are present we can not decide which is the correct one.

    Returns:
        Tuple(coordinates, N_k)
    """

    nr_of_samples = 5_000
    nr_of_steps = 1_000
    coordinates = []
    N_k = np.zeros(len(lambda_scheme))

    # loop over lambda scheme and collect samples in nanometer
    for idx, lamb in enumerate(lambda_scheme):
        file = glob.glob(
            f"{path}/{name}_samples_{nr_of_samples}_steps_{nr_of_steps}_lamb_{lamb:.4f}.pickle"
        )
        if len(file) == 2:
            raise RuntimeError("Multiple traj files present. Abort.")
        elif len(file) == 0:
            print("WARNING! Incomplete equ sampling. Proceed with cautions.")
        else:
            if only_endstates and not (
                idx == 0 or idx == len(lambda_scheme) - 1
            ):  # take only first or last samples
                print(f"skipping {idx}")
                continue
            coords_ = pickle.load(open(file[0], "rb"))
            coords_ = coords_[1_000:]  # remove the first 1k samples
            coords_ = coords_[::every_nth_frame]  # take only every nth sample
            N_k[idx] = len(coords_)
            coordinates.extend([c_.value_in_unit(unit.nanometer) for c_ in coords_])

    number_of_samples = len(coordinates)
    print(f"Number of samples loaded: {number_of_samples}")
    return coordinates * unit.nanometer, N_k


def calculate_u_kn(
    smiles: str,
    forcefield: str,
    path_to_files: str,
    name: str,
    every_nth_frame: int = 2,
    reload: bool = True,
    override: bool = False,
) -> np.ndarray:

    """
    Calculate the u_kn matrix to be used by the mbar estimator

    Args:
        smiles (str): smiles string describing the system
        forcefield (str): which force field is used (allowed options are `openff` or `charmmmff`)
        path_to_files (str): path to location where samples are stored
        name (str): name of the system (used in the sample files)
        every_nth_frame (int, optional): prune the samples further by taking only every nth sample. Defaults to 2.
        reload (bool, optional): do you want to reload a previously saved mbar pickle file if present (every time the free energy is calculated the mbar pickle file is saved --- only loading is optional)
        override (bool, optional) : override
    Returns:
        Tuple(np.array, np.ndarray): (N_k, u_kn)
    """

    from os import path

    from endstate_rew.system import (
        generate_molecule,
        initialize_simulation_with_charmmff,
        initialize_simulation_with_openff,
    )

    # NOTE: NNPOps only runs on CUDA
    implementation, platform = check_implementation()

    pickle_path = f"{path_to_files}/mbar_{every_nth_frame}.pickle"
    if path.isfile(pickle_path) and reload:  # if already generated reuse
        print(f"trying to load: {pickle_path}")
        N_k, u_kn = pickle.load(open(pickle_path, "rb"))
        print(f"Reusing pregenerated mbar object: {pickle_path}")
    else:
        # generate molecule
        m = generate_molecule(smiles=smiles, forcefield=forcefield)
        # initialize simulation
        # first, modify path to point to openff molecule object
        w_dir = path_to_files.split("/")
        w_dir = "/".join(w_dir[:-3])
        # initialize simualtion and reload if already generated
        if forcefield == "openff":
            sim = initialize_simulation_with_openff(m, w_dir=w_dir)
        elif forcefield == "charmmff":
            sim = initialize_simulation_with_charmmff(m, zinc_id=name)
        else:
            raise NotImplementedError("only charmmff or openff are implemented.")
        lambda_scheme = np.linspace(0, 1, 11)
        samples, N_k = _collect_equ_samples(
            path_to_files, name, lambda_scheme, every_nth_frame=every_nth_frame
        )
        samples = np.array(
            samples.value_in_unit(unit.nanometer)
        )  # positions in nanometer
        u_kn = np.zeros(
            (len(N_k), int(N_k[0] * len(N_k))), dtype=np.float64
        )  # NOTE: assuming that N_k[0] is the maximum number of samples drawn from any state k
        for k, lamb in enumerate(lambda_scheme):
            sim.context.setParameter("lambda_interpolate", lamb)
            us = []
            for x in tqdm(range(len(samples))):
                sim.context.setPositions(samples[x])
                u_ = sim.context.getState(getEnergy=True).getPotentialEnergy()
                us.append(u_)
            us = np.array([u / kBT for u in us], dtype=np.float64)
            u_kn[k] = us

        if not path.isfile(pickle_path) or override == True:
            pickle.dump((N_k, u_kn), open(f"{pickle_path}", "wb+"))

    # total number of samples
    total_nr_of_samples = 0
    for n in N_k:
        total_nr_of_samples += n

    assert total_nr_of_samples != 0  # make sure that there are samples present

    return (N_k, u_kn)


def plot_overlap_for_equilibrium_free_energy(
    N_k: np.array, u_kn: np.ndarray, name: str
):
    """
    Calculate the overlap for each state with each other state. THe overlap is normalized to be 1 for each row.

    Args:
        N_k (np.array): numnber of samples for each state k
        u_kn (np.ndarray): each of the potential energy functions `u` describing a state `k` are applied to each sample `n` from each of the states `k`
        name (str): name of the system in the plot
    """
    from pymbar import MBAR

    # initialize the MBAR maximum likelihood estimate

    mbar = MBAR(u_kn, N_k)
    plt.figure(figsize=[8, 8], dpi=300)
    overlap = mbar.computeOverlap()["matrix"]
    sns.heatmap(
        overlap,
        cmap="Blues",
        linewidth=0.5,
        annot=True,
        fmt="0.2f",
        annot_kws={"size": "small"},
    )
    plt.title(f"Free energy estimate for {name}", fontsize=15)
    plt.savefig(f"{name}_equilibrium_free_energy.png")
    plt.show()
    plt.close()


def plot_results_for_equilibrium_free_energy(
    N_k: np.array, u_kn: np.ndarray, name: str
):
    """
    Calculate the accumulated free energy along the mutation progress.


    Args:
        N_k (np.array): numnber of samples for each state k
        u_kn (np.ndarray): each of the potential energy functions `u` describing a state `k` are applied to each sample `n` from each of the states `k`
        name (str): name of the system in the plot
    """
    from pymbar import MBAR

    # initialize the MBAR maximum likelihood estimate

    mbar = MBAR(u_kn, N_k)
    print(
        f'ddG = {mbar.getFreeEnergyDifferences(return_dict=True)["Delta_f"][0][-1]} +- {mbar.getFreeEnergyDifferences(return_dict=True)["dDelta_f"][0][-1]}'
    )

    plt.figure(figsize=[8, 8], dpi=300)
    r = mbar.getFreeEnergyDifferences(return_dict=True)["Delta_f"]

    x = [a for a in np.linspace(0, 1, len(r[0]))]
    y = r[0]
    y_error = mbar.getFreeEnergyDifferences(return_dict=True)["dDelta_f"][0]
    print()
    plt.errorbar(x, y, yerr=y_error, label="ddG +- stddev [kT]")
    plt.legend()
    plt.title(f"Free energy estimate for {name}", fontsize=15)
    plt.ylabel("Free energy estimate in kT", fontsize=15)
    plt.xlabel("lambda state (0 to 1)", fontsize=15)
    plt.savefig(f"{name}_equilibrium_free_energy.png")
    plt.show()
    plt.close()


def _collect_work_values(file: str) -> list:

    ws = pickle.load(open(file, "rb")).value_in_unit(unit.kilojoule_per_mole)
    number_of_samples = len(ws)
    print(f"Number of samples used: {number_of_samples}")
    return ws * unit.kilojoule_per_mole


def collect_results_from_neq_and_equ_free_energy_calculations(
    w_dir: str,
    forcefield: str,
    run_id: int,
    name: str,
    smiles: str,
    every_nth_frame: int = 10,
    switching_length: int = 5001,
) -> NamedTuple:

    """collects the pregenerated equilibrium free energies and non-equilibrium work values (and calculates the free energies)

    Raises:
        FileNotFoundError: _description_
        NotImplementedError: _description_

    Returns:
        _type_: _description_
    """
    from os import path

    from pymbar import MBAR

    from endstate_rew.neq import perform_switching
    from endstate_rew.system import (
        generate_molecule,
        initialize_simulation_with_charmmff,
        initialize_simulation_with_openff,
    )
    from endstate_rew.analysis import _collect_equ_samples

    # collect equ results
    equ_samples_path = f"{w_dir}/sampling_{forcefield}/run{run_id:0>2d}"
    mbar_pickle_path = f"{equ_samples_path}/mbar_{every_nth_frame}.pickle"
    neq_samples_path = f"{w_dir}/switching_{forcefield}/"

    print(f"{equ_samples_path=}")
    print(f"{neq_samples_path=}")

    if not path.isfile(mbar_pickle_path):
        raise FileNotFoundError(
            f"Equilibrium mbar results are not saved: {mbar_pickle_path}"
        )

    N_k, u_kn = pickle.load(open(mbar_pickle_path, "rb"))
    mbar = MBAR(u_kn, N_k)
    r = mbar.getFreeEnergyDifferences(return_dict=True)["Delta_f"]

    # load equ samples
    samples, N_k = _collect_equ_samples(
        equ_samples_path, name=name, lambda_scheme=[0, 1], only_endstates=True
    )
    # split them in mm/qml samples
    mm_samples = samples[: int(N_k[0])]
    qml_samples = samples[int(N_k[0]) :]
    assert len(mm_samples) == N_k[0]
    assert len(qml_samples) == N_k[0]

    # get pregenerated work values
    ws_from_mm_to_qml = np.array(
        _collect_work_values(
            f"{neq_samples_path}/{name}_neq_ws_from_mm_to_qml_500_{switching_length}.pickle"
        )
        / kBT
    )
    ws_from_qml_to_mm = np.array(
        _collect_work_values(
            f"{neq_samples_path}/{name}_neq_ws_from_qml_to_mm_500_{switching_length}.pickle"
        )
        / kBT
    )

    ##############################
    # perform inst switching
    ##############################
    switching_length = 2
    nr_of_switches = 500
    # create molecule
    molecule = generate_molecule(forcefield=forcefield, smiles=smiles)

    if forcefield == "openff":
        sim = initialize_simulation_with_openff(molecule, w_dir=w_dir)
    elif forcefield == "charmmff":
        sim = initialize_simulation_with_charmmff(molecule, zinc_id=name)
    else:
        raise NotImplementedError("only charmmff or openff are implemented.")

    # perform switching
    lambs = np.linspace(0, 1, switching_length)

    dEs_from_mm_to_qml = np.array(
        perform_switching(
            sim,
            lambs,
            samples=mm_samples,
            nr_of_switches=nr_of_switches,
        )[0]
        / kBT
    )
    lambs = np.linspace(1, 0, switching_length)
    dEs_from_qml_to_mm = np.array(
        perform_switching(
            sim,
            lambs,
            samples=qml_samples,
            nr_of_switches=nr_of_switches,
        )[0]
        / kBT
    )
    ##############################

    # pack everything in a namedtuple
    Results = namedtuple(
        "Results",
        "equ_mbar dWs_from_mm_to_qml dWs_from_qml_to_mm dEs_from_mm_to_qml dEs_from_qml_to_mm",
    )
    results = Results(
        mbar,
        ws_from_mm_to_qml,
        ws_from_qml_to_mm,
        dEs_from_mm_to_qml,
        dEs_from_qml_to_mm,
    )
    return results


def plot_resutls_of_switching_experiments(name: str, results: NamedTuple):

    print("################################")
    ddG = results.equ_mbar.getFreeEnergyDifferences(return_dict=True)["Delta_f"][0][-1]
    dddG = results.equ_mbar.getFreeEnergyDifferences(return_dict=True)["dDelta_f"][0][
        -1
    ]
    print(f"Equilibrium free energy: {ddG}+/-{dddG}")
    print(
        f"Crooks' equation: {BAR(results.dWs_from_mm_to_qml, results.dWs_from_qml_to_mm)}"
    )
    print(f"Jarzynski's equation: {EXP(results.dWs_from_mm_to_qml)}")
    print(f"Zwanzig's equation: {EXP(results.dEs_from_mm_to_qml)}")
    print(
        f"Zwanzig's equation bidirectional: {BAR(results.dEs_from_mm_to_qml, results.dEs_from_qml_to_mm)}"
    )
    print("################################")

    sns.set_context("talk")
    fig, axs = plt.subplots(3, 1, figsize=(11.0, 9), dpi=600)
    # plot distribution of dE and dW
    #########################################
    axs[0].set_title(rf"{name} - distribution of $\Delta$W and $\Delta$E")
    palett = sns.color_palette(n_colors=8)
    palett_as_hex = palett.as_hex()
    c1, c2, c3, c4, c5, c7 = (
        palett_as_hex[0],
        palett_as_hex[1],
        palett_as_hex[2],
        palett_as_hex[3],
        palett_as_hex[4],
        palett_as_hex[6],
    )
    axs[0].ticklabel_format(axis="x", style="sci", useOffset=True, scilimits=(0, 0))
    # axs[1].ticklabel_format(axis='x', style='sci', useOffset=False,scilimits=(0,0))

    sns.histplot(
        ax=axs[0],
        alpha=0.5,
        data=results.dWs_from_mm_to_qml * -1,
        kde=True,
        stat="density",
        label=r"$\Delta$W(MM$\rightarrow$QML)",
        color=c1,
    )
    sns.histplot(
        ax=axs[0],
        alpha=0.5,
        data=results.dEs_from_mm_to_qml * -1,
        kde=True,
        stat="density",
        label=r"$\Delta$E(MM$\rightarrow$QML)",
        color=c2,
    )
    sns.histplot(
        ax=axs[0],
        alpha=0.5,
        data=results.dWs_from_qml_to_mm,
        kde=True,
        stat="density",
        label=r"$\Delta$W(QML$\rightarrow$MM)",
        color=c3,
    )
    sns.histplot(
        ax=axs[0],
        alpha=0.5,
        data=results.dEs_from_qml_to_mm,
        kde=True,
        stat="density",
        label=r"$\Delta$E(QML$\rightarrow$MM)",
        color=c4,
    )
    axs[0].legend()

    # plot results
    #########################################
    axs[1].set_title(rf"{name} - offset $\Delta$G(MM$\rightarrow$QML)")
    ddG_list, dddG_list = [], []
    # Equilibrium free energy
    ddG = results.equ_mbar.getFreeEnergyDifferences(return_dict=True)["Delta_f"][0][-1]
    dddG = results.equ_mbar.getFreeEnergyDifferences(return_dict=True)["dDelta_f"][0][
        -1
    ]
    ddG_list.append(ddG)
    dddG_list.append(dddG)

    # Crooks' equation
    ddG, dddG = BAR(results.dWs_from_mm_to_qml, results.dWs_from_qml_to_mm)
    if np.isnan(dddG):
        print("#######################")
        print("BEWARE: dddG is nan!")
        print("WILL BE REPLACED BY 1. for plotting")
        print("#######################")
        dddG = 1.0
    ddG_list.append(ddG)
    dddG_list.append(dddG)
    # Jarzynski's equation
    ddG, dddG = EXP(results.dWs_from_mm_to_qml)
    if np.isnan(dddG):
        print("#######################")
        print("BEWARE: dddG is nan!")
        print("WILL BE REPLACED BY 1. for plotting")
        print("#######################")
        dddG = 1.0
    ddG_list.append(ddG)
    dddG_list.append(dddG)
    # FEP
    ddG, dddG = EXP(results.dEs_from_mm_to_qml)
    if np.isnan(dddG):
        print("#######################")
        print("BEWARE: dddG is nan!")
        print("WILL BE REPLACED BY 1. for plotting")
        print("#######################")
        dddG = 1.0
    ddG_list.append(ddG)
    dddG_list.append(dddG)
    # FEP + BAR
    ddG, dddG = BAR(results.dEs_from_mm_to_qml, results.dEs_from_qml_to_mm)
    if np.isnan(dddG):
        print("#######################")
        print("BEWARE: dddG is nan!")
        print("WILL BE REPLACED BY 1. for plotting")
        print("#######################")
        dddG = 1.0
    ddG_list.append(ddG)
    dddG_list.append(dddG)

    axs[1].errorbar(
        [i for i in range(len(ddG_list))],
        # ddG_list - np.min(ddG_list),
        ddG_list - ddG_list[0],
        dddG_list,
        fmt="o",
    )
    axs[1].set_xticklabels(
        ["", "Equilibrium", "", "Crooks", "", "Jazynski", "", "FEP+EXP", "", "FEP+BAR"]
    )
    axs[1].set_ylabel("kT")
    # axs[1].legend()

    axs[1].set_ylim([-5, 5])

    axs[1].axhline(y=0.0, color=c1, linestyle=":")

    # plot cummulative stddev of dE and dW
    #########################################
    axs[2].set_title(rf"{name} - cummulative stddev of $\Delta$W and $\Delta$E")

    cum_stddev_ws_from_mm_to_qml = [
        results.dWs_from_mm_to_qml[:x].std()
        for x in range(1, len(results.dWs_from_mm_to_qml) + 1)
    ]
    cum_stddev_ws_from_qml_to_mm = [
        results.dWs_from_qml_to_mm[:x].std()
        for x in range(1, len(results.dWs_from_qml_to_mm) + 1)
    ]

    cum_stddev_dEs_from_mm_to_qml = [
        results.dEs_from_mm_to_qml[:x].std()
        for x in range(1, len(results.dEs_from_mm_to_qml) + 1)
    ]
    cum_stddev_dEs_from_qml_to_mm = [
        results.dEs_from_qml_to_mm[:x].std()
        for x in range(1, len(results.dEs_from_qml_to_mm) + 1)
    ]
    axs[2].plot(
        cum_stddev_ws_from_mm_to_qml,
        label=r"stddev $\Delta$W(MM$\rightarrow$QML)",
        color=c1,
    )
    axs[2].plot(
        cum_stddev_dEs_from_mm_to_qml,
        label=r"stddev $\Delta$E(MM$\rightarrow$QML)",
        color=c2,
    )
    axs[2].plot(
        cum_stddev_ws_from_qml_to_mm,
        label=r"stddev $\Delta$W(QML$\rightarrow$MM)",
        color=c3,
    )
    axs[2].plot(
        cum_stddev_dEs_from_qml_to_mm,
        label=r"stddev $\Delta$E(QML$\rightarrow$MM)",
        color=c4,
    )
    # plot 1 kT limit
    axs[2].axhline(y=1.0, color=c7, linestyle=":")
    axs[2].axhline(y=2.0, color=c5, linestyle=":")

    axs[2].set_ylabel("kT")

    axs[2].legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(f"{name}_r_20ps.png")
    plt.show()


# plotting torsion profiles
###########################################################################################################################################################################

# generate molecule picture with atom indices
def save_mol_pic(zinc_id: str, ff: str):
    from rdkit.Chem import AllChem
    from rdkit import Chem
    from rdkit.Chem.Draw import IPythonConsole
    from rdkit.Chem import Draw

    IPythonConsole.drawOptions.addAtomIndices = True
    from rdkit.Chem.Draw import rdMolDraw2D

    # get name
    name, _ = zinc_systems[zinc_id]
    # generate openff Molecule
    mol = generate_molecule(name=name, forcefield=ff)
    # convert openff object to rdkit mol object
    mol_rd = mol.to_rdkit()

    # remove explicit H atoms
    if zinc_id == 4:
        # NOTE: FIXME: this is a temporary workaround to fix the wrong indexing in rdkit
        # when using the RemoveHs() function
        mol_draw = Chem.RWMol(mol_rd)
        # remove all explicit H atoms, except the ones on the ring and on N atoms (for correct indexing)
        for run in range(1, 7):
            n_atoms = mol_draw.GetNumAtoms()
            mol_draw.RemoveAtom(n_atoms - 7)
    else:
        # remove explicit H atoms
        mol_draw = Chem.RemoveHs(mol_rd)

    # get 2D representation
    AllChem.Compute2DCoords(mol_draw)
    # formatting
    d = rdMolDraw2D.MolDraw2DCairo(1500, 1000)
    d.drawOptions().fixedFontSize = 90
    d.drawOptions().fixedBondLength = 110
    d.drawOptions().annotationFontScale = 0.7
    d.drawOptions().addAtomIndices = True

    d.DrawMolecule(mol_draw)
    d.FinishDrawing()
    if not os.path.isdir(f"mol_pics_{ff}"):
        os.makedirs(f"mol_pics_{ff}")
    d.WriteDrawingText(f"mol_pics_{ff}/{name}_{ff}.png")


# get trajectory
def get_traj(
    samples: str,
    name: str,
    ff: str,
    w_dir: str,
    switching: bool,
    switching_length: int = 5,
):

    # get sampling data
    if not switching:

        # depending on endstate, get correct label
        if samples == "mm":
            endstate = "0.0000"
        elif samples == "qml":
            endstate = "1.0000"

        # get pickle files for traj
        pickle_files = glob.glob(
            f"{w_dir}/{name}/sampling_{ff}/run*/{name}_samples_5000_steps_1000_lamb_{endstate}.pickle"
        )

        # list for collecting sampling data
        coordinates = []

        # generate traj instance only if at least one pickle file exists
        if pickle_files:
            for run in pickle_files:
                # load pickle file
                coord = pickle.load(open(run, "rb"))
                # check, if sampling data is complete (MODIFY IF NR OF SAMPLING STEPS != 5000)
                if len(coord) == 5000:
                    # remove first 1k samples
                    coordinates.extend(coord[1000:])
                    # load topology from pdb file
                    top = md.load("mol.pdb").topology
                    # generate trajectory instance
                    traj = md.Trajectory(xyz=coordinates, topology=top)
                    return traj
                else:
                    print(f"{run} file contains incomplete sampling data")

    # get trajectory data after switching
    else:

        if switching_length == 5:
            swi_length = "5001"
        elif switching_length == 10:
            swi_length = "10001"
        elif switching_length == 20:
            swi_length = "20001"

        # get pickle file for traj
        pickle_file = f"{w_dir}/{name}/switching_{ff}/{name}_samples_5000_steps_1000_lamb_{samples}_endstate_nr_samples_500_switching_length_{swi_length}.pickle"

        if os.path.isfile(pickle_file):
            # load pickle file
            coordinates = pickle.load(open(pickle_file, "rb"))
            # load topology from pdb file
            top = md.load("mol.pdb").topology
            # generate trajectory instance
            traj = md.Trajectory(xyz=coordinates, topology=top)

            return traj
        else:
            print("No pickle file found.")


# get indices of dihedral bonds
def get_indices(rot_bond: int, rot_bond_list: list, bonds: list):

    print(f"---------- Investigating bond nr {rot_bond} ----------")

    # get indices of both atoms forming an rotatable bond
    atom_1_idx = (rot_bond_list[rot_bond]).atom1_index
    atom_2_idx = (rot_bond_list[rot_bond]).atom2_index

    # create lists to collect neighbors of atom_1 and atom_2
    neighbors1 = []
    neighbors2 = []

    # find neighbors of atoms forming the rotatable bond and add to index list (if heavy atom torsion)
    for bond in bonds:

        # get neighbors of atom_1 (of rotatable bond)
        # check, if atom_1 (of rotatable bond) is the first atom in the current bond
        if bond.atom1_index == atom_1_idx:

            # make sure, that neighboring atom is not an hydrogen, nor atom_2
            if (
                not bond.atom2.element.name == "hydrogen"
                and not bond.atom2_index == atom_2_idx
            ):
                neighbors1.append(bond.atom2_index)

        # check, if atom_1 (of rotatable bond) is the second atom in the current bond
        elif bond.atom2_index == atom_1_idx:

            # make sure, that neighboring atom is not an hydrogen, nor atom_2
            if (
                not bond.atom1.element.name == "hydrogen"
                and not bond.atom1_index == atom_2_idx
            ):
                neighbors1.append(bond.atom1_index)

        # get neighbors of atom_2 (of rotatable bond)
        # check, if atom_2 (of rotatable bond) is the first atom in the current bond
        if bond.atom1_index == atom_2_idx:

            # make sure, that neighboring atom is not an hydrogen, nor atom_1
            if (
                not bond.atom2.element.name == "hydrogen"
                and not bond.atom2_index == atom_1_idx
            ):
                neighbors2.append(bond.atom2_index)

        # check, if atom_2 (of rotatable bond) is the second atom in the current bond
        elif bond.atom2_index == atom_2_idx:

            # make sure, that neighboring atom is not an hydrogen, nor atom_1
            if (
                not bond.atom1.element.name == "hydrogen"
                and not bond.atom1_index == atom_1_idx
            ):
                neighbors2.append(bond.atom1_index)

    # check, if both atoms forming the rotatable bond have neighbors
    if len(neighbors1) > 0 and len(neighbors2) > 0:

        # list for final atom indices defining torsion
        indices = [[neighbors1[0], atom_1_idx, atom_2_idx, neighbors2[0]]]
        return indices

    else:

        print(f"No heavy atom torsions found for bond {rot_bond}")
        indices = []
        return indices


# plot torsion profiles
def vis_torsions(
    zinc_id: int,
    ff: str,
    w_dir: str = "/data/shared/projects/endstate_rew/",
    switching: bool = False,
    switching_length: int = 5,
):
    ############################################ LOAD MOLECULE AND GET BOND INFO ##########################################################################################

    # get zinc_id(name of the zinc system)
    name, _ = zinc_systems[zinc_id]

    print(
        f"################################## SYSTEM {name} ##################################"
    )

    # generate mol from name
    mol = generate_molecule(forcefield=ff, name=name)

    # write mol as pdb
    # NOTE: pdb file is needed for mdtraj, which reads the topology in get_traj()
    # this is not very elegant # FIXME: try to load topology directly
    mol.to_file("mol.pdb", file_format="pdb")

    # get all bonds
    bonds = mol.bonds

    # get all rotatable bonds
    rot_bond_list = mol.find_rotatable_bonds()
    print(len(rot_bond_list), "rotatable bonds found.")

    ################################################## GET HEAVY ATOM TORSIONS ##########################################################################################

    # list for collecting bond nr, which form a dihedral angle
    torsions = []
    # list for collecting all atom indices, which form a dihedral angle
    all_indices = []
    # lists for traj data
    torsions_mm = []
    torsions_qml = []
    # lists for traj data after switching
    torsions_mm_switching = []
    torsions_qml_switching = []
    # boolean which enables plotting, if data can be retrieved
    plotting = False

    for rot_bond in range(len(rot_bond_list)):

        # get atom indices of current rotatable bond forming a torsion
        indices = get_indices(
            rot_bond=rot_bond, rot_bond_list=rot_bond_list, bonds=bonds
        )
        print(indices)

        # compute dihedrals only if heavy atom torsion was found for rotatable bond
        if len(indices) > 0:
            print(f"Dihedrals are computed for bond nr {rot_bond}")
            # add bond nr to list
            torsions.append(rot_bond)
            # add corresponding atom indices to list
            all_indices.extend(indices)

            # check if traj data can be retrieved
            traj_mm = get_traj(
                samples="mm",
                name=name,
                ff=ff,
                w_dir=w_dir,
                switching=False,
            )
            traj_qml = get_traj(
                samples="qml",
                name=name,
                ff=ff,
                w_dir=w_dir,
                switching=False,
            )

            # if also 'post-switching' data has to be plotted, check if it can be retrieved
            if switching:
                traj_mm_switching = get_traj(
                    samples="mm",
                    name=name,
                    ff=ff,
                    w_dir=w_dir,
                    switching=True,
                    switching_length=switching_length,
                )
                traj_qml_switching = get_traj(
                    samples="qml",
                    name=name,
                    ff=ff,
                    w_dir=w_dir,
                    switching=True,
                    switching_length=switching_length,
                )

            # if both, mm and qml samples are found, compute dihedrals
            if traj_mm and traj_qml:
                torsions_mm.append(
                    md.compute_dihedrals(traj_mm, indices, periodic=True, opt=True)
                )  # * 180.0 / np.pi
                torsions_qml.append(
                    md.compute_dihedrals(traj_qml, indices, periodic=True, opt=True)
                )  # * 180.0 / np.pi
                plotting = True

                # additionally, compute dihedrals from 'post-switching' data
                if switching and traj_mm_switching and traj_qml_switching:
                    torsions_mm_switching.append(
                        md.compute_dihedrals(
                            traj_mm_switching, indices, periodic=True, opt=True
                        )
                    )  # * 180.0 / np.pi
                    torsions_qml_switching.append(
                        md.compute_dihedrals(
                            traj_qml_switching, indices, periodic=True, opt=True
                        )
                    )  # * 180.0 / np.pi
                elif switching and not traj_mm_switching and not traj_qml_switching:
                    plotting = False

            else:
                print(f"Trajectory data cannot be found for {name}")
        else:
            print(f"No dihedrals will be computed for bond nr {rot_bond}")

    ################################################## PLOT TORSION PROFILES ##########################################################################################

    if plotting:

        # generate molecule picture
        save_mol_pic(zinc_id=zinc_id, ff=ff)

        # counter for addressing axis
        counter = 0

        # create corresponding nr of subplots
        fig, axs = plt.subplots(
            len(torsions) + 1, 1, figsize=(8, len(torsions) * 2 + 6), dpi=400
        )
        fig.suptitle(f"Torsion profile of {name} ({ff})", fontsize=13, weight="bold")

        # flip the image, so that it is displayed correctly
        image = mpimg.imread(f"mol_pics_{ff}/{name}_{ff}.png")

        # plot the molecule image on the first axis
        axs[0].imshow(image)
        axs[0].axis("off")

        # set counter to 1 (for torsion profiles)
        counter += 1
        # counter for atom indices
        idx_counter = 0

        # iterate over all torsions and plot results
        for torsion in torsions:

            # plot only sampling data
            if not switching:
                data_histplot = {
                    "mm samples": torsions_mm[idx_counter].squeeze(),
                    "qml samples": torsions_qml[idx_counter].squeeze(),
                }

            # compare to data after switching
            else:
                data_histplot = {
                    "mm samples": torsions_mm[idx_counter].squeeze(),
                    "qml samples": torsions_qml[idx_counter].squeeze(),
                    rf"qml$\rightarrow$mm endstate ({switching_length}ps switch)": torsions_mm_switching[
                        idx_counter
                    ].squeeze(),
                    rf"mm$\rightarrow$qml endstate ({switching_length}ps switch)": torsions_qml_switching[
                        idx_counter
                    ].squeeze(),
                }

                # if needed, compute wasserstein distance
                """  # compute wasserstein distance
                w_distance = wasserstein_distance(u_values = list(chain.from_iterable(torsions_mm[idx_counter])), v_values = list(chain.from_iterable(torsions_qml[idx_counter])))
                w_distance_qml_switch_mm = wasserstein_distance(u_values = list(chain.from_iterable(torsions_qml[idx_counter])), v_values = list(chain.from_iterable(torsions_mm_switching[idx_counter])))
                w_distance_mm_switch_qml = wasserstein_distance(u_values = list(chain.from_iterable(torsions_mm[idx_counter])), v_values = list(chain.from_iterable(torsions_qml_switching[idx_counter]))) """

            sns.histplot(
                ax=axs[counter],
                data=data_histplot,
                bins=100,  # not sure how many bins to use
                kde=True,
                alpha=0.5,
                stat="density",
                common_norm=False,
            )
            # add atom indices as subplot title
            axs[counter].set_title(f"Torsion {all_indices[idx_counter]}", fontsize=13)

            # adjust axis labelling
            unit = np.arange(-np.pi, np.pi + np.pi / 4, step=(1 / 4 * np.pi))
            axs[counter].set(xlim=(-np.pi, np.pi))
            axs[counter].set_xticks(
                unit, ["-π", "-3π/4", "-π/2", "-π/4", "0", "π/4", "π/2", "3π/4", "π"]
            )
            axs[counter].yaxis.set_major_formatter(FormatStrFormatter("%.3f"))

            # if wasserstein distance is computed, it can be added as an annotation box next to the plot
            """ text_div = f'Wasserstein distance\n\nmm (sampling) & qml (sampling): {w_distance:.3f}\nmm (sampling) & qml ({switching_length}ps switch): {w_distance_mm_switch_qml:.3f}\nqml (sampling) & mm ({switching_length}ps switch): {w_distance_qml_switch_mm:.3f}'
            offsetbox = TextArea(text_div,
                                 textprops=dict(ha='left', size = 13))
            xy = (0,0)
            if switching_length == 5:
                x_box = 1.56
            elif switching_length == 10 or switching_length == 20:
                x_box = 1.575
            ab = AnnotationBbox(offsetbox, xy,
                    xybox=(x_box, 10),
                    xycoords='axes points',
                    boxcoords=("axes fraction", "axes points"),
                    box_alignment=(1, 0.08))
                    #arrowprops=dict(arrowstyle="->"))
            axs[counter].add_artist(ab) """

            counter += 1
            idx_counter += 1
        axs[-1].set_xlabel("Dihedral angle")

        plt.tight_layout()

        if not os.path.isdir(f"torsion_profiles_{ff}"):
            os.makedirs(f"torsion_profiles_{ff}")
        plt.savefig(f"torsion_profiles_{ff}/{name}_{ff}_{switching_length}ps.png")
    else:
        print(f"No torsion profile can be generated for {name}")
