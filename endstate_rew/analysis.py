import glob
import pickle
from collections import namedtuple
from typing import NamedTuple, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from openmm import unit
from pymbar import BAR, EXP
from tqdm import tqdm

from endstate_rew.constant import kBT, check_implementation


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
    platform: str = "CPU",
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
            sim = initialize_simulation_with_openff(m, w_dir=w_dir, platform=platform)
        elif forcefield == "charmmff":
            sim = initialize_simulation_with_charmmff(
                m, zinc_id=name, platform=platform
            )
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
            if implementation == "NNPOps":
                sim.context.setParameter("scale", lamb)
            else:
                sim.context.setParameter("lambda", lamb)
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
    implementation: str = "",
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
            f"{neq_samples_path}/{name}_neq_ws_from_mm_to_qml_200_{switching_length}.pickle"
        )
        / kBT
    )
    ws_from_qml_to_mm = np.array(
        _collect_work_values(
            f"{neq_samples_path}/{name}_neq_ws_from_qml_to_mm_200_{switching_length}.pickle"
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

    # NOTE: 'NNPOps' works only with CUDA platform
    if implementation == "NNPOps":
        platform = "CUDA"
    else:
        platform = "CPU"

    if forcefield == "openff":
        sim = initialize_simulation_with_openff(
            molecule, w_dir=w_dir, platform=platform
        )
    elif forcefield == "charmmff":
        sim = initialize_simulation_with_charmmff(
            molecule, zinc_id=name, platform=platform
        )
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
            implementation=implementation,
        )
        / kBT
    )
    lambs = np.linspace(1, 0, switching_length)
    dEs_from_qml_to_mm = np.array(
        perform_switching(
            sim,
            lambs,
            samples=qml_samples,
            nr_of_switches=nr_of_switches,
            implementation=implementation,
        )
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
    c1, c2, c3, c4 = (
        palett_as_hex[0],
        palett_as_hex[1],
        palett_as_hex[2],
        palett_as_hex[3],
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
    # Crooks' equation
    ddG_list, dddG_list = [], []
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
        ddG_list - np.min(ddG_list),
        dddG_list,
        fmt="o",
    )
    axs[1].set_xticklabels(["", "Crooks", "", "Jazynski", "", "FEP+EXP", "", "FEP+BAR"])
    axs[1].set_ylabel("kT")
    # axs[1].legend()

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
        cum_stddev_ws_from_mm_to_qml, label=r"stddev $\Delta$W(MM$\rightarrow$QML)"
    )
    axs[2].plot(
        cum_stddev_ws_from_qml_to_mm, label=r"stddev $\Delta$W(QML$\rightarrow$MM)"
    )
    axs[2].plot(
        cum_stddev_dEs_from_mm_to_qml, label=r"stddev $\Delta$E(MM$\rightarrow$QML)"
    )
    axs[2].plot(
        cum_stddev_dEs_from_qml_to_mm, label=r"stddev $\Delta$E(QML$\rightarrow$MM)"
    )
    # plot 1 kT limit
    axs[2].axhline(y=1.0, color="yellow", linestyle=":")
    axs[2].axhline(y=2.0, color="orange", linestyle=":")

    axs[2].set_ylabel("kT")

    axs[2].legend()

    plt.tight_layout()
    plt.savefig(f"{name}_r_10ps.png")
    plt.show()
