"""Provide the analysis functions."""

import os
from dataclasses import dataclass, fields
from typing import List, Tuple, Union

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
import seaborn as sns
from matplotlib.offsetbox import (AnnotationBbox, DrawingArea, OffsetImage,
                                  TextArea)
from matplotlib.ticker import FormatStrFormatter
from pymbar import bar, exp

from endstate_correction.constant import zinc_systems
from endstate_correction.protocol import Results


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
    overlap = mbar.compute_overlap()["matrix"]
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
        f'ddG = {mbar.compute_free_energy_differences()["Delta_f"][0][-1]} +- {mbar.compute_free_energy_differences()["dDelta_f"][0][-1]}'
    )

    plt.figure(figsize=[8, 8], dpi=300)
    r = mbar.compute_free_energy_differences()["Delta_f"]

    x = [a for a in np.linspace(0, 1, len(r[0]))]
    y = r[0]
    y_error = mbar.compute_free_energy_differences()["dDelta_f"][0]
    print()
    plt.errorbar(x, y, yerr=y_error, label="ddG +- stddev [kT]")
    plt.legend()
    plt.title(f"Free energy estimate for {name}", fontsize=15)
    plt.ylabel("Free energy estimate in kT", fontsize=15)
    plt.xlabel("lambda state (0 to 1)", fontsize=15)
    plt.savefig(f"{name}_equilibrium_free_energy.png")
    plt.show()
    plt.close()


def return_endstate_correction(
    results: Results, method: str = "NEQ", direction: str = "forw"
) -> Tuple[float, float]:
    # generate docstring
    """Return the endstate correction for a given method and direction.
    Args:
        results (Results): instance of the Results class
        method (str): NEQ or FEP
        direction (str): forw, rev or bid
    Returns:
        float: endstate correction delta_f
        float: endstate correction error
    """
    assert type(results) == Results
    assert method in ["NEQ", "FEP"]
    print(f"method: {method}, direction: {direction}")
    
    if method == "FEP" and direction == "forw":
        print(
            f"FEP(forw): {exp(results.dE_reference_to_target)['Delta_f']}"
        )
        est = exp(results.dE_reference_to_target)
        return est["Delta_f"], est["dDelta_f"]
    elif method == "FEP" and direction == "rev":
        print(
            f"FEP(rev): {exp(results.dE_target_to_reference)['Delta_f']}"
        )
        est = exp(results.dE_target_to_reference)
        return est["Delta_f"], est["dDelta_f"]
    elif method == "FEP" and direction == "bid":
        
        print(
            f"FEP(bid): {bar(results.dE_reference_to_target, results.dE_target_to_reference)['Delta_f']}"
        )
        est = bar(results.dE_reference_to_target, results.dE_target_to_reference)
        return est["Delta_f"], est["dDelta_f"]
    elif method == "NEQ" and direction == "forw":
        print(
            f"NEQ(forw): {exp(results.W_reference_to_target)['Delta_f']}"
        )
        est = exp(results.W_reference_to_target)
        return est["Delta_f"], est["dDelta_f"]
    elif method == "NEQ" and direction == "rev":
        print(
            f"NEQ(rev): {exp(results.W_target_to_reference)['Delta_f']}"
        )
        est = exp(results.W_target_to_reference)
        return est["Delta_f"], est["dDelta_f"]
    elif method == "NEQ" and direction == "bid":
        
        print(
            f"NEQ(bid): {bar(results.W_reference_to_target, results.W_target_to_reference)['Delta_f']}"
        )
        est = bar(results.W_reference_to_target, results.W_target_to_reference)
        return est["Delta_f"], est["dDelta_f"]
    else:
        raise ValueError("method and direction combination not supported")

def summarize_endstate_correction_results(results: Results):
    """Summarize the results of the endstate correction analysis.

    Args:
        results (Results): instance of the Results class
    """

    assert type(results) == Results
    print("#--------------- SUMMARY ---------------#")

    if results.dE_reference_to_target.size:
        print(
            f"Zwanzig's equation (from mm to qml): {exp(results.dE_reference_to_target)['Delta_f']}"
        )
    if results.dE_target_to_reference.size:
        print(
            f"Zwanzig's equation (from qml to mm): {exp(results.dE_target_to_reference)['Delta_f']}"
        )
    if results.dE_reference_to_target.size and results.dE_target_to_reference.size:
        print(
            f"Zwanzig's equation bidirectional: {bar(results.dE_reference_to_target, results.dE_target_to_reference)['Delta_f']}"
        )
    if results.W_reference_to_target.size:
        print(
            f"Jarzynski's equation (from mm to qml): {exp(results.W_reference_to_target)['Delta_f']}"
        )
    if results.W_target_to_reference.size:
        print(
            f"Jarzynski's equation (from qml to mm): {exp(results.W_target_to_reference)['Delta_f']}"
        )
    if results.W_reference_to_target.size and results.W_target_to_reference.size:
        print(
            f"Crooks' equation: {bar(results.W_reference_to_target, results.W_target_to_reference)['Delta_f']}"
        )
    if results.equ_mbar:
        ddG_equ = np.average(
            [
                r.compute_free_energy_differences()["Delta_f"][0][-1]
                for r in results.equ_mbar
            ]
        )
        dddG_equ = np.average(
            [
                r.compute_free_energy_differences()["dDelta_f"][0][-1]
                for r in results.equ_mbar
            ]
        )
        print(f"Equilibrium free energy: {ddG_equ}+/-{dddG_equ}")


def plot_endstate_correction_results(
    name: str, results: Results, filename: str = "plot.png"
):
    assert type(results) == Results

    ###########################################################
    # count how many results are available
    multiple_results = 0
    for field in fields(results):
        print(field.name, getattr(results, field.name))
        try:
            if getattr(results, field.name).size:
                multiple_results += 1
        except AttributeError:
            continue

    ax_index = 0
    ###########################################################
    summarize_endstate_correction_results(results)
    ###########################################################
    # ------------------- Plot distributions ------------------

    sns.set_context("talk")
    if multiple_results > 1:
        fig, axs = plt.subplots(3, 1, figsize=(11.0, 9), dpi=600)
    else:
        fig, axs = plt.subplots(2, 1, figsize=(11.0, 9), dpi=600)

    # set up color palette
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

    axs[ax_index].set_title(rf"{name} - distribution of W and $\Delta$E")
    axs[ax_index].ticklabel_format(
        axis="x", style="sci", useOffset=True, scilimits=(0, 0)
    )

    if results.W_reference_to_target.size:
        sns.histplot(
            ax=axs[ax_index],
            alpha=0.5,
            data=results.W_reference_to_target,
            kde=True,
            stat="density",
            label=r"W(MM$\rightarrow$QML)",
            color=c1,
        )

    if results.dE_reference_to_target.size:
        sns.histplot(
            ax=axs[ax_index],
            alpha=0.5,
            data=results.dE_reference_to_target,
            kde=True,
            stat="density",
            label=r"$\Delta$E(MM$\rightarrow$QML)",
            color=c2,
        )

    if results.W_target_to_reference.size:
        sns.histplot(
            ax=axs[ax_index],
            alpha=0.5,
            data=results.W_target_to_reference * -1,
            kde=True,
            stat="density",
            label=r"W(QML$\rightarrow$MM)",
            color=c3,
        )

    if results.dE_target_to_reference.size:
        sns.histplot(
            ax=axs[ax_index],
            alpha=0.5,
            data=results.dE_target_to_reference * -1,
            kde=True,
            stat="density",
            label=r"$\Delta$E(QML$\rightarrow$MM)",
            color=c4,
        )
        axs[ax_index].legend()

    ###########################################################
    # ------------------- Plot results ------------------------
    if multiple_results > 1:
        ax_index += 1
        axs[ax_index].set_title(rf"{name} - offset $\Delta$G(MM$\rightarrow$QML)")
        ddG_list, dddG_list, names = [], [], []

        if results.equ_mbar:
            # Equilibrium free energy
            ddG_list.append(ddG_equ)
            dddG_list.append(dddG_equ)
            names.append("Equilibrium")
        if results.W_reference_to_target.size and results.W_target_to_reference.size:
            # Crooks' equation
            ddG, dddG = -1, -1
            r = bar(results.W_reference_to_target, results.W_target_to_reference)
            ddG, dddG = r["Delta_f"], r["dDelta_f"]
            ddG_list.append(ddG)
            dddG_list.append(dddG)
            names.append("NEQ+Crooks")
        if results.W_reference_to_target.size:
            # Jarzynski's equation
            ddG, dddG = -1, -1
            r = exp(results.W_reference_to_target)
            ddG, dddG = r["Delta_f"], r["dDelta_f"]
            ddG_list.append(ddG)
            dddG_list.append(dddG)
            names.append("NEQ+Jazynski")
        if results.dE_reference_to_target.size and results.dE_target_to_reference.size:
            # FEP + bar
            ddG, dddG = -1, -1
            r = bar(results.dE_reference_to_target, results.dE_target_to_reference)
            ddG, dddG = r["Delta_f"], r["dDelta_f"]
            ddG_list.append(ddG)
            dddG_list.append(dddG)
            names.append("FEP+BAR")
        if results.dE_reference_to_target.size:
            # FEP + EXP
            ddG, dddG = -1, -1
            r = exp(results.dE_reference_to_target)
            ddG, dddG = r["Delta_f"], r["dDelta_f"]
            ddG_list.append(ddG)
            dddG_list.append(dddG)
            names.append("FEP+EXP")

        print("#################")
        print(ddG_list)
        axs[ax_index].errorbar(
            [i for i in range(len(ddG_list))],
            # ddG_list - np.min(ddG_list),
            ddG_list - ddG_list[0],
            dddG_list,
            fmt="o",
        )

        axs[ax_index].set_xticks([i for i in range(len(names))], labels=names)
        # axs[ax_index].set_xticklabels(names)

        axs[ax_index].set_ylabel("kT")
        axs[ax_index].set_ylim([-5, 5])
        axs[ax_index].axhline(y=0.0, color=c1, linestyle=":")

    ######################################################################
    # ------------------- Plot cumulative stddev ------------------------
    ax_index += 1
    axs[ax_index].set_title(rf"{name} - cumulative stddev of W and $\Delta$E")

    if results.W_reference_to_target.size:
        cum_stddev_ws_from_mm_to_qml = [
            results.W_reference_to_target[:x].std()
            for x in range(1, len(results.W_reference_to_target) + 1)
        ]
        axs[ax_index].plot(
            cum_stddev_ws_from_mm_to_qml,
            label=r"stddev W(MM$\rightarrow$QML)",
            color=c1,
        )

    if results.W_target_to_reference.size:
        cum_stddev_ws_from_qml_to_mm = [
            results.W_target_to_reference[:x].std()
            for x in range(1, len(results.W_target_to_reference) + 1)
        ]
        axs[ax_index].plot(
            cum_stddev_ws_from_qml_to_mm,
            label=r"stddev W(QML$\rightarrow$MM)",
            color=c3,
        )

    if results.dE_reference_to_target.size:
        if results.W_reference_to_target.size or results.W_target_to_reference.size:
            size = max(
                results.W_reference_to_target.size, results.W_target_to_reference.size
            )
        else:
            size = results.dE_reference_to_target.size
        cum_stddev_dEs_from_mm_to_qml = [
            results.dE_reference_to_target[:x].std() for x in range(1, size + 1)
        ]
        axs[ax_index].plot(
            cum_stddev_dEs_from_mm_to_qml,
            label=r"stddev $\Delta$E(MM$\rightarrow$QML)",
            color=c2,
        )

    if results.dE_target_to_reference.size:
        if results.W_reference_to_target.size or results.W_target_to_reference.size:
            size = max(
                results.W_reference_to_target.size, results.W_target_to_reference.size
            )
        else:
            size = results.dE_reference_to_target.size
        cum_stddev_dEs_from_qml_to_mm = [
            results.dE_target_to_reference[:x].std() for x in range(1, size + 1)
        ]
        axs[ax_index].plot(
            cum_stddev_dEs_from_qml_to_mm,
            label=r"stddev $\Delta$E(QML$\rightarrow$MM)",
            color=c4,
        )
    # plot 1 kT limit
    axs[ax_index].axhline(y=1.0, color=c7, linestyle=":")
    axs[ax_index].axhline(y=2.0, color=c5, linestyle=":")

    axs[ax_index].set_ylabel("kT")

    axs[ax_index].legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


# plotting torsion profiles
###########################################################################################################################################################################


# generate molecule picture with atom indices
def save_mol_pic(zinc_id: str, ff: str):
    from rdkit import Chem
    from rdkit.Chem import AllChem, Draw
    from rdkit.Chem.Draw import IPythonConsole

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
def visualize_torsion_profile(mol, trajectories: dataclass):
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
            traj_mm = trajectories.equilibrium_mm_trajectory
            traj_qml = trajectories.equilibrium_qml_trajectory

            # if also 'post-switching' data has to be plotted, check if it can be retrieved

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
        import matplotlib.gridspec as gridspec

        plt.style.use("fivethirtyeight")
        sns.set_theme()
        sns.set_palette("bright")

        # generate molecule picture
        save_mol_pic(zinc_id=zinc_id, ff=ff)

        # create corresponding nr of subplots
        fig = plt.figure(tight_layout=True, figsize=(8, len(torsions) * 2 + 6), dpi=400)
        gs = gridspec.GridSpec(
            len(torsions) + 1,
            2,
        )

        fig.suptitle(f"Torsion profile of {name} ({ff})", fontsize=15, weight="bold")

        # flip the image, so that it is displayed correctly
        image = mpimg.imread(f"mol_pics_{ff}/{name}_{ff}.png")

        # plot the molecule image on the first axis
        ax = fig.add_subplot(gs[0, :])

        ax.imshow(image)
        ax.axis("off")

        # iterate over all torsions and plot results
        for counter in range(1, len(torsions) + 1):
            # counter for atom indices
            idx_counter = counter - 1
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

            ax_violin = fig.add_subplot(gs[counter, 0])
            sns.violinplot(
                ax=ax_violin,
                data=[
                    torsions_mm[idx_counter].squeeze(),
                    torsions_qml[idx_counter].squeeze(),
                    torsions_mm_switching[idx_counter].squeeze(),
                    torsions_qml_switching[idx_counter].squeeze(),
                ],
                orient="h",
                inner="point",
                split=True,
                scale="width",
                saturation=0.5,
            )
            ax_kde = fig.add_subplot(gs[counter, 1])
            sns.kdeplot(
                ax=ax_kde,
                data=data_histplot,
                common_norm=False,
                shade=True,
                linewidth=2,
                # kde=True,
                # alpha=0.5,
                # stat="density",
                # common_norm=False,
            )

            # adjust axis labelling
            unit = np.arange(-np.pi, np.pi + np.pi / 4, step=(1 / 4 * np.pi))
            for ax in [ax_violin, ax_kde]:
                # add atom indices as subplot title
                ax.set_title(f"Torsion {all_indices[idx_counter]}", fontsize=13)
                ax.set(xlim=(-np.pi, np.pi))
                ax.set_xticks(
                    unit,
                    ["-π", "-3π/4", "-π/2", "-π/4", "0", "π/4", "π/2", "3π/4", "π"],
                )
                ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
                ax.set_yticks([])  # remove tick values on y axis

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
            axs[counter][0].add_artist(ab) """

        # axs[-1][0].set_xlabel("Dihedral angle")
        plt.tight_layout()
        if not os.path.isdir(f"torsion_profiles_{ff}"):
            os.makedirs(f"torsion_profiles_{ff}")
        plt.savefig(f"torsion_profiles_{ff}/{name}_{ff}_{switching_length}ps.png")
        plt.show()

    else:
        print(f"No torsion profile can be generated for {name}")
