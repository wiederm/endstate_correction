"""Provide the analysis functions."""

import os
from dataclasses import dataclass, fields
from typing import Tuple, Union

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
from pymbar import bar, exp

from endstate_correction.constant import zinc_systems
from endstate_correction.protocol import BaseResults, AllResults, FEPResults, NEQResults, SMCResults


def plot_overlap_for_equilibrium_free_energy(
    N_k: np.array, u_kn: np.ndarray, name: str
):
    """
    Calculate the overlap for each state with each other state. The overlap is normalized to be 1 for each row.

    Args:
        N_k (np.array): number of samples for each state k
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
    plt.savefig(f"{name}_overlap_equilibrium_free_energy.png")
    plt.show()
    plt.close()


def plot_results_for_equilibrium_free_energy(
    N_k: np.array, u_kn: np.ndarray, name: str
):
    """
    Calculate the accumulated free energy along the mutation progress.

    Args:
        N_k (np.array): number of samples for each state k
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
    plt.savefig(f"{name}_results_equilibrium_free_energy.png")
    plt.show()
    plt.close()

def return_endstate_correction(results: Union[FEPResults, NEQResults], direction: str = "forw") -> Tuple[float, float]:
    """Return the endstate correction for a given method and direction.

    Args:
        results (Union[FEPResults, NEQResults]): instance of the FEPResults or NEQResults class
        direction (str, optional): forw, rev or bid. Defaults to "forw".

    Raises:
        ValueError: if method or direction is not supported

    Returns:
        Tuple[float, float]: endstate correction delta_f, endstate correction error
    """

    assert isinstance(results, (FEPResults, NEQResults))

    print(f"method: {results.__class__.__name__}, direction: {direction}")

    if isinstance(results, FEPResults) and direction == "forw":
        assert results.dE_reference_to_target.size
        print(f"FEP(forw): {exp(results.dE_reference_to_target)['Delta_f']}")
        est = exp(results.dE_reference_to_target)
        return est["Delta_f"], est["dDelta_f"]   
    elif isinstance(results, FEPResults) and direction == "rev":
        assert results.dE_target_to_reference.size
        print(f"FEP(rev): {exp(results.dE_target_to_reference)['Delta_f']}")
        est = exp(results.dE_target_to_reference)
        return est["Delta_f"], est["dDelta_f"] 
    elif isinstance(results, FEPResults) and direction == "bid":
        assert results.dE_reference_to_target.size and results.dE_target_to_reference.size
        print(
        f"FEP(bid): {bar(results.dE_reference_to_target, results.dE_target_to_reference)['Delta_f']}"
        )
        est = bar(results.dE_reference_to_target, results.dE_target_to_reference)
        return est["Delta_f"], est["dDelta_f"]
    elif isinstance(results, NEQResults) and direction == "forw":
        assert results.W_reference_to_target.size
        print(f"NEQ(forw): {exp(results.W_reference_to_target)['Delta_f']}")
        est = exp(results.W_reference_to_target)
        return est["Delta_f"], est["dDelta_f"]
    elif isinstance(results, NEQResults) and direction == "rev":
        assert results.W_target_to_reference.size
        print(f"NEQ(rev): {exp(results.W_target_to_reference)['Delta_f']}")
        est = exp(results.W_target_to_reference)
        return est["Delta_f"], est["dDelta_f"]
    elif isinstance(results, NEQResults) and direction == "bid":
        assert results.W_reference_to_target.size and results.W_target_to_reference.size
        print(
            f"NEQ(bid): {bar(results.W_reference_to_target, results.W_target_to_reference)['Delta_f']}"
        )
        est = bar(results.W_reference_to_target, results.W_target_to_reference)
        return est["Delta_f"], est["dDelta_f"]
    else:
        print(type(results))
        raise ValueError("method and direction combination not supported")


def summarize_endstate_correction_results(results: AllResults):
    """Summarize the results of the endstate correction analysis.

    Args:
        results (AllResults): instance of the AllResults class
    """

    assert isinstance(results, AllResults)
    print("#--------------- SUMMARY ---------------#")
    
    if results.fep_results:
        if results.fep_results.dE_reference_to_target.size:
            print(
                f"Zwanzig's equation (from mm to nnp): {exp(results.fep_results.dE_reference_to_target)['Delta_f']}"
            )
        if results.fep_results.dE_target_to_reference.size:
            print(
                f"Zwanzig's equation (from nnp to mm): {exp(results.fep_results.dE_target_to_reference)['Delta_f']}"
            )
        if results.fep_results.dE_reference_to_target.size and results.fep_results.dE_target_to_reference.size:
            print(
                f"Zwanzig's equation bidirectional: {bar(results.fep_results.dE_reference_to_target, results.fep_results.dE_target_to_reference)['Delta_f']}"
            )
    if results.neq_results:
        if results.neq_results.W_reference_to_target.size:
            print(
                f"Jarzynski's equation (from mm to nnp): {exp(results.neq_results.W_reference_to_target)['Delta_f']}"
            )
        if results.neq_results.W_target_to_reference.size:
            print(
                f"Jarzynski's equation (from nnp to mm): {exp(results.neq_results.W_target_to_reference)['Delta_f']}"
            )
        if results.neq_results.W_reference_to_target.size and results.neq_results.W_target_to_reference.size:
            print(
                f"Crooks' equation: {bar(results.neq_results.W_reference_to_target, results.neq_results.W_target_to_reference)['Delta_f']}"
            )
    if results.equ_results:
        if results.equ_results.equ_mbar:
            ddG_equ = np.average(
                [
                    r.compute_free_energy_differences()["Delta_f"][0][-1]
                    for r in results.equ_results.equ_mbar
                ]
            )
            dddG_equ = np.average(
                [
                    r.compute_free_energy_differences()["dDelta_f"][0][-1]
                    for r in results.equ_results.equ_mbar
                ]
            )
            print(f"Equilibrium free energy: {ddG_equ}+/-{dddG_equ}")


def plot_endstate_correction_results(
    name: str, results: AllResults, filename: str = "plot.png"
):
    """Plot endstate correction results.

    Args:
        name (str): name of the system in the plot
        results (Results): instance of the AllResults class
        filename (str, optional): Defaults to "plot.png".
    """
    assert isinstance(results, AllResults)

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

    if results.neq_results:
        if results.neq_results.W_reference_to_target.size:
            sns.histplot(
                ax=axs[ax_index],
                alpha=0.5,
                data=results.neq_results.W_reference_to_target,
                kde=True,
                stat="density",
                label=r"W(MM$\rightarrow$NNP)",
                color=c1,
            )
        if results.neq_results.W_target_to_reference.size:
            sns.histplot(
                ax=axs[ax_index],
                alpha=0.5,
                data=results.neq_results.W_target_to_reference * -1,
                kde=True,
                stat="density",
                label=r"W(NNP$\rightarrow$MM)",
                color=c3,
            )
    if results.fep_results:
        if results.fep_results.dE_reference_to_target.size:
                sns.histplot(
                    ax=axs[ax_index],
                    alpha=0.5,
                    data=results.fep_results.dE_reference_to_target,
                    kde=True,
                    stat="density",
                    label=r"$\Delta$E(MM$\rightarrow$NNP)",
                    color=c2,
                )
        if results.fep_results.dE_target_to_reference.size:
            sns.histplot(
                ax=axs[ax_index],
                alpha=0.5,
                data=results.fep_results.dE_target_to_reference * -1,
                kde=True,
                stat="density",
                label=r"$\Delta$E(NNP$\rightarrow$MM)",
                color=c4,
            )

        axs[ax_index].legend()

    ###########################################################
    # ------------------- Plot results ------------------------
    if multiple_results > 1:
        ax_index += 1
        axs[ax_index].set_title(rf"{name} - offset $\Delta$G(MM$\rightarrow$NNP)")
        ddG_list, dddG_list, names = [], [], []

        if results.equ_results:
            if results.equ_results.equ_mbar:
                # Equilibrium free energy
                ddG_equ = np.average(
                [
                    r.compute_free_energy_differences()["Delta_f"][0][-1]
                    for r in results.equ_results.equ_mbar
                ]
                )
                dddG_equ = np.average(
                    [
                        r.compute_free_energy_differences()["dDelta_f"][0][-1]
                        for r in results.equ_results.equ_mbar
                    ]
                )
                ddG_list.append(ddG_equ)
                dddG_list.append(dddG_equ)
                names.append("Equilibrium")
        if results.neq_results:
            if results.neq_results.W_reference_to_target.size and results.neq_results.W_target_to_reference.size:
                # Crooks' equation
                ddG, dddG = -1, -1
                r = bar(results.neq_results.W_reference_to_target, results.neq_results.W_target_to_reference)
                ddG, dddG = r["Delta_f"], r["dDelta_f"]
                ddG_list.append(ddG)
                dddG_list.append(dddG)
                names.append("NEQ+Crooks")
            if results.neq_results.W_reference_to_target.size:
                # Jarzynski's equation (reference to target)
                ddG, dddG = -1, -1
                r = exp(results.neq_results.W_reference_to_target)
                ddG, dddG = r["Delta_f"], r["dDelta_f"]
                ddG_list.append(ddG)
                dddG_list.append(dddG)
                names.append("NEQ+Jazynski")
        if results.fep_results:
            if results.fep_results.dE_reference_to_target.size and results.fep_results.dE_target_to_reference.size:
                # FEP + bar
                ddG, dddG = -1, -1
                r = bar(results.fep_results.dE_reference_to_target, results.fep_results.dE_target_to_reference)
                ddG, dddG = r["Delta_f"], r["dDelta_f"]
                ddG_list.append(ddG)
                dddG_list.append(dddG)
                names.append("FEP+BAR")
            if results.fep_results.dE_reference_to_target.size:
                # FEP + EXP
                ddG, dddG = -1, -1
                r = exp(results.fep_results.dE_reference_to_target)
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

    if results.neq_results:
        if results.neq_results.W_reference_to_target.size:
            cum_stddev_ws_from_mm_to_nnp = [
                results.neq_results.W_reference_to_target[:x].std()
                for x in range(1, len(results.neq_results.W_reference_to_target) + 1)
            ]
            axs[ax_index].plot(
                cum_stddev_ws_from_mm_to_nnp,
                label=r"stddev W(MM$\rightarrow$NNP)",
                color=c1,
            )
        if results.neq_results.W_target_to_reference.size:
            cum_stddev_ws_from_nnp_to_mm = [
                results.neq_results.W_target_to_reference[:x].std()
                for x in range(1, len(results.neq_results.W_target_to_reference) + 1)
            ]
            axs[ax_index].plot(
                cum_stddev_ws_from_nnp_to_mm,
                label=r"stddev W(NNP$\rightarrow$MM)",
                color=c3,
            )

    if results.fep_results:
        if results.fep_results.dE_reference_to_target.size:
            if results.neq_results:
                size = max(
                    results.neq_results.W_reference_to_target.size, results.neq_results.W_target_to_reference.size
                )
            else:
                size = results.fep_results.dE_reference_to_target.size
            cum_stddev_dEs_from_mm_to_nnp = [
                results.fep_results.dE_reference_to_target[:x].std() for x in range(1, size + 1)
            ]
            axs[ax_index].plot(
                cum_stddev_dEs_from_mm_to_nnp,
                label=r"stddev $\Delta$E(MM$\rightarrow$NNP)",
                color=c2,
            )
        if results.fep_results.dE_target_to_reference.size:
            if results.neq_results:
                size = max(
                    results.neq_results.W_reference_to_target.size, results.neq_results.W_target_to_reference.size
                )
            else:
                size = results.fep_results.dE_reference_to_target.size
            cum_stddev_dEs_from_nnp_to_mm = [
                results.fep_results.dE_target_to_reference[:x].std() for x in range(1, size + 1)
            ]
            axs[ax_index].plot(
                cum_stddev_dEs_from_nnp_to_mm,
                label=r"stddev $\Delta$E(NNP$\rightarrow$MM)",
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
