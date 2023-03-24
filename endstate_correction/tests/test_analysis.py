import pathlib

import endstate_correction
from openmm.app import CharmmParameterSet, CharmmPsfFile
import pytest
import os
from .test_system import setup_vacuum_simulation

@pytest.mark.skipif(
    os.getenv("CI") == "true",
    reason="Skipping tests that take too long in github actions",
)
def test_plotting_equilibrium_free_energy():
    "Test that plotting functions can be called"
    from endstate_correction.analysis import (
        plot_overlap_for_equilibrium_free_energy,
        plot_results_for_equilibrium_free_energy,
    )
    from endstate_correction.equ import calculate_u_kn
    from .test_equ import load_equ_samples

    """test if we are able to plot overlap and """

    ########################################################
    ########################################################
    # ----------------- vacuum -----------------------------
    # get all relevant files
    path = pathlib.Path(endstate_correction.__file__).resolve().parent
    hipen_testsystem = f"{path}/data/hipen_data"

    system_name = "ZINC00077329"
    psf = CharmmPsfFile(f"{hipen_testsystem}/{system_name}/{system_name}.psf")
    params = CharmmParameterSet(
        f"{hipen_testsystem}/top_all36_cgenff.rtf",
        f"{hipen_testsystem}/par_all36_cgenff.prm",
        f"{hipen_testsystem}/{system_name}/{system_name}.str",
    )


    trajs = load_equ_samples(system_name)
    sim = setup_vacuum_simulation(psf, params)
    N_k, u_kn = calculate_u_kn(
        trajs=trajs,
        every_nth_frame=50,
        sim=sim,
    )

    plot_overlap_for_equilibrium_free_energy(N_k=N_k, u_kn=u_kn, name=system_name)
    plot_results_for_equilibrium_free_energy(N_k=N_k, u_kn=u_kn, name=system_name)


def test_plot_results_for_FEP_protocol():
    """Perform FEP uni- and bidirectional protocol"""
    from endstate_correction.protocol import perform_endstate_correction, Protocol
    from .test_neq import load_endstate_system_and_samples
    from endstate_correction.analysis import plot_endstate_correction_results

    system_name = "ZINC00079729"
    # start with FEP
    sim, mm_samples, qml_samples = load_endstate_system_and_samples(
        system_name=system_name
    )

    ####################################################
    # ----------------------- FEP ----------------------
    ####################################################

    fep_protocol = Protocol(
        method="FEP",
        sim=sim,
        reference_samples=mm_samples,
        target_samples=qml_samples,
        nr_of_switches=50,
    )

    r = perform_endstate_correction(fep_protocol)
    print(r)
    plot_endstate_correction_results(
        system_name, r, f"{system_name}_results_fep_unidirectional.png"
    )

    fep_protocol = Protocol(
        method="FEP",
        sim=sim,
        reference_samples=mm_samples,
        target_samples = qml_samples,
        nr_of_switches=100,
    )

    r = perform_endstate_correction(fep_protocol)
    plot_endstate_correction_results(
        system_name, r, f"{system_name}_results_fep_bidirectional.png"
    )


def test_plot_results_for_NEQ_protocol():
    """Perform FEP uni- and bidirectional protocol"""
    from endstate_correction.protocol import Protocol, perform_endstate_correction
    from .test_neq import load_endstate_system_and_samples
    from endstate_correction.analysis import (
        plot_endstate_correction_results,
    )
    import pickle

    system_name = "ZINC00079729"
    # start with NEQ
    sim, mm_samples, qml_samples = load_endstate_system_and_samples(
        system_name=system_name
    )

    ####################################################
    # ----------------------- NEQ ----------------------
    ####################################################

    fep_protocol = Protocol(
        method="NEQ",
        sim=sim,
        reference_samples=mm_samples,
        target_samples = qml_samples,
        nr_of_switches=100,
    )

    r = pickle.load(
        open(
            f"data/{system_name}/switching_charmmff/{system_name}_neq_unid.pickle", "rb"
        )
    )
    plot_endstate_correction_results(
        system_name, r, f"{system_name}_results_neq_unidirectional.png"
    )

    fep_protocol = Protocol(
        method="NEQ",
        sim=sim,
        reference_samples=mm_samples,
        target_samples=qml_samples,
        nr_of_switches=100,
    )
    # load pregenerated data
    r = pickle.load(
        open(
            f"data/{system_name}/switching_charmmff/{system_name}_neq_bid.pickle", "rb"
        )
    )
    plot_endstate_correction_results(
        system_name, r, f"{system_name}_results_neq_bidirectional.png"
    )


def test_plot_results_for_all_protocol():
    """Perform FEP uni- and bidirectional protocol"""
    from endstate_correction.protocol import Protocol
    from .test_neq import load_endstate_system_and_samples
    from endstate_correction.analysis import (
        plot_endstate_correction_results,
    )
    import pickle

    system_name = "ZINC00079729"
    # start with NEQ
    sim, mm_samples, qml_samples = load_endstate_system_and_samples(
        system_name=system_name
    )

    ####################################################
    # ---------------- All corrections -----------------
    ####################################################

    fep_protocol = Protocol(
        method="All",
        sim=sim,
        reference_samples=mm_samples,
        target_samples=qml_samples,
        nr_of_switches=100,
    )

    # load data
    r = pickle.load(
        open(
            f"data/{system_name}/switching_charmmff/{system_name}_all_corrections.pickle",
            "rb",
        )
    )
    print(r)
    plot_endstate_correction_results(system_name, r, f"{system_name}_results_all.png")
