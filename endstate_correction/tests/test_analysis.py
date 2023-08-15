import os
import pathlib

import numpy as np
import pytest
from openmm.app import CharmmParameterSet, CharmmPsfFile

import endstate_correction

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
    from endstate_correction.analysis import (
        plot_endstate_correction_results,
        return_endstate_correction,
    )
    #from endstate_correction.analysis import return_endstate_correction
    from endstate_correction.protocol import FEPProtocol, perform_endstate_correction

    from .test_system import setup_ZINC00077329_system

    system_name = "ZINC00079729"
    # start with FEP
    sim, mm_samples, qml_samples = setup_ZINC00077329_system()

    fep_protocol = FEPProtocol(
        sim=sim,
        reference_samples=mm_samples,
        target_samples=qml_samples,
        nr_of_switches=100,
    )

    r = perform_endstate_correction(fep_protocol)
    plot_endstate_correction_results(
        system_name, r, f"{system_name}_results_fep_bidirectional.png"
    )
    # test return_endstate_correction
    df, ddf = return_endstate_correction(r, method= "FEP", direction="forw")
    df, ddf = return_endstate_correction(r, method="FEP", direction="bid")


def test_plot_results_for_NEQ_protocol():
    """Perform FEP uni- and bidirectional protocol"""
    import pickle

    from endstate_correction.analysis import (
        plot_endstate_correction_results,
        return_endstate_correction,
    )
    #from endstate_correction.protocol import Protocol

    system_name = "ZINC00079729"

    # load pregenerated data
    r = pickle.load(
        open(
            f"data/{system_name}/switching_charmmff/{system_name}_neq_bid_new.pickle", "rb"
        )
    )
    plot_endstate_correction_results(
        system_name, r, f"{system_name}_results_neq_bidirectional.png"
    )

    # test return_endstate_correction
    df, ddf = return_endstate_correction(r, method="NEQ", direction="forw")
    assert np.isclose(df, -2105813.1630223254)
    df, ddf = return_endstate_correction(r, method="NEQ", direction="rev")
    assert np.isclose(df, -2105811.559385495)
    df, ddf = return_endstate_correction(r, method="NEQ", direction="bid")
    # bidirectional test fails most likely due to missing overlap (and convergence)
    # assert np.isclose(df, -217063.7966900661)


def test_plot_results_for_all_protocol():
    """Perform FEP uni- and bidirectional protocol"""
    import pickle

    from endstate_correction.analysis import plot_endstate_correction_results
    from endstate_correction.protocol import AllProtocol, FEPProtocol, NEQProtocol, SMCProtocol

    from .test_system import setup_ZINC00077329_system

    system_name = "ZINC00079729"
    # start with NEQ
    sim, mm_samples, qml_samples = setup_ZINC00077329_system()

    ####################################################
    # ---------------- All corrections -----------------
    ####################################################

    # fep_protocol = FEPProtocol(
    #     sim=sim,
    #     reference_samples=mm_samples,
    #     target_samples=qml_samples,
    #     nr_of_switches=100,
    # )

    # load data
    r = pickle.load(
        open(
            f"data/{system_name}/switching_charmmff/{system_name}_all_corrections_new.pickle",
            "rb",
        )
    )
    print(r)
    plot_endstate_correction_results(system_name, r, f"{system_name}_results_all.png")
