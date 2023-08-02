"""
Unit and regression test for the endstate_correction package.
"""

# Import package, test suite, and other packages as needed
import sys, pickle, os
import pytest
import numpy as np
from endstate_correction.protocol import perform_endstate_correction, Protocol
from .test_system import setup_ZINC00077329_system


def test_endstate_correction_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "endstate_correction" in sys.modules


def save_pickle_results(sim, mm_samples, nnp_samples, system_name):
    # generate data for plotting tests
    protocol = Protocol(
        method="NEQ",
        sim=sim,
        target_samples=mm_samples,
        reference_samples=nnp_samples,
        nr_of_switches=100,
        neq_switching_length=100,
    )

    r = perform_endstate_correction(protocol)
    pickle.dump(
        r,
        open(
            f"data/{system_name}/switching_charmmff/{system_name}_neq_bid.pickle", "wb"
        ),
    )

    protocol = Protocol(
        method="NEQ",
        sim=sim,
        reference_samples=mm_samples,
        nr_of_switches=100,
        neq_switching_length=100,
    )

    r = perform_endstate_correction(protocol)
    pickle.dump(
        r,
        open(
            f"data/{system_name}/switching_charmmff/{system_name}_neq_forw.pickle", "wb"
        ),
    )


def test_FEP_protocol():
    """Perform FEP uni- and bidirectional protocol"""

    # load samples
    sim, mm_samples, nnp_samples = setup_ZINC00077329_system()

    ####################################################
    # ----------------------- FEP ----------------------
    ####################################################

    fep_protocol = Protocol(
        method="FEP",
        sim=sim,
        reference_samples=mm_samples,
        nr_of_switches=50,
    )

    r = perform_endstate_correction(fep_protocol)
    print(r)
    assert len(r.dE_reference_to_target) == fep_protocol.nr_of_switches
    assert len(r.dE_target_to_reference) == 0
    assert len(r.W_reference_to_target) == 0
    assert len(r.W_target_to_reference) == 0
    assert np.all(r.dE_reference_to_target < 0)  # the dE_forw has negative values

    fep_protocol = Protocol(
        sim=sim,
        method="FEP",
        reference_samples=mm_samples,
        target_samples=nnp_samples,
        nr_of_switches=50,
    )

    r = perform_endstate_correction(fep_protocol)
    assert len(r.dE_reference_to_target) == fep_protocol.nr_of_switches
    assert len(r.dE_target_to_reference) == fep_protocol.nr_of_switches
    assert len(r.W_reference_to_target) == 0
    assert len(r.W_target_to_reference) == 0
    assert np.all(r.dE_reference_to_target < 0)  # the dE_forw have negative values
    assert np.all(r.dE_target_to_reference > 0)  # the dE_rev have positive values


@pytest.mark.skipif(
    os.getenv("CI") == "true",
    reason="Skipping tests that take too long in github actions",
)
def test_NEQ_protocol():
    """Perform NEQ uni- and bidirectional protocol"""
    from endstate_correction.protocol import perform_endstate_correction, Protocol

    # load samples
    sim, mm_samples, nnp_samples = setup_ZINC00077329_system()

    protocol = Protocol(
        method="NEQ",
        sim=sim,
        reference_samples=mm_samples,
        target_samples=nnp_samples,
        nr_of_switches=10,
        neq_switching_length=50,
    )

    r = perform_endstate_correction(protocol)
    assert len(r.dE_reference_to_target) == 0
    assert len(r.dE_target_to_reference) == 0
    assert len(r.W_reference_to_target) == protocol.nr_of_switches
    assert len(r.W_target_to_reference) == protocol.nr_of_switches
    assert np.all(r.W_reference_to_target < 0)  # the dE_forw have negative values
    assert np.all(r.W_target_to_reference > 0)  # the dE_rev have positive values

    protocol = Protocol(
        method="NEQ",
        sim=sim,
        reference_samples=mm_samples,
        nr_of_switches=10,
        neq_switching_length=50,
    )

    r = perform_endstate_correction(protocol)
    assert len(r.dE_reference_to_target) == 0
    assert len(r.dE_target_to_reference) == 0
    assert len(r.W_reference_to_target) == protocol.nr_of_switches
    assert len(r.W_target_to_reference) == 0
    assert np.all(r.W_reference_to_target < 0)  # the dE_forw have negative values


@pytest.mark.skipif(
    os.getenv("CI") == "true",
    reason="Skipping tests that take too long in github actions",
)
def test_ALL_protocol():
    """Perform FEP and NEQ uni- and bidirectional protocol"""
    from endstate_correction.protocol import Protocol, perform_endstate_correction

    # load samples
    sim, mm_samples, nnp_samples = setup_ZINC00077329_system()

    ####################################################
    # ---------------- All corrections -----------------
    ####################################################

    protocol = Protocol(
        method="All",
        sim=sim,
        reference_samples=mm_samples,
        target_samples=nnp_samples,
        nr_of_switches=10,
        neq_switching_length=50,
    )

    r = perform_endstate_correction(protocol)
    assert len(r.dE_reference_to_target) == protocol.nr_of_switches
    assert len(r.dE_target_to_reference) == protocol.nr_of_switches
    assert len(r.W_reference_to_target) == protocol.nr_of_switches
    assert len(r.W_target_to_reference) == protocol.nr_of_switches

    assert not np.isclose(
        r.dE_reference_to_target[0], r.dE_target_to_reference[0], rtol=1e-8
    )
    assert not np.isclose(
        r.dE_reference_to_target[0], r.W_reference_to_target[0], rtol=1e-8
    )
    assert not np.isclose(
        r.dE_reference_to_target[0], r.W_target_to_reference[0], rtol=1e-8
    )

    assert not np.isclose(
        r.W_target_to_reference[0], r.W_reference_to_target[0], rtol=1e-8
    )

    assert np.all(r.W_reference_to_target < 0)  # the dE_forw have negative values
    assert np.all(r.W_target_to_reference > 0)  # the dE_rev have positive values
    assert np.all(r.dE_reference_to_target < 0)  # the dE_forw have negative values
    assert np.all(r.dE_target_to_reference > 0)  # the dE_rev have positive values


def test_each_protocol():
    """Test FEP and NEQ uni-and bidirectional protocols separately"""
    from endstate_correction.protocol import perform_endstate_correction, Protocol

    # load samples
    sim, mm_samples, nnp_samples = setup_ZINC00077329_system()

    ####################################################
    # ----------------------- FEP ----------------------
    ####################################################

    fep_protocol = Protocol(
        method="FEP",
        sim=sim,
        reference_samples=mm_samples,
        nr_of_switches=10,
    )

    r = perform_endstate_correction(fep_protocol)
    assert len(r.dE_reference_to_target) == fep_protocol.nr_of_switches
    assert len(r.dE_target_to_reference) == 0
    assert len(r.W_reference_to_target) == 0
    assert len(r.W_target_to_reference) == 0

    fep_protocol = Protocol(
        method="FEP",
        sim=sim,
        target_samples=mm_samples,
        reference_samples=nnp_samples,
        nr_of_switches=10,
    )

    r = perform_endstate_correction(fep_protocol)
    assert len(r.dE_reference_to_target) == fep_protocol.nr_of_switches
    assert len(r.dE_target_to_reference) == fep_protocol.nr_of_switches
    assert len(r.W_reference_to_target) == 0
    assert len(r.W_target_to_reference) == 0

    ####################################################
    # ----------------------- NEQ ----------------------
    ####################################################

    neq_protocol = Protocol(
        method="NEQ",
        sim=sim,
        reference_samples=mm_samples,
        target_samples=nnp_samples,
        nr_of_switches=10,
        neq_switching_length=50,
    )

    r = perform_endstate_correction(neq_protocol)
    assert len(r.dE_reference_to_target) == 0
    assert len(r.dE_target_to_reference) == 0
    assert len(r.W_reference_to_target) == neq_protocol.nr_of_switches
    assert len(r.W_target_to_reference) == neq_protocol.nr_of_switches
    assert len(r.endstate_samples_reference_to_target) == 0
    assert len(r.endstate_samples_reference_to_target) == 0
    assert len(r.switching_traj_reference_to_target) == 0
    assert len(r.switching_traj_target_to_reference) == 0

    neq_protocol = Protocol(
        method="NEQ",
        sim=sim,
        reference_samples=mm_samples,
        nr_of_switches=10,
        neq_switching_length=50,
    )

    r = perform_endstate_correction(neq_protocol)
    assert len(r.dE_reference_to_target) == 0
    assert len(r.dE_target_to_reference) == 0
    assert len(r.W_reference_to_target) == neq_protocol.nr_of_switches
    assert len(r.W_target_to_reference) == 0
    assert len(r.endstate_samples_reference_to_target) == 0
    assert len(r.endstate_samples_reference_to_target) == 0
    assert len(r.switching_traj_reference_to_target) == 0
    assert len(r.switching_traj_target_to_reference) == 0

    # test saving endstates and saving trajectory option
    protocol = Protocol(
        method="NEQ",
        sim=sim,
        reference_samples=mm_samples,
        target_samples=nnp_samples,
        nr_of_switches=10,
        neq_switching_length=50,
        save_endstates=True,
        save_trajs=True,
    )

    r = perform_endstate_correction(protocol)
    assert len(r.dE_reference_to_target) == 0
    assert len(r.dE_target_to_reference) == 0
    assert len(r.W_reference_to_target) == protocol.nr_of_switches
    assert len(r.W_target_to_reference) == protocol.nr_of_switches
    assert len(r.endstate_samples_reference_to_target) == protocol.nr_of_switches
    assert len(r.endstate_samples_reference_to_target) == protocol.nr_of_switches
    assert len(r.switching_traj_reference_to_target) == protocol.nr_of_switches
    assert len(r.switching_traj_target_to_reference) == protocol.nr_of_switches
    assert len(r.switching_traj_reference_to_target[0]) == protocol.neq_switching_length
    assert len(r.switching_traj_target_to_reference[0]) == protocol.neq_switching_length
