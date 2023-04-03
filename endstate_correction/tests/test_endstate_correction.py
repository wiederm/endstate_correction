"""
Unit and regression test for the endstate_correction package.
"""

# Import package, test suite, and other packages as needed
import sys, pickle, os
import pytest
import numpy as np
from endstate_correction.protocol import perform_endstate_correction, Protocol
from .test_neq import load_endstate_system_and_samples


def test_endstate_correction_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "endstate_correction" in sys.modules


def save_pickle_results(sim, mm_samples, qml_samples, system_name):
    # generate data for plotting tests
    protocol = Protocol(
        method="NEQ",
        sim=sim,
        target_samples=mm_samples,
        reference_samples=qml_samples,
        nr_of_switches=100,
        neq_switching_length=100
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
        neq_switching_length=100
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

    # start with FEP
    sim, mm_samples, qml_samples = load_endstate_system_and_samples(
        system_name="ZINC00079729"
    )

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

    fep_protocol = Protocol(
        sim=sim,
        method="FEP",
        reference_samples=mm_samples,
        target_samples=qml_samples,
        nr_of_switches=50,
    )

    r = perform_endstate_correction(fep_protocol)
    assert len(r.dE_reference_to_target) == fep_protocol.nr_of_switches
    assert len(r.dE_target_to_reference) == fep_protocol.nr_of_switches
    assert len(r.W_reference_to_target) == 0
    assert len(r.W_target_to_reference) == 0


@pytest.mark.skipif(
    os.getenv("CI") == "true",
    reason="Skipping tests that take too long in github actions",
)
def test_NEQ_protocol():
    """Perform NEQ uni- and bidirectional protocol"""
    from endstate_correction.protocol import perform_endstate_correction, Protocol
    from .test_neq import load_endstate_system_and_samples

    system_name = "ZINC00079729"
    # start with FEP
    sim, mm_samples, qml_samples = load_endstate_system_and_samples(
        system_name=system_name
    )

    protocol = Protocol(
        method="NEQ",
        sim=sim,
        reference_samples=mm_samples,
        target_samples=qml_samples,
        nr_of_switches=10,
        neq_switching_length=50,
    )

    r = perform_endstate_correction(protocol)
    assert len(r.dE_reference_to_target) == 0
    assert len(r.dE_target_to_reference) == 0
    assert len(r.W_reference_to_target) == protocol.nr_of_switches
    assert len(r.W_target_to_reference) == protocol.nr_of_switches

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

    # longer NEQ switching
    #save_pickle_results(sim, mm_samples, qml_samples, system_name)


@pytest.mark.skipif(
    os.getenv("CI") == "true",
    reason="Skipping tests that take too long in github actions",
)
def test_ALL_protocol():
    """Perform FEP uni- and bidirectional protocol"""
    from endstate_correction.protocol import Protocol, perform_endstate_correction
    from .test_neq import load_endstate_system_and_samples
    import pickle

    system_name = "ZINC00079729"
    # start with NEQ
    sim, mm_samples, qml_samples = load_endstate_system_and_samples(
        system_name=system_name
    )

    ####################################################
    # ---------------- All corrections -----------------
    ####################################################

    protocol = Protocol(
        method="All",
        sim=sim,
        reference_samples=mm_samples,
        target_samples=qml_samples,
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


def test_each_protocol():
    """Perform FEP uni- and bidirectional protocol"""
    from endstate_correction.protocol import perform_endstate_correction, Protocol
    from .test_neq import load_endstate_system_and_samples

    # start with FEP
    sim, mm_samples, qml_samples = load_endstate_system_and_samples(
        system_name="ZINC00079729"
    )

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
        reference_samples=qml_samples,
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
        target_samples=qml_samples,
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
        target_samples=qml_samples,
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
