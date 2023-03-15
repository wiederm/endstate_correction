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
        direction="bidirectional",
        sim=sim,
        trajectories=[mm_samples, qml_samples],
        nr_of_switches=100,
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
        direction="unidirectional",
        sim=sim,
        trajectories=[mm_samples, qml_samples],
        nr_of_switches=100,
    )

    r = perform_endstate_correction(protocol)
    pickle.dump(
        r,
        open(
            f"data/{system_name}/switching_charmmff/{system_name}_neq_unid.pickle", "wb"
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
        direction="unidirectional",
        sim=sim,
        trajectories=[mm_samples, qml_samples],
        nr_of_switches=50,
    )

    r = perform_endstate_correction(fep_protocol)
    assert len(r.dE_mm_to_qml) == fep_protocol.nr_of_switches
    assert len(r.dE_qml_to_mm) == 0
    assert len(r.W_mm_to_qml) == 0
    assert len(r.W_qml_to_mm) == 0

    fep_protocol = Protocol(
        sim=sim,
        method="FEP",
        direction="bidirectional",
        trajectories=[mm_samples, qml_samples],
        nr_of_switches=50,
    )

    r = perform_endstate_correction(fep_protocol)
    assert len(r.dE_mm_to_qml) == fep_protocol.nr_of_switches
    assert len(r.dE_qml_to_mm) == fep_protocol.nr_of_switches
    assert len(r.W_mm_to_qml) == 0
    assert len(r.W_qml_to_mm) == 0


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

    ####################################################
    # ----------------------- NEQ ----------------------
    ####################################################

    protocol = Protocol(
        method="NEQ",
        direction="unidirectional",
        sim=sim,
        trajectories=[mm_samples, qml_samples],
        nr_of_switches=10,
        neq_switching_length=50,
    )

    r = perform_endstate_correction(protocol)
    assert len(r.dE_mm_to_qml) == 0
    assert len(r.dE_qml_to_mm) == 0
    assert len(r.W_mm_to_qml) == protocol.nr_of_switches
    assert len(r.W_qml_to_mm) == 0

    protocol = Protocol(
        method="NEQ",
        direction="bidirectional",
        sim=sim,
        trajectories=[mm_samples, qml_samples],
        nr_of_switches=10,
        neq_switching_length=50,
    )

    r = perform_endstate_correction(protocol)
    assert len(r.dE_mm_to_qml) == 0
    assert len(r.dE_qml_to_mm) == 0
    assert len(r.W_mm_to_qml) == protocol.nr_of_switches
    assert len(r.W_qml_to_mm) == protocol.nr_of_switches

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
        direction="bidirectional",
        sim=sim,
        trajectories=[mm_samples, qml_samples],
        nr_of_switches=10,
        neq_switching_length=50,
    )

    r = perform_endstate_correction(protocol)
    assert len(r.dE_mm_to_qml) == protocol.nr_of_switches
    assert len(r.dE_qml_to_mm) == protocol.nr_of_switches
    assert len(r.W_mm_to_qml) == protocol.nr_of_switches
    assert len(r.W_qml_to_mm) == protocol.nr_of_switches

    assert not np.isclose(r.dE_mm_to_qml[0], r.dE_qml_to_mm[0], rtol=1e-8)
    assert not np.isclose(r.dE_mm_to_qml[0], r.W_mm_to_qml[0], rtol=1e-8)
    assert not np.isclose(r.dE_mm_to_qml[0], r.W_qml_to_mm[0], rtol=1e-8)

    assert not np.isclose(r.W_qml_to_mm[0], r.W_mm_to_qml[0], rtol=1e-8)


def test_each_protocol():
    """Perform FEP uni- and bidirectional protocol"""
    from endstate_correction.protocol import perform_endstate_correction, Protocol
    from .test_neq import load_endstate_system_and_samples

    # start with FEP
    sim, mm_samples, qml_samples = load_endstate_system_and_samples(
        system_name="ZINC00079729"
    )

    ####################################################
    # -------------------Nonsense ----------------------
    ####################################################

    try:
        fep_protocol = Protocol(
            method="EQU",
            direction="unidirectional",
            sim=sim,
            trajectories=[mm_samples, qml_samples],
            nr_of_switches=10,
        )
    except AttributeError:
        pass

    try:
        fep_protocol = Protocol(
            method="NEQ",
            direction="tridirectional",
            sim=sim,
            trajectories=[mm_samples, qml_samples],
            nr_of_switches=10,
        )
    except AttributeError:
        pass
    try:
        fep_protocol = Protocol(
            method="NEQ",
            direction="bidirectional",
            sim=sim,
            trajectories=[mm_samples],
            nr_of_switches=10,
        )
    except AssertionError:
        pass

    ####################################################
    # ----------------------- FEP ----------------------
    ####################################################

    fep_protocol = Protocol(
        method="FEP",
        direction="unidirectional",
        sim=sim,
        trajectories=[mm_samples, qml_samples],
        nr_of_switches=10,
    )

    r = perform_endstate_correction(fep_protocol)
    assert len(r.dE_mm_to_qml) == fep_protocol.nr_of_switches
    assert len(r.dE_qml_to_mm) == 0
    assert len(r.W_mm_to_qml) == 0
    assert len(r.W_qml_to_mm) == 0

    fep_protocol = Protocol(
        sim=sim,
        method="FEP",
        direction="bidirectional",
        trajectories=[mm_samples, qml_samples],
        nr_of_switches=10,
    )

    r = perform_endstate_correction(fep_protocol)
    assert len(r.dE_mm_to_qml) == fep_protocol.nr_of_switches
    assert len(r.dE_qml_to_mm) == fep_protocol.nr_of_switches
    assert len(r.W_mm_to_qml) == 0
    assert len(r.W_qml_to_mm) == 0

    ####################################################
    # ----------------------- NEQ ----------------------
    ####################################################

    neq_protocol = Protocol(
        method="NEQ",
        direction="unidirectional",
        sim=sim,
        trajectories=[mm_samples, qml_samples],
        nr_of_switches=10,
        neq_switching_length=50,
    )

    r = perform_endstate_correction(neq_protocol)
    assert len(r.dE_mm_to_qml) == 0
    assert len(r.dE_qml_to_mm) == 0
    assert len(r.W_mm_to_qml) == neq_protocol.nr_of_switches
    assert len(r.W_qml_to_mm) == 0
    assert len(r.endstate_samples_mm_to_qml) == 0
    assert len(r.endstate_samples_qml_to_mm) == 0
    assert len(r.switching_traj_mm_to_qml) == 0
    assert len(r.switching_traj_qml_to_mm) == 0

    neq_protocol = Protocol(
        method="NEQ",
        direction="unidirectional_reverse",
        sim=sim,
        trajectories=[mm_samples, qml_samples],
        nr_of_switches=10,
        neq_switching_length=50,
    )

    r = perform_endstate_correction(neq_protocol)
    assert len(r.dE_mm_to_qml) == 0
    assert len(r.dE_qml_to_mm) == 0
    assert len(r.W_mm_to_qml) == 0
    assert len(r.W_qml_to_mm) == neq_protocol.nr_of_switches
    assert len(r.endstate_samples_mm_to_qml) == 0
    assert len(r.endstate_samples_qml_to_mm) == 0
    assert len(r.switching_traj_mm_to_qml) == 0
    assert len(r.switching_traj_qml_to_mm) == 0
    

    neq_protocol = Protocol(
        sim=sim,
        method="NEQ",
        direction="bidirectional",
        trajectories=[mm_samples, qml_samples],
        nr_of_switches=10,
        neq_switching_length=50,
    )

    r = perform_endstate_correction(neq_protocol)
    assert len(r.dE_mm_to_qml) == 0
    assert len(r.dE_qml_to_mm) == 0
    assert len(r.W_mm_to_qml) == neq_protocol.nr_of_switches
    assert len(r.W_qml_to_mm) == neq_protocol.nr_of_switches
    assert len(r.endstate_samples_mm_to_qml) == 0
    assert len(r.endstate_samples_qml_to_mm) == 0
    assert len(r.switching_traj_mm_to_qml) == 0
    assert len(r.switching_traj_qml_to_mm) == 0
        
    # test saving endstates and saving trajectory option
    protocol = Protocol(
        method="NEQ",
        direction="bidirectional",
        sim=sim,
        trajectories=[mm_samples, qml_samples],
        nr_of_switches=10,
        neq_switching_length=50,
        save_endstates = True,
        save_trajs= True
    )

    r = perform_endstate_correction(protocol)
    assert len(r.dE_mm_to_qml) == 0
    assert len(r.dE_qml_to_mm) == 0
    assert len(r.W_mm_to_qml) == protocol.nr_of_switches
    assert len(r.W_qml_to_mm) == protocol.nr_of_switches   
    assert len(r.endstate_samples_mm_to_qml) == protocol.nr_of_switches
    assert len(r.endstate_samples_qml_to_mm) == protocol.nr_of_switches
    assert len(r.switching_traj_mm_to_qml) == protocol.nr_of_switches
    assert len(r.switching_traj_qml_to_mm) == protocol.nr_of_switches
    assert len(r.switching_traj_mm_to_qml[0]) == protocol.neq_switching_length
    assert len(r.switching_traj_qml_to_mm[0]) == protocol.neq_switching_length