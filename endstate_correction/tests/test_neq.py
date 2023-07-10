import pathlib
from typing import Tuple

import endstate_correction
import mdtraj
import numpy as np
from endstate_correction.neq import perform_switching
from openmm import unit
from openmm.app import Simulation
from .test_system import setup_vacuum_simulation, setup_ZINC00077329_system


def test_collect_work_values():
    """test if we are able to collect samples as anticipated"""
    from endstate_correction.neq import _collect_work_values

    nr_of_switches = 200
    path = f"data/ZINC00077329/switching_charmmff/ZINC00077329_neq_ws_from_mm_to_qml_{nr_of_switches}_5001.pickle"
    ws = _collect_work_values(path)
    assert len(ws) == nr_of_switches


def test_switching():

    # load simulation and samples for ZINC00077329
    sim, samples_mm, samples_qml = setup_ZINC00077329_system()
    # perform instantaneous switching with predetermined coordinate set
    # here, we evaluate dU_forw = dU(x)_qml - dU(x)_mm and make sure that it is the same as
    # dU_rev = dU(x)_mm - dU(x)_qml
    lambs = np.linspace(0, 1, 2)
    print(lambs)
    dE_list, _, _ = perform_switching(
        sim, lambdas=lambs, samples=samples_mm[:1], nr_of_switches=1
    )
    assert np.isclose(
        dE_list[0].value_in_unit(unit.kilojoule_per_mole), -2345981.1035673507
    )
    lambs = np.linspace(1, 0, 2)

    dE_list, _, _ = perform_switching(
        sim, lambdas=lambs, samples=samples_mm[:1], nr_of_switches=1
    )
    print(dE_list)

    assert np.isclose(
        dE_list[0].value_in_unit(unit.kilojoule_per_mole), 2345981.1035673507
    )

    # perform NEQ switching
    lambs = np.linspace(0, 1, 21)
    dW_forw, _, _ = perform_switching(
        sim, lambdas=lambs, samples=samples_mm, nr_of_switches=2
    )
    print(dW_forw)
    assert dW_forw[0] != dW_forw[1]

    # perform NEQ switching
    lambs = np.linspace(0, 1, 101)
    dW_forw, _, _ = perform_switching(
        sim, lambdas=lambs, samples=samples_mm, nr_of_switches=2
    )
    print(dW_forw)
    assert dW_forw[0] != dW_forw[1]

    # check return values
    lambs = np.linspace(0, 1, 3)
    list_1, list_2, list_3 = perform_switching(
        sim,
        lambdas=lambs,
        samples=samples_mm[:1],
        nr_of_switches=1,
        save_endstates=False,
        save_trajs=False,
    )
    assert len(list_1) == 1 and len(list_2) == 0 and len(list_3) == 0

    list_1, list_2, list_3 = perform_switching(
        sim,
        lambdas=lambs,
        samples=samples_mm[:1],
        nr_of_switches=1,
        save_endstates=False,
        save_trajs=True,
    )

    assert (
        len(list_1) == 1
        and len(list_2) == 0
        and len(list_3) == 1
        and len(list_3[0]) == 3
    )

    list_1, list_2, list_3 = perform_switching(
        sim,
        lambdas=lambs,
        samples=samples_mm[:1],
        nr_of_switches=1,
        save_endstates=True,
        save_trajs=False,
    )
    assert len(list_1) == 1 and len(list_2) == 1 and len(list_3) == 0

    list_1, list_2, list_3 = perform_switching(
        sim,
        lambdas=lambs,
        samples=samples_mm[:1],
        nr_of_switches=1,
        save_endstates=True,
        save_trajs=True,
    )
    assert (
        len(list_1) == 1
        and len(list_2) == 1
        and len(list_3) == 1
        and len(list_3[0]) == 3
    )
