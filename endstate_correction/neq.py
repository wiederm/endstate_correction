"""Provide the functions for non-equilibrium switching."""


import pickle
import random
from typing import Tuple

import numpy as np
from openmm import unit
from tqdm import tqdm

from endstate_correction.constant import distance_unit, temperature
from endstate_correction.system import get_positions


def perform_switching(
    sim, lambdas: list, samples: list, nr_of_switches: int = 50, save_traj: bool = False
) -> Tuple[list, list]:
    """performs NEQ switching using the lambda sheme passed from randomly dranw samples"""

    # list  of work values
    ws = []
    # list of conformations
    endstate_samples = []

    inst_switching = False
    if len(lambdas) == 2:
        print("Instantanious switching: dE will be calculated")
        inst_switching = True
    elif len(lambdas) < 2:
        raise RuntimeError("increase the number of lambda states")
    else:
        print("NEQ switching: dW will be calculated")

    # start with switch
    for _ in tqdm(range(nr_of_switches)):
        # select a random sample
        x = (
            np.array(random.choice(samples).value_in_unit(distance_unit))
            * distance_unit
        )
        # set position
        sim.context.setPositions(x)

        # reseed velocities
        sim.context.setVelocitiesToTemperature(temperature)
        # initialize work
        w = 0.0
        # perform NEQ switching
        for idx_lamb in range(1, len(lambdas)):
            # set lambda parameter
            sim.context.setParameter("lambda_interpolate", lambdas[idx_lamb])
            # test if neq or instantaneous swithching: if neq, perform integration step
            if not inst_switching:
                # perform 1 simulation step
                sim.step(1)
            # calculate work
            # evaluate u_t(x_t) - u_{t-1}(x_t)
            # calculate u_t(x_t)
            u_now = sim.context.getState(getEnergy=True).getPotentialEnergy()
            # calculate u_{t-1}(x_t)
            sim.context.setParameter("lambda_interpolate", lambdas[idx_lamb - 1])
            u_before = sim.context.getState(getEnergy=True).getPotentialEnergy()
            # add to accumulated work
            w += (u_now - u_before).value_in_unit(unit.kilojoule_per_mole)
            # TODO: expand to reduced potential
        if save_traj:
            endstate_samples.append(get_positions(sim))
        ws.append(w)
    return np.array(ws) * unit.kilojoule_per_mole, endstate_samples


def _collect_work_values(file: str) -> list:

    ws = pickle.load(open(file, "rb")).value_in_unit(unit.kilojoule_per_mole)
    number_of_samples = len(ws)
    print(f"Number of samples used: {number_of_samples}")
    return ws * unit.kilojoule_per_mole
