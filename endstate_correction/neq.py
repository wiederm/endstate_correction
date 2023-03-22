"""Provide the functions for non-equilibrium switching."""


import pickle
import random
from typing import Tuple

import numpy as np
from mdtraj import Trajectory
from openmm import unit
from openmm.app import Simulation
from tqdm import tqdm

from endstate_correction.constant import temperature
from endstate_correction.system import get_positions


def perform_switching(
    sim: Simulation,
    lambdas: list,
    samples: Trajectory,
    nr_of_switches: int = 50,
    save_trajs: bool = False,
    save_endstates: bool = False,
) -> Tuple[list, list, list]:
    """Perform NEQ switching using the provided lambda schema on the passed simulation instance.

    Args:
        sim (Simulation): simulation instance
        lambdas (list): list of lambda values
        samples (Trajectory): samples from which the starting points fo the NEQ switching simulation are drawn
        nr_of_switches (int, optional): number of switches. Defaults to 50.
        save_trajs (bool, optional): save switching trajectories. Defaults to False.
        save_endstates (bool, optional): save endstate of switching trajectory. Defaults to False.

    Raises:
        RuntimeError: if the number of lambda states is less than 2

    Returns:
        Tuple[list, list, list]: work values, endstate samples, switching trajectories
    """

    if save_endstates:
        print("Endstate of each switch will be saved.")
    if save_trajs:
        print(f"Switching trajectory of each switch will be saved")

    # list  of work values
    ws = []
    # list for all endstate samples (can be empty if saving is not needed)
    endstate_samples = []
    # list for all switching trajectories (can be empty if saving is not needed)
    all_switching_trajectories = []

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
        if save_trajs:
            # if switching trajectories need to be saved, create an empty list at the beginning
            # of each switch for saving conformations
            switching_trajectory = []

        # select a random frame
        random_frame_idx = random.randint(0, len(samples.xyz) - 1)
        # select the coordinates of the random frame
        coord = samples.openmm_positions(random_frame_idx)
        if samples.unitcell_lengths is not None:
            box_length = samples.openmm_boxes(random_frame_idx)
        else:
            box_length = None
        # set position
        sim.context.setPositions(coord)
        if box_length is not None:
            sim.context.setPeriodicBoxVectors(*box_length)
        # reseed velocities
        sim.context.setVelocitiesToTemperature(temperature)
        # initialize work
        w = 0.0

        # perform NEQ switching
        for idx_lamb in range(1, len(lambdas)):
            # set lambda parameter
            sim.context.setParameter("lambda_interpolate", lambdas[idx_lamb])
            if save_trajs:
                # save conformation at the beginning of each switch
                switching_trajectory.append(get_positions(sim))
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
        if save_trajs:
            # at the end of each switch save the last conformation
            switching_trajectory.append(get_positions(sim))
            # collect all switching trajectories as a list of lists
            all_switching_trajectories.append(switching_trajectory)
        if save_endstates:
            # save the endstate conformation
            endstate_samples.append(get_positions(sim))
        # get all work values
        ws.append(w)
    return (
        np.array(ws) * unit.kilojoule_per_mole,
        endstate_samples,
        all_switching_trajectories,
    )


def _collect_work_values(file: str) -> list:
    ws = pickle.load(open(file, "rb")).value_in_unit(unit.kilojoule_per_mole)
    number_of_samples = len(ws)
    print(f"Number of samples used: {number_of_samples}")
    return ws * unit.kilojoule_per_mole
