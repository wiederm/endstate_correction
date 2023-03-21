"""Provide the functions for non-equilibrium switching."""


import pickle
import random
from typing import Tuple

import numpy as np
from openmm import unit
from tqdm import tqdm

from endstate_correction.constant import distance_unit, temperature
from endstate_correction.system import get_positions
import openmm


def gen_box(positions: np.array) -> openmm.unit.quantity.Quantity:

    coords = positions

    min_crds = [coords[0][0], coords[0][1], coords[0][2]]
    max_crds = [coords[0][0], coords[0][1], coords[0][2]]

    for coord in coords:
        min_crds[0] = min(min_crds[0], coord[0])
        min_crds[1] = min(min_crds[1], coord[1])
        min_crds[2] = min(min_crds[2], coord[2])
        max_crds[0] = max(max_crds[0], coord[0])
        max_crds[1] = max(max_crds[1], coord[1])
        max_crds[2] = max(max_crds[2], coord[2])

    boxlx = max_crds[0] - min_crds[0]
    boxly = max_crds[1] - min_crds[1]
    boxlz = max_crds[2] - min_crds[2]

    return boxlx, boxly, boxlz


def perform_switching(
    sim, lambdas: list, 
    samples: list, 
    nr_of_switches: int = 50, 
    save_trajs: bool = False,
    save_endstates:bool = False,
) -> Tuple[list, list, list]:
    """performs NEQ switching using the lambda sheme passed from randomly dranw samples"""
   
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

        # select a random sample
        x = (
            np.array(random.choice(samples).value_in_unit(distance_unit))
            * distance_unit
        )

        a, b, c = gen_box(x)

        vecA = openmm.Vec3(a.value_in_unit(unit.nanometer), 0, 0)
        vecB = openmm.Vec3(0, b.value_in_unit(unit.nanometer), 0)
        vecC = openmm.Vec3(0, 0, c.value_in_unit(unit.nanometer))

        sim.context.setPeriodicBoxVectors(vecA, vecB, vecC)
        print(f"Using this BoxVectors:", sim.context.getState().getPeriodicBoxVectors())
        # set position
        sim.context.setPositions(x)

        # reseed velocities
        try:
            sim.context.setVelocitiesToTemperature(temperature)
        except openmm.OpenMMException:
            from endstate_correction.equ import _seed_velocities, _get_masses

            sim.context.setVelocities(_seed_velocities(_get_masses(sim.system)))
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
    return np.array(ws) * unit.kilojoule_per_mole, endstate_samples, all_switching_trajectories


def _collect_work_values(file: str) -> list:

    ws = pickle.load(open(file, "rb")).value_in_unit(unit.kilojoule_per_mole)
    number_of_samples = len(ws)
    print(f"Number of samples used: {number_of_samples}")
    return ws * unit.kilojoule_per_mole
