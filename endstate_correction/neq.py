"""Provide the functions for non-equilibrium switching."""
import os
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
    workdir: str = ".",
) -> Tuple[list, list]:
    """Perform NEQ or instantaneous switching using the provided lambda schema on the passed simulation instance.

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
        Tuple[list, list]: work or dE values, endstate samples
    """
    os.makedirs(workdir, exist_ok=True)
    if save_endstates:
        print("Endstate of each switch will be saved.")
    if save_trajs:
        print(f"Switching trajectory of each switch will be saved")

    # list  of work values
    ws = []
    # list for all endstate samples (can be empty if saving is not needed)
    endstate_samples = []

    inst_switching = False
    if len(lambdas) == 2:
        print("Instantanious switching: dE will be calculated")
        inst_switching = True
        if nr_of_switches == -1: # if no specific nr_of_switches is provided (-1 is the default value), use all provided equilibrium samples
            nr_of_switches = len(samples)
            print(f"{nr_of_switches} dE values will be calculated using all provided equilibrium samples")
        else:
            print(f"{nr_of_switches} dE values will be calculated using {nr_of_switches} random equilibrium samples")
    elif len(lambdas) < 2:
        raise RuntimeError("increase the number of lambda states")
    else:
        print("NEQ switching: dW will be calculated")

    # start with switch
    for switch_index in tqdm(range(nr_of_switches)):
        if inst_switching and nr_of_switches == len(samples): # if all samples should be used for instantanious switching
            coord = samples.openmm_positions(switch_index)
            if samples.unitcell_lengths is not None:
                box_length = samples.openmm_boxes(switch_index)
            else:
                box_length = None
        else: # if a specific number of instantaneous switches should be calculated, random conformations will be drawn from the provided equlibirum samples
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

        if save_trajs:
            # if switching trajectories need to be saved, create a list at the beginning
            # of each switch to save the starting conformation
            switching_trajectory = [get_positions(sim).value_in_unit(unit.nanometer)]

        # perform NEQ switching
        for idx_lamb in range(1, len(lambdas)):
            # set lambda parameter
            sim.context.setParameter("lambda_interpolate", lambdas[idx_lamb])
            if save_trajs and idx_lamb % 1000 == 0:
                # Save every 1000 steps
                switching_trajectory.append(
                    get_positions(sim).value_in_unit(unit.nanometer)
                )
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
            switching_trajectory.append(
                get_positions(sim).value_in_unit(unit.nanometer)
            )

            topology = samples.topology
            unitcell_lengths = samples[0].unitcell_lengths
            unitcell_angles = samples[0].unitcell_lengths
            switching_trajectory_length = len(switching_trajectory)
            if unitcell_lengths is None:
                switching_trajectory = Trajectory(
                    topology=topology,
                    xyz=np.stack(switching_trajectory),
                )
            else:
                switching_trajectory = Trajectory(
                    topology=topology,
                    xyz=np.stack(switching_trajectory),
                    unitcell_lengths=np.ones((switching_trajectory_length, 3))
                    * unitcell_lengths,
                    unitcell_angles=np.ones((switching_trajectory_length, 3))
                    * unitcell_angles,
                )
            switching_trajectory.save(
                f"{workdir}/switching_trajectory_{switch_index}.dcd"
            )
        if save_endstates:
            # save the endstate conformation
            endstate_samples.append(get_positions(sim).value_in_unit(unit.nanometer))
        # get all work values
        ws.append(w)
    return (
        np.array(ws) * unit.kilojoule_per_mole,
        endstate_samples
    )


def _collect_work_values(file: str) -> list:
    """Return a list of work values

    Args:
        file (str): pickle file containing work values

    Returns:
        list: list of work values in kJ/mol
    """
    ws = pickle.load(open(file, "rb")).value_in_unit(unit.kilojoule_per_mole)
    number_of_samples = len(ws)
    print(f"Number of samples used: {number_of_samples}")
    return ws * unit.kilojoule_per_mole
