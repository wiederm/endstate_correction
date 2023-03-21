"""Provide the functions for equilibrium sampling."""

from typing import Tuple
from openmm import unit
import numpy as np
from openmm.app import Simulation
from tqdm import tqdm
from endstate_correction.constant import kBT


def _collect_equ_samples(
    trajs: list, every_nth_frame: int = 10
) -> Tuple[list, np.array]:
    """
    Given a list of k trajectories with n samples a dictionary with the number of samples per trajektory
    and a list with all samples [n_1, n_2, ...] is generated

    Returns:
        Tuple(coordinates, N_k)
    """

    coordinates = []
    N_k = np.zeros(len(trajs))

    # loop over lambda scheme and collect samples in nanometer
    for idx, xyz in enumerate(trajs):
        xyz = xyz[1_000:]  # remove the first 1k samples
        xyz = xyz[::every_nth_frame]  # take only every nth sample
        N_k[idx] = len(xyz)
        coordinates.extend([c_.value_in_unit(unit.nanometer) for c_ in xyz])

    number_of_samples = len(coordinates)
    print(f"Number of samples loaded: {number_of_samples}")
    return coordinates * unit.nanometer, N_k


def calculate_u_kn(
    trajs: list,  # list of trajectories
    sim: Simulation,
    every_nth_frame: int = 10,  # prune the samples further by taking only every nth sample
) -> np.ndarray:
    """
    Calculate the u_kn matrix to be used by the mbar estimator

    Args:
        trajs (list): list of trajectories
        sim (Simulation): simulation object
        every_nth_frame (int, optional): prune the samples further by taking only every nth sample. Defaults to 2.
    Returns:
        np.ndarray: u_kn matrix
    """

    lambda_scheme = np.linspace(0, 1, 11)  # equilibrium lambda scheme
    samples, N_k = _collect_equ_samples(trajs, every_nth_frame)  # collect samples

    samples = np.array(samples.value_in_unit(unit.nanometer))  # positions in nanometer
    u_kn = np.zeros(
        (len(N_k), int(N_k[0] * len(N_k))), dtype=np.float64
    )  # NOTE: assuming that N_k[0] is the maximum number of samples drawn from any state k
    for k, lamb in enumerate(lambda_scheme):
        sim.context.setParameter("lambda_interpolate", lamb)
        us = []
        for x in tqdm(range(len(samples))):
            sim.context.setPositions(samples[x])
            u_ = sim.context.getState(getEnergy=True).getPotentialEnergy()
            us.append(u_)
        us = np.array([u / kBT for u in us], dtype=np.float64)
        u_kn[k] = us

    # total number of samples
    total_nr_of_samples = 0
    for n in N_k:
        total_nr_of_samples += n

    assert total_nr_of_samples > 20  # make sure that there are samples present

    return (N_k, u_kn)
