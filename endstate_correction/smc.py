# Perform SMC sampling for endstate correction

import random
from typing import Tuple

import numpy as np
from mdtraj import Trajectory
from openmm import unit
from openmm.app import Simulation
from tqdm import tqdm

from endstate_correction.constant import temperature
from endstate_correction.system import get_positions


def perform_SMC(
    sim: Simulation,
    nr_of_steps: int = 1_000,
    samples: Trajectory,
    nr_of_particles: int = 1_000,
) -> float:
    """Perform sequential MC to interpolate between reference and target potential.

    Args:
        sim (Simulation): simulation instance
        samples (Trajectory): samples from which the starting points fo the NEQ switching simulation are drawn
        nr_of_steps (int, optional): number of interpolation steps. Defaults to 1000.
        nr_of_particles (int, optional): number of particles. Defaults to 1000.

    Returns:
        np.array: weights of the particles
    """

    # list  of weights
    weights = np.ones(nr_of_particles) / nr_of_particles
    t_values = np.linspace(0, 1, nr_of_steps)  # temperature values for the intermediate potentials

    # start with SMC
    for lamb in tqdm(t_values):
        # set lambda parameter
        sim.context.setParameter("lambda_interpolate", lamb)
        # calculate the current potentials for each particle
        u_intermediate = np.zeros(nr_of_particles)
        
        for p_idx, p in enumerate(samples):
            sim.context.setPositions(p)
            e_pot = sim.context.getState(getEnergy=True).getPotentialEnergy()            
            u_intermediate[p_idx] = e_pot
        
        log_weights = np.log(weights) + u_intermediate - np.roll(u_intermediate, shift=1)
        log_weights -= np.max(log_weights)
        weights = np.exp(log_weights)
        weights /= np.sum(weights)
        
        # Resample the particles
        indices = np.random.choice(nr_of_particles, size=nr_of_particles, p=weights)
        particles = particles[indices]

        # Propagate the particles
        _intermediate_samples = []
        for p_idx, p in enumerate(samples):
            sim.context.setPositions(p)
            sim.step(1)
            _intermediate_samples.append(sim.context.getState(getPositions=True).getPositions(asNumpy=True))
        
        samples = _intermediate_samples

    # Calculate the free energy difference
    free_energy_diff = -np.log(np.mean(np.exp(weights))))

    print(free_energy_diff)
 
    return np.mean(u_intermediate)