# Perform SMC sampling for endstate correction

import numpy as np
from mdtraj import Trajectory
from openmm import unit
from openmm.app import Simulation
from tqdm import tqdm

from endstate_correction.constant import kBT


def perform_SMC(
    sim: Simulation,
    samples: Trajectory,
    nr_of_particles: int = 100,
    nr_of_steps: int = 1_000,
) -> float:
    """Perform sequential MC to interpolate between reference and target potential.

    Args:
        sim (Simulation): simulation instance
        samples (Trajectory): samples from which the starting points fo the SMC switching simulation are drawn
        nr_of_particles (int, optional): number of particles. Defaults to 100.
        nr_of_steps (int, optional): number of interpolation steps. Defaults to 1000.

    Returns:
        np.array: weights of the particles
    """

    # list  of weights
    weights = np.ones(nr_of_particles) / nr_of_particles
    t_values = np.linspace(
        0, 1, nr_of_steps
    )  # temperature values for the intermediate potentials

    # select initial samples
    random_frame_idxs = np.random.choice(len(samples.xyz) - 1, size=nr_of_particles)

    # select the coordinates of the random frame
    particles = [
        samples.openmm_positions(random_frame_idx)
        for random_frame_idx in random_frame_idxs
    ]

    assert len(particles) == nr_of_particles
    # start with SMC
    for lamb in tqdm(t_values):
        # set lambda parameter
        sim.context.setParameter("lambda_interpolate", lamb)
        # calculate the current potentials for each particle
        u_intermediate = np.zeros(nr_of_particles)

        for p_idx, p in enumerate(particles):
            sim.context.setPositions(p)
            e_pot = sim.context.getState(getEnergy=True).getPotentialEnergy()
            print(e_pot / kBT)
            print(p_idx)
            u_intermediate[p_idx] = e_pot / kBT

        log_weights = (
            np.log(weights) + u_intermediate - np.roll(u_intermediate, shift=1)
        )
        log_weights -= np.max(log_weights)
        weights = np.exp(log_weights)
        weights /= np.sum(weights)

        # Resample the particles
        particles = np.random.choice(particles, size=nr_of_particles, p=weights)
        print(particles.shape)
        print(particles)
        # Propagate the particles
        _intermediate_particles = []
        for p_idx, p in enumerate(particles):
            sim.context.setPositions(p)
            sim.step(1)
            _intermediate_particles.append(
                sim.context.getState(getPositions=True).getPositions(asNumpy=True)
            )

        particles = _intermediate_particles

    # Calculate the free energy difference
    free_energy_diff = -np.log(np.mean(np.exp(weights)))

    print(free_energy_diff)

    return np.mean(u_intermediate)
