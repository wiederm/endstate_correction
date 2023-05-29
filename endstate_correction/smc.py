# Perform SMC sampling for endstate correction

import numpy as np
from mdtraj import Trajectory
from openmm import unit
from openmm.app import Simulation
from tqdm import tqdm

from endstate_correction.constant import kBT, temperature


class SMC:
    def __init__(
        self,
        sim: Simulation,
        samples: Trajectory,
    ) -> None:
        """Initialize the SMC class

        Args:
            sim (Simulation): simulation instance
            samples (Trajectory): samples from which the starting points fo the SMC switching simulation are drawn
            nr_of_particles (int, optional): number of particles. Defaults to 100.
            nr_of_steps (int, optional): number of interpolation steps. Defaults to 1000.
        """

        self.sim = sim
        self.samples = samples

    @staticmethod
    def _calculate_potential_E_for_particles(
        lamb: float, particles, sim: Simulation
    ) -> np.ndarray:
        # set lambda parameter
        sim.context.setParameter("lambda_interpolate", lamb)

        u_intermediate = np.zeros(len(particles))
        # for each particle calculate the intermediate potential in kBT
        for p_idx, p in enumerate(particles):
            sim.context.setPositions(p)
            e_pot = sim.context.getState(getEnergy=True).getPotentialEnergy() / kBT
            u_intermediate[p_idx] = e_pot

        return u_intermediate

    @staticmethod
    def _propagate_particles(particles, sim: Simulation):
        _intermediate_particles = []
        for p in particles:
            sim.context.setPositions(p)
            sim.context.setVelocitiesToTemperature(temperature)
            sim.step(10)
            _intermediate_particles.append(
                sim.context.getState(getPositions=True).getPositions(asNumpy=True)
            )
        # update particles
        return _intermediate_particles

    def perform_SMC(
        self,
        nr_of_particles: int = 100,
        nr_of_steps: int = 1_000,
    ):
        # ------------------------- #
        # Outline of the algorithm:
        # ------------------------- #
        # initialize particles with random values
        # initialize weights to be uniform

        # FOR each lambda lamb in a sequence from 0 to 1:
        #     calculate the intermediate potential for each particle U = U_ref + lamb * (U_target - U_ref)
        #     calculate the intermediate gradient for each particle

        # update the weights based on the ratio of successive intermediate potentials
        # normalize the weights

        # resample the particles based on the weights

        # FOR each particle:
        #     initialize velocity from a Maxwell-Boltzmann distribution corresponding to the temperature
        #     update the state using Langevin dynamics, combining the gradient, the velocity, and Gaussian noise

        # After all steps are completed, calculate the free energy difference:
        #     compute the reference and target potentials for the final particles
        #     compute the unnormalized weights as the exponential of the difference between the reference and target potentials
        #     estimate the free energy difference as the negative logarithm of the average unnormalized weight
        # ------------------------- #

        # initialize weights
        weights = np.ones(nr_of_particles) / nr_of_particles
        # initialize lambda values
        lamb_values = np.linspace(0, 1, nr_of_steps)
        # initialize potential energy matrix
        pot_e = np.zeros((nr_of_particles, nr_of_steps))

        # select initial samples
        random_frame_idxs = np.random.choice(
            len(self.samples.xyz) - 1, size=nr_of_particles
        )
        particles = [
            self.samples.openmm_positions(random_frame_idx)
            for random_frame_idx in random_frame_idxs
        ]

        assert len(particles) == nr_of_particles
        sim = self.sim

        u_before = self._calculate_potential_E_for_particles(0.0, particles, sim)

        # for each lambda value calculate the intermediate potential for each particle
        # and update the weights based on the ratio of successive intermediate potentials
        for lamb in tqdm(lamb_values[1:]):  # exclude the first lambda value
            u_now = self._calculate_potential_E_for_particles(lamb, particles, sim)
            # update the weights based on the ratio of successive intermediate potentials
            log_weights = np.log(weights) + (
                u_now - u_before
            )  # NOTE: unsure if this is actually ture
            log_weights -= np.max(log_weights)
            weights = np.exp(log_weights)
            weights /= np.sum(weights)

        # Resample the particles based on the weights
        random_frame_idxs = np.random.choice(
            nr_of_particles, size=nr_of_particles, p=weights
        )
        particles = [
            particles[random_frame_idx] for random_frame_idx in random_frame_idxs
        ]

        # Propagate the particles
        particles = self._propagate_particles(particles, sim)

        # Calculate the free energy difference
        free_energy_diff = -np.log(np.mean(np.exp(weights)))

        print(free_energy_diff)

        return (free_energy_diff, pot_e)
