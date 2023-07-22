# Perform SMC sampling for endstate correction

import logging
from typing import List

import numpy as np
from mdtraj import Trajectory
from openmm.app import Simulation
from scipy.special import logsumexp
from tqdm import tqdm
from itertools import accumulate
from endstate_correction.constant import kBT, temperature

logger = logging.getLogger(__name__)


class Resampler:
    def __init__(self):
        pass

    def stratified_resampling(self, samples: List, weights: List[float]) -> List:
        """Stratified resampling of the walkers based on the weights
        Implementation is taken and slightly modified from @msuruzhon's openmmslicer package
        https://github.com/openmmslicer/openmmslicer/blob/main/openmmslicer/resampling_methods.py


        Returns:
            list: resampled walkers
        """

        n_walkers = len(samples)
        weights = np.asarray(weights)
        weights /= sum(weights)

        # stratified resampling
        cdf = np.array([0] + list(np.cumsum(weights)))
        random_numbers = np.random.uniform(size=n_walkers) / n_walkers
        self.random_numbers = random_numbers
        rational_weights = np.linspace(0, 1, endpoint=False, num=n_walkers)
        all_cdf_points = random_numbers + rational_weights

        int_weights = [
            np.histogram(cdf_points, bins=cdf)[0] for cdf_points in all_cdf_points
        ]
        all_samples = [
            sum([i * [x] for i, x in zip(int_weight, samples)], [])
            for int_weight in int_weights
        ]
        # return resampled walkers
        return all_samples


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
        """

        self.sim = sim
        self.samples = samples
        self.logZ = 0.0
        self.resampler = Resampler()

    @staticmethod
    def _calculate_potential_E_for_particles(
        lamb: float, walkers, sim: Simulation
    ) -> np.ndarray:
        # set lambda parameter
        sim.context.setParameter("lambda_interpolate", lamb)

        u_intermediate = np.zeros(len(walkers))
        # for each particle calculate the intermediate potential in kBT
        for p_idx, p in enumerate(walkers):
            sim.context.setPositions(p)
            e_pot = sim.context.getState(getEnergy=True).getPotentialEnergy() / kBT
            u_intermediate[p_idx] = e_pot

        return u_intermediate

    @staticmethod
    def _propagate_walkers(walkers, sim: Simulation, nr_of_steps: int = 1_000):
        _intermediate_walkers = []
        for p in walkers:
            sim.context.setPositions(p)
            sim.context.setVelocitiesToTemperature(temperature)
            sim.step(nr_of_steps)
            _intermediate_walkers.append(
                sim.context.getState(getPositions=True).getPositions(asNumpy=True)
            )
        # update particles
        return _intermediate_walkers

    def _calculate_deltaEs(self, lamb_idx: int, walkers: list) -> np.ndarray:
        # calculate work
        # evaluate U_(\lamb_(i+1))(x_i) -  U_(\lamb_(i))(x_i)
        # calculate U_(\lamb_(i))(x_i)
        u_now = self._calculate_potential_E_for_particles(
            self.lambdas[lamb_idx], walkers, self.sim
        )
        # calculate U_(\lamb_(i+1))(x_i)
        u_future = self._calculate_potential_E_for_particles(
            self.lambdas[lamb_idx + 1], walkers, self.sim
        )
        # calculate weights (equation 2 in 10.1021/acs.jctc.1c01198)
        current_deltaEs = u_future - u_now  # calculate difference in energy
        return current_deltaEs

    def _calculate_weights(self, current_deltaEs: np.ndarray) -> np.ndarray:
        current_weights = np.exp(
            np.nanmin(current_deltaEs) - current_deltaEs
        )  # subtract minimum to avoid numerical issues
        current_weights /= np.sum(current_weights)  # normalize
        return current_weights

    def _calculate_effective_sample_size(
        self, current_weights: np.ndarray, lamb_idx: int
    ) -> float:
        # calculate ESS
        ESS = 1.0 / np.sum(current_weights**2)
        logger.info(
            f"Effective Sample Size at lambda = {self.lambdas[lamb_idx]}: {ESS}"
        )
        logger.info(current_weights)

    def perform_SMC(
        self,
        nr_of_walkers: int = 100,
        nr_of_steps: int = 1_000,
    ) -> None:
        """Perform SMC sampling

        Args:
            nr_of_walkers (int, optional): number of walkers. Defaults to 100.
            nr_of_steps (int, optional): number of interpolation steps. Defaults to 1000.
        """

        # ------------------------- #
        # Outline of the basic Sequential Importance Sampling (SIS) algorithm as outlined in
        # 10.1021/acs.jctc.1c01198
        # ------------------------- #

        # Goal: Calculate free energy estimate between reference distribution \pi(\lamb=0; x) and
        # target distribution \pi(\lamb=1; x) with resampling steps

        # Motivation: SMC might help focus the non equilibrium sampling on the region of interest;
        # might help with RBFE calculations in analogy to the work of 10.1101/2020.07.29.227959 and avoid sampling of the
        # target distribution \pi(\lamb=1; x) for bidirectional NEQ protocols

        # obtain IID samples from the reference distribution \pi(\lamb=0; x)
        # initialize weights to be uniform
        # select random subset of walkers from the samples
        # for each interpolation step i between \lamb=0 and \lamb=1
        #   propagate the walkers for a fixed number of steps with u(x; \lamb=i) to generate pool of IID samples
        #   for each walker j
        #       calculate the unnormalized weights w = exp(u(x; \lamb=i+1) - u(x; \lamb=i))
        #   resample the walkers according to the normalized weights

        # calculate the free energy estimate using the weights of each walker and the Zwanzig relation

        # initialize weights
        self.weights = np.ones(nr_of_walkers) / nr_of_walkers
        # initialize lambda values
        self.lambdas = np.linspace(0, 1, nr_of_steps)

        # select initial, equally spaced samples
        equally_spaces_idx = np.linspace(
            0, len(self.samples.xyz) - 1, nr_of_walkers, dtype=int
        )
        walkers = [
            self.samples.openmm_positions(frame_idx) for frame_idx in equally_spaces_idx
        ]

        assert len(walkers) == nr_of_walkers
        # start with switch
        for lamb_idx in tqdm(range(len(self.lambdas) - 1)):
            # set lambda parameter
            self.sim.context.setParameter("lambda_interpolate", self.lambdas[lamb_idx])
            # Propagate the walkers
            walkers = self._propagate_walkers(walkers, self.sim)
            current_deltaEs = self._calculate_deltaEs(lamb_idx, walkers)
            current_weights = self._calculate_weights(current_deltaEs)
            # report effective sample size
            self._calculate_effective_sample_size(current_weights, lamb_idx)
            # add to accumulated logZ
            self.logZ += logsumexp(-current_deltaEs) - np.log(nr_of_walkers)

            # Resample the particles based on the weights
            walkers = self.resampler.stratified_resampling(walkers, current_weights)
        # reset lambda value
        self.sim.context.setParameter("lambda_interpolate", 0.0)
