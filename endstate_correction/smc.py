# Perform SMC sampling for endstate correction

import numpy as np
from mdtraj import Trajectory
from openmm import unit
from openmm.app import Simulation
from scipy.special import logsumexp
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
        """

        self.sim = sim
        self.samples = samples
        self.logZ = 0.0

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
    def _propagate_particles(walkers, sim: Simulation):
        _intermediate_particles = []
        for p in walkers:
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
        weights = np.ones(nr_of_walkers) / nr_of_walkers
        # initialize lambda values
        lambdas = np.linspace(0, 1, nr_of_steps)

        # select initial samples
        random_frame_idxs = np.random.choice(
            len(self.samples.xyz) - 1, size=nr_of_walkers
        )
        walkers = [
            self.samples.openmm_positions(random_frame_idx)
            for random_frame_idx in random_frame_idxs
        ]

        assert len(walkers) == nr_of_walkers

        # start with switch
        for lamb_idx in tqdm(range(len(lambdas) - 1)):
            # set lambda parameter
            self.sim.context.setParameter("lambda_interpolate", lambdas[lamb_idx])
            # Propagate the walkers
            walkers = self._propagate_particles(walkers, self.sim)
            # calculate work
            # evaluate U_(\lamb_(i+1))(x_i) -  U_(\lamb_(i))(x_i)
            # calculate U_(\lamb_(i))(x_i)
            u_now = self._calculate_potential_E_for_particles(
                lambdas[lamb_idx], walkers, self.sim
            )
            # calculate U_(\lamb_(i+1))(x_i)
            u_future = self._calculate_potential_E_for_particles(
                lambdas[lamb_idx + 1], walkers, self.sim
            )
            # calculate weights (equation 2 in 10.1021/acs.jctc.1c01198)
            current_deltaEs = u_future - u_now  # calculate difference in energy
            current_weights = np.exp(
                np.nanmin(current_deltaEs) - current_deltaEs
            )  # subtract minimum to avoid numerical issues
            current_weights /= np.sum(current_weights)  # normalize
            # calculate ESS
            ESS = 1.0 / np.sum(current_weights**2)
            print(f"Effective Sample Size at lambda = {lambdas[lamb_idx]}: {ESS}")
            print(current_weights)
            # add to accumulated logZ
            self.logZ += logsumexp(-current_deltaEs) - np.log(nr_of_walkers)

            # Resample the particles based on the weights
            random_frame_idxs = np.random.choice(
                nr_of_walkers, size=nr_of_walkers, p=weights
            )
            walkers = [walkers[idx] for idx in random_frame_idxs]

        # reset lambda value
        self.sim.context.setParameter("lambda_interpolate", 0.0)
