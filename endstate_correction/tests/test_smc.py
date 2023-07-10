import os

import numpy as np
import pytest
from scipy.special import logsumexp

from endstate_correction.smc import SMC

from .test_system import setup_ZINC00077329_system


@pytest.fixture
def _am_I_on_GH() -> bool:
    # will return True if on GH, and false locally
    if os.getenv("CI") == None:
        return False
    else:
        return True


def test_SMC_algorithm():
    # Define the potential function
    def u(x, lam):
        return -(x**2) + 100 * lam

    # Function to evolve the configuration. In this toy example,
    # let's assume it's a Gaussian random walk with standard deviation of 1.0
    def evolve_configuration(x):
        return x + np.random.normal(0, 1)

    N_samples = 1000  # Number of samples
    N_steps = 100  # Number of steps from lam=0 to lam=1

    # Generate initial configurations
    configurations = np.random.normal(0, 1, N_samples)
    current_deltaEs = np.ones(N_samples)
    logZ = 0.0
    for i in range(N_steps):
        lamb = i / N_steps
        next_lamb = (i + 1) / N_steps

        # importance sampling
        for j in range(N_samples):
            old_x = configurations[j]
            new_x = evolve_configuration(old_x)
            configurations[j] = new_x

            # adjust weights based on energy difference
            current_deltaEs[j] = u(new_x, next_lamb) - u(new_x, lamb)
        current_weights = np.exp(
            np.nanmin(current_deltaEs) - current_deltaEs
        )  # subtract minimum to avoid numerical issues
        current_weights /= np.sum(current_weights)  # normalize

        # calculate ESS
        ESS = 1.0 / np.sum(current_weights**2)
        print(f"Effective Sample Size at lambda = {lamb}: {ESS}")

        # resampling
        indexes = np.random.choice(range(N_samples), size=N_samples, p=current_weights)
        configurations = configurations[indexes]

        # calculate free energy difference
        logZ += logsumexp(-current_deltaEs) - np.log(N_samples)

    assert np.isclose(logZ, -100.0, atol=0.1)


def test_SMC(_am_I_on_GH):
    sim, samples_mm, samples_mm_qml = setup_ZINC00077329_system()
    smc_sampler = SMC(sim=sim, samples=samples_mm)
    # perform SMC switching
    print("Performing SMC switching")
    if _am_I_on_GH == True:
        smc_sampler.perform_SMC(nr_of_steps=1, nr_of_walkers=2)
    else:
        smc_sampler.perform_SMC(nr_of_steps=5, nr_of_walkers=100)
    print(smc_sampler.logZ)


def test_SMC_stratified_resampling():
    from endstate_correction.smc import Resampler

    N_samples = 1000  # Number of samples
    samples = np.arange(N_samples)

    # Generate initial weights
    weights = abs(np.random.normal(1, 1, N_samples))

    resampler = Resampler()

    resampled_samples = resampler.stratified_resampling(samples, weights)

    assert len(resampled_samples) == len(samples)
    resampled_samples_with_uniform_weights_try1 = resampler.stratified_resampling(
        samples, 0.1 * np.ones(N_samples)
    )
    resampled_samples_with_uniform_weights_try2 = resampler.stratified_resampling(
        samples, 0.5 * np.ones(N_samples)
    )
    
    assert np.allclose(resampled_samples_with_uniform_weights_try1, resampled_samples_with_uniform_weights_try2)
