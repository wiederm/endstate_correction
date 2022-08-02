# implement everything related to NCMC
from endstate_correction.neq import perform_switching
from endstate_correction.equ import generate_samples
from openmm.app import Simulation
import numpy as np


def perform_ncmc_simulations(sim: Simulation):

    # initiallize sampling from initial conformation

    # samples for 1 ns
    sample = generate_samples(sim, n_samples=1, n_steps_per_sample=1_000_000)
    # (START): use sample after 1 ns to initialize a NEQ switching protocoll (with propagation and perturbation kernal)
    # from a source level of theory to a target level of theory and collect work values
    lambdas = np.linspace(0, 1, 5_000)
    w, sample = perform_switching(
        sim, lambdas, sample, nr_of_switches=1, save_traj=True
    )
    # both are lists with only one element
    w = w[0], sample = sample[0]
    # use work value in metropolis criteria to either acceptor or reject the new conformation at the target level of theory
    # if accepted: this new conformation is now used with the target level of theory to continue sampling (for 1ps), then (START) again

    # if rejected: neither the conformation nor the work value are used, the velocity reversed initial conformation is used to continue sampling (for 1ps), then (START) again
