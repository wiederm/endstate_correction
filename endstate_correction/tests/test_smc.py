from .test_neq import load_endstate_system_and_samples
from endstate_correction.smc import perform_SMC


def test_SMC():
    system_name = "ZINC00077329"
    print(f"{system_name=}")

    # load simulation and samples for ZINC00077329
    sim, samples_mm, _ = load_endstate_system_and_samples(
        system_name=system_name,
    )

    # perform SMC switching
    perform_SMC(sim=sim, nr_of_steps=100, samples=samples_mm, nr_of_particles=10)
