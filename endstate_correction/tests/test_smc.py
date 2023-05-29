from .test_neq import load_endstate_system_and_samples
from endstate_correction.smc import SMC
import pytest, os


@pytest.fixture
def _am_I_on_GH() -> bool:
    # will return True if on GH, and false locally
    if os.getenv("CI") == None:
        return False
    else:
        return True


def test_SMC(_am_I_on_GH):
    system_name = "ZINC00077329"
    print(f"{system_name=}")

    # load simulation and samples for ZINC00077329
    sim, samples_mm, _ = load_endstate_system_and_samples(
        system_name=system_name,
    )

    smc_sampler = SMC(sim=sim, samples=samples_mm)
    # perform SMC switching
    print("Performing SMC switching")
    if _am_I_on_GH == True:
        free_energy, pot_e = smc_sampler.perform_SMC(
            nr_of_steps=100, nr_of_particles=10
        )
        print(f"Am I on GH: {_am_I_on_GH}")

    else:
        free_energy, pot_e = smc_sampler.perform_SMC(
            nr_of_steps=1000, nr_of_particles=100
        )
        print(f"Am I on GH: {_am_I_on_GH}")
