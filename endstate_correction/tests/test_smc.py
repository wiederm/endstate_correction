from .test_neq import load_endstate_system_and_samples
from endstate_correction.smc import perform_SMC
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

    # perform SMC switching
    print("Performing SMC switching")
    print(_am_I_on_GH)
    if _am_I_on_GH() == True:
        free_energy, pot_e = perform_SMC(
            sim=sim, nr_of_steps=100, samples=samples_mm, nr_of_particles=10
        )
    else:
        free_energy, pot_e = perform_SMC(
            sim=sim, nr_of_steps=1000, samples=samples_mm, nr_of_particles=100
        )
