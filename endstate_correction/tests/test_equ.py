import glob
import pathlib
import pickle

import endstate_correction
import mdtraj as md
import numpy as np
from endstate_correction.equ import calculate_u_kn
from openmm import unit
from openmm.app import CharmmParameterSet, CharmmPsfFile
from pymbar import MBAR
from openmm.app import Simulation
from typing import Tuple
from .test_system import setup_vacuum_simulation

def load_equ_samples(
    system_name: str,
) -> Tuple[Simulation, list]:
    """Helper function that loads trajectories from the test data"""
    nr_of_samples = 5_000
    nr_of_steps = 1_000
    trajs = []
    path = f"data/{system_name}/sampling_charmmff/run01/"
    for lamb in np.linspace(0, 1, 11):
        print(lamb)
        file = glob.glob(
            f"{path}/{system_name}_samples_{nr_of_samples}_steps_{nr_of_steps}_lamb_{lamb:.4f}.dcd"
        )
        if len(file) == 2:
            raise RuntimeError("Multiple traj files present. Abort.")
        if len(file) == 0:
            raise RuntimeError(
                "WARNING! Incomplete equ sampling. Proceed with cautions."
            )

        trajs.append(md.open(file[0]).read()[0] * unit.angstrom)
        print(trajs[-1][0])
    return trajs


def test_collect_equ_samples():
    """test if we are able to collect samples as anticipated"""
    from endstate_correction.equ import _collect_equ_samples

    name = "ZINC00077329"
    # load trajs
    trajs = load_equ_samples(name)
    # collect samples from traj
    samples, N_k = _collect_equ_samples(trajs)
    # test that tendstates have the correct number of samples (remove the first 20%, then take every 10th frame)
    assert N_k[0] == 400
    assert N_k[-1] == 400
    assert len(samples) == 4400


def test_equilibrium_free_energy():
    "test that u_kn can be calculated and that results are consistent whether we reload mbar pickle or regernerate it"
    ########################################################
    ########################################################
    # ----------------- vacuum -----------------------------
    # get all relevant files and initialize sim
    path = pathlib.Path(endstate_correction.__file__).resolve().parent
    hipen_testsystem = f"{path}/data/hipen_data"

    system_name = "ZINC00077329"
    psf = CharmmPsfFile(f"{hipen_testsystem}/{system_name}/{system_name}.psf")
    params = CharmmParameterSet(
        f"{hipen_testsystem}/top_all36_cgenff.rtf",
        f"{hipen_testsystem}/par_all36_cgenff.prm",
        f"{hipen_testsystem}/{system_name}/{system_name}.str",
    )
    sim = setup_vacuum_simulation(psf, params)
    # load samples
    trajs = load_equ_samples(system_name)
    # calculate u_kn
    N_k, u_kn = calculate_u_kn(sim=sim, trajs=trajs, every_nth_frame=50)

    # calculate free energy
    mbar = MBAR(u_kn, N_k)
    f = mbar.compute_free_energy_differences()
    print(f)
    assert np.isclose(mbar.f_k[-1], f["Delta_f"][0][-1])
    assert np.isclose(f["Delta_f"][0][-1], -940544.0390218807, rtol=1e-06)

    # save N_k and u_kn
    pickle_path = f"{path}/mbar_20.pickle"
    pickle.dump((N_k, u_kn), open(pickle_path, "wb"))

    # reload
    (N_k, u_kn) = pickle.load(open(pickle_path, "rb"))
    # calculate again
    mbar = MBAR(u_kn, N_k)
    f = mbar.compute_free_energy_differences()
    assert np.isclose(mbar.f_k[-1], f["Delta_f"][0][-1])
    assert np.isclose(f["Delta_f"][0][-1], -940544.0390218807, rtol=1e-06)
