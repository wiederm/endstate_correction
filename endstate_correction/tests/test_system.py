import numpy as np
from openmm import unit
import endstate_correction
import pathlib
from openmm.app import (
    CharmmParameterSet,
    CharmmPsfFile,
    PDBFile,
    CharmmCrdFile,
    NoCutoff,
    Simulation,
    PME,
)
from openmm import LangevinIntegrator
from openmmml import MLPotential
import mdtraj as md
from typing import Tuple
import pytest

path = pathlib.Path(endstate_correction.__file__).resolve().parent
hipen_testsystem = f"{path}/data/hipen_data"
jctc_testsystem = f"{path}/data/jctc_data"


def load_endstate_system_and_samples(
    system_name: str,
) -> Tuple[Simulation, list, list]:
    """Test if samples can be loaded and system can be created

    Args:
        system_name (str): name of the system

    Returns:
        Tuple[Simulation, list, list]: instance of Simulation class, MM samples, NNP samples
    """
    # initialize simulation and load pre-generated samples

    from openmm.app import CharmmCrdFile, CharmmParameterSet, CharmmPsfFile

    ########################################################
    ########################################################
    # ----------------- vacuum -----------------------------
    # get all relevant files
    path = pathlib.Path(endstate_correction.__file__).resolve().parent
    hipen_testsystem = f"{path}/data/hipen_data"

    psf = CharmmPsfFile(f"{hipen_testsystem}/{system_name}/{system_name}.psf")
    crd = CharmmCrdFile(f"{hipen_testsystem}/{system_name}/{system_name}.crd")
    params = CharmmParameterSet(
        f"{hipen_testsystem}/top_all36_cgenff.rtf",
        f"{hipen_testsystem}/par_all36_cgenff.prm",
        f"{hipen_testsystem}/{system_name}/{system_name}.str",
    )
    # define region that should be treated with the nnp
    sim = setup_vacuum_simulation(psf=psf, params=params)
    sim.context.setPositions(crd.positions)
    n_samples = 5_000
    n_steps_per_sample = 1_000
    ###########################################################################################
    pdb_file = f"data/{system_name}/{system_name}.pdb"
    mm_samples = md.load_dcd(
        f"data/{system_name}/sampling_charmmff/run01/{system_name}_samples_{n_samples}_steps_{n_steps_per_sample}_lamb_0.0000.dcd",
        top=pdb_file,
    )

    nnp_samples = []
    nnp_samples = md.load_dcd(
        f"data/{system_name}/sampling_charmmff/run01/{system_name}_samples_{n_samples}_steps_{n_steps_per_sample}_lamb_1.0000.dcd",
        top=pdb_file,
    )

    return sim, mm_samples, nnp_samples


def setup_ZINC00077329_system():
    system_name = "ZINC00077329"
    print(f"{system_name=}")

    # load simulation and samples for ZINC00077329
    sim, samples_mm, samples_mm_nnp = load_endstate_system_and_samples(
        system_name=system_name,
    )
    return sim, samples_mm, samples_mm_nnp


def setup_vacuum_simulation(
    psf: CharmmPsfFile, params: CharmmParameterSet
) -> Simulation:
    """Test setup simulation in vacuum.

    Args:
        psf (CharmmPsfFile): topology instance
        params (CharmmParameterSet): parameter

    Returns:
        Simulation: instance of Simulation class
    """
    chains = list(psf.topology.chains())
    ml_atoms = [atom.index for atom in chains[0].atoms()]
    print(f"{ml_atoms=}")
    # set up system
    mm_system = psf.createSystem(params=params, nonbondedMethod=NoCutoff)
    potential = MLPotential("ani2x")
    ml_system = potential.createMixedSystem(
        psf.topology, mm_system, ml_atoms, interpolate=True
    )
    return Simulation(psf.topology, ml_system, LangevinIntegrator(300, 1, 0.001))


def setup_waterbox_simulation(
    psf: CharmmPsfFile,
    params: CharmmParameterSet,
    r_off: float = 1.2,
    r_on: float = 0.0,
) -> Simulation:
    """Test setup simulation in waterbox.

    Args:
        psf (CharmmPsfFile): topology instance
        params (CharmmParameterSet): parameter
        r_off (float, optional): _description_. Defaults to 1.2.
        r_on (float, optional): _description_. Defaults to 0.0.

    Returns:
        Simulation: instance of Simulation class
    """
    chains = list(psf.topology.chains())
    ml_atoms = [atom.index for atom in chains[0].atoms()]
    print(f"{ml_atoms=}")
    mm_system = psf.createSystem(
        params,
        nonbondedMethod=PME,
        # nonbondedCutoff=r_off * unit.nanometers,
        # switchDistance=r_on * unit.nanometers,
    )

    # set up system
    mm_system = psf.createSystem(params=params, nonbondedMethod=NoCutoff)
    potential = MLPotential("ani2x")
    ml_system = potential.createMixedSystem(
        psf.topology, mm_system, ml_atoms, interpolate=True
    )
    return Simulation(psf.topology, ml_system, LangevinIntegrator(300, 1, 0.001))


def test_initializing_ZINC00077329_system():
    sim, samples_mm, samples_mm_nnp = setup_ZINC00077329_system()
    assert len(samples_mm) == 5000


def test_generate_simulation_instances_with_charmmff():
    """Test if we can generate a simulation instance with charmmff"""
    from endstate_correction.system import get_energy, read_box

    ########################################################
    ########################################################
    # ----------------- vacuum -----------------------------
    # get all relevant files

    system_name = "ZINC00079729"
    psf = CharmmPsfFile(f"{hipen_testsystem}/{system_name}/{system_name}.psf")
    crd = CharmmCrdFile(f"{hipen_testsystem}/{system_name}/{system_name}.crd")
    params = CharmmParameterSet(
        f"{hipen_testsystem}/top_all36_cgenff.rtf",
        f"{hipen_testsystem}/par_all36_cgenff.prm",
        f"{hipen_testsystem}/{system_name}/{system_name}.str",
    )
    # define region that should be treated with the nnp
    sim = setup_vacuum_simulation(psf, params)
    # set up system
    sim.context.setPositions(crd.positions)

    ############################
    ############################
    # check potential at endpoints
    # at lambda=0.0 (mm endpoint)
    sim.context.setParameter("lambda_interpolate", 0.0)
    e_sim_mm_interpolate_endstate = get_energy(sim).value_in_unit(
        unit.kilojoule_per_mole
    )
    print(e_sim_mm_interpolate_endstate)
    assert np.isclose(e_sim_mm_interpolate_endstate, 156.9957913623657)

    ############################
    ############################
    # at lambda=1.0 (nnp endpoint)
    sim.context.setParameter("lambda_interpolate", 1.0)
    e_sim_nnp_endstate = get_energy(sim).value_in_unit(unit.kilojoule_per_mole)
    print(e_sim_nnp_endstate)
    assert np.isclose(e_sim_nnp_endstate, -5252411.066221259)

    ########################################################
    ########################################################
    # ----------------- vacuum -----------------------------
    # get all relevant files
    system_name = "1_octanol"
    psf = CharmmPsfFile(f"{jctc_testsystem}/{system_name}/charmm-gui/openmm/vac.psf")
    pdb = PDBFile(f"{jctc_testsystem}/{system_name}/charmm-gui/openmm/vac.pdb")
    params = CharmmParameterSet(
        f"{jctc_testsystem}/{system_name}/charmm-gui/unk/unk.rtf",
        f"{jctc_testsystem}/{system_name}/charmm-gui/unk/unk.prm",
        f"{jctc_testsystem}/toppar/top_all36_cgenff.rtf",
        f"{jctc_testsystem}/toppar/par_all36_cgenff.prm",
        f"{jctc_testsystem}/toppar/toppar_water_ions.str",
    )
    # define region that should be treated with the nnp
    sim = setup_vacuum_simulation(psf, params)
    sim.context.setPositions(pdb.positions)

    ############################
    ############################
    # check potential at endpoints
    # at lambda=0.0 (mm endpoint)
    sim.context.setParameter("lambda_interpolate", 0.0)
    e_sim_mm_interpolate_endstate = get_energy(sim).value_in_unit(
        unit.kilojoule_per_mole
    )
    print(e_sim_mm_interpolate_endstate)
    assert np.isclose(e_sim_mm_interpolate_endstate, 316.4088125228882)

    ############################
    ############################
    # at lambda=1.0 (nnp endpoint)
    sim.context.setParameter("lambda_interpolate", 1.0)
    e_sim_nnp_endstate = get_energy(sim).value_in_unit(unit.kilojoule_per_mole)
    print(e_sim_nnp_endstate)
    assert np.isclose(e_sim_nnp_endstate, -1025774.735780582)

    ########################################################
    ########################################################
    # ----------------- waterbox ---------------------------
    # get all relevant files
    system_name = "1_octanol"
    psf = CharmmPsfFile(
        f"{jctc_testsystem}/{system_name}/charmm-gui/openmm/step3_input.psf"
    )
    pdb = PDBFile(f"{jctc_testsystem}/{system_name}/charmm-gui/openmm/step3_input.pdb")
    params = CharmmParameterSet(
        f"{jctc_testsystem}/{system_name}/charmm-gui/unk/unk.rtf",
        f"{jctc_testsystem}/{system_name}/charmm-gui/unk/unk.prm",
        f"{jctc_testsystem}/toppar/top_all36_cgenff.rtf",
        f"{jctc_testsystem}/toppar/par_all36_cgenff.prm",
        f"{jctc_testsystem}/toppar/toppar_water_ions.str",
    )
    psf = read_box(psf, f"{jctc_testsystem}/{system_name}/charmm-gui/input.config.dat")
    sim = setup_waterbox_simulation(psf, params)
    sim.context.setPositions(pdb.positions)

    ############################
    ############################
    # check potential at endpoints
    # at lambda=0.0 (mm endpoint)
    sim.context.setParameter("lambda_interpolate", 0.0)
    e_sim_mm_interpolate_endstate = get_energy(sim).value_in_unit(
        unit.kilojoule_per_mole
    )
    print(e_sim_mm_interpolate_endstate)
    assert np.isclose(e_sim_mm_interpolate_endstate, -36663.29543876648)

    ############################
    ############################
    # at lambda=1.0 (nnp endpoint)
    sim.context.setParameter("lambda_interpolate", 1.0)
    e_sim_nnp_endstate = get_energy(sim).value_in_unit(unit.kilojoule_per_mole)
    print(e_sim_nnp_endstate)
    assert np.isclose(e_sim_nnp_endstate, -1062775.8348574494)


def test_simulating():
    """Test if we can generate a simulation instance with charmmff"""

    ########################################################
    ########################################################
    # ----------------- vacuum -----------------------------
    # get all relevant files

    system_name = "ZINC00079729"
    psf = CharmmPsfFile(f"{hipen_testsystem}/{system_name}/{system_name}.psf")
    crd = CharmmCrdFile(f"{hipen_testsystem}/{system_name}/{system_name}.crd")
    params = CharmmParameterSet(
        f"{hipen_testsystem}/top_all36_cgenff.rtf",
        f"{hipen_testsystem}/par_all36_cgenff.prm",
        f"{hipen_testsystem}/{system_name}/{system_name}.str",
    )
    sim = setup_vacuum_simulation(psf, params)
    sim.context.setPositions(crd.positions)
    sim.step(100)
