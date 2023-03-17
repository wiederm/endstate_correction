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
from openmmtools.utils import get_fastest_platform

path = pathlib.Path(endstate_correction.__file__).resolve().parent
hipen_testsystem = f"{path}/data/hipen_data"
jctc_testsystem = f"{path}/data/jctc_data"


def setup_vacuum_simulation(
    psf: CharmmPsfFile, params: CharmmParameterSet
) -> Simulation:
    chains = list(psf.topology.chains())
    ml_atoms = [atom.index for atom in chains[0].atoms()]
    print(f"{ml_atoms=}")
    # set up system
    mm_system = psf.createSystem(params=params, nonbondedMethod=NoCutoff)
    potential = MLPotential("ani2x")
    ml_system = potential.createMixedSystem(
        psf.topology, mm_system, ml_atoms, interpolate=True
    )
    platform = get_fastest_platform(minimum_precision="mixed")
    return Simulation(psf.topology, ml_system, LangevinIntegrator(300, 1, 0.001), platform=platform)


def setup_waterbox_simulation(
    psf: CharmmPsfFile,
    params: CharmmParameterSet,
    r_off: float = 1.2,
    r_on: float = 0.,
) -> Simulation:
    chains = list(psf.topology.chains())
    ml_atoms = [atom.index for atom in chains[0].atoms()]
    print(f"{ml_atoms=}")
    mm_system = psf.createSystem(
        params,
        nonbondedMethod=PME,
        #nonbondedCutoff=r_off * unit.nanometers,
        #switchDistance=r_on * unit.nanometers,
    )

    # set up system
    mm_system = psf.createSystem(params=params, nonbondedMethod=NoCutoff)
    potential = MLPotential("ani2x")
    ml_system = potential.createMixedSystem(
        psf.topology, mm_system, ml_atoms, interpolate=True
    )
    platform = get_fastest_platform(minimum_precision="mixed")
    return Simulation(psf.topology, ml_system, LangevinIntegrator(300, 1, 0.001), platform=platform)


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
    # define region that should be treated with the qml
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
    # at lambda=1.0 (qml endpoint)
    sim.context.setParameter("lambda_interpolate", 1.0)
    e_sim_qml_endstate = get_energy(sim).value_in_unit(unit.kilojoule_per_mole)
    print(e_sim_qml_endstate)
    assert np.isclose(e_sim_qml_endstate, -5252411.066221259)

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
    # define region that should be treated with the qml
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
    # at lambda=1.0 (qml endpoint)
    sim.context.setParameter("lambda_interpolate", 1.0)
    e_sim_qml_endstate = get_energy(sim).value_in_unit(unit.kilojoule_per_mole)
    print(e_sim_qml_endstate)
    assert np.isclose(e_sim_qml_endstate, -1025774.735780582)

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
    # at lambda=1.0 (qml endpoint)
    sim.context.setParameter("lambda_interpolate", 1.0)
    e_sim_qml_endstate = get_energy(sim).value_in_unit(unit.kilojoule_per_mole)
    print(e_sim_qml_endstate)
    assert np.isclose(e_sim_qml_endstate, -1062775.8348574494)


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
