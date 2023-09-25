from openmm.app import (
    CharmmParameterSet,
    CharmmPsfFile,
    CharmmCrdFile,
    PDBFile,
)
import endstate_correction
import pathlib
from openmm.app import DCDReporter
from .test_system import setup_vacuum_simulation, setup_waterbox_simulation

# define path to test systems
path = pathlib.Path(endstate_correction.__file__).resolve().parent
hipen_testsystem = f"{path}/data/hipen_data"
jctc_testsystem = f"{path}/data/jctc_data"


def test_sampling():
    """Test if we can sample with simulation instance in vacuum and watervox"""
    from endstate_correction.system import (
        read_box,
    )

    ########################################################
    ########################################################
    # ----------------- vacuum-- ---------------------------

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
    sim.context.setPositions(crd.positions)
    sim.context.setVelocitiesToTemperature(300)
    sim.reporters.append(DCDReporter("test.dcd", 100))
    sim.step(1000)

    ########################################################
    ########################################################
    # ----------------- waterbox ---------------------------
    # get all relevant files and initialize SIMulation

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
    # define region that should be treated with the nnp
    chains = list(psf.topology.chains())
    ml_atoms = [atom.index for atom in chains[0].atoms()]
    # set up system
    sim = setup_waterbox_simulation(psf, params, ml_atoms)
    sim.context.setPositions(pdb.positions)
    sim.step(50)
