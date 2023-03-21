# general imports
import json

from openmm import unit
from openmm.app import (
    CharmmPsfFile,
    CharmmCrdFile,
)


def gen_box(psf: CharmmPsfFile, crd: CharmmCrdFile) -> CharmmPsfFile:
    """
    Function to create psf file containing information about the box used (only for waterbox or commplex simulations). Usful
    when information about box size is not available (e.g. when using TF)
    """
    coords = crd.positions

    min_crds = [coords[0][0], coords[0][1], coords[0][2]]
    max_crds = [coords[0][0], coords[0][1], coords[0][2]]

    for coord in coords:
        min_crds[0] = min(min_crds[0], coord[0])
        min_crds[1] = min(min_crds[1], coord[1])
        min_crds[2] = min(min_crds[2], coord[2])
        max_crds[0] = max(max_crds[0], coord[0])
        max_crds[1] = max(max_crds[1], coord[1])
        max_crds[2] = max(max_crds[2], coord[2])

    boxlx = max_crds[0] - min_crds[0]
    boxly = max_crds[1] - min_crds[1]
    boxlz = max_crds[2] - min_crds[2]

    psf.setBox(boxlx, boxly, boxlz)
    return psf


def read_box(psf: CharmmPsfFile, filename: str) -> CharmmPsfFile:
    """set waterbox dimensions given the sysinfo.dat file containing the boxlengths (x,y,z) provided by CHARMM-GUI

    Args:
        psf (CharmmPsfFile): topology instance
        filename (str): filename for sysinfo file

    Returns:
        CharmmPsfFile: topology instance with box dimensions set
    """

    try:
        sysinfo = json.load(open(filename, "r"))
        boxlx, boxly, boxlz = map(float, sysinfo["dimensions"][:3])
    except:
        for line in open(filename, "r"):
            segments = line.split("=")
            if segments[0].strip() == "BOXLX":
                boxlx = float(segments[1])
            if segments[0].strip() == "BOXLY":
                boxly = float(segments[1])
            if segments[0].strip() == "BOXLZ":
                boxlz = float(segments[1])
    psf.setBox(boxlx * unit.angstroms, boxly * unit.angstroms, boxlz * unit.angstroms)
    return psf


# def create_charmm_system(
#     psf: CharmmPsfFile,
#     parameters: CharmmParameterSet,
#     env: str,
#     ml_atoms: list,
#     r_off: int = 1.0,
#     r_on: int = 0,
# ) -> Simulation:
#     """Generate an openMM simulation object using CHARMM topology and parameter files

#     Args:
#         psf (CharmmPsfFile): topology instance
#         parameters (CharmmParameterSet): parameter instance
#         env (str): either complex, waterbox or vacuum
#         ml_atoms (list): list of atoms described by the QML potential


#     Returns:
#         Simulation: openMM simulation instance
#     """

#     ###################
#     print(f"Generating charmm system in {env}")
#     platform = get_fastest_platform(minimum_precision='mixed')

#     ###################
#     print(f"{platform=}")
#     print(f"{env=}")
#     ###################
#     # TODO: add additional parameters for complex
#     if env == "vacuum":
#         mm_system = psf.createSystem(parameters, nonbondedMethod=NoCutoff)
#     else:
#         mm_system = psf.createSystem(
#             parameters,
#             nonbondedMethod=PME,
#             nonbondedCutoff=r_off * unit.nanometers,
#             switchDistance=r_on * unit.nanometers,
#         )

#     print(f"{ml_atoms=}")

#     #####################
#     potential = MLPotential("ani2x")
#     ml_system = potential.createMixedSystem(
#         psf.topology, mm_system, ml_atoms, interpolate=True
#     )
#     #####################

#     integrator = mm.LangevinIntegrator(temperature, collision_rate, stepsize)

#     return Simulation(psf.topology, ml_system, integrator, platform=platform)


def get_positions(sim):
    """get position of system in a state"""
    return sim.context.getState(getPositions=True).getPositions(asNumpy=True)


def get_energy(sim):
    """get energy of system in a state"""
    return sim.context.getState(getEnergy=True).getPotentialEnergy()
