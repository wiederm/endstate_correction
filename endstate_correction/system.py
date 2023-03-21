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


def get_positions(sim):
    """get position of system in a state"""
    return sim.context.getState(getPositions=True).getPositions(asNumpy=True)


def get_energy(sim):
    """get energy of system in a state"""
    return sim.context.getState(getEnergy=True).getPotentialEnergy()
