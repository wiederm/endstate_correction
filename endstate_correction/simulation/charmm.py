from openmm import unit
from openmm.app import CharmmParameterSet, CharmmPsfFile

from base import EndstateCorrectionBase
from ..system import read_box


class EndstateCorrectionCHARMM(EndstateCorrectionBase):
    def _get_mm_topology(self) -> CharmmPsfFile:
        psf = CharmmPsfFile(self.top.Psf)
        params = CharmmParameterSet(*self.top.ParameterSet)
        # set up the treatment of the system for the specific environment
        if self.env == "waterbox":
            psf = read_box(psf, self.top.input_config)
        return psf

    def createSystem(self, **kwargs):
        kwargs = {
            "params": CharmmParameterSet(*self.top.ParameterSet),
            "switchDistance": self.protocol.switchDistance * unit.nanometers,
            **kwargs,
        }
        return super().createSystem(env, **kwargs)
