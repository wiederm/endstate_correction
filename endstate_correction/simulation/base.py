import abc
import logging
import mdtraj
import os
from mdtraj.core.trajectory import Trajectory as mdtraj_trajectory
from mdtraj.reporters import HDF5Reporter
from openmm import Integrator, LangevinIntegrator, OpenMMException, Platform
from openmm import unit
from openmm.app import (
    AmberPrmtopFile,
    CharmmPsfFile,
    NoCutoff,
    PME,
    Simulation,
    Topology,
)
from openmmml import MLPotential
from typing import List, Literal, Union

from ..protocol import BSSProtocol
from ..topology import AMBERTopology, CHARMMTopology
from openmmtools.utils import get_fastest_platform


class EndstateCorrectionBase(abc.ABC):
    def __init__(
        self,
        top: Union[AMBERTopology, CHARMMTopology],
        env: Literal["complex", "waterbox", "vacuum"],
        ml_atoms: List[int],
        protocol: BSSProtocol,
        name: str = "endstate_correction",
        work_dir: str = "./",
        potential: str = "ani2x",
        implementation: str = "nnpops",
        interpolate: bool = True,
    ):
        self.logger = logging.getLogger("EndstateCorrectionBase")
        self.top = top
        self.protocol = protocol
        self.env = env
        self.name = name
        self.work_dir = work_dir
        self._mm_topology = None
        self._initial_coordinates = None

        mm_system = self._createSystem()
        ml_potential = MLPotential(potential)
        ml_system = ml_potential.createMixedSystem(
            self.get_mm_topology().topology,
            mm_system,
            ml_atoms,
            interpolate=interpolate,
            implementation="nnpops",
        )
        integrator = self.get_integrator()
        platform = get_fastest_platform(minimum_precision="mixed")
        self.simulation = Simulation(
            self.get_mm_topology().topology, ml_system, integrator, platform=platform
        )

    def get_mm_topology(self) -> Topology:
        if self._mm_topology is None:
            self._mm_topology = self._get_mm_topology()
        return self._mm_topology

    def get_initial_coordinates(self) -> unit.Quantity:
        if self._initial_coordinates is None:
            self._initial_coordinates = self._get_initial_coordinates()
        return self._initial_coordinates

    @abc.abstractmethod
    def _get_mm_topology(self) -> Union[CharmmPsfFile, AmberPrmtopFile]:
        pass

    @abc.abstractmethod
    def _get_initial_coordinates(self) -> unit.Quantity:
        pass

    def get_integrator(self) -> Integrator:
        return LangevinIntegrator(
            self.protocol.temperature,
            self.protocol.collision_rate,
            self.protocol.timestep,
        )

    def _createSystem(self, **kwargs):
        if self.env == "vacuum":
            mm_system = self.get_mm_topology().createSystem(
                nonbondedMethod=NoCutoff, **kwargs
            )
        else:
            mm_system = self.get_mm_topology().createSystem(
                nonbondedMethod=PME,
                nonbondedCutoff=self.protocol.rlist,
                **kwargs,
            )
        return mm_system

    def start(self):
        # define lambda states
        lamb = self.protocol.lam["ml-lambda"]
        self.logger.info(f"{lamb=}")
        # define where to store samples
        trajectory_dir = f"{self.work_dir}/lambda_{lamb:.4f}"
        # path where samples should be stored (will be created if it doesn't exist)
        os.makedirs(trajectory_dir, exist_ok=True)
        trajectory_file = f"{self.work_dir}/lambda_{lamb:.4f}/{self.name}.h5"
        self._traj_file = trajectory_file
        self.logger.info(f"Trajectory saved to: {trajectory_file}")
        # set lambda
        self.simulation.context.setParameter("lambda_interpolate", lamb)
        # set coordinates
        self.simulation.context.setPositions(self.get_initial_coordinates())
        # try to set velocities using openMM, fall back to manual velocity seeding if it fails
        if not self.protocol.restart:
            try:
                self.simulation.context.setVelocitiesToTemperature(
                    self.protocol.temperature
                )
            except OpenMMException:
                from endstate_correction.equ import _seed_velocities, _get_masses

                self.simulation.context.setVelocities(
                    _seed_velocities(_get_masses(self.simulation.system))
                )

        # define HDF5Reporter
        self.simulation.reporters.append(
            HDF5Reporter(
                trajectory_file,
                self.protocol.restart_interval,
            )
        )
        # perform sampling
        self.simulation.step(self.protocol.n_integration_steps)
        self.simulation.reporters.clear()

    def get_trajectory(self) -> mdtraj_trajectory:
        traj = mdtraj.load_hdf5(self._traj_file)
        if self.env == "waterbox":
            traj.image_molecules()
        return traj

    def get_simulation(self) -> Simulation:
        return self.simulation

    def get_xyz(self) -> List[unit.Quantity]:
        traj = self.get_trajectory()
        return [traj.openmm_positions(i) for i in range(traj.n_frames)]
