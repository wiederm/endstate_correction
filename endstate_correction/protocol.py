"""Provide functions for the endstate correction workflow."""


from dataclasses import dataclass, field
from typing import List, Optional, Union

import mdtraj as md
import numpy as np
import pandas as pd
from openmm import unit
from openmm.app import Simulation
from pymbar import MBAR

from endstate_correction.smc import SMC


class BSSProtocol:
    """This is a dataclass mimicking the BioSimSpace.Protocol."""

    def __init__(
        self,
        timestep: Union[int, unit.Quantity],
        n_integration_steps: int,
        temperature: Union[float, unit.Quantity],
        pressure: Union[float, unit.Quantity],
        report_interval: int,
        restart_interval: int,
        rlist: Union[float, unit.Quantity],
        collision_rate: Union[float, unit.Quantity],
        switchDistance: Union[float, unit.Quantity],
        lam: float,
        restart: bool,
    ):
        """Class for storing the run information

        :param timestep: fs
        :param n_integration_steps:
        :param temperature: K
        :param pressure: atm
        :param report_interval: The frequency at which energy are recorded (In integration steps).
        :param restart_interval: The frequency at which frames are recorded (In integration steps).
        :param rlist: short-range cutoff in nanometers.
        :param collision_rate: 1/picosecond
        :param switchDistance: nanometers
        :param lam: Current lambda between 0 and 1
        :param restart: Whether to reset the velocity or not
        """
        if isinstance(timestep, unit.Quantity):
            try:
                self.timestep = timestep.value_in_unit(unit.femtosecond)
            except Exception as e:
                raise ValueError(f"`timestep` should be a time unit.") from e
        else:
            self.timestep = timestep * unit.femtosecond

        self.n_integration_steps = n_integration_steps

        if isinstance(temperature, unit.Quantity):
            try:
                self.temperature = temperature.value_in_unit(unit.kelvin)
            except Exception as e:
                raise ValueError(f"`temperature` should be a temperature unit.") from e
        else:
            self.temperature = temperature * unit.kelvin

        if isinstance(pressure, unit.Quantity):
            try:
                self.pressure = pressure.value_in_unit(unit.atmosphere)
            except Exception as e:
                raise ValueError(f"`pressure` should be a pressure unit.") from e
        else:
            self.pressure = pressure * unit.atmosphere

        self.report_interval = report_interval
        self.restart_interval = restart_interval

        if isinstance(rlist, unit.Quantity):
            try:
                self.rlist = rlist.value_in_unit(unit.nanometer)
            except Exception as e:
                raise ValueError(f"`rlist` should be a length unit.") from e
        else:
            self.rlist = rlist * unit.nanometer

        if isinstance(collision_rate, unit.Quantity):
            try:
                self.collision_rate = collision_rate.value_in_unit(
                    unit.picosecond**-1
                )
            except Exception as e:
                raise ValueError(f"`collision_rate` should be a 1/time unit.") from e
        else:
            self.collision_rate = collision_rate / unit.picosecond

        if isinstance(switchDistance, unit.Quantity):
            try:
                self.collision_rate = collision_rate.value_in_unit(unit.nanometer)
            except Exception as e:
                raise ValueError(f"`switchDistance` should be a length unit.") from e
        else:
            self.switchDistance = switchDistance * unit.nanometer

        self.lam = lam
        self.restart = restart


@dataclass
class BaseProtocol:
    """Base class for all endstate correction protocols"""

    sim: Simulation  # simulation object
    reference_samples: Optional[md.Trajectory] = None  # reference samples
    target_samples: Optional[md.Trajectory] = None  # target samples

    def __post_init__(self):
        if self.reference_samples is None and self.target_samples is None:
            raise ValueError("reference_samples or target_samples must be provided!")


@dataclass
class FEPProtocol(BaseProtocol):
    """FEP-specific protocol"""

    nr_of_switches: int = -1  # number of switches for NEQ

    def __post_init__(self):
        super().__post_init__()  # Call base class's post-init


@dataclass
class NEQProtocol(BaseProtocol):
    """NEQ-specific protocol"""

    nr_of_switches: int = -1  # number of switches for NEQ
    switching_length: int = 5_000  # switching length in steps
    save_endstates: bool = False
    save_trajs: bool = False

    def __post_init__(self):
        super().__post_init__()  # Call base class's post-init


@dataclass
class SMCProtocol(BaseProtocol):
    """SMC-specific protocol"""

    nr_of_walkers: int = -1  # number of walkers for SMC
    nr_of_resampling_steps: int = 1_000  # number of resampling steps for SMC
    save_endstates: bool = False

    def __post_init__(self):
        super().__post_init__()  # Call base class's post-init


@dataclass
class AllProtocol():
    """Dataclass for running all protocols"""

    fep_protocol: Union[None, FEPProtocol] = None
    neq_protocol: Union[None, NEQProtocol] = None
    smc_protocol: Union[None, SMCProtocol] = None

    # check if reference or target samples are provided
    def __post_init__(self):
        self.fep_protocol.__post_init__()  
        self.neq_protocol.__post_init__()
        self.smc_protocol.__post_init__()

class BaseResults:
    """Base class for all protocol results"""


@dataclass
class EquResults(BaseResults):
    """Equilibrium simulation-specific results"""

    equ_mbar: List[MBAR] = field(default_factory=list)  # MBAR object for each lambda


@dataclass
class FEPResults(BaseResults):
    """FEP-specific results"""

    dE_reference_to_target: np.array = field(default_factory=lambda: np.array([]))  # dE from reference to target
    dE_target_to_reference: np.array = field(default_factory=lambda: np.array([]))  # dE from target to reference

@dataclass
class NEQResults(BaseResults):
    """Provides a dataclass containing the results of a protocol"""

    W_reference_to_target: np.array = field(default_factory=lambda: np.array([]))  # W from reference to target
    W_target_to_reference: np.array = field(default_factory=lambda: np.array([]))  # W from target to reference
    endstate_samples_reference_to_target: np.array = field(default_factory=lambda: np.array([]))  # endstate samples from reference to target
    endstate_samples_target_to_reference: np.array = field(default_factory=lambda: np.array([]))  # endstate samples from target to reference
    switching_traj_reference_to_target: np.array = field(default_factory=lambda: np.array([]))  # switching traj from reference to target
    switching_traj_target_to_reference: np.array = field(default_factory=lambda: np.array([]))  # switching traj from target to reference


@dataclass
class SMCResults(BaseResults):
    logZ: float = 0.0  # free energy difference
    effective_sample_size: list = field(default_factory=list)  # effective sample size
    endstate_samples_reference_to_target: np.array = field(default_factory=lambda: np.array([]))  # endstate samples from reference to target
    endstate_samples_target_to_reference: np.array = field(default_factory=lambda: np.array([]))  # endstate samples from target to reference


@dataclass
class AllResults():
    """Dataclass for combined results of all protocols"""

    equ_results: Union[None, EquResults] = None
    fep_results: Union[None, FEPResults] = None
    neq_results: Union[None, NEQResults] = None
    smc_results: Union[None, SMCResults] = None


def perform_endstate_correction(protocol: Union[BaseProtocol, AllProtocol]) -> AllResults:
    """Perform endstate correction using the provided protocol.

    Args:
        protocol (Union[BaseProtocol, AllProtocol]): defines the endstate correction. 
            Either a specific protocol or a collection of protocols.

    Returns:
        BaseResults: results generated using the passed protocol
    """
    from endstate_correction.constant import kBT
    from endstate_correction.neq import perform_switching

    r = AllResults()
    if isinstance(protocol, AllProtocol) or isinstance(protocol, FEPProtocol):
        print(f"Performing endstate correction using FEP")
        if isinstance(protocol, AllProtocol):
            protocol_ = protocol.fep_protocol
        else:
            protocol_ = protocol

        print("#####################################################")
        print("# ------------------- FEP ---------------------------")
        print("#####################################################")
        sim = protocol_.sim
        r_fep = FEPResults()
        list_of_lambda_values = np.linspace(0, 1, 2)  # lambda values
        # from reference to target potential
        if protocol_.reference_samples is not None:  # if reference samples are provided
            print("Performing FEP from reference to target potential")
            dEs, _, _ = perform_switching(
                sim,
                lambdas=list_of_lambda_values,
                samples=protocol_.reference_samples,
                nr_of_switches=protocol_.nr_of_switches,
            )
            dE_reference_to_target = np.array(dEs / kBT)  # remove units
            r_fep.dE_reference_to_target = dE_reference_to_target

        # from target to reference potential
        if protocol_.target_samples is not None:  # if target samples are provided
            print("Performing FEP from target to reference potential")
            dEs, _, _ = perform_switching(
                sim,
                lambdas=np.flip(
                    list_of_lambda_values
                ),  # NOTE: we reverse the list of provided lamba values to indicate switching from the target to the reference potential
                samples=protocol_.target_samples,
                nr_of_switches=protocol_.nr_of_switches,
            )
            dE_target_to_reference = np.array(dEs / kBT)  # remove units
            r_fep.dE_target_to_reference = dE_target_to_reference
        r.fep_results = r_fep

    if isinstance(protocol, AllProtocol) or isinstance(protocol, SMCProtocol):
        print(f"Performing endstate correction using SMC")
        if isinstance(protocol, AllProtocol):
            protocol_ = protocol.smc_protocol
        else:
            protocol_ = protocol
        print("#####################################################")
        print("# ------------------- SMC ---------------------------")
        print("#####################################################")
        sim = protocol_.sim
        r_smc = SMCResults()
        if protocol_.reference_samples is not None:  # if reference samples are provided
            print("Performing SMC from reference to target potential")
            smc_sampler = SMC(sim=sim, samples=protocol_.reference_samples)
            smc_sampler.perform_SMC(
                nr_of_steps=protocol_.nr_of_resampling_steps,
                nr_of_walkers=protocol_.nr_of_walkers,
            )

            r_smc.logZ = smc_sampler.logZ
            r_smc.effective_sample_size = smc_sampler.effective_sample_size
            r_smc.endstate_samples_reference_to_target = (
                smc_sampler.current_set_of_walkers
            )
        r.smc_results = r_smc

    if isinstance(protocol, AllProtocol) or isinstance(protocol, NEQProtocol):
        print(f"Performing endstate correction using NEQ")
        if isinstance(protocol, AllProtocol):
            protocol_ = protocol.neq_protocol
        else:
            protocol_ = protocol
        print("#####################################################")
        print("# ------------------- NEQ ---------------------------")
        print("#####################################################")
        sim = protocol_.sim
        r_neq = NEQResults()
        list_of_lambda_values = np.linspace(0, 1, protocol_.switching_length)
        # from reference to target potential
        if protocol_.reference_samples is not None:
            print("Performing NEQ from reference to target potential")
            (
                Ws,
                endstates_reference_to_target,
                trajs_reference_to_target,
            ) = perform_switching(
                sim,
                lambdas=list_of_lambda_values,
                samples=protocol_.reference_samples,
                nr_of_switches=protocol_.nr_of_switches,
                save_endstates=protocol_.save_endstates,
                save_trajs=protocol_.save_trajs,
            )
            Ws_reference_to_target = np.array(Ws / kBT)  # remove units
            r_neq.W_reference_to_target = Ws_reference_to_target
            r_neq.endstate_samples_reference_to_target = endstates_reference_to_target
            r_neq.switching_traj_reference_to_target = trajs_reference_to_target

        # from target to reference potential
        if protocol_.target_samples is not None:
            print("Performing NEQ from target to reference potential")
            (
                Ws,
                endstates_target_to_reference,
                trajs_target_to_reference,
            ) = perform_switching(
                sim,
                lambdas=np.flip(
                    list_of_lambda_values
                ),  # NOTE: we reverse the list of provided lamba values to indicate switching from the target to the reference potential
                samples=protocol_.target_samples,
                nr_of_switches=protocol_.nr_of_switches,
                save_endstates=protocol_.save_endstates,
                save_trajs=protocol_.save_trajs,
            )
            Ws_target_to_reference = np.array(Ws / kBT)
            r_neq.W_target_to_reference = Ws_target_to_reference
            r_neq.endstate_samples_target_to_reference = endstates_target_to_reference
            r_neq.switching_traj_target_to_reference = trajs_target_to_reference
        r.neq_results = r_neq

    return r
