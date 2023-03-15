"""Provide functions for the endstate correction workflow."""


from openmm.app import Simulation
import numpy as np
from dataclasses import dataclass, field
from typing import List, Union

import numpy as np
import pandas as pd
from openmm import unit
from openmm.app import Simulation
from pymbar import MBAR
from openmmtools.utils import get_fastest_platform

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
        lam: pd.Series,
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
        :param lam: Current lambda
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
                self.collision_rate = collision_rate.value_in_unit(unit.picosecond**-1)
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
class Protocol:
    """Defining the endstate correction protocol"""

    method: str
    sim: Simulation
    trajectories: List
    direction: str = "None"
    nr_of_switches: int = -1
    neq_switching_length: int = 5_000
    save_endstates: bool = False # True makes only sense for NEQ 
    save_trajs: bool = False # True makes only sense for NEQ 


@dataclass
class Results:
    """Provides a dataclass containing the results of a protocol"""

    equ_mbar: List[MBAR] = field(default_factory=list)
    dE_mm_to_qml: np.array = np.array([])
    dE_qml_to_mm: np.array = np.array([])
    W_mm_to_qml: np.array = np.array([])
    W_qml_to_mm: np.array = np.array([])
    endstate_samples_mm_to_qml : np.array = np.array([])
    endstate_samples_qml_to_mm : np.array = np.array([])
    switching_traj_mm_to_qml : np.array = np.array([])
    switching_traj_qml_to_mm : np.array = np.array([])
   


def perform_endstate_correction(protocol: Protocol) -> Results:
    """Perform endstate correction using the provided protocol.

    Args:
        protocol (Protocol): defines the endstatte correction

    Raises:
        AttributeError: _description_
        AttributeError: _description_
        RuntimeError: _description_
        RuntimeError: _description_

    Returns:
        Results: results generated using the passed protocol
    """

    from endstate_correction.neq import perform_switching
    from endstate_correction.constant import kBT

    print(protocol.method)
    # check that all necessary keywords are present
    if protocol.method.upper() not in ["FEP", "NEQ", "ALL"]:
        raise AttributeError(
            "Only `FEP`, 'NEQ` or 'ALL'  are supported methods for endstate corrections"
        )
    if protocol.method.upper() in [
        "FEP",
        "NEQ",
    ] and protocol.direction.lower() not in ["bidirectional", "unidirectional", "unidirectional_forward", "unidirectional_reverse"]:
        raise AttributeError(
            "Only `bidirectional`, `unidirectional_forward` or `unidirectional_reverse` protocols are supported"
        )

    sim = protocol.sim
    # initialize Results with default values
    r = Results()
    if protocol.method.upper() == "FEP" or protocol.method.upper() == "ALL":
        ####################################################
        # ------------------- FEP ---------------------------
        ####################################################
        print("#####################################################")
        print("# ------------------- FEP ---------------------------")
        print("#####################################################")
        # from MM to QML
        if (
            protocol.direction.lower() == "bidirectional"
            or protocol.method.upper() == "ALL"
        ):
            ####################################################
            # ------------------- bidirectional-----------------
            # perform switching from mm to qml

            assert len(protocol.trajectories) == 2

            print("Performing bidirectional protocol ...")
            lambs = np.linspace(0, 1, 2)
            dEs, _, _ = perform_switching(
                    sim,
                    lambs,
                    samples=protocol.trajectories[0],
                    nr_of_switches=protocol.nr_of_switches,
                )
            dEs_from_mm_to_qml = np.array(dEs/kBT)
            
            # perform switching from qml to mm
            lambs = np.linspace(1, 0, 2)
            dEs, _, _ = perform_switching(
                    sim,
                    lambs,
                    samples=protocol.trajectories[-1],
                    nr_of_switches=protocol.nr_of_switches,
                )
            dEs_from_qml_to_mm = np.array(dEs/kBT)

            # set results
            r.dE_mm_to_qml = dEs_from_mm_to_qml
            r.dE_qml_to_mm = dEs_from_qml_to_mm

            ####################################################
            # ------------------- unidirectional---------------
        elif (
            protocol.direction.lower() == "unidirectional_forward"
            or protocol.direction == "unidirectional"
            ):
            # perform switching from mm to qml
            print("Performing unidirectional protocol from MM to QML...")
            lambs = np.linspace(0, 1, 2)
            dEs, _, _ = perform_switching(
                    sim,
                    lambs,
                    samples=protocol.trajectories[0],
                    nr_of_switches=protocol.nr_of_switches,
                )
            dEs_from_mm_to_qml = np.array(dEs/kBT)

            r.dE_mm_to_qml = dEs_from_mm_to_qml

        elif (protocol.direction.lower() == "unidirectional_reverse"):
            # perform switching qml to mm
            print("Performing unidirectional protocol from QML to MM...")
            lambs = np.linspace(1, 0, 2)
            dEs, _, _  = perform_switching(
                    sim,
                    lambs,
                    samples=protocol.trajectories[-1],
                    nr_of_switches=protocol.nr_of_switches,
                )
            dEs_from_qml_to_mm = np.array(dEs/kBT)

            r.dE_qml_to_mm = dEs_from_qml_to_mm
        else:
            raise RuntimeError()

    if protocol.method.upper() == "NEQ" or protocol.method.upper() == "ALL":
        ####################################################
        # ------------------- NEQ ---------------------------
        ####################################################
        print("#####################################################")
        print("# ------------------- NEQ ---------------------------")
        print("#####################################################")
        if protocol.direction == "bidirectional" or protocol.method.upper() == "ALL":
            ####################################################
            # ------------------- bidirectional-----------------
            # perform switching from mm to qml

            assert len(protocol.trajectories) == 2

            print("Performing bidirectional protocol...")
            lambs = np.linspace(0, 1, protocol.neq_switching_length)
            Ws, endstates_mm_to_qml, trajs_mm_to_qml = perform_switching(
                    sim,
                    lambs,
                    samples=protocol.trajectories[0],
                    nr_of_switches=protocol.nr_of_switches,
                    save_endstates=protocol.save_endstates,
                    save_trajs=protocol.save_trajs
                )
            Ws_from_mm_to_qml = np.array(Ws/ kBT)

            # perform switching from qml to mm
            lambs = np.linspace(1, 0, protocol.neq_switching_length)
            Ws, endstates_qml_to_mm, trajs_qml_to_mm = perform_switching(
                    sim,
                    lambs,
                    samples=protocol.trajectories[-1],
                    nr_of_switches=protocol.nr_of_switches,
                    save_endstates=protocol.save_endstates,
                    save_trajs=protocol.save_trajs
                )
            Ws_from_qml_to_mm = np.array(Ws/ kBT)   

            r.W_mm_to_qml = Ws_from_mm_to_qml
            r.W_qml_to_mm = Ws_from_qml_to_mm

            r.endstate_samples_mm_to_qml = endstates_mm_to_qml
            r.endstate_samples_qml_to_mm = endstates_qml_to_mm

            r.switching_traj_mm_to_qml = trajs_mm_to_qml
            r.switching_traj_qml_to_mm = trajs_qml_to_mm

            ####################################################
            # ------------------- unidirectional----------------
        elif (
            protocol.direction == "unidirectional_forward"
            or protocol.direction == "unidirectional"
            ):
            # perform switching from mm to qml
            print("Performing unidirectional protocol from MM to QML...")
            lambs = np.linspace(0, 1, protocol.neq_switching_length)
            Ws, endstates_mm_to_qml, trajs_mm_to_qml = perform_switching(
                    sim,
                    lambs,
                    samples=protocol.trajectories[0],
                    nr_of_switches=protocol.nr_of_switches,
                    save_endstates=protocol.save_endstates,
                    save_trajs=protocol.save_trajs
                )
            Ws_from_mm_to_qml = np.array(Ws/ kBT)

            r.W_mm_to_qml = Ws_from_mm_to_qml
            r.endstate_samples_mm_to_qml = endstates_mm_to_qml
            r.switching_traj_mm_to_qml = trajs_mm_to_qml

        elif (protocol.direction == "unidirectional_reverse"):
            # perform switching from qml to mm
            lambs = np.linspace(1, 0, protocol.neq_switching_length)
            Ws, endstates_qml_to_mm, trajs_qml_to_mm = perform_switching(
                    sim,
                    lambs,
                    samples=protocol.trajectories[-1],
                    nr_of_switches=protocol.nr_of_switches,
                    save_endstates=protocol.save_endstates,
                    save_trajs=protocol.save_trajs
                )
            Ws_from_qml_to_mm = np.array(Ws/ kBT)   

            r.W_qml_to_mm = Ws_from_qml_to_mm
            r.endstate_samples_qml_to_mm = endstates_qml_to_mm            
            r.switching_traj_qml_to_mm = trajs_qml_to_mm

        else:
            raise RuntimeError()

    return r
