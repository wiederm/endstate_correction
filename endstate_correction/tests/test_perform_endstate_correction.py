import pandas as pd
import pytest
from importlib_resources import files

import endstate_correction
from endstate_correction.protocol import BSSProtocol, Results
from endstate_correction.protocol import perform_endstate_correction, Protocol
from endstate_correction.simulation import (
    EndstateCorrectionAMBER,
)
from endstate_correction.topology import AMBERTopology


class TestPerformCorrection:
    @staticmethod
    @pytest.fixture(scope="module")
    def bss_protocol():
        protocol = BSSProtocol(
            timestep=1,
            n_integration_steps=100,  # 10 * 10 steps
            temperature=300,
            pressure=1,
            report_interval=10,
            restart_interval=50,
            rlist=1,
            collision_rate=1,
            switchDistance=0,
            restart=False,
            lam=pd.Series(data={"ml-lambda": 0}),
        )
        return protocol

    @staticmethod
    @pytest.fixture(scope="module")
    def ec(tmp_path_factory, bss_protocol):
        package_path = files(endstate_correction)
        system_name = "methane"
        env = "waterbox"
        # define the output directory
        output_base = tmp_path_factory.mktemp(system_name)
        parameter_base = package_path / "data" / "amber"
        # load the charmm specific files (psf, pdb, rtf, prm and str files)
        top = AMBERTopology(
            prm7_file_path=str(parameter_base / f"{system_name}.prm7"),
            rst7_file_path=str(parameter_base / f"{system_name}.rst7"),
        )

        simulation = EndstateCorrectionAMBER(
            top,
            env=env,
            ml_atoms=list(range(5)),
            protocol=bss_protocol,
            name=system_name,
            work_dir=str(output_base),
        )
        simulation._traj_file = str(parameter_base / f"{system_name}.h5")
        return simulation

    @staticmethod
    @pytest.fixture(scope="module")
    def perform_correction(ec):
        sim = ec.get_simulation()
        traj = ec.get_xyz()
        fep_protocol = Protocol(
            method="NEQ",
            sim=sim,
            reference_samples=traj,
            target_samples=traj,
            nr_of_switches=5,
            neq_switching_length=10,
        )

        r = perform_endstate_correction(fep_protocol)
        return r

    def test_sanity(self, perform_correction):
        assert isinstance(perform_correction, Results)
