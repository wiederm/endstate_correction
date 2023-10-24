import mdtraj
import pytest
from importlib_resources import files

import endstate_correction
from endstate_correction.protocol import BSSProtocol, AllResults
from endstate_correction.protocol import (
    perform_endstate_correction,
    NEQProtocol,
)
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
            lam=0,
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
            implementation="torchani",
        )
        simulation.set_trajectory(str(parameter_base / f"{system_name}.h5"))
        return simulation

    @staticmethod
    @pytest.fixture(scope="module")
    def perform_correction(ec, tmp_path_factory):
        sim = ec.get_simulation()
        traj = ec.get_trajectory()
        outdir = tmp_path_factory.mktemp("out")
        neq_protocol = NEQProtocol(
            sim=sim,
            reference_samples=traj,
            nr_of_switches=5,
            switching_length=10,
            save_endstates=False,
            save_trajs=True,
        )
        r = perform_endstate_correction(
            neq_protocol,
            workdir=outdir,
        )
        r.workdir = outdir
        return r

    def test_sanity(self, perform_correction):
        assert isinstance(perform_correction, AllResults)

    def test_traj_num(self, perform_correction):
        # Test if the correct number of trajectory has been generated
        r = perform_correction
        assert len(list((r.workdir / "reference_to_target").glob("*.dcd"))) == 5

    def test_traj_length(self, perform_correction):
        package_path = files(endstate_correction)
        system_name = "methane"
        parameter_base = package_path / "data" / "amber"
        r = perform_correction
        traj = mdtraj.load_dcd(
            r.workdir / "reference_to_target" / "switching_trajectory_0.dcd",
            top=str(parameter_base / f"{system_name}.prm7"),
        )
        assert len(traj) == 10
