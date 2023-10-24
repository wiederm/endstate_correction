from pathlib import Path

import pytest
from importlib_resources import files
from mdtraj.core.trajectory import Trajectory as mdtraj_trajectory

import endstate_correction
from endstate_correction.protocol import BSSProtocol
from endstate_correction.simulation import (
    EndstateCorrectionAMBER,
    EndstateCorrectionCHARMM,
)
from endstate_correction.simulation.base import EndstateCorrectionBase
from endstate_correction.topology import AMBERTopology, CHARMMTopology


class TestEndstateCorrectionCharmm:
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
    def setup(tmp_path_factory, bss_protocol):
        package_path = files(endstate_correction)
        system_name = "1_octanol"
        env = "waterbox"
        # define the output directory
        output_base = tmp_path_factory.mktemp(system_name)
        parameter_base = package_path / "data" / "jctc_data"
        # load the charmm specific files (psf, pdb, rtf, prm and str files)
        top = CHARMMTopology(
            psf_file_path=str(
                parameter_base / system_name / "charmm-gui/openmm/step3_input.psf"
            ),
            parameter_set=[
                str(parameter_base / system_name / "charmm-gui/unk/unk.rtf"),
                str(parameter_base / system_name / "charmm-gui/unk/unk.prm"),
                str(parameter_base / "toppar/top_all36_cgenff.rtf"),
                str(parameter_base / "toppar/par_all36_cgenff.prm"),
                str(parameter_base / "toppar/toppar_water_ions.str"),
            ],
            input_config=str(
                parameter_base / system_name / "charmm-gui/input.config.dat"
            ),
            crd_file_path=str(
                parameter_base / system_name / "charmm-gui/openmm/step3_input.pdb"
            ),
        )

        simulation = EndstateCorrectionCHARMM(
            top,
            env=env,
            ml_atoms=list(range(27)),
            protocol=bss_protocol,
            name="openMM",
            work_dir=str(output_base),
            implementation="torchani",
        )
        simulation.start()
        return simulation

    def test_sanity(self, setup):
        """Test if the setup is fine."""
        assert isinstance(setup, EndstateCorrectionBase)

    def test_trajfile_exist(self, setup):
        assert Path(setup._traj_file).is_file()

    def test_traj(self, setup):
        assert isinstance(setup.get_trajectory(), mdtraj_trajectory)

    def test_get_xyz(self, setup):
        assert len(setup.get_xyz()) == 2


class TestEndstateCorrectionAmber(TestEndstateCorrectionCharmm):
    @staticmethod
    @pytest.fixture(scope="module")
    def setup(tmp_path_factory, bss_protocol):
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
        simulation.start()
        return simulation
