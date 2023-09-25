"""Dataclass for AMBER and CHARMM topology"""
from dataclasses import dataclass
from typing import List


@dataclass
class AMBERTopology:
    """This is a dataclass for amber topology."""

    rst7_file_path: str
    prm7_file_path: str


@dataclass
class CHARMMTopology:
    """This is a dataclass for CHARMM topology."""

    psf_file_path: str
    crd_file_path: str
    parameter_set: List[str]
    input_config: str
