# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright: Fiona Klute
import piculet
import pytest
from pathlib import Path


@pytest.fixture
def matrix_build():
    return Path(__file__).parent / 'build.yaml'


def test_pipeline(matrix_build):
    piculet.run_pipeline(matrix_build)
