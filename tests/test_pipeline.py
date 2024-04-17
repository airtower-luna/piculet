# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright: Fiona Klute
import piculet
import pytest
from pathlib import Path


@pytest.fixture
def pipeline_dir():
    return Path(__file__).parent / 'pipelines'


async def test_pipeline(pipeline_dir):
    await piculet.run(['--search', str(pipeline_dir)])
