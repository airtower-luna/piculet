# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright: Fiona Klute
import piculet
import pytest
from pathlib import Path

ALPINE_IMAGE = 'docker.io/library/alpine:3.19.1'


@pytest.fixture
def pipeline_dir():
    return Path(__file__).parent / 'pipelines'


@pytest.fixture
async def workspace():
    async with piculet.Workspace(None) as w:
        yield w


async def test_pipeline(pipeline_dir):
    await piculet.run(['--search', str(pipeline_dir)])


async def test_step_success(workspace):
    result = await piculet.run_step(
        ALPINE_IMAGE, workspace, ['ls -l src/piculet.py'], dict())
    assert result.returncode == 0
    assert result.stdout.startswith('-rw')
    assert 'root' in result.stdout
    assert 'src/piculet.py' in result.stdout
    assert result.stderr == ''


async def test_step_fail(workspace):
    result = await piculet.run_step(
        ALPINE_IMAGE, workspace, ['grep -h'], dict())
    assert result.returncode != 0
    assert result.stdout == ''
    assert result.stderr.startswith('BusyBox')
