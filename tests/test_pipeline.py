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


def test_pipeline(pipeline_dir):
    piculet.main(['--search', str(pipeline_dir)])


async def test_step_success(workspace):
    result = await piculet.run_step(
        ALPINE_IMAGE, workspace, ['ls -l .git/HEAD'], dict())
    assert result.returncode == 0
    assert result.stdout.startswith('-rw')
    assert 'root' in result.stdout
    assert '.git/HEAD' in result.stdout
    assert result.stderr == ''


async def test_step_fail(workspace):
    with pytest.raises(piculet.StepFail) as excinfo:
        await piculet.run_step(
            ALPINE_IMAGE, workspace, ['grep -h'], dict())
    assert excinfo.value.returncode != 0
    assert excinfo.value.stdout == ''
    assert excinfo.value.stderr.startswith('BusyBox')


async def test_ci_ref(workspace):
    results = await piculet.run_job(
        {
            'steps': [
                {
                    'name': 'echo',
                    'image': ALPINE_IMAGE,
                    'commands': [
                        'echo ${CI_COMMIT_REF}',
                        'echo "${CAT}"',
                    ],
                    'environment': {'CAT': 'Meow!'},
                },
            ]
        },
        await piculet.commit_info(), {})
    assert results.success
    assert len(results.steps) == 1
    lines = results.steps[0].stdout.splitlines()
    lines[0].strip() == (
        Path(__file__).parent.parent / '.git' / 'HEAD').read_text().split()[1]
    lines[1].strip() == 'Meow!'
