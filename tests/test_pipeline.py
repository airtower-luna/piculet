# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright: Fiona Klute
import piculet
import pytest
import yaml
from pathlib import Path

ALPINE_IMAGE = 'docker.io/library/alpine:3.19.1'
FAIL_PIPELINE = {
    'steps': [
        {
            'name': 'echo',
            'image': ALPINE_IMAGE,
            'commands': ['echo test'],
        },
        {
            'name': 'fail',
            'image': ALPINE_IMAGE,
            'commands': ['false'],
        },
    ]
}


@pytest.fixture
def pipeline_dir():
    return Path(__file__).parent / 'pipelines'


@pytest.fixture
async def workspace():
    async with piculet.Workspace(None) as w:
        yield w


async def test_pipeline(pipeline_dir):
    pipelines = piculet.find_pipelines(pipeline_dir)
    results = [result async for result in piculet.run_pipelines_ordered(
        pipelines, await piculet.commit_info())]
    assert len(results) == 6
    for r in results:
        assert r.success
    # ensure the task that depends on the others is last in the results
    results[-1].steps[0].stdout == 'Meow, Meow.\n'
    results[-1].steps[0].stderr == ''


def test_matrix_load(pipeline_dir):
    """Load matrix pipeline, check for expected elements"""
    with open(pipeline_dir / 'build.yaml') as fh:
        p = yaml.safe_load(fh)
    jobs = piculet.Pipeline.load('build', p, {})
    assert len(jobs) == 4
    for j in jobs:
        assert len(j.steps) == 2
    elements = [j.matrix_element for j in jobs]
    for e in (
            {'IMAGE': 'docker.io/library/alpine:3.19.1', 'WORD': 'Bye'},
            {'IMAGE': 'docker.io/library/alpine:3.19.1', 'WORD': 'Hello'},
            {'IMAGE': 'docker.io/library/debian:bookworm', 'WORD': 'Hello'},
            {'IMAGE': 'docker.io/library/debian:bookworm', 'WORD': 'Bye'}):
        assert e in elements


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
    pipeline = piculet.Pipeline.load(
        'test ci vars',
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
        await piculet.commit_info())
    assert len(pipeline) == 1
    results = await pipeline[0].run()
    assert results.success
    assert len(results.steps) == 1
    lines = results.steps[0].stdout.splitlines()
    assert lines[0].strip() == (
        Path(__file__).parent.parent / '.git' / 'HEAD').read_text().split()[1]
    assert lines[1].strip() == 'Meow!'


async def test_single_pipeline_fail(workspace):
    pipeline = piculet.Pipeline(
        'test fail', FAIL_PIPELINE, await piculet.commit_info())
    with pytest.raises(piculet.PipelineFail) as excinfo:
        await pipeline.run()
    assert excinfo.value.success is False
    steps = excinfo.value.steps
    assert len(steps) == 2
    assert steps[0].stdout == 'test\n'
    assert steps[0].stderr == ''
    assert isinstance(steps[1], piculet.StepFail)
    assert steps[1].stdout == ''
    assert steps[1].stderr == ''


async def test_pipeline_ordered_fail(workspace):
    results = [result async for result in piculet.run_pipelines_ordered(
        {'test fail': FAIL_PIPELINE}, await piculet.commit_info())]
    assert len(results) == 1
    assert isinstance(results[0], piculet.PipelineFail)
    assert results[0].success is False
    steps = results[0].steps
    assert len(steps) == 2
    assert steps[0].stdout == 'test\n'
    assert steps[0].stderr == ''
    assert isinstance(steps[1], piculet.StepFail)
    assert steps[1].stdout == ''
    assert steps[1].stderr == ''


def test_missing_dependency(pipeline_dir):
    with pytest.raises(ValueError) as excinfo:
        piculet.main(['--search', str(pipeline_dir / 'last.yaml')])
    assert 'dependency on non-existing task' in str(excinfo.value)


def test_run(pipeline_dir):
    ret = piculet.main(['--search', str(pipeline_dir / 'test.yaml')])
    assert ret == 0
