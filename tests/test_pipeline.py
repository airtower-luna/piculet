# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright: Fiona Klute
import asyncio
import json
import piculet
import pytest
import yaml
from pathlib import Path

ALPINE_IMAGE = 'docker.io/library/alpine:3.19.1'
FAIL_PIPELINE = {
    'skip_clone': True,
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
def pipeline_fail_dir():
    return Path(__file__).parent / 'pipelines-fail'


@pytest.fixture
async def picu_conf():
    repo = Path(__file__).parent.parent.resolve()
    return piculet.PiculetConfig(
        repo=repo,
        ci_env=await piculet.commit_info('HEAD', repo),
        keep_workspace=False)


@pytest.fixture
async def workspace():
    async with piculet.Workspace(None) as w:
        yield w


async def test_pipeline(picu_conf, pipeline_dir):
    pipelines = piculet.find_pipelines(pipeline_dir)
    results = [result async for result in piculet.run_pipelines_ordered(
        pipelines, picu_conf)]
    assert len(results) == 6
    for r in results:
        assert r.success
    # ensure the task that depends on the others is last in the results
    results[-1].steps[0] == piculet.StepResult('meow', 0, 'Meow, Meow.\n', '')


async def test_dependency_fail(picu_conf, pipeline_fail_dir):
    pipelines = piculet.find_pipelines(pipeline_fail_dir)
    results = [result async for result in piculet.run_pipelines_ordered(
        pipelines, picu_conf)]
    assert len(results) == 2
    assert isinstance(results[0], piculet.PipelineFail)
    assert results[0].steps[0] == piculet.StepFail('fail', 1, '', '')
    assert isinstance(results[1], piculet.PipelineCancelled)
    assert results[1].steps == []


def test_matrix_load(picu_conf, pipeline_dir):
    """Load matrix pipeline, check for expected elements"""
    with open(pipeline_dir / 'build.yaml') as fh:
        p = yaml.safe_load(fh)
    jobs = piculet.Pipeline.load('build', p, picu_conf)
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
    await piculet.clone_into(workspace, repo=Path(__file__).parent.parent)
    result = await piculet.run_step(
        'ls', ALPINE_IMAGE, workspace, ['ls -l .git/HEAD'], dict())
    assert result.returncode == 0
    assert result.stdout.startswith('-rw')
    assert 'root' in result.stdout
    assert '.git/HEAD' in result.stdout
    assert result.stderr == ''


async def test_step_fail(workspace):
    with pytest.raises(piculet.StepFail) as excinfo:
        await piculet.run_step(
            'grep', ALPINE_IMAGE, workspace, ['grep -h'], dict())
    assert excinfo.value.returncode != 0
    assert excinfo.value.stdout == ''
    assert excinfo.value.stderr.startswith('BusyBox')


def test_step_report(caplog):
    result = piculet.StepResult('meow', 0, 'Meow, meow.', 'Hiss!')
    with caplog.at_level('INFO', logger=piculet.__name__):
        assert result.report() == 'step "meow" returned 0'
    assert result.report(verbose=True) == 'step "meow" returned 0\n' \
        '------ stderr ------\nHiss!\n' \
        '------ stdout ------\nMeow, meow.'


def test_pipeline_report(caplog):
    result = piculet.PipelineResult(
        'Meow', True, [piculet.StepResult('meow', 0, 'Meow, meow.', 'Hiss!')],
        1.234)
    with caplog.at_level('INFO', logger=piculet.__name__):
        assert result.report() == 'Meow: passed (1.23s)'
    assert result.report(verbose=True) == 'Meow: passed (1.23s)\n' \
        'step "meow" returned 0\n' \
        '------ stderr ------\nHiss!\n' \
        '------ stdout ------\nMeow, meow.'


def test_pipeline_report_volume(caplog):
    result = piculet.PipelineResult(
        'Meow', True, [piculet.StepResult('meow', 0, 'Meow, meow.', 'Hiss!')],
        1.234, 'litterbox')
    with caplog.at_level('INFO', logger=piculet.__name__):
        assert result.report() == 'Meow: passed (1.23s)\n' \
            'preserved volume: litterbox'
    assert result.report(verbose=True) == 'Meow: passed (1.23s)\n' \
        'preserved volume: litterbox\n' \
        'step "meow" returned 0\n' \
        '------ stderr ------\nHiss!\n' \
        '------ stdout ------\nMeow, meow.'


async def test_ci_ref(picu_conf, workspace):
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
                        'cat .git/HEAD',
                    ],
                    'environment': {'CAT': 'Meow!'},
                },
            ]
        },
        picu_conf)
    assert len(pipeline) == 1
    results = await pipeline[0].run()
    assert results.success
    assert len(results.steps) == 1
    lines = results.steps[0].stdout.splitlines()
    head = (Path(__file__).parent.parent / '.git' / 'HEAD').read_text()
    assert lines[0].strip() == head.split()[1]
    assert lines[1].strip() == 'Meow!'
    assert lines[2].strip() == head.strip()


async def test_single_pipeline_fail(picu_conf, workspace):
    pipeline = piculet.Pipeline('test fail', FAIL_PIPELINE, picu_conf)
    with pytest.raises(piculet.PipelineFail) as excinfo:
        await pipeline.run()
    assert excinfo.value.success is False
    steps = excinfo.value.steps
    assert steps == [
        piculet.StepResult('echo', 0, 'test\n', ''),
        piculet.StepFail('fail', 1, '', ''),
    ]


async def test_pipeline_ordered_fail(picu_conf, workspace):
    results = [result async for result in piculet.run_pipelines_ordered(
        {'test fail': FAIL_PIPELINE}, picu_conf)]
    assert len(results) == 1
    assert isinstance(results[0], piculet.PipelineFail)
    assert results[0].success is False
    steps = results[0].steps
    assert steps == [
        piculet.StepResult('echo', 0, 'test\n', ''),
        piculet.StepFail('fail', 1, '', ''),
    ]


async def test_workspace_labels(caplog):
    labels = {
        'repo': '/cat/nest',
        'pipeline': 'meow',
    }
    async with piculet.Workspace(None, labels, keep=True) as w:
        proc = await asyncio.create_subprocess_exec(
            'podman', 'volume', 'inspect', w.volume,
            stdout=asyncio.subprocess.PIPE)
        try:
            stdout, _ = await proc.communicate()
            assert proc.returncode == 0
        finally:
            await proc.wait()
        j = json.loads(stdout)
        assert len(j) == 1
        assert j[0]['Labels'] == {
            'piculet.repo': '/cat/nest',
            'piculet.pipeline': 'meow',
        }

    with caplog.at_level('DEBUG', logger=piculet.__name__):
        await piculet.Workspace.prune_repo_volumes(Path('/cat/nest'))
    assert caplog.records[-1].message == f"repo cleanup done: ['{w.volume}']"


def test_missing_dependency(pipeline_dir):
    with pytest.raises(ValueError) as excinfo:
        piculet.main(['--search', str(pipeline_dir / 'last.yaml')])
    assert 'dependency on non-existing task' in str(excinfo.value)


def test_run(pipeline_dir):
    ret = piculet.main(['--search', str(pipeline_dir / 'test.yaml')])
    assert ret == 0


def test_run_fail(pipeline_fail_dir):
    ret = piculet.main(['--search', str(pipeline_fail_dir / 'fail.yaml')])
    assert ret == 1
