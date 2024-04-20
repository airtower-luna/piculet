# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright: Fiona Klute
import argparse
import asyncio
import logging
import re
import shlex
import shutil
import subprocess
import sys
import uuid
import yaml
from collections import ChainMap
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from io import StringIO
from itertools import product, repeat
from pathlib import Path
from string import Template
from tempfile import NamedTemporaryFile, TemporaryDirectory
from toposort import toposort
from typing import Any, Self

WORKSPACE = '/work'
ENV_NAME = re.compile(r'^\w[\w\d]*$')
logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')


class Workspace:
    volume: str
    base: Path
    path: Path

    def __init__(self, config: dict[str, str] | None):
        if config is None:
            self.base = Path('/work')
            self.path = Path('.')
        else:
            self.base = Path(config.get('base', '/work'))
            self.path = Path(config.get('path', '.'))
            if self.path.is_absolute():
                raise ValueError('workspace.path must be relative')
        self.volume = str(uuid.uuid4())

    @property
    def workdir(self):
        return str(self.base / self.path)

    async def cleanup(self):
        proc = await asyncio.create_subprocess_exec(
            'podman', 'volume', 'rm', self.volume,
            stdout=subprocess.DEVNULL)
        await proc.wait()

    async def __aenter__(self) -> Self:
        proc = await asyncio.create_subprocess_exec(
            'podman', 'volume', 'create', self.volume,
            stdout=subprocess.PIPE)
        try:
            stdout, _ = await proc.communicate()
            assert proc.returncode == 0
            vol_id = stdout.strip().decode()
            assert self.volume == vol_id
            await clone_into(self)
        except Exception:
            await self.cleanup()
            raise
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.cleanup()


@dataclass(frozen=True)
class StepResult:
    name: str
    returncode: int
    stdout: str
    stderr: str

    def report(self, verbose=False):
        buf = StringIO()
        buf.write(f'step {self.name} returned {self.returncode}')
        if self.returncode != 0 or verbose:
            if self.stderr:
                buf.write(f'\n------ stderr ------\n{self.stderr}')
            if self.stdout:
                buf.write(f'\n------ stdout ------\n{self.stdout}')
        return buf.getvalue()


@dataclass(frozen=True)
class PipelineResult:
    name: str
    success: bool
    steps: list[StepResult]

    @property
    def status(self):
        if self.success:
            return 'passed'
        elif isinstance(self, PipelineCancelled):
            return 'cancelled'
        else:
            return 'failed'

    def report(self, verbose=False):
        buf = StringIO()
        buf.write(self.name)
        buf.write(': ')
        buf.write(self.status)
        if not self.success or verbose:
            for s in self.steps:
                buf.write('\n')
                buf.write(s.report(verbose))
        return buf.getvalue()


class StepFail(StepResult, Exception):
    pass


class PipelineFail(PipelineResult, Exception):
    pass


class PipelineCancelled(PipelineResult, Exception):
    pass


async def clone_into(workspace: Workspace, repo: Path = Path('.')):
    """copy source into volume"""
    logger.debug(f'cloning {repo} into volume {workspace.volume}')
    with ExitStack() as stack:
        tmpdir = Path(stack.enter_context(TemporaryDirectory(
            prefix='piculet-clone-', ignore_cleanup_errors=True)))
        volume = tmpdir / 'volume'
        git = await asyncio.create_subprocess_exec(
            'git', 'clone', '--depth=1',
            f'file://{repo.resolve()}', volume / workspace.path,
            stderr=subprocess.DEVNULL)
        await git.wait()
        assert git.returncode == 0
        tarfile = shutil.make_archive(
            str(tmpdir / 'workspace'), 'tar',
            owner='root', group='root',
            root_dir=volume, base_dir=workspace.path)
        recv = await asyncio.create_subprocess_exec(
            'podman', 'volume', 'import', workspace.volume, tarfile)
        await recv.wait()
        assert recv.returncode == 0


async def commit_info(rev: str = 'HEAD', repo=Path('.')) -> dict[str, str]:
    get_hash = await asyncio.create_subprocess_exec(
        'git', 'rev-parse', rev,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
    get_ref = await asyncio.create_subprocess_exec(
        'git', 'rev-parse', '--symbolic-full-name', rev,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
    h, _ = await get_hash.communicate()
    r, _ = await get_ref.communicate()
    return {
        'CI_COMMIT_SHA': h.decode(),
        'CI_COMMIT_REF': r.decode(),
    }


async def run_step(name: str, image: str, workspace: Workspace,
                   commands: list[str],
                   environment: Mapping[str, str]) -> StepResult:
    with NamedTemporaryFile(mode='w+', suffix='.sh') as script:
        script.write('set -e\n')
        for var, value in environment.items():
            if ENV_NAME.match(var) is None:
                raise ValueError('invalid environment variable name')
            script.write(f'{var}={shlex.quote(value)}\n')
        script.write('\n')
        for c in commands:
            script.write(f'{c}\n')
        script.flush()
        script_mount = Path(script.name).name
        try:
            proc = await asyncio.create_subprocess_exec(
                'podman', 'run', '--rm',
                f'--mount=type=bind,'
                f'source={script.name},target=/{script_mount},ro=true',
                f'--mount=type=volume,'
                f'source={workspace.volume},target={workspace.base}',
                '--workdir', workspace.workdir,
                image, 'sh', f'/{script_mount}',
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = await proc.communicate()
        except asyncio.CancelledError:
            await proc.wait()
            raise
        assert isinstance(proc.returncode, int)
        if proc.returncode != 0:
            raise StepFail(
                name, proc.returncode, stdout.decode(), stderr.decode())
        return StepResult(
            name, proc.returncode, stdout.decode(), stderr.decode())


class Pipeline:
    def __init__(self, name: str, config, ci_env: dict[str, str],
                 matrix_element=None):
        self._name = name
        self.config = config
        self.ci_env = ci_env
        self.matrix_element = matrix_element or dict()

    @property
    def name(self):
        if self.matrix_element:
            return f'{self._name} {self.matrix_element}'
        return self._name

    @property
    def steps(self):
        return self.config['steps']

    @property
    def workspace(self):
        return Workspace(self.config.get('workspace'))

    def step_env(self, env):
        return ChainMap(self.ci_env, env, self.matrix_element)

    async def run(self) -> PipelineResult:
        async with self.workspace as work:
            results = list()
            for s in self.steps:
                image = Template(s['image']).safe_substitute(
                    self.matrix_element)
                try:
                    result = await run_step(
                        s['name'], image, work, s['commands'],
                        self.step_env(s.get('environment', {})))
                    logger.debug(
                        '%s, step %s: done', self.name, s['name'])
                    results.append(result)
                except StepFail as result:
                    logger.error(
                        '%s, step %s: fail', self.name, s['name'])
                    results.append(result)
                    raise PipelineFail(self.name, False, results)
            return PipelineResult(self.name, True, results)

    @classmethod
    def load(cls, name: str, pipeline, ci_env: dict[str, str]) \
            -> list[Self]:
        """Create Pipeline objects from pipeline config. If the config
        defines a matrix the returned list will contain one Pipeline
        per matrix combination, otherwise exactly one Pipeline."""
        if 'matrix' in pipeline:
            return [
                cls(name, pipeline, ci_env, dict(x))
                for x in product(*([*zip(repeat(k), pipeline['matrix'][k])]
                                   for k in pipeline['matrix'].keys()))]
        else:
            return [cls(name, pipeline, ci_env)]


async def run_pipelines_ordered(pipelines: dict[str, Any],
                                ci_env: dict[str, str]):
    tasks: dict[str, asyncio.Task] = dict()
    for batch in toposort(dict(
            (name, set(x.get('depends_on', [])))
            for name, x in pipelines.items())):
        for name in batch:
            if name not in pipelines:
                raise ValueError(f'dependency on non-existing task {name}')
            logger.debug(f'creating pipeline task {name}')

            async def pipeline_task(name):
                logger.debug(f'starting pipeline task {name}')
                if 'depends_on' in pipelines[name]:
                    done, _ = await asyncio.wait(
                        [tasks[dep] for dep in pipelines[name]['depends_on']])
                    for task in done:
                        for r in task.result():
                            if isinstance(r, Exception):
                                logger.warning(
                                    'pipeline task %s cancelled because of '
                                    'failed dependency', name)
                                return [PipelineCancelled(name, False, [])]
                    logger.debug('pipeline task %s awaited dependencies', name)
                return await asyncio.gather(
                    *(asyncio.create_task(job.run())
                      for job in Pipeline.load(name, pipelines[name], ci_env)),
                    return_exceptions=True)

            tasks[name] = asyncio.create_task(pipeline_task(name))

    for t in asyncio.as_completed(tasks.values()):
        for result in await t:
            if result.success:
                logger.info(result.report())
            else:
                logger.error(result.report())
            yield result


def find_pipelines(search: Path, endings=('*.yaml', '*.yml')):
    if not search.is_dir():
        with search.open() as fh:
            return {'pipeline': yaml.safe_load(fh)}

    pipelines = dict()
    for e in endings:
        for pipeline in search.glob(e):
            logger.debug(f'found pipeline: {pipeline}')
            with pipeline.open() as fh:
                p = yaml.safe_load(fh)
            pipelines[pipeline.stem.lstrip('.')] = p
    return pipelines


async def run(cmdline=None):
    parser = argparse.ArgumentParser(
        description='tiny local CI engine')
    parser.add_argument(
        '--search', metavar='DIR', dest='search', default='.woodpecker/',
        type=Path, help='directory to search for pipelines (*.yaml)')

    # enable bash completion if argcomplete is available
    try:
        import argcomplete
        argcomplete.autocomplete(parser)
    except ImportError:
        pass

    args = parser.parse_args(args=cmdline)

    ci_env = await commit_info()
    pipelines = find_pipelines(args.search)

    fails = 0
    async for ret in run_pipelines_ordered(pipelines, ci_env):
        if not ret.success:
            fails += 1
    return fails


def main(cmdline=None):
    return asyncio.run(run(cmdline))


if __name__ == '__main__':
    sys.exit(main())
