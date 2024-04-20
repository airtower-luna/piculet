# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright: Fiona Klute
import argparse
import asyncio
import logging
import re
import shlex
import shutil
import subprocess
import uuid
import yaml
from collections import ChainMap
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from itertools import product, repeat
from pathlib import Path
from string import Template
from tempfile import NamedTemporaryFile, TemporaryDirectory
from toposort import toposort
from typing import Self

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
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class PipelineResult:
    success: bool
    steps: list[StepResult]


class StepFail(StepResult, Exception):
    pass


class PipelineFail(PipelineResult, Exception):
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


async def run_step(image: str, workspace: Workspace,
                   commands: list[str],
                   environment: Mapping[str, str]) -> StepResult:
    with NamedTemporaryFile(mode='w+', suffix='.sh') as script:
        script.write('set -e\n')
        for name, value in environment.items():
            if ENV_NAME.match(name) is None:
                raise ValueError('invalid environment variable name')
            script.write(f'{name}={shlex.quote(value)}\n')
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
            raise StepFail(proc.returncode, stdout.decode(), stderr.decode())
        return StepResult(proc.returncode, stdout.decode(), stderr.decode())


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
                        image, work, s['commands'],
                        self.step_env(s.get('environment', {})))
                    logger.info(
                        '%s, step %s: done', self.name, s['name'])
                    results.append(result)
                except StepFail as result:
                    logger.error(
                        '%s, step %s: fail', self.name, s['name'])
                    results.append(result)
                    raise PipelineFail(False, results)
            return PipelineResult(True, results)

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

    pipelines = dict()
    for pipeline in args.search.glob('*.yaml'):
        logger.debug(f'found pipeline: {pipeline}')
        with pipeline.open() as fh:
            p = yaml.safe_load(fh)
        pipelines[pipeline.stem.lstrip('.')] = p

    for batch in toposort(dict(
            (name, set(x.get('depends_on', [])))
            for name, x in pipelines.items())):
        # TODO: retrieve and log pipeline results
        async with asyncio.TaskGroup() as tg:
            for p in batch:
                for job in Pipeline.load(p, pipelines[p], ci_env):
                    tg.create_task(job.run())


def main(cmdline=None):
    asyncio.run(run(cmdline))


if __name__ == '__main__':
    main()
