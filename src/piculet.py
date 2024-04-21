# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright: Fiona Klute
import argparse
import asyncio
import json
import logging
import re
import shlex
import shutil
import subprocess
import sys
import time
import uuid
import yaml
from collections import ChainMap
from collections.abc import Iterable, Mapping
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
logger = logging.getLogger(__name__ if __name__ != '__main__' else 'piculet')


@dataclass(frozen=True)
class PiculetConfig:
    repo: Path
    ci_env: dict[str, str]
    keep_workspace: bool = False
    output: Path | None = None


class Workspace:
    volume: str
    base: Path
    path: Path

    def __init__(self, config: dict[str, str] | None,
                 labels: dict[str, str] | None = None,
                 keep=False):
        if config is None:
            self.base = Path('/work')
            self.path = Path('.')
        else:
            self.base = Path(config.get('base', '/work'))
            self.path = Path(config.get('path', '.'))
            if self.path.is_absolute():
                raise ValueError('workspace.path must be relative')

        self.labels: list[str] = []
        if labels is not None:
            for name, value in labels.items():
                self.labels.extend(['--label', f'piculet.{name}={value}'])

        self.keep = keep
        self.volume = f'piculet-{uuid.uuid4()}'

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
            'podman', 'volume', 'create', *self.labels, self.volume,
            stdout=subprocess.PIPE)
        try:
            stdout, _ = await proc.communicate()
            assert proc.returncode == 0
            vol_id = stdout.strip().decode()
            assert self.volume == vol_id
        except Exception:
            await self.cleanup()
            raise
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if not self.keep:
            await self.cleanup()
        else:
            logger.debug(
                'keeping volume %s %s', self.volume, self.labels)

    @staticmethod
    async def prune_repo_volumes(repo: Path):
        proc = await asyncio.create_subprocess_exec(
            'podman', 'volume', 'prune', '--force',
            '--filter', f'label=piculet.repo={repo.resolve()}',
            stdout=asyncio.subprocess.PIPE)
        try:
            stdout, _ = await proc.communicate()
            logger.debug('repo cleanup done: %s', stdout.decode().split())
            assert proc.returncode == 0
        finally:
            await proc.wait()


@dataclass(frozen=True)
class StepResult:
    name: str
    returncode: int
    stdout: str
    stderr: str

    def report(self, verbose=False):
        buf = StringIO()
        buf.write(f'step "{self.name}" returned {self.returncode}')
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
    time: float | int
    volume: str | None = None

    @property
    def cancelled(self) -> bool:
        return isinstance(self, PipelineCancelled)

    @property
    def status(self):
        if self.success:
            return 'passed'
        elif self.cancelled:
            return 'cancelled'
        else:
            return 'failed'

    def report(self, verbose: bool | None = None):
        if verbose is None:
            verbose = logger.isEnabledFor(logging.DEBUG)
        buf = StringIO()
        buf.write(self.name)
        buf.write(': ')
        buf.write(self.status)
        if not self.cancelled:
            buf.write(f' ({self.time:.2f}s)')
        if self.volume is not None:
            buf.write('\npreserved volume: ')
            buf.write(self.volume)
        if not self.success or verbose:
            for s in self.steps:
                buf.write('\n')
                buf.write(s.report(verbose))
        return buf.getvalue()

    @staticmethod
    def _nonword_repl(matchobj):
        if any(g is not None for g in matchobj.groups()):
            return ''
        return '-'

    def to_file(self, output: Path):
        name = re.sub(r'(^)?[^\w\.]+($)?', self._nonword_repl, self.name)
        log = output / f'{name}.log'
        log.write_text(self.report(verbose=True))


class StepFail(StepResult, Exception):
    pass


class PipelineFail(PipelineResult, Exception):
    pass


class PipelineCancelled(PipelineResult, Exception):
    pass


async def clone_into(workspace: Workspace, repo: Path):
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


async def commit_info(rev: str, repo: Path) -> dict[str, str]:
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
        finally:
            await proc.wait()
        assert isinstance(proc.returncode, int)
        if proc.returncode != 0:
            raise StepFail(
                name, proc.returncode, stdout.decode(), stderr.decode())
        return StepResult(
            name, proc.returncode, stdout.decode(), stderr.decode())


class Pipeline:
    def __init__(self, name: str, config, picu: PiculetConfig,
                 matrix_element=None):
        self._name = name
        self.config = config
        assert isinstance(picu, PiculetConfig)
        self.picu = picu
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
        return Workspace(
            self.config.get('workspace'),
            {
                'repo': str(self.picu.repo),
                'pipeline': self.name,
            },
            keep=self.picu.keep_workspace)

    def step_env(self, env):
        return ChainMap(self.picu.ci_env, env, self.matrix_element)

    async def run(self) -> PipelineResult:
        start = time.time()
        async with self.workspace as work:
            if not self.config.get('skip_clone', False):
                await clone_into(work, self.picu.repo)
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
                    raise PipelineFail(
                        self.name, False, results, time.time() - start,
                        work.volume if self.picu.keep_workspace else None)
        return PipelineResult(
            self.name, True, results, time.time() - start,
            work.volume if self.picu.keep_workspace else None)

    @classmethod
    def load(cls, name: str, pipeline, picu: PiculetConfig) -> list[Self]:
        """Create Pipeline objects from pipeline config. If the config
        defines a matrix the returned list will contain one Pipeline
        per matrix combination, otherwise exactly one Pipeline."""
        if 'matrix' in pipeline:
            return [
                cls(name, pipeline, picu, dict(x))
                for x in product(*([*zip(repeat(k), pipeline['matrix'][k])]
                                   for k in pipeline['matrix'].keys()))]
        else:
            return [cls(name, pipeline, picu)]


async def run_pipelines_ordered(pipelines: dict[str, Any],
                                picu: PiculetConfig):
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
                                return [PipelineCancelled(name, False, [], 0)]
                    logger.debug('pipeline task %s awaited dependencies', name)
                return await asyncio.gather(
                    *(asyncio.create_task(job.run())
                      for job in Pipeline.load(name, pipelines[name], picu)),
                    return_exceptions=True)

            tasks[name] = asyncio.create_task(pipeline_task(name))

    for t in asyncio.as_completed(tasks.values()):
        for result in await t:
            if result.success:
                logger.info(result.report())
            else:
                logger.error(result.report())
            if picu.output is not None:
                result.to_file(picu.output)
            yield result


def find_pipelines(search: Iterable[Path], endings=('*.yaml', '*.yml')):
    """For each element of "search", if it is a directory, search
    using all patterns in endings, otherwise just use the file."""
    pipelines = dict()
    for s in search:
        for e in endings if s.is_dir() else (None,):
            for pipeline in s.glob(e) if s.is_dir() else (s,):
                logger.debug(f'found pipeline: {pipeline}')
                name = pipeline.stem.lstrip('.')
                if name in pipelines:
                    raise ValueError('duplicate pipeline name')
                with pipeline.open() as fh:
                    pipelines[name] = yaml.safe_load(fh)
    return pipelines


RUN_DEFAULTS = {
    'repo': Path('.'),
    'config': '.piculet.yaml',
    'log_level': 'INFO',
    'keep_workspace': False,
    'pipelines': ['.woodpecker/'],
}


class DebugEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, ChainMap):
            return dict(obj)
        return super().default(obj)


async def run(cmdline: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description='Tiny local CI engine using Podman',
        argument_default=argparse.SUPPRESS,
        epilog='Any relative paths will be interpreted relative to REPO.')
    parser.add_argument(
        '--repo', metavar='REPO',
        type=Path, help='repository to run CI for, '
        f'default: {RUN_DEFAULTS["repo"]!s}')
    parser.add_argument(
        '--config', type=Path,
        help=f'configuration file to use, default: {RUN_DEFAULTS["config"]!s}')
    parser.add_argument(
        '--log-level', metavar='LEVEL',
        help=f'log level for Piculet, default: {RUN_DEFAULTS["log_level"]}')
    parser.add_argument(
        '--output', metavar='DIR',
        help='if set, logs and report will be written to this directory')
    parser.add_argument(
        '--keep-workspace', action='store_true',
        help='keep pipeline workspace volumes for debugging, '
        'any old ones for this repository are deleted before the run')
    parser.add_argument(
        'pipelines', metavar='PIPELINE',
        type=Path, nargs='*',
        help='Pipeline locations, can be files or directories to search '
        f'for pipelines (*.yaml/*.yml). Default: {RUN_DEFAULTS["pipelines"]}')

    # enable bash completion if argcomplete is available
    try:
        import argcomplete
        argcomplete.autocomplete(parser)
    except ImportError:
        pass

    cmdline_args = parser.parse_args(args=cmdline)
    args = ChainMap(vars(cmdline_args), RUN_DEFAULTS)
    conffile = args['repo'] / args['config']
    if conffile.is_file():
        with conffile.open() as fh:
            config = yaml.safe_load(fh)
        args.maps.insert(1, config)
    elif 'config' in cmdline_args:
        raise ValueError('config file given on command line does not exist')
    logger.setLevel(args['log_level'])
    logging.basicConfig(level=args['log_level'])
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            'effective config: %s', json.dumps(args, cls=DebugEncoder))

    picu = PiculetConfig(
        repo=Path(args['repo']).resolve(),
        ci_env=await commit_info('HEAD', args['repo']),
        keep_workspace=args['keep_workspace'],
        output=(Path(args['repo']) / Path(args['output'])
                if 'output' in args else None),
    )
    pipelines = find_pipelines(picu.repo / p for p in args['pipelines'])

    if picu.keep_workspace:
        await Workspace.prune_repo_volumes(picu.repo)

    if picu.output is not None:
        if not picu.output.exists():
            picu.output.mkdir()
            (picu.output / '.gitignore').write_text('*\n')
        elif not picu.output.is_dir():
            raise NotADirectoryError('output must be a directory')

    fails = 0
    async for ret in run_pipelines_ordered(pipelines, picu):
        if not ret.success:
            fails += 1
    return fails


def main(cmdline=None):
    return asyncio.run(run(cmdline))


if __name__ == '__main__':
    sys.exit(main())
