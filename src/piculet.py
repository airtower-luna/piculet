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
class JobResult:
    success: bool
    steps: list[StepResult]


async def clone_into(workspace: Workspace, repo: Path = Path('.')):
    """copy source into volume"""
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
                '-v', f'{script.name}:/{script_mount}',
                '-v', f'{workspace.volume}:{workspace.base}',
                '--workdir', workspace.workdir,
                image, 'sh', f'/{script_mount}',
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = await proc.communicate()
        except asyncio.CancelledError:
            await proc.wait()
            raise
        assert isinstance(proc.returncode, int)
        return StepResult(proc.returncode, stdout.decode(), stderr.decode())


async def run_job(pipeline, ci_env: dict[str, str], matrix_element) \
          -> JobResult:
    async with Workspace(pipeline.get('workspace')) as work:
        results = list()
        for s in pipeline['steps']:
            image = Template(s['image']).safe_substitute(matrix_element)
            result = await run_step(
                image, work, s['commands'],
                ChainMap(ci_env, s.get('environment', {}), matrix_element))
            results.append(result)
            if result.returncode != 0:
                print(f'step {s["name"]} ({matrix_element}) failed')
                print('------ stdout ------')
                print(result.stdout)
                print('------ stderr ------')
                print(result.stderr)
                print('------------')
                return JobResult(False, results)
            else:
                print(f'step {s["name"]} ({matrix_element}) done')
        return JobResult(True, results)


async def run_pipeline(pipeline: Path, ci_env: dict[str, str]):
    with pipeline.open() as fh:
        p = yaml.safe_load(fh)

    if 'matrix' in p:
        matrix = [
            dict(x)
            for x in product(*([*zip(repeat(k), p['matrix'][k])]
                               for k in p['matrix'].keys()))]
    else:
        matrix = [dict()]

    async with asyncio.TaskGroup() as tg:
        for elem in matrix:
            tg.create_task(run_job(p, ci_env, elem))


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
    async with asyncio.TaskGroup() as tg:
        for pipeline in args.search.glob('*.yaml'):
            print(pipeline)
            tg.create_task(run_pipeline(pipeline, ci_env))


def main(cmdline=None):
    asyncio.run(run(cmdline))


if __name__ == '__main__':
    main()
