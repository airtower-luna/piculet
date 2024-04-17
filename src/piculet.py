# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright: Fiona Klute
import argparse
import asyncio
import re
import shlex
import socket
import subprocess
import uuid
import yaml
from contextlib import asynccontextmanager
from itertools import product, repeat
from pathlib import Path
from string import Template
from tempfile import NamedTemporaryFile

WORKSPACE = '/work'
ENV_NAME = re.compile(r'^\w[\w\d]*$')


@asynccontextmanager
async def workspace():
    vol_name = str(uuid.uuid4())
    proc = await asyncio.create_subprocess_exec(
        'podman', 'volume', 'create', vol_name,
        stdout=subprocess.PIPE)
    try:
        stdout, _ = await proc.communicate()
        assert proc.returncode == 0
        vol_id = stdout.strip().decode()
        assert vol_name == vol_id
        # copy source into volume
        source, dest = socket.socketpair()
        git = await asyncio.create_subprocess_exec(
            'git', 'archive', '--format=tar', 'HEAD',
            stdout=source.fileno())
        recv = await asyncio.create_subprocess_exec(
            'podman', 'volume', 'import', vol_name, '-',
            stdin=dest.fileno())
        await git.wait()
        source.shutdown(socket.SHUT_RDWR)
        await recv.wait()
        source.close()
        dest.close()
        assert git.returncode == 0
        assert recv.returncode == 0

        yield vol_name
    finally:
        await asyncio.create_subprocess_exec(
            'podman', 'volume', 'rm', vol_name,
            stdout=subprocess.DEVNULL)


async def run_step(image: str, workspace: str,
                   commands: list[str], environment: dict[str, str]):
    with NamedTemporaryFile(mode='w+', suffix='.sh') as script:
        for name, value in environment.items():
            if ENV_NAME.match(name) is None:
                raise ValueError('invalid environment variable name')
            script.write(f'{name}={shlex.quote(value)}\n')
        script.write('\n')
        for c in commands:
            script.write(f'{c}\n')
        script.flush()
        script_mount = Path(script.name).name
        proc = await asyncio.create_subprocess_exec(
            'podman', 'run', '--rm',
            '-v', f'{script.name}:/{script_mount}',
            '-v', f'{workspace}:{WORKSPACE}', '--workdir', WORKSPACE,
            image, 'sh', f'/{script_mount}')
        await proc.wait()


async def run_job(steps, matrix_element):
    async with workspace() as work:
        for s in steps:
            image = Template(s['image']).safe_substitute(matrix_element)
            await run_step(image, work, s['commands'], matrix_element)


async def run_pipeline(pipeline: Path):
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
            print(elem)
            tg.create_task(run_job(p['steps'], elem))


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

    async with asyncio.TaskGroup() as tg:
        for pipeline in args.search.glob('*.yaml'):
            print(pipeline)
            tg.create_task(run_pipeline(pipeline))


def main():
    asyncio.run(run())


if __name__ == '__main__':
    main()
