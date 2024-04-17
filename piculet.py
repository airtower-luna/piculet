# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright: Fiona Klute
import re
import shlex
import socket
import subprocess
import uuid
import yaml
from contextlib import contextmanager
from itertools import product, repeat
from pathlib import Path
from string import Template
from tempfile import NamedTemporaryFile

WORKSPACE = '/work'
ENV_NAME = re.compile(r'^\w[\w\d]*$')


@contextmanager
def workspace():
    vol_name = str(uuid.uuid4())
    proc = subprocess.run(
        ['podman', 'volume', 'create', vol_name],
        check=True, stdout=subprocess.PIPE)
    try:
        vol_id = proc.stdout.strip().decode()
        assert vol_name == vol_id
        # copy source into volume
        source, dest = socket.socketpair()
        git = subprocess.Popen(
            ['git', 'archive', '--format=tar', 'HEAD'],
            stdout=source.fileno())
        recv = subprocess.Popen(
            ['podman', 'volume', 'import', vol_name, '-'],
            stdin=dest.fileno())
        git.wait()
        source.shutdown(socket.SHUT_RDWR)
        recv.wait()
        source.close()
        dest.close()
        assert git.returncode == 0
        assert recv.returncode == 0

        yield vol_name
    finally:
        subprocess.run(
            ['podman', 'volume', 'rm', vol_name],
            check=True, stdout=subprocess.DEVNULL)


def run_step(image: str, workspace: str,
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
        subprocess.run(
            [
                'podman', 'run', '--rm',
                '-v', f'{script.name}:/{script_mount}',
                '-v', f'{workspace}:{WORKSPACE}', '--workdir', WORKSPACE,
                image, 'sh', f'/{script_mount}'],
            check=True)


def run_pipeline(pipeline: Path):
    with pipeline.open() as fh:
        p = yaml.safe_load(fh)

    if 'matrix' in p:
        params = [
            dict(x)
            for x in product(*([*zip(repeat(k), p['matrix'][k])]
                               for k in p['matrix'].keys()))]
    else:
        params = [dict()]

    for param in params:
        print(param)
        with workspace() as work:
            for s in p['steps']:
                image = Template(s['image']).safe_substitute(param)
                run_step(image, work, s['commands'], param)


if __name__ == '__main__':
    for pipeline in Path('.woodpecker/').glob('*.yaml'):
        print(pipeline)
        run_pipeline(pipeline)
