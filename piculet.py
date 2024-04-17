# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright: Fiona Klute
import socket
import subprocess
import uuid
import yaml
from contextlib import contextmanager
from itertools import product, repeat
from pathlib import Path
from string import Template

WORKSPACE = '/work'


@contextmanager
def work_container(image: str, opts: list[str] = []):
    proc = subprocess.run(
        [
            'podman', 'run', '--detach', '--rm', *opts,
            image, 'sleep', '1200'],
        check=True, stdout=subprocess.PIPE)
    try:
        container_id = proc.stdout.strip().decode()
        yield container_id
    finally:
        subprocess.run(
            ['podman', 'stop', '-t', '1', container_id],
            check=True)
        subprocess.run(
            ['podman', 'wait', '--ignore', container_id],
            check=True)


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
            check=True)


def run_ci(image: str, workspace: str, commands: list[str]):
    with work_container(image, ['-v', f'{workspace}:{WORKSPACE}']) \
         as container_id:
        # run commands
        for c in commands:
            subprocess.run(
                [
                    'podman', 'exec', '--workdir', WORKSPACE, container_id,
                    'sh', '-c', c],
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
                # TODO: add params to env
                run_ci(image, work, s['commands'])


if __name__ == '__main__':
    for pipeline in Path('.woodpecker/').glob('*.yaml'):
        print(pipeline)
        run_pipeline(pipeline)
