# Piculet

Piculet is a tiny local-only CI engine that implements a subset of the
[Woodpecker CI](https://woodpecker-ci.org/) syntax, highly
experimental. It uses Podman to run containers for steps.

The name comes from a [subfamily of tiny
woodpeckers](https://en.wikipedia.org/wiki/Piculet).

## Usage

Call `src/piculet.py`, or install using `pip`/`pipx` and use the
`piculet` command. Pass pipeline files or directories holding pipeline
files. For example, if you've just cloned the repository and want to
run the test pipelines in `tests/pipelines/`:

```sh
python3 src/piculet.py tests/pipelines/
```

Or just one:

```sh
python3 src/piculet.py tests/pipelines/build.yaml
```

Piculet generally expects to be called from the root of your
repository, use `--repo` with the path if you want to call if from
elsewhere. Check `--help`/`-h` for more options.

## Configuration

Piculet will check for a `.piculet.yaml` file in the root of your
repository, or wherever you point it with the `--config` option. See
[example-config.yaml](example-config.yaml) for an annotated
example. Config options are the same as command line options, with the
latter taking precedence.

## Debugging builds

If you want to inspect build data after a build, use
`--keep-workspace` or set `keep-workspace: true` in the config. In
that case Piculet will preserve the latest set of workspace volumes
after the build (old ones for the same repository are deleted at the
start of a new build if the option is enabled). You can then use
`podman unshare` and `podman mount` to mount and inspect the
workspace(s) you're interested in.

## Features

Piculet supports a subset of the [Woodpecker pipeline
syntax](https://woodpecker-ci.org/docs/usage/workflow-syntax). Supported
are:

* `matrix`: Matrix values are substituted in `steps.*.image` and
  provided to `commands` as environment variables. Only `${VAR}`
  substitutions, no more complex pre-processing.
* `workspace`
* `skip_clone`
* `steps` with `name`, `image`, and `commands` (no plugins). Note that
  the `steps` element of the pipeline must be a list, mappings are not
  supported.
* `depends_on` between pipelines (not steps)
* The `CI_COMMIT_SHA` and `CI_COMMIT_REF` default environment
  variables.

The aim is that Piculet pipelines should work unmodified in Woodpecker
CI (assuming images are available, etc.).
