"""Locating and driving the Mojo compiler."""

from __future__ import annotations

import functools
import os
import shutil
import subprocess


class ToolchainError(RuntimeError):
    pass


@functools.lru_cache(maxsize=1)
def mojo_command() -> list[str]:
    """The argv prefix that runs `mojo`.

    `MOJOSUB_MOJO` wins; then `mojo` on PATH; then `pixi run mojo` inside
    `MOJOSUB_PIXI_PROJECT` (or the repo's own pixi environment).
    """
    override = os.environ.get("MOJOSUB_MOJO")
    if override:
        return override.split()
    found = shutil.which("mojo")
    if found:
        return [found]
    project = os.environ.get("MOJOSUB_PIXI_PROJECT")
    pixi = shutil.which("pixi") or os.path.expanduser("~/.pixi/bin/pixi")
    if project and os.path.exists(pixi):
        return [pixi, "run", "--manifest-path", os.path.join(project, "pixi.toml"), "mojo"]
    raise ToolchainError(
        "mojo not found: set MOJOSUB_MOJO=/path/to/mojo or MOJOSUB_PIXI_PROJECT"
    )


@functools.lru_cache(maxsize=1)
def mojo_version() -> str:
    out = subprocess.run(
        mojo_command() + ["--version"], capture_output=True, text=True, timeout=120
    )
    if out.returncode != 0:
        raise ToolchainError(out.stderr.strip() or "mojo --version failed")
    return out.stdout.strip()


def build_shared_lib(src_path: str, out_path: str, opt: int = 3) -> None:
    cmd = mojo_command() + [
        "build", "--emit", "shared-lib", f"--optimization-level={opt}",
        src_path, "-o", out_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if proc.returncode != 0 or not os.path.exists(out_path):
        raise ToolchainError((proc.stderr or proc.stdout).strip()[:4000])
