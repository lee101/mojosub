"""Locating and driving the Mojo compiler."""

from __future__ import annotations

import functools
import os
import shutil
import signal
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


# A build that does not finish has to end as a refusal, like any other build
# failure, because the tiered design can only fall back to CPython if something
# comes back. Some units make the compiler spin: one program out of the fuzzer
# ran `mojo build` for two days on a single 35-line function before it was
# noticed, holding a core the whole time. Minutes are already far outside the
# 3-5s a real build takes.
BUILD_TIMEOUT = 300.0


def build_timeout() -> float:
    return float(os.environ.get("MOJOSUB_BUILD_TIMEOUT", BUILD_TIMEOUT))


def build_shared_lib(src_path: str, out_path: str, opt: int = 3,
                     timeout: float | None = None) -> None:
    cmd = mojo_command() + [
        "build", "--emit", "shared-lib", f"--optimization-level={opt}",
        src_path, "-o", out_path,
    ]
    limit = build_timeout() if timeout is None else timeout
    # Its own session, so the timeout can kill the compiler's children too.
    # `subprocess.run(timeout=...)` kills only the process it started and then
    # waits on the pipes, which a surviving grandchild holds open - the timeout
    # that was supposed to bound the build hangs on the build.
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        start_new_session=True,
    )
    try:
        out, err = proc.communicate(timeout=limit)
    except subprocess.TimeoutExpired:
        _kill_group(proc)
        raise ToolchainError(
            f"mojo build did not finish within {limit:g}s "
            f"(raise MOJOSUB_BUILD_TIMEOUT): {src_path}"
        ) from None
    if proc.returncode != 0 or not os.path.exists(out_path):
        raise ToolchainError((err or out).strip()[:4000])


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (OSError, AttributeError):
        proc.kill()
    try:
        proc.communicate(timeout=30)
    except subprocess.TimeoutExpired:  # pragma: no cover - the pipes are gone
        pass
