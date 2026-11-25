"""Content-addressed compile cache.

A unit's key is the hash of its Mojo source, the compiler version and flags,
and the instruction set of the machine doing the building (see `host_target`) —
so a cache hit is safe across processes, and across machines that can actually
run each other's code. Builds are done to a temp file and renamed, which makes a
concurrent build racy but never corrupt.

The cache also remembers **inferred signatures**. Knowing that a library exists
is not enough to call it: the transpiler needs the return type, which is
normally learned by running the function under CPython once. Recording it here
means a function whose library is already on disk can be dispatched natively on
its *first* call in a new process - which is the difference between the JIT
helping and not helping at all for code that runs once per request, like an
agent's generated pandas loop.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import platform
import tempfile

from .toolchain import build_shared_lib, mojo_version

DEFAULT_ROOT = os.path.expanduser(
    os.environ.get("MOJOSUB_CACHE", "~/.cache/mojosub")
)


@functools.lru_cache(maxsize=1)
def host_target() -> str:
    """An identifier for the instruction set this machine's builds will use.

    `mojo build` documents `--target-cpu` as "Defaults to the host CPU", so an
    artifact is only loadable on a machine whose instruction set is a superset
    of the builder's. That is invisible on one box and fatal across a fleet: a
    unit built on an AVX-512 worker, synced through the shared object store, and
    `dlopen`ed on an AVX2 box does not fail to load — it runs, and dies on the
    first vector instruction it was compiled to use.

    So the CPU feature set goes in the cache key. Two machines that can run each
    other's code share entries; two that cannot get their own. The flags are
    what actually decides that, which is why they are hashed rather than the
    model name — the same model number ships with features fused off.
    """
    parts = [platform.machine(), platform.system()]
    try:
        with open("/proc/cpuinfo", "r") as fh:
            for line in fh:
                if line.startswith("flags") or line.startswith("Features"):
                    parts.append(" ".join(sorted(line.split(":", 1)[1].split())))
                    break
    except OSError:
        # No /proc: fall back to the coarse identity. Over-sharing is not
        # possible here, only under-sharing, because the machine string still
        # separates architectures.
        pass
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:12]


def key_for(source: str, opt: int) -> str:
    h = hashlib.sha256()
    h.update(mojo_version().encode())
    h.update(f"|O{opt}|v2|{host_target()}|".encode())
    h.update(source.encode())
    return h.hexdigest()[:32]


def lib_path(key: str, root: str | None = None) -> str:
    return os.path.join(_root(root), key + ".so")


def lookup(source: str, opt: int = 3, root: str | None = None) -> str | None:
    path = lib_path(key_for(source, opt), root)
    return path if os.path.exists(path) else None


def compile_unit(source: str, opt: int = 3, root: str | None = None) -> str:
    """Return the path to a shared library for `source`, building if needed."""
    key = key_for(source, opt)
    final = lib_path(key, root)
    if os.path.exists(final):
        return final
    os.makedirs(_root(root), exist_ok=True)
    with tempfile.TemporaryDirectory(dir=_root(root)) as tmp:
        src_path = os.path.join(tmp, "unit.mojo")
        with open(src_path, "w") as fh:
            fh.write(source)
        out_path = os.path.join(tmp, "unit.so")
        build_shared_lib(src_path, out_path, opt=opt)
        os.replace(out_path, final)
        with open(final + ".mojo", "w") as fh:
            fh.write(source)
    return final


# -- remembered signatures -------------------------------------------------


def _root(root: str | None) -> str:
    """Resolved at call time, so a test (or a caller) can redirect the cache
    by setting `DEFAULT_ROOT` without every default argument having captured
    the old value at import."""
    return root or DEFAULT_ROOT


def _sig_path(source_key: str, root: str | None = None) -> str:
    return os.path.join(_root(root), "sigs", source_key + ".json")


def source_key(py_source: str) -> str:
    """Key for a *Python* function body, independent of the toolchain.

    Deliberately not `key_for`: a signature is a fact about the function, not
    about the compiler, so it survives a Mojo upgrade that invalidates every
    compiled library.
    """
    h = hashlib.sha256()
    h.update(b"pysig|v1|")
    h.update(py_source.encode())
    return h.hexdigest()[:32]


def _read_table(py_source: str, root: str) -> dict:
    try:
        with open(_sig_path(source_key(py_source), root)) as fh:
            table = json.load(fh)
    except (OSError, ValueError):
        return {}
    # An older cache stored the return type as a bare string.
    return {
        k: (v if isinstance(v, dict) else {"ret": v})
        for k, v in table.items()
    }


def _write_entry(py_source: str, argtypes: tuple, patch: dict, root: str) -> None:
    path = _sig_path(source_key(py_source), root)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    table = _read_table(py_source, root)
    entry = table.get("|".join(argtypes), {})
    entry.update(patch)
    table["|".join(argtypes)] = entry
    tmp = path + f".{os.getpid()}.tmp"
    with open(tmp, "w") as fh:
        json.dump(table, fh)
    os.replace(tmp, path)  # concurrent writers race, never tear


def remember_signature(py_source: str, argtypes: tuple, ret: str,
                       root: str | None = None) -> None:
    """Record what this function returned for these argument types."""
    _write_entry(py_source, argtypes, {"ret": ret}, root)


def recall_signature(py_source: str, argtypes: tuple,
                     root: str | None = None) -> str | None:
    return _read_table(py_source, root).get("|".join(argtypes), {}).get("ret")


def remember_verdict(py_source: str, argtypes: tuple, *, verified: bool | None = None,
                     retired: str | None = None, raced: bool | None = None,
                     speedup: float | None = None, root: str | None = None) -> None:
    """Record the outcome of checking a compiled variant against CPython.

    Verification costs a full interpreted run, so doing it once per *process*
    means a function called once per request is checked on every request and
    the compiled path never actually replaces anything - which measured as a
    1.4x gain where the compiled code itself is 200x. The check is a fact about
    a (source, signature, toolchain) triple, so it belongs on disk with the
    library it is about.

    `retired` is the same decision in the other direction: a variant that
    disagreed with CPython, or was slower than it, is not worth re-discovering
    in every worker.

    `raced` records that the two tiers have been timed against each other and
    the compiled one won, with `speedup` as the measured ratio. Without it the
    race repeats in every process, and the race costs interpreted calls -
    exactly what compiling was supposed to stop paying for.
    """
    patch: dict = {}
    if verified is not None:
        patch["verified"] = verified
    if retired is not None:
        patch["retired"] = retired
    if raced is not None:
        patch["raced"] = raced
    if speedup is not None:
        patch["speedup"] = round(float(speedup), 3)
    if patch:
        _write_entry(py_source, argtypes, patch, root)


def recall_verdict(py_source: str, argtypes: tuple,
                   root: str | None = None) -> dict:
    return _read_table(py_source, root).get("|".join(argtypes), {})


def clear(root: str | None = None) -> int:
    n = 0
    root = _root(root)
    if not os.path.isdir(root):
        return 0
    for name in os.listdir(root):
        if name.endswith(".so") or name.endswith(".so.mojo"):
            os.remove(os.path.join(root, name))
            n += 1
    sigs = os.path.join(root, "sigs")
    if os.path.isdir(sigs):
        for name in os.listdir(sigs):
            os.remove(os.path.join(sigs, name))
            n += 1
    return n
