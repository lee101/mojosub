"""Content-addressed compile cache.

A unit's key is the hash of its Mojo source plus the compiler version and
flags, so a cache hit is safe across processes and machines with the same
toolchain. Builds are done to a temp file and renamed, which makes a
concurrent build racy but never corrupt.
"""

from __future__ import annotations

import hashlib
import os
import tempfile

from .toolchain import build_shared_lib, mojo_version

DEFAULT_ROOT = os.path.expanduser(
    os.environ.get("MOJOSUB_CACHE", "~/.cache/mojosub")
)


def key_for(source: str, opt: int) -> str:
    h = hashlib.sha256()
    h.update(mojo_version().encode())
    h.update(f"|O{opt}|v1|".encode())
    h.update(source.encode())
    return h.hexdigest()[:32]


def lib_path(key: str, root: str = DEFAULT_ROOT) -> str:
    return os.path.join(root, key + ".so")


def lookup(source: str, opt: int = 3, root: str = DEFAULT_ROOT) -> str | None:
    path = lib_path(key_for(source, opt), root)
    return path if os.path.exists(path) else None


def compile_unit(source: str, opt: int = 3, root: str = DEFAULT_ROOT) -> str:
    """Return the path to a shared library for `source`, building if needed."""
    key = key_for(source, opt)
    final = lib_path(key, root)
    if os.path.exists(final):
        return final
    os.makedirs(root, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root) as tmp:
        src_path = os.path.join(tmp, "unit.mojo")
        with open(src_path, "w") as fh:
            fh.write(source)
        out_path = os.path.join(tmp, "unit.so")
        build_shared_lib(src_path, out_path, opt=opt)
        os.replace(out_path, final)
        with open(final + ".mojo", "w") as fh:
            fh.write(source)
    return final


def clear(root: str = DEFAULT_ROOT) -> int:
    n = 0
    if not os.path.isdir(root):
        return 0
    for name in os.listdir(root):
        if name.endswith(".so") or name.endswith(".so.mojo"):
            os.remove(os.path.join(root, name))
            n += 1
    return n
