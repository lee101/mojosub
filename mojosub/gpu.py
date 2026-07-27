"""GPU code generation for the per-element loop shape.

A function whose whole body is `for i in range(n): ...` — one independent
iteration per element — maps directly onto a CUDA grid. mojosub emits the loop
body as a kernel, copies the buffers across, launches, and copies back.

**When this is worth doing.** Measured on an RTX 5090 over 1e6 float64
elements, varying the arithmetic per element:

| flops/byte | GPU | CPU (SIMD) | speedup |
| ---: | ---: | ---: | ---: |
| 0.2 | 4.2 ms | 2.6 ms | 0.6x |
| 1.9 | 3.6 ms | 8.9 ms | 2.5x |
| 18.8 | 6.2 ms | 220 ms | 35x |
| 187.5 | 9.1 ms | 2396 ms | **263x** |

The crossover is at roughly 2 flops per byte. Below it the PCIe round trip
costs more than the arithmetic saves and the GPU cannot win, no matter how
large the array — a single elementwise pass over host memory is transfer
bound at every size tested up to 50M elements. So this is opt-in
(`target="gpu"`), not automatic: mojosub cannot see how much arithmetic a
runtime trip count implies.
"""

from __future__ import annotations

import ast

from .transpile import Signature, Unit, Unsupported, _Fn
from .types import INT, NONE, Array

BLOCK = 256

GPU_PRELUDE = (
    "from std.gpu.host import DeviceContext\n"
    "from std.gpu import thread_idx, block_idx, block_dim\n"
    "from std.math import sqrt, exp, log, log2, log10, sin, cos, tan, tanh, "
    "atan, asin, acos, sinh, cosh, floor, ceil, erf, atan2\n"
)


def _element_loop(fn: ast.FunctionDef) -> ast.For:
    """The single `for i in range(...)` that makes up the function body."""
    body = [s for s in fn.body if not (isinstance(s, ast.Expr)
                                       and isinstance(s.value, ast.Constant))]
    loops = [s for s in body if isinstance(s, ast.For)]
    others = [s for s in body if not isinstance(s, ast.For)]
    if len(loops) != 1:
        raise Unsupported("a GPU function must be exactly one element loop")
    for stmt in others:
        if not isinstance(stmt, ast.Return):
            raise Unsupported("statements outside the element loop do not map to a grid")
    loop = loops[0]
    if not isinstance(loop.target, ast.Name):
        raise Unsupported("element loop must bind a single index")
    if not (isinstance(loop.iter, ast.Call)
            and isinstance(loop.iter.func, ast.Name)
            and loop.iter.func.id == "range"
            and len(loop.iter.args) == 1):
        raise Unsupported("element loop must be range(n)")
    return loop


def transpile_gpu(tree: ast.Module, sig: Signature, entry: str) -> Unit:
    """Render `entry` as a GPU kernel plus a host-side launching wrapper.

    The wrapper owns the whole round trip: allocate, copy in, launch, copy
    back, synchronize. Every buffer argument is copied both ways — telling
    which ones are outputs would need an escape analysis that is not worth its
    error modes here, and the copy is symmetric anyway.
    """
    fndef = next(
        (n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == entry),
        None,
    )
    if fndef is None:
        raise Unsupported(f"no function named {entry}")
    if sig.ret is not NONE:
        raise Unsupported("a GPU function returns through its buffers, not a value")
    if not any(isinstance(t, Array) for _, t in sig.params):
        raise Unsupported("a GPU function needs at least one buffer argument")

    loop = _element_loop(fndef)
    index = loop.target.id

    # The kernel takes the same parameters, plus the element count.
    kernel_sig = Signature(f"__kernel_{entry}", list(sig.params), NONE)
    emitter = _Fn({}, kernel_sig, vectorize=False)
    emitter.env[index] = INT
    count_code, count_ty = emitter.expr(loop.iter.args[0])
    if count_ty is not INT:
        raise Unsupported("element count must be an integer")
    emitter.block(loop.body, 2)
    body = "\n".join(emitter.lines)

    params = []
    for name, ty in sig.params:
        params.append(f"{name}: {ty.mojo}")
        if isinstance(ty, Array):
            params.append(f"{name}_len: Int")

    kernel = (
        f"def __kernel_{entry}({', '.join(params)}):\n"
        f"    var {index} = Int(thread_idx.x + block_idx.x * block_dim.x)\n"
        f"    if {index} < {count_code}:\n"
        f"{body}\n"
    )

    # host wrapper
    wrap_params, allocs, copies_in, launch_args, copies_back = [], [], [], [], []
    for name, ty in sig.params:
        if isinstance(ty, Array):
            wrap_params.append(f"{name}_addr: Int")
            wrap_params.append(f"{name}_len: Int")
            host = f"__h_{name}"
            dev = f"__d_{name}"
            allocs.append(
                f"        var {host} = {ty.mojo}(unsafe_from_address={name}_addr)"
            )
            allocs.append(
                f"        var {dev} = ctx.enqueue_create_buffer"
                f"[DType.float64]({name}_len)"
            )
            copies_in.append(f"        ctx.enqueue_copy({dev}, {host})")
            copies_back.append(f"        ctx.enqueue_copy({host}, {dev})")
            launch_args.append(f"{dev}.unsafe_ptr()")
            launch_args.append(f"{name}_len")
        else:
            wrap_params.append(f"{name}: {ty.mojo}")
            launch_args.append(name)

    # the element count already refers to a `<buffer>_len` parameter, which
    # the wrapper takes verbatim
    grid_source = count_code

    lines = [
        f'@export("mojosub_{entry}")',
        f'def mojosub_wrap_{entry}({", ".join(wrap_params)}) abi("C") -> Int:',
        "    try:",
        "        var ctx = DeviceContext()",
    ]
    lines += allocs + copies_in
    lines.append(f"        var __n = {grid_source}")
    lines.append(f"        ctx.enqueue_function[__kernel_{entry}](")
    lines.append(f"            {', '.join(launch_args)},")
    lines.append(
        f"            grid_dim=(__n + {BLOCK - 1}) // {BLOCK}, block_dim={BLOCK},"
    )
    lines.append("        )")
    lines += copies_back
    lines.append("        ctx.synchronize()")
    lines.append("        return 0")
    lines.append("    except:")
    lines.append("        return 1")
    wrapper = "\n".join(lines)

    src = GPU_PRELUDE + "\n" + kernel + "\n" + wrapper + "\n"
    return Unit(source=src, signatures={entry: sig})
