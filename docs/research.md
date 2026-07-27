# How fast can arbitrary Python be executed as Mojo?

Notes from building mojosub. Everything here is measured on one machine
(Ryzen workstation, Mojo `1.0.0b3.dev2026072406`, CPython 3.13); the numbers
move, the shape of the result does not.

## The premise

An AI coding agent writes a numeric function, runs it, looks at the answer, and
writes another one. Under CPython the run is 50–100x slower than it needs to be.
Mojo compiles to native code through MLIR, so the ceiling is C. The question is
whether the *floor* — the cost of getting from a string to an answer — can be
brought anywhere near `exec`.

## Measurement 1: the compile is fixed cost

The single most important number. Compiling one generated unit:

| unit | wall clock | per function |
| --- | ---: | ---: |
| 1 function | 5.17 s | 5.17 s |
| 5 functions | 4.98 s | 1.00 s |
| 20 functions | 4.96 s | 0.25 s |

`mojo build` spends its time on process start, standard library load, and
linking. The actual code being compiled is nearly free at this size. Optimization
level barely registers either — `-O0` through `-O3` all land in the 3–5 s band,
inside the noise of a busy machine.

So there is no version of this where a single snippet compiles in milliseconds.
The design has to route around the compiler, not through it, and there are
exactly three ways to do that: cache it, hide it, or batch it. mojosub does all
three.

## Measurement 2: dispatch is nearly free

Once the library exists, calling into it is cheap:

| path | time |
| --- | ---: |
| raw `ctypes` call into an `@export`ed symbol | 0.87 us |
| bound `Callable` with scalar arguments | 1.16 us |
| `@jit` wrapper (type key + variant lookup + stats) | 2.04 us |
| in-process memo, from source string to answer | 8.6 us |
| disk cache hit (parse + transpile + hash + `dlopen`) | 0.64 ms |

The `@jit` wrapper costs about 0.9 us over the bare symbol, almost all of it
argument-type keying. Keying scalars off `type(arg)` rather than running full
inference cut that path from 3.4 us to 2.0 us. For anything running longer than
about 20 us of real work, dispatch overhead is invisible.

The disk-cache path is 500x slower than the memo path and it is *all* Python:
`ast.parse`, transpile, `sha256`, `dlopen`. Hence the memo.

## Measurement 3: where Mojo wins, and where it loses

Speedups run 24x to 112x on scalar and buffer kernels. Two things break the
pattern:

- **`list[float]` arguments.** A million-element list must be copied into an
  `array.array` before the call and copied back after, which costs more than the
  kernel saves — `dot(1e6)` goes from 36x faster on a numpy array to 3x *slower*
  on a list. The buffer protocol is the whole ballgame; anything exposing
  `__array_interface__` is passed by address.
- **Kernels already vectorized in numpy.** mojosub competes with the Python
  interpreter, not with BLAS. `np.dot` beats a scalar Mojo loop.

The sweet spot is exactly what agents write and numpy cannot express: loops with
data-dependent control flow. `sma_crossover` — a rolling window with branches —
is 112x, and there is no clean vectorization of it.

## Measurement 4: what the first call has to be

An unannotated `def kernel(n):` has no return type, and the transpiler needs one
before it can emit a signature. Running the function once under CPython supplies
it — and the same first call is the one that would otherwise block for five
seconds waiting on the compiler. So the tiered design is not a compromise; the
Python tier is doing necessary work. It answers immediately, it types the
signature, and the compiler runs behind it.

## Things that did not work

- **`@export` on a parametric function.** Any generic parameter, including an
  inferred origin (`UnsafePointer[Float64, _]`), makes the function parametric
  and `@export` refuses it. Pointers must cross the ABI as `Int` addresses and
  be reconstructed with an explicit `AnyOrigin[mut=True]`.
- **`mojo run` as the execution path.** 1.2 s of JIT per invocation, and no way
  to keep the result. A shared library plus `dlopen` is three orders of
  magnitude better on the second call.
- **Turning down the optimizer to compile faster.** No measurable effect; the
  cost is not in optimization.
- **Arbitrary-precision integers.** Mojo's `Int` is 64-bit and wraps. A Monte
  Carlo kernel with a 64-bit LCG multiplier silently disagreed with CPython
  until its constants were brought inside the range. This is a real correctness
  cliff and it is documented rather than fixed — fixing it means either an
  overflow check on every operation or a bignum runtime, and both give back
  most of the speedup.

## What is next

- A persistent compile worker, so process start is paid once rather than per
  snippet. The compiler has no server mode, so this means a resident process
  holding a queue and building batched units on a timer.
- Speculative batching in `@jit`: collect every function that misses the cache
  during the first second of a run and compile them as one unit.
- SIMD and `parallelize` in generated code for reduction and map shapes, which
  is where the remaining order of magnitude is.
- 2-D buffers, which unlock most of the scikit-learn-shaped work.
