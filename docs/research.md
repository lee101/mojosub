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
  `array.array` before the call, which costs more than the kernel saves —
  `dot(1e6)` went from 36x faster on a numpy array to 3x *slower* on a list.
  The buffer protocol is the whole ballgame; anything exposing
  `__array_interface__` is passed by address. **Half of this turned out to be
  self-inflicted — see measurement 7.**
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

## Measurement 5: SIMD, and where hand-vectorizing loses

Emitting width-4 `SIMD` accumulations for reduction loops is worth about 2x
(`dot` over 1e6 float64: 2.2 ms scalar, 1.35 ms vectorized). Doing the same for
store loops is not always a win — LLVM already vectorizes the simple ones, and
the explicit form was measurably worse:

| loop | scalar | vectorized |
| --- | ---: | ---: |
| `b[i] = a[i] * k + 1.0` | 0.68 ms | 1.24 ms |
| `b[i] = (a[i]*a[i]*k + a[i]/3 - a[i]*0.5) * (a[i]+k)` | 3.09 ms | 1.83 ms |

So store loops are vectorized only above three arithmetic operations, which is
roughly where the loop stops being memory-bound. Reductions have no such
crossover — the sequential dependency is what LLVM will not break on its own.

## Measurement 6: the GPU crossover is arithmetic intensity, not size

Same per-element loop on an RTX 5090, 1e6 float64 elements, varying only how
much arithmetic each element gets:

| flops/byte | GPU | CPU (SIMD) | speedup |
| ---: | ---: | ---: | ---: |
| 0.2 | 4.2 ms | 2.6 ms | 0.6x |
| 1.9 | 3.6 ms | 8.9 ms | 2.5x |
| 18.8 | 6.2 ms | 220 ms | 35x |
| 187.5 | 9.1 ms | 2396 ms | 263x |

A single elementwise pass loses at *every* size tested — 100k, 1M, 10M, 50M —
because the PCIe round trip costs more than the arithmetic saves. Scaling up
the array does not help; only scaling up the work per element does. The
crossover sits near 2 flops per byte.

That kills the obvious feature (automatically move big arrays to the GPU) and
justifies the one that is implemented: `target="gpu"` is explicit, refuses
anything that is not a single element loop, and returns results through the
caller's buffers.

## Measurement 7: half the list penalty was a copy nobody needed

The copy *in* is unavoidable: a Python list is an array of pointers to boxed
floats and the ABI wants sixty-four contiguous bits. The copy *back* is not. It
existed because the kernel might have stored through the buffer — and for a
reduction, it provably cannot have.

The transpiler already knows: it has just decided whether to emit a store
through each parameter. Recording that on the signature (`Signature.writes`) and
skipping the write-back for read-only parameters:

| n | CPython | with copy back | without | saved |
| ---: | ---: | ---: | ---: | ---: |
| 100,000 | 33.7 ms | 21.3 ms | 11.6 ms | 45% |
| 400,000 | 127.8 ms | 101.1 ms | 45.5 ms | 55% |
| 1,000,000 | 253.1 ms | 259.0 ms | 94.7 ms | 63% |

The bottom row is the interesting one. At a million elements the compiled
version was *slower than the interpreter* — the headline finding of measurement
3 — and more than half of that was a `tolist()` and a slice assignment throwing
away work the kernel never did. With it gone, the same kernel on the same list:

| n | list | numpy array |
| ---: | ---: | ---: |
| 1,000 | 2.2x | 16.9x |
| 10,000 | 5.1x | 179x |
| 100,000 | 1.9x | 277x |
| 400,000 | 2.2x | 336x |
| 1,000,000 | 2.3x | 147x |

There is no size at which *this* list argument loses any more, but the general
claim does not survive the benchmark: `dot(1e6)` on a list still measures 0.8x,
because the copy *in* is the half that cannot be removed. The gap to a buffer is
still two orders of magnitude, and it is entirely the caller's to close, so
mojosub counts the elements it had to copy (`Stats.copied_elements`) and the
service says so in the run trace rather than leaving the user to wonder why
their 300x is a 2x.

Escape analysis stays deliberately shallow: a buffer handed to a helper is
assumed written. Following it properly is a real analysis, and the failure mode
of getting it wrong is a silently stale array in the caller's hands.

## Measurement 8: the emitted names were a correctness bug

Running a corpus of ordinary statistics code through the pipeline and reading
what the compiler said found three bugs at once, all in the same place — user
identifiers were being emitted verbatim into Mojo:

```python
var = sq / n - mean * mean     # -> `var var: Float64 = ...`  syntax error
fn = xs[i] - mean              # -> `var fn: Float64 = ...`   syntax error
xs_len = 3                     # -> reassigned the length of buffer `xs`
```

`var`, `fn`, `alias`, `struct`, `ref`, `owned` and `inout` are all legal Python
names and reserved Mojo ones. `var` for a variance is not an unusual thing to
write, and the cost of writing it was five seconds of compiler per submission,
for ever, with the kernel silently staying on CPython — the failure was in the
build, which the tiered design is specifically built to swallow.

The third is worse than a compile failure. `xs_len` is the parameter synthesised
for `len(xs)`, so a program with its own variable of that name got a wrong
answer, quietly, in whatever direction the collision happened to point.

The fix is not a keyword blacklist — that is a race against the language. Every
user identifier is emitted into a `u_` namespace and every synthesised length
into an `n_` one. Both maps are injective and their images are disjoint, so no
user name can collide with a keyword, with a generated length, or with each
other, including the awkward `xs` and `xs_len` together. One kernel in the corpus
went from *never compiling* to 359x.

The general lesson is about where to look: this was not found by testing the
transpiler, which had 30 passing tests. It was found by running realistic code
through the whole pipeline and reading the compiler's stderr, which nothing was
doing because the tiered design is built to keep going when a compile fails.
A fallback that hides failures needs something else watching them.

## Measurement 9: the cache key was not machine-specific enough

`mojo build --target-cpu` documents its default as "the host CPU". The cache key
was the compiler version, the optimization level and the source — so two
machines with different instruction sets shared entries. On one box that is
invisible. Across a fleet with a shared object store it is a unit built where
AVX-512 exists, `dlopen`ed where it does not, running until the first vector
instruction and then dying.

The host's CPU *feature flags* are now part of the key — the flags rather than
the model name, because the same model number ships with features fused off.
Machines that can run each other's code still share; machines that cannot get
their own entries, which is what they always needed.

## Measurement 10: 2-D buffers, and the ML-shaped work they unlock

"2-D buffers, which unlock most of the scikit-learn-shaped work" sat at the
bottom of this file as future work for a while. It is done, and it was worth
more than expected, because the alternative was not "slower" — it was "cannot be
expressed". A kernel over a feature matrix had to be rewritten against a flat
array with the stride arithmetic done by hand, which is a different program from
the one the caller debugged in numpy.

A logistic-regression gradient pass — the shape of nearly every linear-model and
gradient-boosting inner loop — over 20,000 x 32 float64:

| | |
| --- | ---: |
| CPython | 898.2 ms |
| Mojo | 3.44 ms |
| | **261x** |

Exact agreement on both the loss and the gradient vector.

Three things were harder than the indexing arithmetic:

**A matrix and a flat array emit identical Mojo.** Both are a bare
`UnsafePointer[Float64]`; only the number of shape arguments beside them differs.
The variant cache and the signature memo were keyed on the emitted text, so a
2-D call would have reused the library compiled for a 1-D one — reading the wrong
addresses and returning a plausible number. Types now carry a `key` separate from
their `mojo` spelling, and `Array.key` deliberately keeps its old text so
signature memos already on disk still resolve.

**Contiguity has to be checked, not assumed.** The emitted index is
`row * cols + col`, which is simply the wrong address for a transposed or sliced
array. Numpy hands those over as happily as any other array, and the result is
not a crash, it is a wrong answer. The array interface reports strides; anything
that is not row-major contiguous is refused at the boundary with the fix named in
the message.

**`m.shape[1]` had to be allowed.** Attribute access is otherwise entirely
outside the subset and stays so — one exact pattern is recognised, a bare name
followed by `.shape` followed by a literal 0 or 1. The alternative was asking the
caller to rewrite their kernel to make it fast, which is most of the way to
asking them to write it in Mojo.

A list of lists is refused rather than copied. It is not one allocation, so there
is no address to hand over; the copy would dominate any kernel it fed, and it
used to be typed as a flat *int* buffer and fail later and confusingly.

## Measurement 11: fuzzing found that Python and Mojo disagree about scope

The transpiler had 59 passing tests. `bench/fuzz.py` generates random programs
inside and just outside the subset — grammar-directed, since random bytes are
rejected by `ast.parse` and test nothing — runs each under CPython and through
the transpiler, and compares. Three of the first dozen programs found bugs, and
all three were one root cause:

> Python scopes a local to its function. Mojo scopes a `var` to its block.

```python
def f(n: int) -> float:
    n = n * 2              # 1. a Mojo parameter is an immutable binding
    total = 0.0
    for i in range(n):
        if i % 2 == 0:
            hit = 1.5      # 2. `var hit` here is scoped to the `if`
            total += hit
    return total + hit     # ...and unknown out here

def g(n: int) -> float:
    for i in range(n):
        pass
    return float(i)        # 3. Python leaves `i` bound; Mojo does not
```

All three are ordinary Python. All three emitted Mojo that did not compile. And
none of them looked like anything from the outside, because the tiered design
falls back to CPython when a compile fails — the program returned the right
answer, slowly, for ever.

The fix is not three special cases; it is to put declarations where Python puts
them. Every local is hoisted to the top of the emitted function, a reassigned
parameter arrives under a separate name and is copied into a mutable local, and a
loop counter that outlives its loop is driven by a generated iteration variable
that copies into a hoisted one. Crashes over a 60-program run: 13 to 0, with no
disagreements.

Two details worth keeping:

- **Hoisting only happens for the functions that need it.** Doing it everywhere
  would rewrite every unit's content hash and invalidate a cache full of correct
  libraries for no benefit.
- **A loop whose counter escapes is not vectorized.** The vectorizer emits its
  own scalar tail loop and would bypass the hoist. Losing SIMD on a loop whose
  counter is read afterwards is a fair trade for not emitting code that does not
  compile.

The generator's accept rate is reported alongside the findings, because a fuzzer
that only produces programs the transpiler refuses is testing the refusal path
and nothing else. It currently accepts about three quarters of what it makes.

## Measurement 12: one function in a file could break every other one

Found by pointing a converter at a whole module instead of hand-picked kernels.
Every function came back refused with `call to sorted` — including functions with
no `sorted` in them.

A unit is emitted whole or not at all, and `_helpers` was putting *every
annotated sibling function* into it. So one unrelated helper made an entire file
uncompilable, and the reason reported for a clean kernel pointed at code that was
not in it. The function's own docstring said "reachable from `entry`"; the
implementation did not compute reachability. It does now, transitively.

The lesson is about the test corpus, not the code: every existing test compiled
one function from a file containing one function. Nothing exercised the case that
every real user hits.

## Measurement 13: a local has one type in Mojo and several in Python

The transpiler declared each local from its *first* assignment. Python rebinds,
so the commonest shape in the whole subset was silently wrong:

```python
t = 0                 # declared Int
for i in range(n):
    t += xs[i]        # every store becomes Int(...)
return float(t)       # 2.0, where CPython says 2.5
```

Nothing was watching for it. The emitted Mojo compiled, the library loaded, the
call was fast, and the number was truncated — the one failure mode the tiered
design cannot absorb, because a wrong answer looks exactly like a right one.
`t = 5; t = t / 2` and `flag = True; flag = 5` are the same bug.

The fix is to declare each local at the **join** of everything assigned to it.
That is not knowable until the body has been walked, so the walk is simply run
again with the wider types seeded — emission is pure, it builds strings, and two
passes settle anything real. Where the join is impossible the answer is a
refusal rather than a guess: a parameter's type is fixed by the ABI and a loop
counter's by its `range`, so widening either is `Unsupported` and the caller
keeps the interpreted answer.

The same sweep found `2 ** -1`: CPython switches to float for a negative
exponent, `Int.__pow__` cannot, and the result was 0 against 0.5. A literal
exponent says which case it is; a runtime one is refused, because the type of
the result genuinely depends on a value nobody has yet.

None of this came from the fuzzer, which keeps its int and float variables in
separate pools and so can never generate the mixed assignment. It came from
asking what a *typed* transpiler does with an untyped language's rebinding.

## Measurement 14: a build that never returns

A generated unit — 35 lines, nothing exotic in it — put `mojo build` into a spin
that was still running two days later, holding a core the whole time, because
the fuzzer that started it had long since been killed. Isolated, none of its
subexpressions reproduce it; the whole function does, at every optimization
level.

The interesting part is what it says about the architecture rather than about
the compiler. A tiered JIT is only allowed to be invisible because every failure
comes *back* as a failure — a refusal, a compile error, a mismatch — and gets
turned into "keep using CPython". A build that never returns is the one outcome
that has no path back to that, so the build is bounded (`MOJOSUB_BUILD_TIMEOUT`,
300s by default, against a real build of 3–5s) and a timeout is reported as an
ordinary `ToolchainError`. The compiler gets its own session so the timeout can
kill what it spawned: `subprocess.run(timeout=...)` kills only the process it
started and then blocks reading pipes a surviving grandchild still holds, which
is a hang inside the code that exists to prevent one.

## What is next

- Following buffers into helpers properly, so a read-only buffer passed to a
  helper keeps the write-back elimination of measurement 7.
- A persistent compile worker, so process start is paid once rather than per
  snippet. The compiler has no server mode, so this means a resident process
  holding a queue and building batched units on a timer.
- Speculative batching in `@jit`: collect every function that misses the cache
  during the first second of a run and compile them as one unit.
- Multi-threading the generated CPU code with `parallelize`, which is the
  remaining order of magnitude on large buffers.
- Keeping GPU buffers resident across calls, so a pipeline of kernels pays the
  transfer once instead of per kernel — the measurement above says that is
  where the rest of the GPU win lives.
