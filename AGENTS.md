# Working in this repository

Read the HornLab workspace `GIT-WORKFLOW.md` before any Git or GitHub action.
This file adds project rules; it cannot weaken that policy.

## What this repository is

A packaged extraction of Boundary Lab's BEAT Engine solver (GPL-3), not
original work. `VENDORING.md` is authoritative for provenance: it records the
upstream commit, what is copied byte for byte, and every difference. Keep it
accurate — it is this package's GPL-3 §5(a) notice, not a courtesy.

## The rule that matters most

**`hornlab_beat_bem/julia/src/*.jl` is a verbatim copy of upstream.** Do not
edit those files here. A fix belongs upstream, in `boundary-lab`, and arrives
here through a re-sync. Verbatim copies are what let the extraction be verified
by identity rather than by tolerance; the moment one is patched locally, that
proof is gone and every future sync becomes a merge.

`BeatEngineDriver.jl` is the one file that carries local changes, and it is a
three-way merge on every sync. `VENDORING.md` explains the three decisions in
it. `solver.jl` used to be that file; since the cold-start sync it is a
verbatim entry point and the body lives in the driver.

## The failure mode to watch for

The engine is precompiled into a package per backend under `julia_engine/`,
and **losing that is silent**: `solver.jl` falls back to including the sources,
so a worker with no bundle still solves and still gives the right answers, at
three to five times the cold start, with nothing logged. A re-vendor that
copies `julia/` and forgets `julia_engine/`, a backend project that does not
declare its bundle, a wheel whose package data misses the new directory, or a
precompile workload the driver rejects all look exactly like success.

So do not verify this by running it and seeing output. Count runtime
compilations (`julia --trace-compile=stderr`) or measure time to first result
through the worker; `tests/test_engine_bundles.py` does both.

## A consumer pins this repository by SHA

Waveguide Generator pins `hornlab-beat-bem` in `pins.json` and
`server/requirements-pins.txt`. Adding commits is safe; **rewriting this
repository's history is not**, because a pinned SHA that becomes unreachable
breaks a clean install of WG. Never force-push, never rewrite published history
here.

A pin bump is its own reviewed change in the consumer, after the commit it
names is pushed. See workspace `GIT-WORKFLOW.md` §5.

## Verification

The Julia CPU suite and the Python suite run anywhere. The Metal scripts need
Apple Silicon, the ROCm scripts need an AMD host.

```bash
J=hornlab_beat_bem/julia
julia --project=hornlab_beat_bem/julia          $J/tests/runtests.jl
julia --project=hornlab_beat_bem/julia_metal    $J/scripts/validate_metal_exterior.jl
julia --project=hornlab_beat_bem/julia_metal    $J/scripts/validate_metal_symmetry.jl
julia --project=hornlab_beat_bem/julia_metal    $J/scripts/validate_metal_coupled.jl
julia --project=hornlab_beat_bem/julia_metal    $J/scripts/validate_metal_fused_burton_miller.jl
julia -t auto --project=hornlab_beat_bem/julia  $J/scripts/validate_gmres_burton_miller.jl
pytest
pytest -m slow          # a real Julia solve; needs a Julia executable
```

`pytest -m slow` includes the bundle probe, which skips rather than fails when
the CPU project has not been instantiated. Instantiate it first, or the check
that matters most here does not run:

```bash
julia --project=hornlab_beat_bem/julia -e "using Pkg; Pkg.instantiate()"
```

One check is deliberately conditional. The Julia suite and the Krylov gate
both compare an unreorthogonalised Float32 Gram-Schmidt against the remedies,
so that the agreement between the remedies is not vacuous. *Whether* it
degrades is a property of the host's floating point rather than of the code:
the synthetic system loses 20x on an M1 Max and reportedly not at all on a
Ryzen 7 5825U. A suite that fails where the code is right is reporting a
microarchitecture, so that half is a **warning**, not an assertion. The
agreement between the remedies stays hard. Do not "fix" a warning here by
tightening it into a failure.

CI measured this rather than assuming it. On the first run of the workflow
(2026-09-03), `ubuntu-latest` fired the warning -- single MGS 39 iterations
against Float64's 20, inside the 4x threshold -- while `macos-latest` did not
fire it at all. Both jobs passed, 420 assertions each. So the note above is
load-bearing and not a caution about something hypothetical: promoted to an
assertion, that check would make `ubuntu-latest` permanently red on code that
is correct, and the obvious way to get it green again would be to loosen the
agreement assertions that are the actual evidence. If you are reading this
because you were about to tighten it, that is the run to look at first.

Do not weaken a tolerance to make a gate pass. The fused and symmetry gates
compare two code paths that differ only in Float32 summation order, so their
tolerances are noise floors and a real regression will not sit just above one.

## Continuous integration

`.github/workflows/ci.yml` runs on pull requests and on pushes to `main`. It
covers the Julia CPU suite and the Python suite on `ubuntu-latest` and
`macos-latest`, and the exterior, symmetry and fused Burton-Miller Metal
validators on `macos-latest` -- the hosted macOS runner has a working Metal
device, so those run for real rather than skipping.

Two things are asserted that an exit code does not cover, because until
2026-09-03 this repository had no CI at all while a consumer pinned it by SHA
and sessions described its branches as gated:

- **Julia has to be present.** Three Python test files take a `julia` fixture
  that skips when `discover_julia()` returns `None`, so without the runtime the
  suite reports 60 passed, 3 skipped and exits 0 -- green over a run that never
  solved anything. `.github/scripts/assert_julia_runtime.py` fails first
  instead, and `check_pytest_report.py` then rejects any skip at all. There is
  no platform `skipif` anywhere in `tests/`, so zero skips is a real invariant
  rather than an aspiration; a new skip should be argued for there.
- **The suites have to be non-empty.** Both checkers enforce a floor and print
  counts, so a job log carries a number instead of the word "green".

The hosted macOS runner reports an `AppleParavirtDevice`, not the bare metal
GPU, which is why the job asserts that a kernel compiles, dispatches and
returns the right answer before trusting `Metal.functional()`.

Nothing in CI compares GPU output byte for byte. The native Metal path
accumulates through atomics and is not bit-reproducible run to run (~8e-7);
only the CPU `reference` mode is. The first CI run showed both halves of that
at once, comparing the paravirtualised runner against a local M1 Max: the
`host` singular-mode operator errors agreed to every printed digit
(`double_layer` 2.4922872e-7 on both), while the `native` ones did not
(2.6501868e-7 local against 2.643291e-7 in CI). Same code, same mesh,
different accumulation order. An equality assertion on those would fail for
the reason the design says it would.

`validate_metal_coupled.jl` and the ROCm scripts are not in CI: the first has
not had a green hosted run yet, and the second needs an AMD host.

### A known intermittent on ubuntu-latest -- do not "fix" it by loosening it

Four assertions in `julia/tests/coupled_condensed_tests.jl` fail
intermittently on `ubuntu-latest`, and have never failed on `macos-latest`.
The failures are the `symmetry == :off` branch of the forked-versus-shared
comparison, which asserts **bitwise** equality on the stated ground that "with
no images the fused sweep reduces to the base sweep". The observed
disagreement is one Float32 ULP: `3.354732f-9` against `3.3547323f-9`. A
failing job reports 416 passed, 4 failed, 1 broken.

Sightings so far, `ubuntu-latest` only:

| Run | Commit changed | `cpu_name` | Result |
|---|---|---|---|
| CI run 1 | -- | not recorded | pass |
| CI run 2 | AGENTS.md only | not recorded | **fail** |
| CI run 3 | workflow only | `znver3` | pass |
| CI run 4 | AGENTS.md only | `znver4` | pass |
| PR #6 | AGENTS.md only | `znver3` | pass |
| PR #7 run 33758796520 | a new script CI did not yet invoke | `znver4` | **fail** |

Two of those changed no Julia code whatsoever, and the sixth added a script
the workflow did not call. So the suite disagrees with itself across runs of
identical code.

`Threads.nthreads()` is 1 on the runners, so the `Threads.@threads` loops in
`BeatEngineCpuAssembly.jl` and `BeatEngineCondensedAssembly.jl` are disabled
and thread scheduling is not the cause. The suite also passes locally at
`-t 4` on Apple Silicon.

**The obvious hypothesis has been tested and does not hold.** `ubuntu-latest`
is genuinely a heterogeneous pool -- these runs saw both `znver3` and `znver4`
-- so the natural explanation was that one host vectorises the two sweeps into
different summation orders. But `znver4` appears once as a pass and once as a
fail, and `znver3` has only ever passed. The same CPU model therefore both
agrees and disagrees, which rules out `Sys.CPU_NAME` as a sufficient
explanation and points at something that varies *within* a host between runs.
Multithreaded OpenBLAS (2 threads on ubuntu, 3 on macOS) is the leading
remaining candidate; allocation-dependent vectorisation of a loop tail is
another. Keep recording `cpu_name` anyway -- a finer-grained difference than
the LLVM target name is still possible -- but do not go on attributing this to
the pool.

This is pre-existing and not owned here. `coupled_condensed_tests.jl` is
vendored (see `VENDORING.md`; only the fixture root differs from upstream), so
a fix belongs in `boundary-lab` and arrives through a re-sync. **Do not relax
that `==` to an `≈` in this repository**, and do not relax it upstream without
first establishing the mechanism -- the assertion is
deliberate, its comment explains why both the strict and the loose halves are
there, and turning it loose would discard the only check that the fused sweep
really does reduce to the base sweep. If it turns out to be a genuine
microarchitecture dependency, the honest repair is to say so where the
assertion is made, exactly as the conditional Krylov check does.

## Benchmarking

Timing on a shared machine is worthless without exclusivity. Check for other
load first — `ps -Ao pcpu,comm -r | awk 'NR>1 && $1>5.0 {n++} END {print n+0}'`
reads 1-6 on an idle machine — and take the minimum of several passes rather
than a single sample. The fused solve has a heavy tail (max/median up to 4.5x),
so a single-sample A/B on that stage can be wrong by 4x.

Watch for a filesystem indexer: on macOS, Spotlight indexing a fresh checkout
held load average at 20-49 for an hour and invalidated two separate A/Bs. Put
`.metadata_never_index` in any new working copy's root.

## Conventions

- Author identity `m-a` / `m3gnus`. No AI-attribution footers.
- No real names, hostnames, absolute user paths, or private references in
  anything committed here — this repository is public.
- Preserve upstream authorship and licence notices exactly.
