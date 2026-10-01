# Provenance and vendoring

This package is a derivative work. The numerical solver in
`hornlab_beat_bem/julia/` is the **BEAT Engine** (Boundary Element Acoustic
Toolkit Engine) from **Boundary Lab**, and it is not original to HornLab.

| | |
|---|---|
| Original project | Boundary Lab, <https://github.com/JWSound/boundary-lab> |
| Long-term upstream | Official engine fork, <https://github.com/m3gnus/BEAT_Engine> |
| Upstream licence | GNU General Public License v3.0 |
| Upstream copyright | The Boundary Lab authors (JWSound) |
| Historical sync fork | <https://github.com/m3gnus/boundary-lab> |

Boundary Lab ships the bare GPL-3 licence text with no per-file copyright
headers, so this repository reproduces the same licence verbatim in `LICENSE`
and records authorship here instead of inventing notices upstream does not
carry. Original solver authorship is retained. The CPU SIMD improvements
below are HornLab's own work, published as m3gnus's official engine PRs.

GPL-3 §5(a) requires a modified work to carry prominent notices of what was
changed. This file is that notice: **every difference from upstream is listed
below**. Since the 2026-10-01 sync, changes are applied directly here and
listed individually with their official engine PR/commit. The baseline is
retained for unchanged files; this package no longer claims that the whole
engine is a verbatim copy of one upstream commit.

## Sync points

| | commit | branch |
|---|---|---|
| Original vendoring (2026-08-19) | `42c87812f70b9ae3ab446bcc543b3789941d0509` (2026-08-17) | upstream `dev`, now also on `main` |
| Sync at first publish (2026-09-02) | `f536d9e6a89c348cb5e071349f788cfe0f078156` | `feat/beat-adaptive-solve` |
| Sync at 3ebc90a (2026-09-02) | `3ebc90aa95743b56dd19cd85ececf190d2672776` | `feat/beat-adaptive-solve` |
| Cold-start sync (2026-09-02) | `cd50b3c64771b242d681aaf440a5b786757ba35a` | `feat/beat-cold-start-adaptive` |
| Cherry-pick (2026-09-03) | `1f90433` on `fix/condensed-entrywise-floor` | the :off condensed comparison becomes an entrywise absolute floor; decided by the maintainer after the AVX-512 experiment |
| Driver sync (2026-09-03) | `724d573d596b60db521a7ba618d04557ed7727d2` | `feat/beat-metal-pipeline-size-aware`, cut from `cd50b3c`: the Metal sweep overlap becomes a per-solve decision from the dof count |
| Singular Burton-Miller fusion (2026-09-05) | `02364b595230db5b73c225c5271a34128bcf7880` | `perf/metal-singular-bm-fusion`, on the `fix/beat-krylov-gate-tolerance` (`531d99fd`) stack and **unlanded upstream**: the singular Burton-Miller pair is combined per quadrature point rather than per pair |
| Metal engine and bundle sync (2026-10-01) | `3ea846eed48e25b88aa339d0188b9a69699a4815` | `perf/metal-speedups-for-hornlab`: reconstruct the shipped engine baseline, cache source-entry Metal kernels, and port PR #15 exterior packed kernels |
| CPU SIMD local differences (2026-10-01) | official engine `3d3e0a4`, `85bac13`, `4aa32e7` | Applied directly after the `3ea846e` sync; file-by-file notice below |
| GMRES sweep reuse port (2026-10-01) | official PR #23, `3157fe0710e6b43677ab4d9c3a0485711fd4b58b` | Applied directly after the CPU SIMD port; compiled integration `458c472` reviewed for lineage applicability; file-by-file notice below |

The 2026-08-19 vendoring was a verbatim copy: all 25 files of
`src/blab/solvers/julia_local/src/` and both project files matched `42c8781`
exactly, and only `solver.jl` was afterwards modified in this repository. That
is what made the current sync a clean three-way merge rather than a rewrite.

`feat/beat-adaptive-solve` is a stack of three branches, so its tip carries all
three pieces of work:

| branch | tip | what it adds |
|---|---|---|
| `feat/beat-metal-backend` | `6025ff1e8a4874ffa8e51ab00fd40bb8aef24e1b` | Apple Metal backend; operators in shared storage, so there is no device-to-host copy; the Burton-Miller right-hand side built without materialising an `N x 2N` operator |
| `feat/beat-bm-fusion` | `bd01028f24389ce423a44483d5a54b08068dffa7` | the Burton-Miller combination fused into the pair kernel |
| `feat/beat-adaptive-solve` | `3ebc90aa95743b56dd19cd85ececf190d2672776` | the adaptive dense solve: LU or diagonally preconditioned GMRES, chosen per solve |

The branch tip has moved twice during this work, so the sync commit is not the
one the extraction started from. First from `e862240` to `f536d9e` while it was
being verified. The two added commits touch only `beat-engine-core.md` and
`validate_gmres_burton_miller.jl` — no solver source — and both are taken:
`6b27f22` extends the Krylov gate to symmetry off/x/xy and to the sliver-rim
meshes, and `f536d9e` records a routing regression on A1r and retires the claim
that this operator never stagnates. Nothing else in this package differs
between the two commits.

Then from `f536d9e` to `3ebc90a`, in four commits that touch five files —
`BeatEngineCore.jl`, `BeatEngineDenseSolve.jl`, `calibrate_dense_solve.jl`,
`validate_gmres_burton_miller.jl` and `tests/runtests.jl`, and **not**
`solver.jl`, so this second sync needed no merge at all. All four are taken,
and the first is a real defect fix rather than documentation:

| commit | what it changes |
|---|---|
| `a533a15` | **bounds a misrouted GMRES** at one LU's worth of matvecs, and refits the calibration script to read its constants at the crossover, warm, with the matvec's linear term clamped at zero |
| `5474cc5` | makes the "unreorthogonalised Float32 is worse" check a warning rather than an assertion, because whether it degrades is a property of the host's floating point |
| `8aa2539` | **drives the Krylov gate with the physical tag-2 excitation** instead of a random right-hand side, which was materially easier than the path it guards |
| `3ebc90a` | retires the never-stagnates claim from the solver header and refreshes two stale figures in it |

The version first published here, `ba72fb0`, predates all four. Anyone reading
that commit should know it ships the adaptive router **without** the misroute
bound.

## The cold-start sync

`feat/beat-cold-start` and `feat/beat-adaptive-solve` were both cut from
`e862240` and this package needs both. Merging them upstream rather than
merging their effects here is what keeps the rule in `AGENTS.md` intact: the
vendored sources stay a verbatim copy of **one** commit.

| branch | tip | what it adds |
|---|---|---|
| `feat/beat-cold-start` | `1d97e03def3abf2c275d87933bf3d7502ae817a6` | the engine and the worker driver moved into precompilable packages under `julia_engine/`, each with a `PrecompileTools` workload |
| `feat/beat-cold-start-adaptive` | `cd50b3c64771b242d681aaf440a5b786757ba35a` | the merge of the two branches, plus the one fix below |

The merge itself was clean (`ort`, no conflicts): the two branches overlap in
`BeatEngineCore.jl` and `beat-engine-core.md` only, and in different places.

### The one commit made upstream for this package

`cd50b3c` changes the precompile workload's request from
`"source_motion" => "piston"` to `"normal"`, in all four bundles.

Upstream has no `source_motion` key at all, so it ignores the value and the
workload solves. **This package's `solver.jl` validates it** — `normal` or
`axial`, one of the three local modifications listed below — so `piston` threw
on the workload's first line. The workload catches everything, deliberately,
so that a build never fails over an optimisation; the result was a bundle that
precompiled a fraction of what it was written to precompile, with nothing
logged anywhere. Measured through the worker on the ATH `250917asro68q`
quarter export: 228 runtime compilations on a first solve rather than 95, with
344 being the no-bundle baseline. `normal` is what a real request from
`hornlab_beat_bem.sweep` carries, and it is as inert upstream as `piston` was.

`tests/test_engine_bundles.py` now validates every bundle's workload request
against this package's own `SolveConfig`, so the next such divergence fails a
test rather than costing two thirds of the win in silence.

## Two later cherry-picks, 2026-09-03

Two fork branches carry work that postdates `3ebc90a` and is on neither
`feat/beat-adaptive-solve` nor the cold-start merge above. Both are taken here
as verbatim file copies, and both had a base identical to `3ebc90a` for every
file they touch, so each is exactly its own change and nothing else. That base
is still current: the cold-start sync from `3ebc90a` to `cd50b3c` left
`src/BeatEngineDenseSolve.jl`, `src/BeatEngineMetalField.jl` and
`tests/runtests.jl` byte-for-byte unchanged, so re-basing these two picks onto
it was a no-op rather than a merge.

| branch | commit | files taken | what it does |
|---|---|---|---|
| `feat/beat-gmres-tolerance-1e-5` | `a52b8f4` | `src/BeatEngineDenseSolve.jl`, `tests/runtests.jl` | **exterior GMRES tolerance 1e-6 → 1e-5.** At 1e-6 the sliver-rim ATH meshes floor above the target below ~6 kHz and fall back to the dense LU, paying a full GMRES budget *and* the factorization. The radiated field moves at most 0.00100 dB. **Scoped to exterior solves**: the coupled FEM/LEM path factorizes directly and was never measured at either tolerance |
| `feat/beat-metal-field-occupancy` | `ca5597a` | `src/BeatEngineMetalField.jl`, `scripts/benchmark_metal_field.jl` | **Metal field-evaluation occupancy.** The kernel launched one thread per evaluation point — a two-cut polar sweep is 74 points, so 74 threads on a 32-core GPU, each walking 13,650 quadrature sources (4× that with symmetry images). Chunking the source loop is ~20× on the stage (0.0571 → 0.0028 s/frequency) |
| `feat/beat-metal-field-occupancy` | `d9ecee6` | `scripts/benchmark_metal_assembly_stages.jl`, `scripts/probe_metal_assembly_concurrency.jl` | the two probes behind the decision **not** to pursue cross-frequency concurrency: concurrent assemblies measure *slower* than sequential (0.85× at N=2), because one assembly already saturates the GPU |

Measured effect on the shipped package, asro68 quarter export, M1 Max, against
`hornlab-metal-bem` `74eca82` as an unchanged control — this is the package as
it now stands, with no environment overrides:

| case | metal-bem | BEAT-Metal | |
|---|---:|---:|---|
| 1,209 dofs, 3 frequencies | 0.539 s | **0.437 s** | BEAT 1.23× |
| 1,209 dofs, 40 frequencies | **2.206 s** | 4.600 s | metal-bem 2.09× |
| 4,552 dofs, 3 frequencies | 3.325 s | **2.322 s** | BEAT 1.43× |
| 4,552 dofs, 40 frequencies | **20.79 s** | 27.95 s | metal-bem 1.34× |

Before these two cherry-picks the 40-frequency quarter case was ~11 s. Far-field
agreement is unchanged at 0.0082 dB main-lobe rms in band.

## A third pick, 2026-09-04: the Krylov gate follows the tolerance

`feat/beat-gmres-tolerance-1e-5` above supplied `src/BeatEngineDenseSolve.jl`
and `tests/runtests.jl`, but not `scripts/validate_gmres_burton_miller.jl`. The
gate's own default tolerance therefore stayed at 1e-6 while the product's moved
to 1e-5, and the gate began measuring a 1e-5 solve against a bound built from
1e-6: six failures on `main`, on a solver that was correct. This branch is the
repair, and it is upstream because the script is vendored verbatim.

| branch | commit | files taken | what it does |
|---|---|---|---|
| `fix/beat-krylov-gate-tolerance` | `531d99f` | `scripts/validate_gmres_burton_miller.jl` | the gate reads `_beat_gmres_tolerance` instead of restating it, so the two cannot drift again; `BLAB_VALIDATE_GMRES_TOLERANCE` is retired because it moved the bound and the Krylov probes but never the `beat_solve_dense_system` call they guard, and `BLAB_BEAT_GMRES_TOL` moves both together |

Cut from `cd50b3c`, the recorded sync point, against which the vendored copy was
byte-identical before this change. Nothing else on the branch.

Measured on the bundled 1,390-dof sample at 500 / 2000 / 6000 Hz, two drives:

| run | result |
|---|---|
| before, gate at 1e-6 against a 1e-5 solve | exit 1, six FAIL, residuals 6.9e-6 to 1.05e-5 against a 4.44e-6 bound |
| after, gate at the solver's 1e-5 | exit 0, PASS |
| after, `BLAB_BEAT_GMRES_TOL=1e-6` | exit 0, PASS — 500 Hz and 2 kHz correctly fall back to the LU |
| after, `BLAB_VALIDATE_GMRES_TOLERANCE=1e-6` | exit 1, refuses with the retirement message |

**No tolerance was weakened.** The `sqrt(N) * eps(Float32)` evaluation floor and
the `4 x tolerance` factor are both unchanged; only the tolerance fed into them
moved, from a stale literal to the value the checked solve actually ran at. The
LU-agreement assertion — the one that says the answers are right — passed
throughout at 1.0e-6 to 1.8e-6 against its unchanged 1e-4 bound, before and
after.

Two things the fix surfaced that the old default had been hiding:

- At 1e-6 the three Krylov variants **do not converge** at 500 Hz and 2 kHz on
  this fixture; they stop at ~1.1e-6. Their reported counts, 53 / 58 / 55, are
  where they gave up rather than where they agreed, and 55 against 58 inverts
  the frequency ordering the script's own header describes. At the solver's
  1e-5 they converge at 32 / 38 / 39 — the ordering the header claims, and the
  same counts the production solve takes. The gate's iteration-agreement checks
  now measure the path the product runs.
- The header's sample counts, `44 / 51 / 63`, matched neither and are replaced
  with the measured `32 / 38 / 39`. The A5 ladder counts beside them are marked
  as 1e-6-era rather than restated as current; that mesh is not in this package.

## GMRES elapsed-deadline re-sync

The two files below are copied verbatim from fork commit
`4e22bc7471d0c45e7da42242bb2d553c890e85f2`
(`fix/beat-gmres-deadline-resync`). This commit starts at `a52b8f4`, merges
the deadline merge `c8de262`, and shares one absolute monotonic deadline
across drive columns, preserves the exact remaining iteration budget, and
reports the reason for a fallback. It retains the `1e-5` exterior tolerance
and later Krylov tests from `a52b8f4`. Deadline tests use an injected clock
for partial-iterate coverage rather than comparing timings across solves.

This package copies these two files byte for byte from that upstream
commit:

| here | upstream |
|---|---|
| `hornlab_beat_bem/julia/src/BeatEngineDenseSolve.jl` | `src/blab/solvers/julia_local/src/BeatEngineDenseSolve.jl` |
| `hornlab_beat_bem/julia/tests/runtests.jl` | `src/blab/solvers/julia_local/tests/runtests.jl` |

Their SHA-256 values are respectively
`813da0d82c82f242e0af8f8c28be74c7efda2a4ec3bf1487888c941fbcb838d0`
and `5999489fcbee66390ce3dc9d4b4628d16ef13523c60edd6e54ed6444515b765f`.
No fixture-path adaptation is needed for either file.

## The singular Burton-Miller fusion re-sync, 2026-09-05

The three files below are copied verbatim from fork commit
`02364b595230db5b73c225c5271a34128bcf7880`, the tip of
`perf/metal-singular-bm-fusion`.

**That branch is unlanded upstream, and it is not on the fork's `main`.** It
sits on the `fix/beat-krylov-gate-tolerance` stack (`531d99fd`), which is the
branch this package's Krylov gate already came from and whose ancestor
`cd50b3c` is the sync commit recorded above. The fork's default branch, `main`
at `4331b8cb` (2026-07-04), predates the whole BEAT Metal backend and carries
no `BeatEngineMetalBurtonMiller.jl` at all, so it is not a base this file may
claim the sync came from. Anyone auditing provenance should fetch the branch,
not the default.

The regular fused kernel already formed the Burton-Miller combination inside
its accumulation loop; the singular kernel did not. It called
`_metal_singular_pair_blocks` — shared with the four-operator path — which
accumulates slp/adj/dlp/hb and `g_total` over the whole Duffy rule, 50 live
accumulator floats, expanding the rank-1 outer product four times per
quadrature point pair, and combined only afterwards. The combination is linear,
so `_metal_singular_pair_fused_bm_blocks` applies it to the per-point scalars
before the expansion and hoists the loop-invariant hypersingular curl term out
of the loop: two 3x3 expansions instead of four, one 3x1 instead of two, 26
live floats instead of 50. Same signature, same scratch buffers, same launch
geometry, same scatter kernel, no new device buffer.
`_metal_singular_pair_blocks` itself is untouched, so the four-operator path
stays an independent reference rather than a copy of the code under test.

This package copies these three files byte for byte from that upstream commit:

| here | upstream | SHA-256 |
|---|---|---|
| `hornlab_beat_bem/julia/src/BeatEngineMetalBurtonMiller.jl` | `src/blab/solvers/julia_local/src/BeatEngineMetalBurtonMiller.jl` | `874875ed50f50e1c61a5e4b65a10e7ba008ae7318fa86605c852e98eacdf9c22` |
| `hornlab_beat_bem/julia/scripts/validate_metal_fused_burton_miller.jl` | `src/blab/solvers/julia_local/scripts/validate_metal_fused_burton_miller.jl` | `5717651ea5a8e2ccb51b0156660159ee6c05aee7b99315d6283ddd4946450fbf` |
| `hornlab_beat_bem/julia/scripts/validate_metal_singular_summation.jl` | `src/blab/solvers/julia_local/scripts/validate_metal_singular_summation.jl` | `bda12085b9c9cf273d25ed85b8525a4ec8a1bcf3429ebf650c2e4c57cf7a9efb` |

No fixture-path adaptation is needed for any of the three. The new script
resolves its mesh as `joinpath(@__DIR__, "..", "test_meshes", ...)`, which is
the same relative shape in this layout as in upstream's, so it is not added to
the repointed-path table below. Nothing in `pyproject.toml` moved either: the
`julia/scripts/*.jl` package-data glob already carries it into the wheel.

**The base was identical, so this pick is exactly its own change.** Before this
re-sync the vendored copies of the two modified files were byte-for-byte equal
to `531d99f`, the commit `02364b5` is built on —
`src/BeatEngineMetalBurtonMiller.jl` at
`2ade503a7ba770a144f86a06474a45cf2fa820e81b21ecea18a8470f78f2f725` and
`scripts/validate_metal_fused_burton_miller.jl` at
`f276557ee1b324881c330ce4ecfce649a4173dc88cf113248eba6993d12834da`. No merge
was needed for either.

Measured upstream on an M1 Max, minimum of 9-11 repeats per cell, and
re-verified here by the gates at this commit:

| | before | after | |
|---|---:|---:|---|
| singular block kernel, 1,209-dof symmetry-reduced mesh | 20.7 ms | 7.7 ms | -63% |
| singular block kernel, 4,552-dof full mesh | 84.1 ms | 30.9 ms | -63% |
| assembly wall | | | -17.1% / -14.2% |
| 40-frequency sweep | | | -13.8% median quarter, -10.7% full |
| device memory high-water | | | +0 bytes |

The evidence, its predeclaration and the independent adversarial review are in
the HornLab workspace at `archive/260905-beat-singular-fusion-prototype/`. The
review's verdict was PROMOTE with three packaging corrections, all of which are
already applied in `02364b5`: the prototype's `BLAB_METAL_SINGULAR_BM_FUSION`
runtime switch is gone (the fused-at-quadrature form replaces the
block-accumulate fused kernel outright rather than becoming a second selectable
kernel), the `BLAB_METAL_SINGULAR_STAGE_SPLIT` stage instrumentation is gone,
and the now-dead `_metal_fused_pair_combination` is removed. **No tolerance was
changed**, here or upstream.

### The two script changes that a caller can see

`validate_metal_fused_burton_miller.jl` takes `BLAB_VALIDATE_SYMMETRY` as a
comma-separated **arm list**, and its default moved from the single arm `off`
to `off,x,ground`; every arm runs and the script fails if any of them fails.
`ci.yml` invokes it with no environment, so that job's workload widened without
its workflow line changing — recorded in `AGENTS.md` under *Continuous
integration* so a failure there is read as an arm rather than as the script.

`xy` is deliberately absent from that default. On the bundled `sample.msh` it
fails `pressure_relative_error` at about 1e-5 against the script's 5e-6
tolerance, and it fails **non-deterministically**: three runs of the same tree
spread 1.27e-5 to 2.29e-5 while their `lhs` and `rhs` errors match to three
digits at 1e-7. The operators agree, so it is not an operator defect; the LU of
the symmetry-reduced matrix at that fixture amplifies the atomic-accumulation
non-determinism of the singular scatter into the pressure
(`BLAB_METAL_SINGULAR_MODE=host` removes the atomics). The review reproduced it
on a pristine `a46b2ba` tree, 3/3 failing at a 2.03x spread, so it predates this
change entirely. Closing it means a better-conditioned `xy` fixture or a
deterministic singular scatter — **not a wider tolerance**.

`validate_metal_singular_summation.jl` is new and has no predecessor here. It
bounds the summation-order difference directly, per singular pair: it evaluates
the same pairs three ways on the host — the four-operator accumulation order in
Float32, the fused per-quadrature-point order in Float32, and both in Float64 —
and gates that the two orders are the same algebra (they must agree in
Float64), that neither Float32 order exceeds the stated bound against the
Float64 reference, and that the fused order is not systematically worse. It is
bit-deterministic: no atomics, a fixed pair selection, a fixed frequency set. It
is a script, not a CI job — the fork's `ci.yml` has no Metal job, and nothing
was wired into this repository's CI by this re-sync.

### `docs/beat-engine-metal.md`

Upstream changed its own `docs/advanced/beat-engine-metal.md` in the same
commit, and this package's copy of that page has local edits of its own (the
source-path rewrites, and the per-solve sweep-pipelining text from the
`267512c` driver sync). The page was therefore three-way merged —
base `531d99f`, ours the vendored copy, theirs `02364b5` — with no conflicts.
The result adds upstream's four additions verbatim: the singular-fusion
paragraph, the note that `_metal_singular_pair_blocks` is deliberately
untouched, the two validator table rows, and the "Known issue: the fused
Burton-Miller gate at symmetry `xy`" section. None of the added prose names an
upstream source path, so no rewrite was needed inside it.

## Metal kernel-cache and packed exterior re-sync, 2026-10-01

The engine sources and all four bundles are copied byte for byte from fork
branch `perf/metal-speedups-for-hornlab`, commit `3ea846eed48e25b88aa339d0188b9a69699a4815`: the baseline commit `e4cbae8`,
the port `7674bd0`, and `3ea846e`, which skips the Metal kernel workload before
Julia 1.11 (as Metal.jl does for its own). This branch starts at `02364b5`; its first change reconstructs the
engine already shipped here before either speed-up is applied:

1. Apply the `BeatEngineCore.jl` and `BeatEngineCpuAssembly.jl` hunks from
   `ec99b75744bc48aff4ae9327d46ebc9864b6c472`, accepting one near-correction
   cache per symmetry image. These two additions were present in this package
   but missing from the sync inventory above; their CUDA counterparts were
   already present in the branch's base.
2. Copy `BeatEngineDenseSolve.jl` and `tests/runtests.jl` from
   `4e22bc7471d0c45e7da42242bb2d553c890e85f2`.
3. Copy `BeatEngineMetalField.jl` from `ca5597a`, retaining the shipped field
   occupancy change.

All 41 engine source files then match this package's pre-sync sources exactly.
The new changes therefore start from the same numerical engine, rather than
from the older singular-fusion branch alone. The new snapshot has 44 source
files, including the three packed-kernel files.

### Source-entry Metal kernel cache

The compile-only workload is adapted from official BEAT Engine commit
`430b6e0`. The Metal project and bundle pin Metal 1.11.1, with the resolved
GPUCompiler 2.9.0 stack. `BeatEngineMetalBundle`, the bundle loaded by
`solver.jl`, compiles production kernel signatures with `Metal.mtlfunction`
without launching them. It gates on Apple Silicon, imports the device-array
aliases explicitly, reports failures and counts, and clears Metal's
process-local state before writing the package image. The entry point loads
Metal before JSON to preserve the bundle's cached call graph.

All four bundles decode their host workload through `JSON.parse(JSON.json(...))`,
matching the worker's request type. A second request uses a quadrant plate
with symmetry `xy`, non-adjacent pairs and image-singular pairs, singular order
4, two frequencies, and the outputs the wrapper requests: polar cuts,
diagonal cut, radiation impedance, surface traces and a 37 by 72 sphere grid.
Boundary Lab's driver uses the equivalent 2,664-point spherical sampling;
this package's retained driver honours the explicit theta-major grid.

`julia/tests/metal_kernel_coverage_tests.jl` observes source-entry requests
under GPUCompiler's scoped compilation hook, including package-image cache
hits, and rejects signatures missing from the generated inventory. It covers
the default Float32 exterior path, symmetry off/x/xy, regular orders 1/2/4,
singular order 4, two frequencies, polar cuts and sphere output. Ordinary runs
do not regenerate the inventory. The existing ground and four-operator gates
remain separate qualification of those paths.

### PR #15 exterior kernels

The packed Float32 field implementation and multi-drive API, packed fused
regular pairs with image accumulation before gathering, and grouped full-Duffy
singular pairs are ported from the official engine's exterior-only port of
closed PR #15. Original source commits are `7e4a39e`, `7c8491a` and the required
helpers from `09388b9`; original author: BumelantPZA. No low-frequency singular
split, global BLAS switch, coupled-only changes, pools or router changes are
included.

Writable pair blocks and timing dictionaries belong to each assembly. Packed
geometry publication is locked; cached tables are read-only. This fork retains
its existing atomic singular scatter. Its singular cache lacks the official
engine's gather-table reference, so packed singular tables are keyed by the
identity of its rule-weight array and released with that cache.
`validate_metal_packed_exterior.jl` races singular-table publication and regular
assembly scratch, checks exact repeated regular assembly, and validates the
multi-drive field API across its eight-drive batch boundary. Existing native
singular validators retain their tolerances.

`BeatEngineDriver.jl` has no upstream change in this re-sync, so its local
features and prior merge are retained unchanged. The CUDA project addition,
fixture-path adjustments and local validators described below are retained.
At that sync, the engine source and bundle inventory came from the single
new fork commit. Later direct edits are listed individually below; the older
per-file sync entries remain historical provenance.

## Local differences since the 2026-10-01 sync

These changes are applied directly to this package. Boundary Lab was not
changed for this port. Their long-term upstream is
[m3gnus/BEAT_Engine](https://github.com/m3gnus/BEAT_Engine):

- [PR #17](https://github.com/JWSound/BEAT_Engine/pull/17),
  `3d3e0a48af711cf9ca3948d7b5d5d48c0798b27f`: CPU regular-pair SIMD.
- [PR #19](https://github.com/JWSound/BEAT_Engine/pull/19),
  `85bac1329832b366ddf437007cf8a6b1e883f0c2`: SIMD field evaluation,
  fused singular corrections and transposed scatter, stacked on #17.
- [PR #20](https://github.com/JWSound/BEAT_Engine/pull/20),
  `4aa32e7f988a49018f850875895f42258265a07e`: representative CPU
  precompile workload, only the coverage missing from the prior sync.

Paths in the following table are relative to `hornlab_beat_bem/` unless
prefixed with `../`.

| File | Source | Local difference from the sync baseline |
|---|---|---|
| `julia/src/BeatEngineCpuSimd.jl` (new) | #17 + #19 | SoA regular pairs, polynomial sincos, transposed square scatter, field-density packing and fused Duffy SIMD; byte-identical to #19 |
| `julia/src/BeatEngineCore.jl` | #17 + #19 | Export selectors; add scalar-default CPU-only dispatcher keyword |
| `julia/src/BeatEngineCpu.jl` | #17 | Include the new SIMD implementation |
| `julia/src/BeatEngineCpuAssembly.jl` | #17 | Add scalar-default regular-kernel selection and SIMD regular-pass hook |
| `julia/src/BeatEngineCpuBurtonMiller.jl` | #17 + #19 | Scalar-default regular/singular selectors, SIMD regular hook, SoA Duffy rules for base and image singular pairs, selection diagnostics |
| `julia/src/BeatEngineCpuField.jl` | #19 | Add scalar-default field-kernel selection and SIMD hook |
| `julia/BeatEngineDriver.jl` | #17 + #19 | Opt CPU source requests into SIMD regular, field and fused singular stages and report selections; read the regular selector only for CPU requests |
| `julia/tests/cpu_simd_kernel_tests.jl` (new) | #17 + #19 | Port entrywise, precision, symmetry, subset, threading and block-boundary gates; adapt convention API tests to signed wavenumbers |
| `julia/tests/runtests.jl` | #17 | Include the new SIMD gates |
| `julia/scripts/benchmark_cpu_regular_kernel.jl` (new) | #17 | Regular-stage scalar/SIMD timing, entrywise parity and LLVM vector-width probe; byte-identical to #17 |
| `julia_engine/BeatEngineCpuBundle/src/BeatEngineCpuBundle.jl` | #20 | Select regular orders 2 and 4 by wavelength and add explicit mesh translation input to the existing representative workload |
| `../docs/beat-engine-cpu-simd.md` (new) | #17 + #19 | Port the official CPU kernel page; rewrite package paths, describe signed-wavenumber gates, state the existing Float64-reference bound precisely and clarify which singular path remains scalar |
| `../README.md` | #17 + #19 | Document the three CPU switches and link the kernel page |
| `../tests/test_engine_bundles.py` | #20 | Pin the added wavelength/order and explicit mesh-input workload coverage |
| `../AGENTS.md`, `../VENDORING.md` | Route decision, 2026-10-01 | Replace the single-commit/verbatim-only development rule with direct edits and individual official-engine provenance |

Only the CPU source-request driver opts in. Library assembly and field
functions still default to scalar, preserving accelerator host-staged paths,
validator CPU references and compiled-system solves. The driver retains the
package's existing request/output decisions listed below.

The official engine's selectable phasor API is absent from this lineage. The
port preserves its existing wavenumber handling: no `outgoing_wavenumber`
conversion or convention switch is imported. The PR's convention tests compare
scalar and SIMD at signed wavenumbers instead, with unchanged bounds.

PR #20 largely overlaps the prior sync's workload: JSON-decoded worker request
types, non-adjacent quadrant pairs, xy images and image-singular pairs, diagonal
and sphere output, and surface traces were already present. The additions
reach order 2 at 1 kHz and order 4 at 20 kHz through wavelength selection, and
explicit mesh translation. The existing 9-node plate and 37-by-72 sphere are
retained; duplicating the PR's 16-node plate or its extra Dict solve would not
add worker coverage. Accelerator bundle workloads are unchanged.


## GMRES sweep reuse port, 2026-10-01

[Official PR #23](https://github.com/JWSound/BEAT_Engine/pull/23),
source commit `3157fe0710e6b43677ab4d9c3a0485711fd4b58b` on
`e6b3037`, is applied directly after the CPU SIMD port. This is a port of
individual changes, not a whole-tree sync. Paths below are relative to
`hornlab_beat_bem/`.

| File | Port and adaptation |
|---|---|
| `julia/BeatEngineDriver.jl` | Own a dense sweep state in each fused Metal request's solve consumer, pass frequency/state to the solve, and expose warm-start, budget, fallback and true-residual diagnostics. Retain this lineage's `metal_pipeline_requested` selector rather than importing the official engine's newer overlap planner; retain every package request/output decision and CPU SIMD hook. |
| `julia/src/BeatEngineDenseSolve.jl` | Port bounded six-solution, per-drive minimum-residual history; reject unhelpful/non-finite guesses; route model-selected frequencies within a 1.5 ratio of an observed fallback to LU; retain explicit overrides and shared deadline/iteration guards. Preserve this lineage's existing routing calibration and phasor behavior. No arithmetic adaptation to the added functions. |
| `julia/src/BeatEngineMetalBurtonMiller.jl` | Pass optional request state/frequency through the shared-host Metal solve wrapper. Preserve this lineage's packed exterior kernels and atomic singular scatter. |
| `julia/scripts/validate_gmres_burton_miller.jl` | Port physical-drive warm-sweep/LU agreement and true-residual gates, optional Metal backend and drive-tag selection; byte-identical to the exact source commit. Default CPU physical-drive gate and numerical bounds remain unchanged. |
| `julia/tests/runtests.jl` | Port request/drive isolation, repeated/reversed frequency, deficient/non-finite history, pipeline consumption/cancellation and fallback-backoff/override tests. Adapt the pipeline test to this lineage's single pending `Threads.@spawn`/`fetch` producer and release-on-cancellation, rather than importing the newer queue helper. Preserve all test assertions, existing SIMD gates and historical package tests. |
| `../VENDORING.md` | Record this port and its lineage applicability. |

The source commit also changes `coupled_solver.jl`. Its later direct fused
Metal/adaptive exterior path is absent here: this package's compiled-system
exterior path assembles four operators and factorizes with fixed LU, so it
cannot retry GMRES or reuse a GMRES iterate. That file is unchanged. Importing
the intervening assembly/router rewrite is not part of this port.

The compiled-driver integration was reviewed separately at official benchmark
commit `458c4729c92d64af7d0255eceece294532850ce0`. Its
`BeatEngineCompiledDriver.jl` hooks duplicate the later `coupled_solver.jl`
fused adaptive path, likewise absent here. This package has no file of that
name and no separate `MetalHostPrecompile.jl` inventory. All four shipped
bundles include `BeatEngineDriver.jl`, so the applicable source-driver change
is also the precompiled-driver change. Their existing parsed-request workloads
and package data remain intact. The source-entry `MetalKernelSignatures.jl`
inventory is independently regenerated and checked for this lineage rather
than copied from the benchmark stack's compiled bundle.

Unmodified engine/bundle files retain their earlier recorded identities;
modified files above do not claim whole-file identity with the PR commit.
Only the validator is newly byte-identical. The port preserves the exterior
GMRES tolerance, iteration/deadline guards, singular/reference accuracy bounds,
CUDA dependency wiring and fixture paths. Qualification uses the package
bundle runtime probe, Julia suite and physical-drive validator, Metal inventory
coverage, and paired package-API agreement on the S/C fixtures; source similarity
alone is not an accuracy claim.

## What is copied verbatim

The following inventory describes the `3ea846e` baseline. It remains
byte-identical except for the explicitly listed local differences above and
the older package modifications below:

| here | upstream |
|---|---|
| `hornlab_beat_bem/julia/src/*.jl` (44 baseline files; SIMD addition listed above) | `src/blab/solvers/julia_local/src/` |
| `hornlab_beat_bem/julia/coupled_solver.jl` | `src/blab/solvers/julia_local/coupled_solver.jl` |
| `hornlab_beat_bem/julia_engine/BeatEngine{Cpu,Cuda,Rocm,Metal}Bundle/` | `src/blab/solvers/julia_engine/` |
| `hornlab_beat_bem/julia/{Project,Manifest}.toml` | `src/blab/solvers/julia_local/` |
| `hornlab_beat_bem/julia_rocm/{Project,Manifest}.toml` | `src/blab/solvers/julia_rocm/` |
| `hornlab_beat_bem/julia_metal/{Project,Manifest}.toml` | `src/blab/solvers/julia_metal/` |
| `hornlab_beat_bem/julia/test_meshes/*.msh` | `src/blab/solvers/julia_local/test_meshes/` |
| `hornlab_beat_bem/julia/test_fixtures/*.msh` | `tests/fixtures/{femvolume,exterior_conforming}.msh` |
| `hornlab_beat_bem/julia/tests/runtests.jl` | `src/blab/solvers/julia_local/tests/` |
| `hornlab_beat_bem/julia/scripts/*.jl`, except the five listed below, `validate_analytic_exterior.jl`, which is new here, and three Metal benchmark/probe scripts kept from earlier syncs that the current sync commit does not carry (`benchmark_metal_assembly_stages.jl`, `benchmark_metal_field.jl`, `probe_metal_assembly_concurrency.jl`; their provenance is in the sections above) | `src/blab/solvers/julia_local/scripts/` |

Before the 2026-10-01 re-sync, ten of those files were taken from the later
branches above rather than from the
`cd50b3c` sync commit; they are verbatim copies of *those* commits. Three of the
ten — `src/BeatEngineMetalBurtonMiller.jl`,
`scripts/validate_metal_fused_burton_miller.jl` and the new
`scripts/validate_metal_singular_summation.jl` — come from `02364b5`, which is
**unlanded upstream**; see the singular-fusion section above.

Unchanged engine files can still be verified by identity against the
recorded baseline. The CPU SIMD files above differ from that baseline and
require numerical qualification; whole-engine byte identity is not claimed.

`BeatEngineDriver.jl` is deliberately absent from that list. It is upstream's
file, produced by a three-way merge and overwhelmingly upstream's code, but it
carries the package's request/output decisions and is not byte-for-byte
identical to any upstream commit. The section below enumerates every difference.

The `julia_cuda/` project files carry one local addition, described below.

## What is modified, and why

### One runtime default, set from `hornlab_beat_bem/worker.py`

Not a Julia source difference, but it means this
package's out-of-the-box behaviour is not upstream's, so it is listed here too.
`julia_threads="auto"` resolves to the performance-core count rather than
`os.cpu_count()`. It is `setdefault`-style: an explicit `julia_threads` wins,
and it was measured rather than assumed. See the README's "Sweep threads and
sweep pipelining".

**There used to be a second one, and it is gone.** `worker.py` set
`BLAB_METAL_PIPELINE=0`, because the sweep overlap is a loss on the small
symmetry-reduced meshes this package usually serves. That was the wrong place
for the answer and, at a large enough mesh, the wrong answer: the worker starts
before any mesh has been read, so a process-wide value could only ever suit one
mesh size, and it cost ~1.4x on a full model. The choice moved upstream into
`BeatEngineDriver.jl`, which knows the dof count and decides per solve — sync
commit `267512c` above. `worker.py` now sets nothing, so an explicit
`BLAB_METAL_PIPELINE` in the caller's environment still reaches the solver and
still wins in both directions, and the default is the solver's own.

### `hornlab_beat_bem/julia/solver.jl` and `BeatEngineDriver.jl`

The cold-start sync split this file. `solver.jl` is now the worker entry point
and nothing else — bundle resolution and dispatch, under 80 lines — and is a
**verbatim** copy of upstream. The body moved to `BeatEngineDriver.jl`, which
is where this package's three local decisions now live.

`BeatEngineDriver.jl` was produced by a three-way merge, not by re-applying
patches: base `e862240:solver.jl`, ours `7b6e6eb:solver.jl`, theirs
`cd50b3c:BeatEngineDriver.jl`. It merged with no conflicts, and the result
differs from this package's previous `solver.jl` by exactly upstream's split
(the docstring, dropping the engine `include` and the load-time BLAS call, and
wrapping the entry point in `main(args)`), so every local decision below is
carried over unchanged rather than re-derived.

Upstream moved the BLAS thread call into `main` on purpose: precompiled into a
bundle, load time is the *build* machine's, and the thread count has to be the
running machine's.

The first three decisions were made when this file first diverged, in a
three-way merge of upstream `42c8781` (base), this repository's `c207139`
(ours) and upstream `e862240` (theirs). They are unchanged; a fourth was added
on 2026-09-03:

1. **One merge conflict, resolved toward upstream.** Upstream restructured the
   per-channel solve to take the fused path when it is available; this
   repository had added a `source_motion` keyword to the four-operator call.
   The resolution keeps upstream's structure and re-adds the keyword.
2. **`channel_neumann_columns` gained a `source_motion` keyword.** Upstream
   builds every channel's right-hand-side column in one pass for the fused
   path, and that pass had no axial-motion case because upstream has no axial
   source motion. Without this, selecting `source_motion="axial"` would have
   been silently ignored on the fused path — the same drive rule as the
   four-operator path, applied in the place the fused path builds its columns.
3. **The Boundary Lab deploy request schemas are not vendored.**
   `include("deploy_solver.jl")` and the `boundary_lab_deploy_*` dispatch in
   `solve_request` are removed. `deploy_solver.jl` implements the Boundary Lab
   application's own solve schemas; it is application scope, not solver scope.
   `solve_request` therefore keeps this repository's shape: it calls
   `solve_request_impl` and always runs the accelerator cleanup.
4. **The rigid half space is reachable, guarded, and counted once.** Three
   changes, all local, added 2026-09-03:
   - `symmetry_mode_from_config` accepts `"ground"`. The engine in `src/` has
     implemented `rigid_ground_transform()` all along and `tests/runtests.jl`
     gates it, but the driver's request parser accepted only `off`/`x`/`xy`,
     so no request could reach it.
   - `impedance_for_radiators` no longer scales the integrated force by
     `symmetry_reduction_factor(mode)` when the mode is `:ground`. That
     function counts image *transforms*, which is the right multiplier only
     when the images are real radiators; a ground image is fictitious, so the
     reported impedance came back a factor 2 — 6.02 dB — high over a pressure
     field that was entirely correct. **This is a behaviour change against
     upstream**: a Boundary Lab `:ground` solve reports the doubled figure.
     Measured end to end at 300 and 500 Hz with the body 1 m above the plane,
     the fix moves reported impedance by 0.008 and 0.119 dB where the
     unfixed path moved it by 6.03 and 6.14 dB.
   - `validate_ground_plane_domain!` is new, and has no upstream counterpart.
     `symmetry_active_axes(:ground)` is empty, so
     `validate_symmetry_fundamental_domain!` is a no-op for this mode and a
     body straddling the plane would assemble against a domain that does not
     exist. It is ported from hornlab-metal-bem
     (`metal/geometry.py`, `validate_native_ground_plane`) and enforces the
     same three things: the whole mesh at Y >= 0, no face lying flat in the
     plane, and an optional minimum clearance. The tolerance is Boundary Lab's
     own fixed 1e-6 m, as `deploy_solve.py` uses on the same geometry.

   Nothing in `src/` is touched by any of this.

Features that exist only in this repository and are preserved by both merges:
diagonal observation cuts, the theta-major spherical grid for balloon/DI
mapping, and axial source motion.

Three further driver-only features, added in earlier rounds and carried
unchanged through the 2026-10-01 re-sync, also differ from the sync commit:

- **Near-correction selection** (`near_correction_selection`, and the routing
  that passes it to assembly): opt-in `near_correction_enabled` /
  `near_correction_cutoff` / `near_correction_order` request keys choosing a
  per-pair correction order from the distance ratio.
- **Double-precision CPU solves** (`solve_precision = "double"`): the CPU
  backend may assemble and solve in Float64 as an arbiter; results stay Float32
  on the wire, and the option is refused on accelerator backends.
- **Surface traces**: `surface_pressure` / `surface_neumann` outputs of the
  mixed boundary solution when requested.

### `hornlab_beat_bem/julia_cuda/{Project,Manifest}.toml`

The one place a bundle is wired up locally. Upstream updates `julia_local`,
`julia_metal` and `julia_rocm` to depend on their bundles but leaves
`julia_cuda` alone, because its CUDA image runs
`Pkg.develop(path=".../BeatEngineCudaBundle")` in the Dockerfile instead. This
package has no Dockerfile: `hornlab_beat_bem.provision` runs a plain
`Pkg.instantiate()`, so a CUDA host would have installed no bundle and taken
the slow path with nothing to say so.

`BeatEngineCudaBundle` is therefore added to `julia_cuda`'s `[deps]` and to its
manifest, as a path dependency exactly like the three upstream ones. Every
package the bundle needs was already in that manifest, so no version moved;
the recorded `project_hash` was recomputed with
`Pkg.Types.workspace_resolve_hash`, and that function was checked first by
confirming it reproduces upstream's own hashes for the other three projects.

`tests/test_engine_bundles.py` asserts that every backend project declares its
bundle and that the relative path resolves, so a future sync that adds the
CUDA dep upstream will not leave two conflicting sources of truth unnoticed.

### Fixture and repository-root paths

Upstream resolves shared fixtures five directory levels up, at the Boundary Lab
repository root. Those meshes live inside the package here, so one path
constant was repointed in each of six files. Nothing else in those six files
changed.

| file | change |
|---|---|
| `julia/tests/coupled_solver_tests.jl` | fixture root -> `../test_fixtures` |
| `julia/tests/coupled_condensed_tests.jl` | fixture root -> `../test_fixtures` |
| `julia/scripts/validate_metal_coupled.jl` | fixture root -> `../test_fixtures` |
| `julia/scripts/validate_rocm_coupled.jl` | fixture root -> `../test_fixtures` |
| `julia/scripts/compare_coupled_precision.jl` | default FEM/BEM mesh -> `../test_fixtures` |
| `julia/scripts/benchmark_cpu.jl` | `repo_root` depth 5 -> 3, to match this layout |

`julia/scripts/validate_metal_symmetry.jl` additionally differs in substance,
not only in a path: it gained the `:ground` cases on 2026-09-03, in both of
that mode's geometries (lifted clear of the plane, and resting on it). Upstream
covers off / x / xy only. The Metal-vs-CPU tolerances are unchanged.

`julia/scripts/validate_analytic_exterior.jl` has no upstream counterpart at
all. It is original to this repository, and it is the only gate here that
scores against a closed form rather than against a second BEAT code path.

### Documentation

`docs/` carries Boundary Lab's five BEAT Engine pages and its coupled-solver
page, with source paths rewritten from `src/blab/solvers/julia_local` to
`hornlab_beat_bem/julia` (and the CUDA/ROCm/Metal project directories likewise),
and the cross-link to `Coupled Solver.md` repointed at `coupled-solver.md`.
One further edit was necessary rather than cosmetic: a PowerShell example in
`beat-engine-CPU.md` contained a real person's Windows home directory
(`C:\Users\<name>\AppData\...`). This repository is public, so the path is
replaced with a placeholder and the two backslash-style source paths in the
same block are rewritten like the rest.

The prose is otherwise unchanged, so it still describes the Boundary Lab
application in places — `blab` CLI commands, `.blab.json` projects, solver
selection in application preferences. Those describe upstream, not this package.

**Where these docs are superseded.** They are upstream's, kept verbatim rather
than corrected, so three statements in them are narrower than they read and the
README carries the measured version:

- `beat-engine-metal.md`'s summary bullet "the default `pair_gather` kernels
  are bitwise reproducible run to run" is true of the *regular* kernels and
  does not extend to the assembly, because the `native` singular correction
  scatters with atomics — the same page says so in its singular-correction
  section. Measured on A3, three passes over identical inputs: regular-only
  operators are byte-identical, and adding the native singular correction
  makes them differ by 7e-8 (single layer) and 3e-7 (hypersingular).
  `BLAB_METAL_SINGULAR_MODE=host` restores byte-identical assembly.
- `beat-engine-core.md` at `f536d9e` already records the A1r routing
  regression, so it is current; earlier copies of it claimed the operator
  never stagnates, which is retired.
- `beat-engine-metal.md` is **current again on the sweep pipelining**, and this
  entry is kept only so that anyone holding an older copy of this file knows
  why it said otherwise. Between 2026-09-02 and 2026-09-03 the page described
  an unconditional overlap while `worker.py` forced `BLAB_METAL_PIPELINE=0`,
  because the overlap is slower on the small symmetry-reduced meshes this
  package usually serves. The sync at `267512c` removed the divergence at its
  source: the page and the solver now both say the sweep decides per solve from
  the dof count, so there is nothing left here to supersede. See the README's
  "Sweep threads and sweep pipelining" for this package's own measurements.

## What is deliberately not vendored

| upstream | why |
|---|---|
| `deploy_solver.jl` and the deploy request schemas | Boundary Lab application scope |
| `src/blab/**` Python, GUI, assets, `ath/`, examples, `.bat` launchers | application scope |
| `docs/` other than the BEAT Engine and coupled-solver pages | unrelated to the solver |
| `julia_local/tests/noncubic_cavity_loss_tests.jl` | not part of `runtests.jl`, and needs 34 MB of cavity fixtures |
| `scripts/analyze_noncubic_ppw.jl`, `scripts/test_noncubic_cavity_loss.jl` | same fixtures |
| `scripts/analyze_curved_fem_convergence.jl` | needs 23 MB of curved-interface fixtures |
| `scripts/benchmark_cuda.jl`, `benchmark_rocm.jl`, `profile_solver.jl`, and the ROCm/CUDA diagnostic scripts | hardware-specific benchmarking, not validation |

`docs/beat-engine-*.md` still mention some of these scripts, because the prose
is upstream's.

## Porting later official engine changes

Use the official engine fork `m3gnus/BEAT_Engine` as the long-term upstream.
Its engine layout is `src/beat_engine/julia_local/` and its bundles live in
`src/beat_engine/julia_engine/`. Apply improvements directly here, adapting
only what this package's existing lineage and API require. Do not replace the
whole source tree with a blind copy: this package retains earlier engine work
and the local differences recorded above.

For each port, record the official PR and exact source commit, every affected
file, and every adaptation in this notice. Compare unchanged files against the
recorded baseline and identify any newly byte-identical files separately.
Preserve the driver decisions, fixture-path changes and CUDA bundle dependency
listed above. Keep all backend bundles in package data and qualify their actual
runtime compilation/first-result behaviour, as `tests/test_engine_bundles.py`
describes. Run the engine suites, numerical gates and package tests before
claiming a qualified port.
