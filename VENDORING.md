# Provenance and vendoring

This package is a derivative work. The numerical solver in
`hornlab_beat_bem/julia/` is the **BEAT Engine** (Boundary Element Acoustic
Toolkit Engine) from **Boundary Lab**, and it is not original to HornLab.

| | |
|---|---|
| Upstream project | Boundary Lab, <https://github.com/JWSound/boundary-lab> |
| Upstream licence | GNU General Public License v3.0 |
| Upstream copyright | The Boundary Lab authors (JWSound) |
| Fork the sync is taken from | <https://github.com/m3gnus/boundary-lab> |

Boundary Lab ships the bare GPL-3 licence text with no per-file copyright
headers, so this repository reproduces the same licence verbatim in `LICENSE`
and records authorship here instead of inventing notices upstream does not
carry. Nothing in `hornlab_beat_bem/julia/` is attributed to HornLab.

GPL-3 §5(a) requires a modified work to carry prominent notices of what was
changed. This file is that notice: **every difference from upstream is listed
below**, and everything not listed is a byte-for-byte copy.

## Sync points

| | commit | branch |
|---|---|---|
| Original vendoring (2026-08-19) | `42c87812f70b9ae3ab446bcc543b3789941d0509` (2026-08-17) | upstream `dev`, now also on `main` |
| Sync at first publish (2026-09-02) | `f536d9e6a89c348cb5e071349f788cfe0f078156` | `feat/beat-adaptive-solve` |
| Sync at 3ebc90a (2026-09-02) | `3ebc90aa95743b56dd19cd85ececf190d2672776` | `feat/beat-adaptive-solve` |
| Current sync (2026-09-02) | `cd50b3c64771b242d681aaf440a5b786757ba35a` | `feat/beat-cold-start-adaptive` |
| Cherry-pick (2026-09-03) | `1f90433` on `fix/condensed-entrywise-floor` | the :off condensed comparison becomes an entrywise absolute floor; decided by Magnus after the AVX-512 experiment |
| Driver sync (2026-09-03) | `724d573d596b60db521a7ba618d04557ed7727d2` | `feat/beat-metal-pipeline-size-aware`, cut from `cd50b3c`: the Metal sweep overlap becomes a per-solve decision from the dof count |
| Singular Burton-Miller fusion (2026-09-05) | `02364b595230db5b73c225c5271a34128bcf7880` | `perf/metal-singular-bm-fusion`, on the `fix/beat-krylov-gate-tolerance` (`531d99fd`) stack and **unlanded upstream**: the singular Burton-Miller pair is combined per quadrature point rather than per pair |

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

## What is copied verbatim

Byte-for-byte identical to the sync commit, with no edits of any kind:

| here | upstream |
|---|---|
| `hornlab_beat_bem/julia/src/*.jl` (41 files) | `src/blab/solvers/julia_local/src/` |
| `hornlab_beat_bem/julia/coupled_solver.jl` | `src/blab/solvers/julia_local/coupled_solver.jl` |
| `hornlab_beat_bem/julia_engine/BeatEngine{Cpu,Cuda,Rocm,Metal}Bundle/` | `src/blab/solvers/julia_engine/` |
| `hornlab_beat_bem/julia/{Project,Manifest}.toml` | `src/blab/solvers/julia_local/` |
| `hornlab_beat_bem/julia_rocm/{Project,Manifest}.toml` | `src/blab/solvers/julia_rocm/` |
| `hornlab_beat_bem/julia_metal/{Project,Manifest}.toml` | `src/blab/solvers/julia_metal/` |
| `hornlab_beat_bem/julia/test_meshes/*.msh` | `src/blab/solvers/julia_local/test_meshes/` |
| `hornlab_beat_bem/julia/test_fixtures/*.msh` | `tests/fixtures/{femvolume,exterior_conforming}.msh` |
| `hornlab_beat_bem/julia/tests/runtests.jl` | `src/blab/solvers/julia_local/tests/` |
| `hornlab_beat_bem/julia/scripts/*.jl`, except the five listed below and `validate_analytic_exterior.jl`, which is new here | `src/blab/solvers/julia_local/scripts/` |

Ten of those files are taken from the later branches above rather than from the
`cd50b3c` sync commit; they are verbatim copies of *those* commits. Three of the
ten — `src/BeatEngineMetalBurtonMiller.jl`,
`scripts/validate_metal_fused_burton_miller.jl` and the new
`scripts/validate_metal_singular_summation.jl` — come from `02364b5`, which is
**unlanded upstream**; see the singular-fusion section above.

Every numerical result this package produces comes from those files, and they
are unmodified. That is deliberate: it is what lets the extraction be verified
by identity rather than by tolerance.

`BeatEngineDriver.jl` is deliberately absent from that list. It is upstream's
file, produced by a three-way merge and overwhelmingly upstream's code, but it
is the one place this package's local decisions live and it is not byte-for-byte
identical to any upstream commit. The section below enumerates every difference.

The `julia_cuda/` project files carry one local addition, described below.

## What is modified, and why

### One runtime default, set from `hornlab_beat_bem/worker.py`

Not a source difference — the vendored solver is unchanged — but it means this
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

## Re-syncing

The layout is a flat rename, so a future sync is mechanical:

1. Copy `src/blab/solvers/julia_local/src/*.jl` over `hornlab_beat_bem/julia/src/`
   and `coupled_solver.jl`, `solver.jl`, `Project.toml`, `Manifest.toml`
   alongside; `src/blab/solvers/julia_engine/` over `hornlab_beat_bem/julia_engine/`;
   and the `{Project,Manifest}.toml` pairs for `julia_metal` and `julia_rocm`.
   These are verbatim, so a plain copy is correct.
2. Three-way merge `BeatEngineDriver.jl` with `git merge-file`, using the
   previous sync commit recorded above as the base.
3. Re-apply the six path constants in the table above, and the `julia_cuda`
   bundle dependency.
4. Update the sync commit in this file, and re-run the verification in
   `README.md`.
5. **Check that the fast path is still taken**, because losing it is silent.
   `pytest tests/test_engine_bundles.py` covers the wiring; `pytest -m slow`
   counts runtime compilations against a live bundle. A re-vendor that copies
   `julia/` but not `julia_engine/`, or that leaves a backend project without
   its bundle dependency, produces correct answers at the old cold start and
   reports nothing.
