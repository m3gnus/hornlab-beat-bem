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

## What is copied verbatim

Byte-for-byte identical to the sync commit, with no edits of any kind:

| here | upstream |
|---|---|
| `hornlab_beat_bem/julia/src/*.jl` (41 files) | `src/blab/solvers/julia_local/src/` |
| `hornlab_beat_bem/julia/coupled_solver.jl` | `src/blab/solvers/julia_local/coupled_solver.jl` |
| `hornlab_beat_bem/julia/BeatEngineDriver.jl` | `src/blab/solvers/julia_local/` |
| `hornlab_beat_bem/julia_engine/BeatEngine{Cpu,Cuda,Rocm,Metal}Bundle/` | `src/blab/solvers/julia_engine/` |
| `hornlab_beat_bem/julia/{Project,Manifest}.toml` | `src/blab/solvers/julia_local/` |
| `hornlab_beat_bem/julia_rocm/{Project,Manifest}.toml` | `src/blab/solvers/julia_rocm/` |
| `hornlab_beat_bem/julia_metal/{Project,Manifest}.toml` | `src/blab/solvers/julia_metal/` |
| `hornlab_beat_bem/julia/test_meshes/*.msh` | `src/blab/solvers/julia_local/test_meshes/` |
| `hornlab_beat_bem/julia/test_fixtures/*.msh` | `tests/fixtures/{femvolume,exterior_conforming}.msh` |
| `hornlab_beat_bem/julia/tests/runtests.jl` | `src/blab/solvers/julia_local/tests/` |
| `hornlab_beat_bem/julia/scripts/*.jl`, except the four listed below | `src/blab/solvers/julia_local/scripts/` |

Every numerical result this package produces comes from those files, and they
are unmodified. That is deliberate: it is what lets the extraction be verified
by identity rather than by tolerance.

The `julia_cuda/` project files carry one local addition, described below.

## What is modified, and why

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

The three decisions were made when this file first diverged, in a three-way
merge of upstream `42c8781` (base), this repository's `c207139` (ours) and
upstream `e862240` (theirs). They are unchanged:

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
constant was repointed in each of six files. Nothing else in these files
changed.

| file | change |
|---|---|
| `julia/tests/coupled_solver_tests.jl` | fixture root -> `../test_fixtures` |
| `julia/tests/coupled_condensed_tests.jl` | fixture root -> `../test_fixtures` |
| `julia/scripts/validate_metal_coupled.jl` | fixture root -> `../test_fixtures` |
| `julia/scripts/validate_rocm_coupled.jl` | fixture root -> `../test_fixtures` |
| `julia/scripts/compare_coupled_precision.jl` | default FEM/BEM mesh -> `../test_fixtures` |
| `julia/scripts/benchmark_cpu.jl` | `repo_root` depth 5 -> 3, to match this layout |

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
than corrected, so two statements in them are narrower than they read and the
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
