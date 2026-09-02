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
| Current sync (2026-09-02) | `3ebc90aa95743b56dd19cd85ececf190d2672776` | `feat/beat-adaptive-solve` |

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

## What is copied verbatim

Byte-for-byte identical to the sync commit, with no edits of any kind:

| here | upstream |
|---|---|
| `hornlab_beat_bem/julia/src/*.jl` (41 files) | `src/blab/solvers/julia_local/src/` |
| `hornlab_beat_bem/julia/coupled_solver.jl` | `src/blab/solvers/julia_local/coupled_solver.jl` |
| `hornlab_beat_bem/julia/{Project,Manifest}.toml` | `src/blab/solvers/julia_local/` |
| `hornlab_beat_bem/julia_metal/{Project,Manifest}.toml` | `src/blab/solvers/julia_metal/` |
| `hornlab_beat_bem/julia/test_meshes/*.msh` | `src/blab/solvers/julia_local/test_meshes/` |
| `hornlab_beat_bem/julia/test_fixtures/*.msh` | `tests/fixtures/{femvolume,exterior_conforming}.msh` |
| `hornlab_beat_bem/julia/tests/runtests.jl` | `src/blab/solvers/julia_local/tests/` |
| `hornlab_beat_bem/julia/scripts/*.jl`, except the four listed below | `src/blab/solvers/julia_local/scripts/` |

Every numerical result this package produces comes from those files, and they
are unmodified. That is deliberate: it is what lets the extraction be verified
by identity rather than by tolerance.

The `julia_cuda/` and `julia_rocm/` project files are unchanged from the
2026-08-19 vendoring, which is also verbatim.

## What is modified, and why

### `hornlab_beat_bem/julia/solver.jl`

The one file that had already diverged. It is a three-way merge of upstream
`42c8781` (base), this repository's `c207139` (ours) and upstream `e862240`
(theirs — the last commit on the branch that touches this file; `6b27f22` and
`f536d9e` do not). Everything merged cleanly except one hunk, and three deliberate
decisions were made:

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

Features that existed only in this repository's `solver.jl` and are preserved
by the merge: diagonal observation cuts, the theta-major spherical grid for
balloon/DI mapping, and axial source motion.

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
   and `coupled_solver.jl`, `Project.toml`, `Manifest.toml` alongside. These are
   verbatim, so a plain copy is correct.
2. Three-way merge `solver.jl` with `git merge-file`, using the previous sync
   commit recorded above as the base.
3. Re-apply the six path constants in the table above.
4. Update the sync commit in this file, and re-run the verification in
   `README.md`.
