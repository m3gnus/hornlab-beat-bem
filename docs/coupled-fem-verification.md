# Does the vendored coupled FEM-BEM-LEM path work?

Measured 2026-09-03 for workspace `PLAN.md` step 7, whose first prerequisite is to
verify the coupled path through this package **using Boundary Lab's own ready-made
FEM models**, not invented fixtures. This file is that measurement. It changes no
code; it is a record of what the package already does.

## Verdict

**The Julia engine carries a complete, correct coupled path. The Python package
does not expose it at all.** Every shipped Boundary Lab FEM example that was tried
solved through this package's Julia tree, and on the CPU backend the results are
**bitwise identical** to the same model solved by Boundary Lab proper. But nothing
in `hornlab_beat_bem/*.py` can build a coupled request or start a coupled worker,
so from the package's public API the path is unreachable. What is missing is a
Python layer, not a solver.

Two consequences for the plan:

- The remaining step-7 prerequisites (what a WG user drives, and a validation gate)
  are not blocked on solver work. The solver is there and it agrees with upstream.
- The work item is a request-compilation layer, and it is substantial — see
  [What is missing](#what-is-missing). It is not a thin wrapper over `sweep.solve`.

One separate defect found on the way: the coupled entry point gets **no bundle
precompilation**, so a coupled worker compiles the whole engine from source on
every start. See [Cold start](#cold-start-a-real-and-separate-defect).

## What was driven, and how

Boundary Lab ships thirteen example projects at the sync commit; eight contain
bounded air regions and so take the coupled path (`2x12_CRAM`, `F2B_FLH`,
`Multi_region_SAWMOD`, `S218BP`, `SKRAM`, `Simple_Sealed`, `Vented_Sub`,
`compression_driver`). Three were driven end to end, chosen to widen the feature
surface rather than to repeat it:

| example | bounded regions | conforming interfaces | transducers | symmetry |
|---|---|---|---|---|
| `Simple_Sealed` | 1 | 0 | 1 | off |
| `Vented_Sub` | 1 | 1 | 1 | xy |
| `Multi_region_SAWMOD` | 3 | 4 | 3 | xy |

All three are `coupled_bem_fem` solves with `electrodynamic_transducer` components
driven from `voltage` excitation ports — exterior BEM, interior FEM and the
lumped voice-coil model in one system, which is the whole of FEM-BEM-LEM.

Because this package ships no Python that can compile a `.blab.json` into a solve
request, the request was built by **Boundary Lab's own** headless compiler and then
routed at the last moment to this package's Julia tree, by overriding the solver
script and Julia project the backend launches. Boundary Lab proper was run the same
way against its own tree. So both sides of every comparison share one request, one
mesh and one result mapping, and the only difference is which engine solved it —
which is exactly the question. Boundary Lab proper means the upstream fork at
`cd50b3c`, the sync commit `VENDORING.md` records, not the older branch a working
checkout may happen to sit on.

## Does it run, and does it agree?

Yes, and yes. `rel` is the largest absolute difference over the whole array,
divided by that array's largest magnitude.

| example | backend | frequencies | exterior pressure | diaphragm velocity | voice-coil current |
|---|---|---|---|---|---|
| `Simple_Sealed` | CPU | 100, 300, 1000 Hz | **bitwise** (722 pts) | **bitwise** | **bitwise** |
| `Simple_Sealed` | Metal | 100, 300, 1000 Hz | 6.4e-06 (722 pts) | 2.5e-07 | **bitwise** |
| `Vented_Sub` | CPU | 40, 120 Hz | **bitwise** (7,322 pts) | **bitwise** | **bitwise** |
| `Multi_region_SAWMOD` | CPU | 100, 800 Hz | **bitwise** (21,966 pts) | **bitwise** | **bitwise** |

Bitwise identity on the CPU backend is the strongest available result and it is the
expected one: `coupled_solver.jl` here is byte-identical to upstream at the sync
commit, so the same deterministic arithmetic runs on both sides. It is worth
stating anyway, because it converts "the files look the same" into a measurement of
the answers.

The Metal row is not a disagreement. Metal's assembly sums with a
non-deterministic atomic order, so 6.4e-06 relative on pressure is about
**5.6e-05 dB** — roughly two orders of magnitude below this programme's 0.01 dB
agreement floor, and the same noise scale the exterior Metal gates already accept.

### The package's own coupled gates

Run in a clean worktree of this repository, against its own fixtures:

- `tests/runtests.jl` with `BLAB_RUN_COUPLED_REFERENCE=1`: **512 passed, 0 failed,
  1 broken**. The broken one is `cuda production pipeline`, which needs an Nvidia
  host. Coupled testsets inside it: direct coupled FEM-BEM reference 17/17, Schur
  condensation algebra 53/53, condensed-vs-monolithic 32/32, condensed interior
  resonance 38/38, condensed regular assembly against the shared CPU assembly
  49/49, condensed per-order quadrature bundles 29/29, blocked exact FEM Schur
  complement 3/3.
- The dense FEM-BEM reference diagnostics: relative residual **9.42e-14**,
  pressure continuity error **0.0**, flux conservation error **0.0**, BEM replay
  error **1.26e-15**.
- `scripts/validate_metal_coupled.jl`: `METAL_COUPLED_VALIDATION_OK` on an M1 Max,
  pressure continuity 0.0, flux conservation 7.6e-08. That last figure moves
  between runs — 4.8e-09 on another — because Metal's assembly sums in a
  non-deterministic atomic order; it is a noise floor, not a fixed number.
- Python suite: **98 passed** — none of which touch the coupled path, which is the
  point of the next section.

### One gate was not run, and it is red on trunk

`scripts/validate_gmres_burton_miller.jl` is an exterior Krylov gate with no bearing
on the coupled path, so it was not part of this measurement. Do not read the list
above as saying it was green: **it fails on unmodified `main` at `284b397`**, six
failures, reproduced here in a clean worktree and independently by another session.

The solver is not wrong and this is not an Apple Silicon property. The gate checks
the *production* solve's recomputed residual against
`max(4 * tolerance, sqrt(N) * eps(Float32))`, where `tolerance` is the **script's
own** default — still `1e-6` at line 79. Production now targets `1e-5`, so it
converges at about `1e-5` and trips a bound of `4.4e-06` built from the older
number. The LU-agreement half of the gate passes throughout, at `1.0e-06` to
`1.8e-06` against its `1e-4` bound: the answers agree, only the bound is stale.

Confirmed by moving the input rather than the bound —
`BLAB_VALIDATE_GMRES_TOLERANCE=1e-5` makes the whole gate exit 0 with no failures.
So the fix is to decide what that default should track, which belongs to whoever
owns the tolerance vendoring, not to this measurement. Recorded, deliberately not
fixed here. It is in `AGENTS.md`'s verification block but in no CI job, which is why
trunk looks green.

## What is missing

The Julia side is complete. `hornlab_beat_bem/julia/` carries `coupled_solver.jl`
and `BeatEngineCoupled.jl`, `BeatEngineCoupledCondensed.jl`,
`BeatEngineCondensedAssembly.jl` and `BeatEngineSpeakerRom.jl`, plus the coupled
test fixtures and the Metal coupled gate. `Project.toml` matches upstream's exactly.

It survives packaging, too, which is worth stating because it is the kind of thing
that silently does not: `pyproject.toml`'s package data globs `julia/*.jl`,
`julia/src/*.jl` and `julia/test_fixtures/*.msh`, so an installed wheel carries
`coupled_solver.jl`, every coupled module and the coupled fixtures. The gap is
entirely on the Python side, not in what ships.

The Python side carries none of it:

- **`hornlab_beat_bem/*.py` contains no occurrence of the word `coupled`.** The
  package exports `solve`, `solve_frequencies`, `warm_up`, `SolveConfig` and
  `ObservationConfig` — an exterior-only surface.
- `runtime.py` hardcodes `DEFAULT_SOLVER_SCRIPT = julia/solver.jl`. The vendored
  `coupled_solver.jl` is never launched by anything in this package. It runs here
  only because a test script or an outside caller invokes it directly.
- There is no equivalent of the request compiler. Building a coupled request means
  resolving gmsh physical groups to tags, inferring component symmetry, computing
  the conforming FEM-BEM interface topology, and validating the whole plan. In
  Boundary Lab the transitive import closure behind `coupled_backend` and
  `physical_compiler` is 29 modules and roughly 12,000 lines; the load-bearing core
  is `physical_compiler`, `physical_model`, `system_contract`, `interface_conform`
  and `coupled_backend`, not all 29. Either way it is a port, not a wrapper.
- The result contract differs too. The exterior path returns this package's
  `SolveResult`; the coupled path streams `SystemFrequencyResult` objects carrying a
  tuple of named quantities — exterior pressure, diaphragm velocity, voice-coil
  current — which WG's existing result mapping does not consume.

Note for whoever scopes that work: the FEM volume meshes are **gmsh 4.1** while the
BEM surface meshes are **gmsh 2.2**, and this package's `mesh.py` reads only
`read_gmsh22_info`.

## Cold start, a real and separate defect

A coupled worker takes about **40 s to its first result** and about **0.65 s per
frequency** after that. The cause is structural, and it is not a re-vendoring slip:

- `solver.jl`, the exterior entry point, does `using BeatEngine<Backend>Bundle` and
  only falls back to including sources when the bundle is absent. That bundle is a
  package, so Julia caches its native code in a pkgimage.
- `coupled_solver.jl` never attempts the bundle. It unconditionally includes
  `BeatEngineCore.jl` **and** the three coupled modules from source, into `Main`.
- Nothing else reaches them: neither `BeatEngineCore.jl` nor `BeatEngineDriver.jl`
  — the only two files the bundles compile — includes any coupled module. So no
  bundle can ever cover the coupled path.

This is upstream's design, inherited verbatim, not something this package broke. But
it is exactly the silent slow path `AGENTS.md` warns about, and a WG integration
would inherit it: correct answers, no error, several times the cold start. It
deserves its own plan item rather than a footnote.

## Two scoping decisions this measurement confirms

**The GMRES tolerance does not reach the coupled path, and must not be widened to
it.** `PLAN.md` records that the 1e-5 default was deliberately scoped to exterior
solves. That is now verified by tracing the call graph, not by failing to grep for
a spelling:

- The adaptive router is `beat_solve_dense_system`, and its only callers anywhere
  in the engine are `BeatEngineCpuBurtonMiller.jl` and
  `BeatEngineMetalBurtonMiller.jl` — the exterior Burton-Miller solve.
- `BeatEngineCoupled.jl`, `BeatEngineCoupledCondensed.jl`,
  `BeatEngineCondensedAssembly.jl` and `BeatEngineSpeakerRom.jl` call **no**
  `BeatEngineDenseSolve` entry point at all — not the router, not `beat_gmres!`,
  not the cost model. They factorize directly: UMFPACK on the sparse FEM interior,
  dense LU on the coupled or condensed system.
- `coupled_solver.jl` does contain one Burton-Miller call site that can reach the
  router, but it sits inside `solve_exterior_request` — the branch taken when a
  project has no bounded regions, which is by definition not a coupled solve.

So there is no tolerance to tune on this path. The question of whether 1e-5 is safe
here does not arise, and the reason the coupled path was never measured at either
tolerance is that neither one applies to it.

**The condensed-assembly assertion was settled by its owner, not here.** This
measurement deliberately did not touch it. Magnus's decision landed as `f5efd5f`
while these runs were in flight, replacing the `symmetry=:off` bitwise `==` with an
entrywise absolute floor. Every figure in this file was re-measured afterwards at
`284b397`, which carries it.

**One nearby commit message is superseded, and this is where you will look.**
`7184162`, which vendored the size-aware Metal sweep overlap, explains the small-mesh
loss as Metal.jl's task-local command queues being rebuilt each frequency. The
session that wrote it then measured that claim and disproved it — a spawned task's
Metal overhead is a fixed 0.23-0.29 ms against a penalty of some 145 ms — and the
real mechanism is that one assembly already saturates the GPU, so overlapping
interleaves two queues over the same units. Their later commits correct the driver,
the README and the upstream doc, but `7184162`'s own message is published history and
stands uncorrected. That measurement is theirs, not reproduced here; it is recorded
because a reader arriving at this file from the cold-start section above will pass
straight through that commit.

## Cost

Treat these as indicative only. A sibling session was benchmarking BEAT-Metal on the
same machine for part of this work, at load averages of 11-18; contamination in that
window was measured at up to a factor of seven, so these are upper bounds and the
correctness results above are the load-independent part of this report.

| example | backend | frequencies | first result | total |
|---|---|---|---|---|
| `Simple_Sealed` | Metal | 12 | 40.5 s | 47.6 s |
| `Simple_Sealed` | CPU | 3 | 28.8 s | 38.2 s |
| `Multi_region_SAWMOD` | CPU | 2 | 42.8 s | 89.3 s |

The shape matters more than the numbers: cold start dominates a short sweep, and the
marginal cost of a frequency on `Simple_Sealed`/Metal is about 0.65 s against 40.5 s
to get started. Fixing the bundle gap would matter more to a WG user than any
per-frequency optimisation. On the same model the coupled Metal gate assembles the
BEM operators in 0.44 s against the CPU's 4.79 s, so the accelerator advantage does
carry into the coupled path — it is simply hidden behind the compile.

## Reproducing this

The package's own gates need nothing external:

```bash
J=hornlab_beat_bem/julia
BLAB_RUN_COUPLED_REFERENCE=1 julia --project=$J $J/tests/runtests.jl
julia --project=hornlab_beat_bem/julia_metal $J/scripts/validate_metal_coupled.jl
```

Driving a shipped Boundary Lab example additionally needs a Boundary Lab checkout at
the `VENDORING.md` sync commit, for its request compiler only. Load the project with
`blab.headless.load_headless_project`, build the request with
`prepare_headless_solve`, then construct
`blab.solvers.coupled_backend.PhysicalSystemProductionBackend` with `solver_script`
pointing at this package's `julia/coupled_solver.jl`, having rebound that module's
`DEFAULT_COUPLED_CPU_PROJECT` and `DEFAULT_BEAT_ENGINE_METAL_PROJECT` to this
package's `julia/` and `julia_metal/`. Compare quantity by quantity across the two
engines. Nothing needs to be written into either checkout.
