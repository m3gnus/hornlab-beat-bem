# hornlab-beat-bem

The **BEAT Engine** Helmholtz solver, packaged so a program can depend on it
without vendoring an application. Exterior Burton-Miller BEM and coupled
FEM-BEM-LEM, in Julia, on CPU, NVIDIA CUDA, AMD ROCm and Apple Metal, behind a
small Python API.

The solver is not original to this repository. It is Boundary Lab's BEAT
Engine, GPL-3, and this package is a derivative work of it — see
[Licence and provenance](#licence-and-provenance) and [`VENDORING.md`](VENDORING.md),
which records the exact upstream commit and every difference from it.

## What is in here

| | |
|---|---|
| `hornlab_beat_bem/julia/` | the solver: `src/*.jl` (41 files), the exterior entry point `solver.jl`, the coupled entry point `coupled_solver.jl` |
| `hornlab_beat_bem/julia/scripts/` | validation gates and the cost-model calibration |
| `hornlab_beat_bem/julia/tests/` | the Julia test suite (`runtests.jl`) |
| `hornlab_beat_bem/julia{,_cuda,_rocm,_metal}/` | one Julia project per backend |
| `hornlab_beat_bem/*.py` | the Python wrapper: config, sweep, worker, capability probe, runtime provisioning |
| `docs/` | Boundary Lab's BEAT Engine and coupled-solver documentation |

**Formulation.** Galerkin Burton-Miller with P1 pressure and DP0 Neumann
traces, `eta = i/k` coupling, Float32 throughout. Exterior solves take a fused
path that forms `0.5 I - D + (i/k) H` and the right-hand side inside the
assembly kernels; the coupled FEM-BEM-LEM solver needs `S`, `K'`, `D` and `H`
individually and keeps the four-operator path.

## Backends

| backend | project | assembly | dense solve |
|---|---|---|---|
| `cpu` | `julia/` | host, multithreaded | OpenBLAS ILP64, adaptive LU/GMRES |
| `metal` | `julia_metal/` | Metal.jl, `pair_gather` kernels | host, adaptive LU/GMRES |
| `cuda` | `julia_cuda/` | CUDA.jl | on device |
| `rocm` | `julia_rocm/` | AMDGPU.jl | on device |

`beat_engine_status()` reports *available* only when a functional accelerator
exists — a CUDA device, a ROCm runtime, or Apple Silicon with a working
Metal.jl. The CPU backend is the reference and CI path and is reported
available only under `HORNLAB_BEAT_FORCE_CPU=1`; that gate is about which
backend a *consumer application* should offer, not about whether the CPU path
works. Call it directly with `beat_backend="cpu"` and it runs.

**Reproducibility.** The regular `pair_gather` kernel is bit-reproducible —
one owner per matrix entry, fixed summation order, no atomics — and measures
exactly so: three assembly passes over identical inputs give byte-identical
operators. The default `native` singular correction is **not**: its scatter is
an atomic add, and adding it makes the same three passes differ by 7e-8 on the
single layer and 3e-7 on the hypersingular operator, which reaches about 4e-7
relative (3e-5 dB) in the radiated field. `BLAB_METAL_SINGULAR_MODE=host`
restores byte-identical assembly, at the cost of doing the corrections on the
CPU.

A byte-exact golden file off the CPU backend therefore needs both that variable
and a single BLAS thread — the multithreaded dense LU is not reproducible
either.

## Install

```bash
pip install git+https://github.com/m3gnus/hornlab-beat-bem.git@<40-char sha>
```

Then Julia >= 1.10 on `PATH`, or `HORNLAB_BEAT_JULIA=<path to julia>`, or the
provisioned runtime below. Instantiate the project for the backend you want:

```bash
julia --project=hornlab_beat_bem/julia_metal -e "using Pkg; Pkg.instantiate()"
```

`julia` (CPU), `julia_cuda` and `julia_rocm` likewise.

**Instantiating is also what installs the engine bundle**, and skipping it
costs a lot more than it looks. Each backend project depends on a package
under `julia_engine/` that holds the whole engine; without it `solver.jl`
falls back to compiling the engine from source in every worker process, and
the only symptom is a cold start three to five times longer. See
[Cold start](#cold-start).

### Provisioned runtime

```bash
python -m hornlab_beat_bem.provision --if-gpu
```

is a strict no-op unless a supported GPU is present. Only then does it resolve
Julia (an existing install always wins; otherwise the official portable Julia
1.12.6 is downloaded, SHA-256 verified, into a per-user runtime directory —
override with `HORNLAB_BEAT_RUNTIME_DIR`), instantiate the matching project,
and force artifact resolution ending in a `functional()` check, so a recorded
*ready* state means the first solve computes instead of downloading. Windows,
Linux x86-64 and macOS arm64 have portable downloads configured. Progress and
failures are recorded in `state.json` and surface as `beat_engine_status()`
reasons; retry with `--force`. CUDA's first run pulls several GB (CUDA.jl ships
its own toolkit artifacts; users need only the driver); Metal pulls a small
package, because the driver is the operating system's.

## Use

```python
import hornlab_beat_bem as beat

config = beat.SolveConfig(
    beat_backend="metal",              # or "cpu" / "cuda" / "rocm"
    native_symmetry_plane="yz+xz",     # quarter mesh
    mesh_scale=0.001,                  # mesh in mm
    observation=beat.ObservationConfig(distance_m=2.0, angle_count=37),
)
result = beat.solve_frequencies("horn.msh", [500.0, 1000.0, 2000.0], config)
beat.shutdown_workers()
```

`solve_frequencies()` is the convention boundary. Every returned array is
rescaled to the shared HornLab **unit normal acceleration** convention
(`p_accel = p_vel / (-i omega)`), the impedance pair is unwound into the raw
area-weighted mean source pressure, and an axis-aligned `ObservationFrame`
override is applied as a rigid mesh translation so the observation origin
matches the caller's frame. Tilted frames are refused, not approximated.

The Julia solver underneath:

- uses the `e^{-i omega t}` time convention with outgoing waves `e^{+ikr}` —
  the same as hornlab-metal-bem and hornlab-bempp-bem;
- drives the source tag with `q = i rho omega v_n` on a **1 m/s
  normal-velocity basis**;
- observes polar cuts around the **global mesh origin** with the forward axis
  fixed to **+z** (horizontal cut in x-z, vertical in y-z);
- reports a legacy-scaled impedance pair `[Re(F)/2, -Im(F)/2]` with
  `F = 10 * sym_factor * sum(p_bar * area) / v`.

Symmetry: plane `yz` -> BEAT `x` (half domain, mesh in x >= 0), `yz+xz` -> BEAT
`xy` (quarter, x >= 0 and y >= 0). A y-only `xz` half domain is not
representable and is rejected by `reject_unsupported_native_symmetry`.

Supported: diagonal observation cuts, spherical balloon/DI grids (theta-major,
so WG's DI integration and 3D balloon both work), axial source motion, and
surface trace retention (`SolveConfig.surface_traces`, off by default).

Not supported: multi-source drive -- exactly one velocity source tag at unit
amplitude -- and WG's y-only `xz` half domain, per the symmetry note above.

### The one local patch to the vendored engine

`hornlab_beat_bem/julia/src/` is otherwise a verbatim copy of upstream, so a
re-sync is a straight file copy. The exception is boundary-lab
`feature/multiple-image-near-caches`: upstream's assembly takes a single
image-near correction cache, which is enough for the one ground reflection it
was written for but not for `yz+xz`, whose three mirror transforms would leave
two of them uncorrected and say nothing about it. The patch makes
`image_near_correction_cache` accept a collection; a single cache behaves
exactly as before. **When that lands upstream, drop the patch and re-sync
normally; until then a re-sync must re-apply it.**

**The patch is applied to the CPU assembly only.** Upstream's CUDA paths take
the same single-cache argument and have the same gap, but no machine available
to this project has an NVIDIA GPU, so a CUDA edit here could not be executed,
let alone verified -- and the correction is exactly the kind of change whose
error is a plausible wrong number rather than a crash. `near_correction` is
therefore accepted on the CPU backend and must not be trusted on `cuda` until
someone with the hardware ports and measures it.

## Cold start

A worker must load and compile the engine before it can solve anything. Julia
caches native code only for **packages**, and until `julia_engine/` existed
none of this was in one: `solver.jl` pulled ~20,700 lines of engine in with
`include` and carried ~1,300 lines of driver itself, so every worker process
compiled all of it again. The engine and the driver now live in a package per
backend, each carrying a `PrecompileTools` workload that solves one frequency
on a four-triangle tetrahedron.

**Packaging alone buys almost nothing — the workload is the fix.** Upstream
measured runtime compilations going 370-378 to 351-354 on packaging alone, and
to 89-92 with the workload.

Measured here on an M1 Max against the ATH `250917asro68q` quarter export
(1,209 P1 dofs, `xy` symmetry), arms interleaved, machine load checked at each
sample (3-8 processes above 5% CPU):

| | include path | bundle |
|---|---|---|
| runtime compilations, first solve (CPU) | 344 | **95** |
| time to first result, CPU | 12.4-14.6 s | **2.7-2.9 s** |
| runtime compilations, first solve (Metal) | 846 | **613** |
| time to first result, Metal | 41.4 s | **17.9 s** |
| second solve, same worker, CPU | 1.18-1.22 s | 1.16-1.21 s |
| second solve, same worker, Metal | 0.53 s | 0.43 s |

The last two rows are the control: a warm solve is unchanged, which is what
says this is start-up and not the solver.

The runtime-compilation count is the instrument that matters, because it does
not move with machine load and a wall clock does. `--trace-compile=stderr` on
the worker's own entry path reports it.

**Metal's residual is not fixable by any cache.** GPUCompiler has no disk
cache, so kernel compilation is paid once per process however much is
precompiled — the 613 that remain are mostly that. Keeping one worker alive
across solves is therefore load-bearing rather than an optimisation, which is
what `hornlab_beat_bem.worker` does.

**A missing bundle is silent.** `solver.jl` falls back to including the
sources, so an installation whose environment was never instantiated, or a
wheel built without `julia_engine/` in its package data, still solves and
still gives the right answers — at the old cost, with nothing logged. That is
why `tests/test_engine_bundles.py` asserts the wiring structurally, and why
its one slow test counts compilations rather than trusting that the package
loaded.

### Ahead-of-time codegen is not bit-identical to the JIT

Compiling the kernels ahead of time does not produce the same machine code as
compiling them on first call, so a bundled worker differs from the include
path in the last bits. On `250917asro68q`, CPU backend, 500 / 2000 / 6000 Hz:
pressures agree to 9.7e-7 of peak (about 8 Float32 ulp) and SPL to 5.4e-4 dB
over a response spanning 59 dB.

**How large that becomes in dB depends on the conditioning of the mesh, not
on the size of the codegen difference.** On `test_meshes/sample_detailed.msh`
— where GMRES diverges to a relative residual of 19 and the solve falls back
to LU — the same comparison gives 0.097 dB and 1.1% in pressure. Float32 LU on
an ill-conditioned operator amplifies any change of summation order by roughly
the condition number; nothing about the bundle is different there.

`BLAB_BEAT_ENGINE_BUNDLE=0` forces the include path when a number has to be
reproduced exactly. It is verified bit-identical: pressures, SPL and impedance
compare byte-for-byte equal to the pre-bundle package on both meshes above.
It costs the old start-up.

On the Metal backend the question answers itself — the `native` singular
correction's atomics already make two runs of the *same* code differ by more
than this does.

## Measured performance

M1 Max, Float32, the ATH reference ladder (real ATH `ABEC_FreeStanding`
exports, welded and orientation-repaired), 500 / 2000 / 6000 Hz, one drive,
exterior Neumann, no symmetry. **Every number below holds over that range and
that implementation, and nothing here says anything about multiple drives,
coupled solves, or Windows.**

### Fusing Burton-Miller into the assembly kernel

Forming the Burton-Miller combination inside the pair kernel, before the
rank-1 expansion, rather than assembling four operators and combining them on
the host:

| mesh | N | four-operator | fused | assembly | operator memory |
|---|---:|---:|---:|---:|---:|
| A1 | 1,974 | 0.221 s | 0.104 s | 2.12x | 187 -> 31 MB |
| A3 | 3,898 | 0.742 s | 0.304 s | 2.44x | 729 -> 122 MB |
| A5 | 5,107 | 1.246 s | 0.502 s | 2.48x | 1252 -> 209 MB |
| A2r | 10,230 | 4.674 s | 1.689 s | 2.77x | 5023 -> 837 MB |
| A5r | 20,422 | 19.64 s | 7.25 s | 2.71x | 20018 -> 3337 MB |

**2.12-2.77x on assembly and 6.00x less operator memory**, the memory factor
exact on every mesh tried. Because operator memory is `O(N^2)`, 6x is
`sqrt(6) ~ 2.45x` more dofs at the same peak.

Two things that sound like they should explain this and do not, both measured:
halving the stored bytes per pair is worth **1.00-1.02x**, and occupancy does
not move (`maxTotalThreadsPerThreadgroup` is 384 either way). The win is
collapsing the rank-1 expansion and halving the live accumulators. Anyone
proposing "fuse the accumulators to save stores or FLOPs" should read
`docs/beat-engine-metal.md` first.

The CPU backend gets the same change and less of it: **1.34-1.38x**.

For scale, assembly against hornlab-metal-bem (a one-operator standard
formulation, so it does strictly less work) on the same meshes is within
+/-20% across 1,974-20,422 dofs, BEAT ahead in the middle of the range and
metal-bem ahead at both ends.

### Adaptive dense solve

The fused path hands over one `N x N` matrix and an `N x drives` right-hand
side, and the solver routes per solve by cost model:

    choose GMRES when   drives * t_gmres(N)  <  T_fact(N) + drives * t_tri(N)

Not a dof threshold. The LU amortises one factorization across every drive and
GMRES has none to share, so the drive crossover moves with N; two independent
thresholds would send a three-way design to the slower path at exactly the size
where a dof test says GMRES wins. The refitted model puts the one-drive
crossover near **3,000 dofs**, inside the 2,000-5,000 range measured
independently, and predicts the drive crossover at 5,107 dofs as **2.4**, a
number it was not fitted to. At 5,107 dofs and one drive GMRES is 2.4x faster
than the LU.

GMRES iterations at tolerance 1e-6 **on the true relative residual**, diagonal
preconditioning. These are the figures the shipped iteration constant is fitted
to, quoted from `BeatEngineDenseSolve.jl`'s own calibration docstring rather
than re-measured here — ladder bench figures, not gate figures, which is a
distinction that matters more than it looks (see the drive caveat below):

| mesh | N | 500 Hz | 2 kHz | 6 kHz |
|---|---:|---:|---:|---:|
| A5 | 5,107 | 70 | 51 | 51 |
| A2r | 10,230 | 119 | 85 | 63 |
| A5r | 20,422 | — | 79 | — |

Roughly flat in N, a factor of ~2 in frequency, worse at the bottom of the band
because the uncapped `eta = i/k` coupling degrades conditioning as k goes to
zero.

Two implementation requirements, both load-bearing rather than stylistic:
the Krylov space is carried in **Float64** over the Float32 operator, and the
convergence test is on the **true** residual recomputed against the operator,
not the preconditioned residual the Givens recursion tracks. An
unreorthogonalised Float32 Arnoldi recurrence loses orthogonality on this
operator and reports iteration counts several times too large, which reads
exactly like bad conditioning; `scripts/validate_gmres_burton_miller.jl` exists
to stop that recurring, and deliberately includes a variant that fails.

GMRES falls back to the dense LU on non-convergence, reporting the fallback
rather than raising, and **the fallback is load-bearing rather than a safety
net**. Measured through the shipped path — the physical tag-2 velocity drive,
Metal backend, one drive, `auto` routing:

| mesh | N | routed to | iterations | true residual | outcome |
|---|---:|---|---:|---:|---|
| A1 | 1,974 | LU | — | — | the model prices the LU below GMRES, so GMRES never runs |
| A1r | 7,890 | GMRES | 146–206 | 2.1e-6 – 6.5e-6 | **misses the 1e-6 tolerance, falls back to the LU** |

A1r is the sliver-rim mesh — a 2.95 mm shell rim meshed with 6.9 mm elements —
and hornlab-metal-bem's Krylov path fails on the same mesh family, so this is
the geometry rather than either implementation. What BEAT has that metal-bem
does not is the fallback, so the answer still comes back correct.

**A misrouted GMRES is bounded, not unbounded.** The router prices a
*converging* GMRES, so a mesh where GMRES converges slowly and then stops is
exactly the case it cannot see. It used to pay for the failed Krylov attempt
*and* the factorization with nothing to cap it. The figures that motivated the
bound were taken upstream on a Ryzen 7 5825U, not on the M1 Max everything else
here is measured on: 3.85x worse than forcing the LU on the worst cell, with
the true iteration count running 39–549 and *rising* with frequency — the
opposite ordering to the ATH ladder the shipped constant was fitted on, which
is the point. The defect is invisible on hardware where the model's quantities
are flat. A model-chosen GMRES now gets one LU's worth
of matvecs (`BLAB_BEAT_GMRES_BUDGET`, default 1.0). Exceeding the budget is the
discovery that the routing was wrong, so the solve falls back having spent at
most about twice the LU rather than an unbounded multiple. A GMRES the caller
asked for explicitly is not budgeted.

**How convergence is measured here, and a trap that was in this gate.** The
table above is the production drive. `validate_gmres_burton_miller.jl` used to
excite with a random right-hand side over every DP0 dof, which is materially
easier on this mesh family: it reported health on precisely the configuration
that fails in production. It now drives the same localised tag-2 velocity the
solver applies, and reports a fallback as a fact rather than forbidding one,
because on these meshes falling back *is* the designed response.

Keep the general form in mind if you extend it: **a gate easier than the path
it guards will pass a change that breaks production**, and this one guards the
most actively changed code in the package. Check what right-hand side a gate
builds before trusting what its numbers mean.

**Record the drive beside every iteration count.** A bare count is not a
measurement here. The two drives do not merely shift the numbers, they can
invert an ordering: on the bundled fixture the physical drive reads 53/58/55 at
500/2000/6000 Hz where the random one read 44/51/63, so an argument that this
fixture is hardest at 6 kHz — which the counts supported under one drive —
is false under the other, with every individual figure still looking plausible.
Treat any count written before the gate changed as describing a different
problem.

### Calibrate on a new machine

The cost model carries four measured constants, and they ship calibrated for an
M1 Max. What the model actually weighs is a machine's ratio of dense-GEMM
throughput to memory bandwidth, so **any other machine should refit them**:

```bash
julia -t auto --project=hornlab_beat_bem/julia_metal \
  hornlab_beat_bem/julia/scripts/calibrate_dense_solve.jl 2048 5107 10230 20422
```

It needs no mesh — the primitives are BLAS, not BEM — and prints `export` lines
for `BLAB_BEAT_LU_GFLOPS`, `BLAB_BEAT_MATVEC_ENTRY_SECONDS`,
`BLAB_BEAT_MATVEC_DOF_SECONDS` and `BLAB_BEAT_TRIANGULAR_GBPS`. Take at least
three sizes: two points cannot tell a wrong exponent from a wrong constant.

Three things it learned from being run on a machine unlike the one it was
written on (a Ryzen 7 5825U, where every quantity the M1 Max holds flat
moves). It warms `lu!`/`mul!`/`ldiv!` before measuring — unwarmed, the sample
nearest the crossover read 51.8 GFLOP/s against 230.2 warm. It reads the LU
and triangular constants **at the crossover** rather than averaging or
maximising over sizes, because the crossover is the only place the two modelled
costs are close and therefore the only place a constant can change a decision.
And it clamps the matvec's linear term at zero rather than letting it fit
negative, which it does wherever effective bandwidth *falls* with N — a
negative constant is one the solver refuses to start with.

The fifth constant, `BLAB_BEAT_GMRES_MODEL_ITERATIONS`, is the operator's own,
and it needs re-measuring whenever the formulation, the coupling parameter or
the quadrature changes. Do not assume it travels between hosts either: it ships
at 70 from this ladder, where the count falls with frequency, and it has been
measured on other hardware both far outside that range and rising with
frequency instead. If the router's choices look wrong on your machine, this is
the constant to check first — and note that a mis-set iteration constant costs
time rather than accuracy, because a GMRES that is routed to and then fails
falls back to the LU.

The CUDA and ROCm backends factor on the device and do not route through this
model.

## Validation

### Against a second solver

On the ATH ladder, against hornlab-metal-bem — a different formulation
(standard operator with a complex-k shift, not Burton-Miller) and a different
implementation — four of the five meshes agree to **<= 0.26 dB on-axis and
<= 0.10 dB in main-lobe shape** at 500 / 2000 / 6000 Hz.

The fifth, A1, disagreed by 0.84 dB, and it is worth knowing why rather than
treating it as a tolerance. It is not quadrature: q6/s6 instead of q4/s4 moves
it by 0.008 dB, and adaptive near-field quadrature on the other solver moves it
not at all. It is discretisation, and it converges away — subdividing A1
one-to-four, which keeps the element *shapes* and so keeps its slivers, takes
the 500 Hz on-axis disagreement from 0.842 dB to **0.021 dB**. A1 at 1,974 dofs
is under-resolved for its own geometry: the 6 el/lambda rule is satisfied, but
a 2.95 mm shell rim is not resolved by 6.9 mm elements, and the two
formulations express that error differently. Neither solver is wrong and the
fix is the mesh.

### Gates in this repository

Every script runs against the bundled fixtures and prints its tolerances.

| script | what it gates |
|---|---|
| `validate_metal_exterior.jl` | Metal operators, boundary pressure and radiated field against the CPU build |
| `validate_metal_symmetry.jl` | the same across symmetry off / x / xy, where a sign slip in the image-singular correction would otherwise be silent |
| `validate_metal_coupled.jl` | the coupled FEM-BEM-LEM paths, monolithic and condensed, against the pure CPU build |
| `validate_metal_fused_burton_miller.jl` | the fused system against the four-operator system on the same mesh, frequency and quadrature — Float32 summation order is the only difference, so the tolerance is noise, not physics |
| `validate_gmres_burton_miller.jl` | the adaptive solve on a real assembled operator across the band: LU agreement, true residual, three independent Krylov variants agreeing on the iteration count, independence from the restart length, and that plain Float32 single Gram-Schmidt is materially worse |
| `validate_rocm_*.jl` | the ROCm equivalents |
| `smoke_coupled_solver.jl` | the coupled suite as a standalone smoke test |

```bash
J=hornlab_beat_bem/julia
julia --project=hornlab_beat_bem/julia          $J/tests/runtests.jl
julia --project=hornlab_beat_bem/julia_metal    $J/scripts/validate_metal_exterior.jl
julia --project=hornlab_beat_bem/julia_metal    $J/scripts/validate_metal_symmetry.jl
julia --project=hornlab_beat_bem/julia_metal    $J/scripts/validate_metal_coupled.jl
julia --project=hornlab_beat_bem/julia_metal    $J/scripts/validate_metal_fused_burton_miller.jl
julia -t auto --project=hornlab_beat_bem/julia  $J/scripts/validate_gmres_burton_miller.jl
pytest                 # Python surface
pytest -m slow         # plus a real Julia solve
```

## Tuning knobs

All are environment variables; the defaults are the shipped configuration.

| variable | default | effect |
|---|---|---|
| `BLAB_BEAT_DENSE_SOLVE` | `auto` | `lu` or `gmres` forces the choice past the cost model |
| `BLAB_BEAT_GMRES_TOL` | `1e-6` | tolerance on the true relative residual |
| `BLAB_BEAT_GMRES_BUDGET` | `1.0` | matvec budget for a *model-chosen* GMRES, in units of one LU; exceeding it falls back. An explicitly requested GMRES is not budgeted |
| `BLAB_BEAT_FUSED_BM` | `1` | `0` restores the four-operator exterior path |
| `BLAB_METAL_REGULAR_KERNEL_MODE` | `pair_gather` | `pair_atomic`, `pair_owned`, `entry_owned` are diagnostics |
| `BLAB_METAL_GATHER_BUDGET_MB` | `512` | trial-chunk memory budget |
| `BLAB_METAL_SINGULAR_MODE` | `native` | `host` does the singular corrections on the CPU, which makes assembly byte-identical run to run |
| `BLAB_BEAT_ENGINE_BUNDLE` | `1` | `0` ignores the precompiled bundle and includes the engine from source: bit-identical to the pre-bundle package, at the old cold start |
| `HORNLAB_BEAT_JULIA` | — | explicit Julia executable |
| `HORNLAB_BEAT_FORCE_CPU` | — | `1` reports the CPU backend as available |

`docs/beat-engine-metal.md` and `docs/beat-engine-core.md` list the rest.
Those pages are Boundary Lab's, kept verbatim; `VENDORING.md` notes the two
places where this README's measurements supersede them.

## Licence and provenance

**GPL-3.0-or-later.** The solver is Boundary Lab's BEAT Engine
(<https://github.com/JWSound/boundary-lab>), GPL-3, copyright the Boundary Lab
authors; this package is a derivative work and stays GPL-3.
[`VENDORING.md`](VENDORING.md) records the upstream commit this was taken from
and every single difference from it — a short list, because the solver core is
copied byte for byte.

Combining this with an AGPL-3 program is explicitly permitted by GPLv3 §13,
which applies AGPL's network clause to the combination. No relicensing is
needed, and none is possible.

## Consumers

Waveguide Generator registers this package as its `beat` engine
(`server/solver/beat.py`) and pins it by commit SHA. The Python API this
sync preserves unchanged: `beat_engine_status`, `SolveConfig`,
`ObservationConfig`, `ObservationFrame`, `reject_unsupported_native_symmetry`,
`solve_frequencies`, `shutdown_workers`.

Two things a consumer should know before bumping its pin:

- **`beat_engine_status()` now reports *available* on Apple Silicon**, with
  backend `metal`, where it previously reported no supported GPU. For an
  application that lists engines by probing, that is a new engine appearing on
  every Mac. It is additive and user-selectable, but it is a behaviour change
  and belongs in a reviewed pin bump rather than arriving as a side effect.
- **`SolveConfig(beat_backend=...)` accepts `"metal"`**, and the rejection
  message for an unknown backend is now generated from `BEAT_BACKENDS` rather
  than hardcoded.
