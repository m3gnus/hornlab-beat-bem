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
| `hornlab_beat_bem/worker{_registry,_host,_client}.py` | the worker that outlives the application: its key and registry, the detached host process, the adopting client |
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
beat.shutdown_workers()      # or detach_workers() to leave it warm for next time
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
  `F = 10 * sym_factor * sum(p_bar * area) / v`, where `sym_factor` counts
  **real** radiators — so it is 1, 2 or 4 for a mirror-reduced mesh and 1 for
  a ground plane, whose image is a fiction.

Symmetry: plane `yz` -> BEAT `x` (half domain, mesh in x >= 0), `yz+xz` -> BEAT
`xy` (quarter, x >= 0 and y >= 0). A y-only `xz` half domain is not
representable and is rejected by `reject_unsupported_native_symmetry`.

### What comes back, and what it is normalised to

`spl_db` is absolute SPL in dB re 20 uPa. `directivity_db` (alias
`spl_norm_db`) is the same field normalised so the reference angle reads
0 dB, which is what hornlab-metal-bem means by that name — the two differ by
the reference sample's own level, about 94 dB for a 1 Pa reference. This
alias previously returned `spl_db` unchanged, so a consumer written against
the metal package's contract read absolute SPL under a name that promises a
pattern.

The reference is the first sample of smallest `|angle|`: on axis for the
ordinary cut, which `ObservationConfig` already requires to span 0 degrees,
and deterministic for a grid that straddles zero without landing on it.
`directivity_reference_index` and `directivity_reference_deg` say which sample
was used. Amplitudes are floored at -120 dB SPL before the dB conversion, as
in hornlab-metal-bem, so a null on the reference angle lifts the cut by a
visibly implausible amount instead of producing `nan`.

`observation.origin` names which feature the arcs are centred on, and this
package cannot find a mouth or a throat in a mesh: the solver measures around
its own coordinate origin and the only freedom here is the rigid translation
`frame_override` asks for. So `"mouth"` or `"throat"` **requires** a frame,
and is refused without one rather than accepted and ignored. Resolve the point
where the CAD model is and pass `ObservationFrame(origin=...)`; the default
`None` observes around the mesh coordinate origin.

A returned sweep is complete or it is an error. `completed` carrying fewer
results than frequencies requested, an event stream that ends without a
terminal event, and a pressure block whose dimensions do not match the
angle grid all raise. The one short result is a cancellation, and it says so:
`cancelled` is true, `requested_frequency_count` records what was asked for,
and `is_partial` compares the two.

### Rigid half space

```python
config = beat.SolveConfig(
    ground_plane={"enabled": True, "axis": "y", "height_m": 1.0},
)
```

One rigid, infinite, perfectly reflecting boundary through the origin, solved
by the image method. **The plane is named by the axis it bounds, never by an
axis-pair token**, because that token means three different things across this
workspace: boundary-lab `"xy"` is a quarter model, hornlab-bempp-bem `"xy"` is
the z=0 plane, and BEAT `"xy"` is x-and-y mirrors. Nothing is passed through by
name here; `_GROUND_AXIS_TO_BEAT_MODE` in `config.py` is the only translation.

The fluid occupies `axis >= 0`. `height_m` is the height of the model origin
above the plane, so the mesh is translated by `+height_m` along the axis and
the containment check is `min(coord) + height_m >= 0`; equality is allowed and
means the model rests on the plane. `enabled` is the on/off signal — a disabled
ground plane still carries an axis, so the field's presence is not enablement.

**Axis `"y"` only.** BEAT's half space is `rigid_ground_transform()`, signs
(1, -1, 1): the Y=0 plane and nothing else. In WG's frame (z the horn axis, y
vertical) that is the floor, the ordinary case. Axis `"x"` is a side wall and
`"z"` a rigid wall behind the throat, and both are refused by name rather than
substituted — hornlab-bempp-bem does all three. `GROUND_PLANE_AXES` advertises
`("y",)` so a capability probe can say which axes exist instead of a boolean.

**It is not a symmetry plane, and does not compose with one.** The two are
separate configuration axes with separate machinery, because they are different
physics that happen to share the solver's one `symmetry` field:

| | `native_symmetry_plane` | `ground_plane` |
|---|---|---|
| the mesh is | half or a quarter of a mirror-symmetric body | the whole body |
| must reach the plane | yes, its rim lies on the cut | no, it may float clear |
| a face lying in the plane | rejected | rejected |
| the image is | part of the real radiator | fictitious |
| reported impedance | multiplied by the copy count | counted once |

A ground plane on axis y also *destroys* the xz mirror outright, since the
model is lifted clear of y=0 and no longer touches it. WG degrades a quarter
domain to a `yz` half for that reason — but BEAT carries a single
image-transform set, so even the surviving half cannot ride alongside the
ground image. A grounded solve here therefore runs the full domain, at roughly
4x the cost of a quarter, and `reject_unsupported_ground_plane` says so rather
than quietly picking one.

The containment guard lives in the Julia driver
(`validate_ground_plane_domain!`), because only it sees the mesh after loading,
scaling and translation. It is ported from hornlab-metal-bem and rejects a body
straddling the plane, any face lying flat in it — such a face is coincident
with its own image and the boundary integral is singular there — and a gap
below `min_clearance_m`. `validate_symmetry_fundamental_domain!` cannot do this
job: `symmetry_active_axes(:ground)` is empty, so it is a no-op for this mode
and a straddling body would otherwise solve in silence.

The numbers are gated against a closed form, not against another BEAT path:
case 3 of `validate_analytic_exterior.jl` scores the half-space **field**
against the exact source-plus-image solution, and case 4 scores the reported
**impedance**, which fails independently — the defect above left the field
entirely correct. See [Validation](#validation).

Supported: diagonal observation cuts, spherical balloon/DI grids (theta-major,
so WG's DI integration and 3D balloon both work), axial source motion, and
surface trace retention (`SolveConfig.surface_traces`, off by default).

Not supported: multi-source drive -- exactly one velocity source tag at unit
amplitude -- WG's y-only `xz` half domain, per the symmetry note above, a
ground plane on any axis but `y`, and a ground plane combined with native
symmetry.

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

### The worker outlives the application

Keeping one worker alive across solves removes the cold start *within* a
session. It does nothing for the next session, and that is where the cost was
actually being paid: an application that starts a worker, solves, and quits
pays the whole thing again on its next launch. Measured on the packaged
Waveguide Generator 0.3.1 on an M1 Max, from the app's own job events: 15-19 s
to the first result after a restart against 3.4-3.9 s warm, with the warm-up
firing correctly. **The warm-up cannot fix it** — the warm-up and the user's
first solve compete for the same single worker, so whichever wins pays the
compilation and the other waits for it. The wait is reshuffled, not removed.

So the worker now outlives the client. A detached **host process** owns the
Julia child on the same inherited-pipe protocol as before and serves it over a
socket; a later application launch finds that host in a per-user registry and
adopts it, Julia runtime and all.

Measured here, three consecutive Python processes against a fresh registry,
`--project=julia_metal`, four-triangle tetrahedron, machine under load:

| | first process | second | third |
|---|---|---|---|
| worker ready | 24.00 s | **0.01 s** | **0.01 s** |
| first solve | 12.15 s | **0.02 s** | **0.02 s** |
| SPL | 37.11 dB | 37.11 dB | 37.11 dB |

The same shape on the CPU backend is 2.00 s against 0.01 s. The SPL row is the
control: adoption returns the same numbers, because it is the same runtime —
the host pid *and* the Julia pid are unchanged across all three.

**Why a host process rather than a socket in Julia.** `julia/src/` is a
verbatim vendored copy (`VENDORING.md`), and a transport change there would
have to be re-applied on every re-sync, permanently. The host keeps the Julia
side untouched: it speaks the JSON lines `BeatEngineDriver.jl` has always
spoken, on the same stdin and stdout.

**Identity is the whole safety argument.** A worker is adopted only when the
Julia executable, the solver script, the project, the sysimage, the thread
count, the installed package version, a **content fingerprint of the wrapper
and its Julia dependency manifests**, the Julia-relevant environment
(`JULIA_DEPOT_PATH`, `JULIA_LOAD_PATH`, `JULIA_PROJECT`, every `BLAB_*`) and
the wire protocol version all match. The last four are in there because an
adoption that crosses them has no symptom: the wrong worker answers every
request perfectly and returns the previous configuration's numbers. The
fingerprint is the one doing the work — **the version number cannot**, because
consumers pin this repository by commit SHA, so the declared version sits
still across a re-vendor, a driver fix or a wrapper change while the solver
underneath it moves. It hashes `hornlab_beat_bem/*.py`, `julia/*.jl`,
`julia/src/*.jl`, every bundled backend and engine-bundle `Project.toml` and
`Manifest.toml`, and `julia_engine/*/src/*.jl`: about 1.3 MB over 72 package
files. A selected custom project and sysimage are content-hashed too. The
result is cached once per runtime choice in each client process. The host
re-checks the key during the handshake, so a client cannot talk its way in
past a record it was handed.

**Lifetime, for an embedding application:**

```python
beat.detach_workers()      # quit hook: let go, leave the worker running
beat.shutdown_workers()    # stop the worker too (the default; scripts, tests)
```

`shutdown_workers(detach=True)` is the same thing as `detach_workers()`. An
application that wants a warm next launch calls the detaching one from its
shutdown hook and nothing else changes: `get_worker`, `solve_frequencies` and
`warm_up` keep their signatures, and a solve is streamed through the host
exactly as it was through the pipe.

Nothing accumulates. A host with no connected client and no running job retires
itself after `HORNLAB_BEAT_WORKER_IDLE_S` (30 minutes by default), and every
way a host can be gone — a dead pid in the registry, a socket file nobody
answers, a wedged process, a client killed mid-solve — resolves to a clean
respawn. Two applications launched together take a cross-process lock, so the
race produces one worker rather than two. A client that dies mid-solve is
noticed by the host, which cancels the job and replaces the runtime rather than
handing a mid-job worker to the next adopter.

**Retiring is no longer expensive.** A failed or abandoned solve still retires
the Julia child — after an accelerator failure the runtime may be in any state
— but the host immediately starts its replacement, so the compilation happens
while nobody is waiting rather than in front of the next solve.

**Transport.** A Unix domain socket where one is available and the path fits in
`sun_path` (about 100 bytes), and a loopback TCP port otherwise. The fallback
is both the Windows path and what a deep registry directory selects on POSIX,
so the two are not separate code. Both are guarded the same way: the registry
directory is owner-only (0700 on POSIX and the user's local application-data
directory on Windows), the key file is 0600 on POSIX, and the handshake
carries a per-host token that only a process able to read that file can
present. The Windows registry defaults to
`%LOCALAPPDATA%\HornLab\BEAT\workers`, so a later application process finds
the same host without an override. If `LOCALAPPDATA` is absent it uses
`%USERPROFILE%\AppData\Local`. A stripped service environment with no profile
is refused because its system temp directory may be shared across accounts;
set `HORNLAB_BEAT_WORKER_DIR` to a service-private local directory in that
environment.

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

### Every other gate here is equivalence-based, and the one thing that cannot catch

Every other exterior gate in this repository compares two BEAT code paths:
Metal against CPU, a `:ground` half space against an explicitly mirrored full
space, LU against GMRES. Each of those shares a kernel with what it checks, and
a shared kernel is what such a comparison cannot see through. Conjugate the
Green's function globally — `exp(+ikr)` becomes `exp(-ikr)`, the time
convention silently inverted — and **both sides of every one of those
equalities conjugate together**. That is measured, not argued: `runtests.jl`'s
`rigid y0 half-space Green function` check reports a relative error of
**1.203e-15 at +k and 1.203e-15 at -k**, bit-identical, and
`validate_analytic_exterior.jl` prints that pair on every run so the blind spot
stays visible.

That script is the answer — a reference the kernel under test did not produce.
Four cases and two controls, CPU backend, Float64:

| case | what fixes the answer |
|---|---|
| 1 pulsating sphere, free field | closed form; includes the icosphere's faceting error, reported separately |
| 2 interior monopole, free field | Neumann data of a point monopole at the centre, so the faceted surface is the exact boundary and geometry error is zero |
| 3 rigid-image monopole over y=0, `symmetry_mode=:ground` | driven with the Neumann data of source-plus-image, which is the exact solution of the half-space Neumann problem |
| 4 the ground image counts once | scores the reported **impedance**, a different failure surface from the field — see the ground-plane section |

Measured at the defaults, identically on ubuntu-latest and macos-latest:
0.0223 / 0.0188 / 0.0203 dB worst level and 0.233 / 0.077 / 0.094 deg worst
phase for cases 1–3, and +0.0004 dB of impedance movement for case 4.

**The controls are asserted on phase, and that is a measured decision, not a
stylistic one.** For a real-velocity drive the Neumann data are purely
imaginary, so `conj(q) = -q`; every operator conjugates under `k -> -k`, and
the radiated field satisfies `field(-k) = -conj(field(+k))` exactly. Measured
on the case-1 drive, the conjugated solve's worst level error is
`0.022304990113225294` dB and the correct solve's is
`0.022304990113225294` dB — **bitwise identical**, field difference
`0.000e+00` — while its phase is out by 90.7 degrees. A gate scoring SPL alone
does not merely pass a solver radiating incoming waves; it awards it a score
indistinguishable from a correct one, and no amount of tightening the level
tolerance can fix that.

That identity is conditional on the drive, and the condition is the production
case rather than a special one: a real normal velocity is what WG drives with,
so the bitwise-invisible regime is the one that ships. Under case 2's
general-phase drive the identity breaks and the two differ in the fifth
significant figure (0.018798 against 0.018793 dB) — still far inside the pass
tolerance, so the conclusion holds either way.

The controls are also not asserted on a relative-error percentage, because that
tracks ka rather than the presence of the defect: 21.7% at 400 Hz, 142.5% at
1000 Hz, 136.2% at 2000 Hz, so a floor calibrated at one frequency silently
stops controlling at another.

### Gates in this repository

Every script runs against the bundled fixtures and prints its tolerances.

| script | what it gates |
|---|---|
| `validate_analytic_exterior.jl` | the radiated field and the reported impedance against a **closed form** rather than another BEAT path, with two controls that must fail — see below |
| `validate_metal_exterior.jl` | Metal operators, boundary pressure and radiated field against the CPU build |
| `validate_metal_symmetry.jl` | the same across symmetry off / x / xy / ground, where a sign slip in the image-singular correction would otherwise be silent. The ground case runs twice: lifted clear of the plane, and resting on it, which is what puts real and image elements in coincident and adjacent pairs (389 image singular pairs against 0 for the lifted case) |
| `validate_metal_coupled.jl` | the coupled FEM-BEM-LEM paths, monolithic and condensed, against the pure CPU build |
| `validate_metal_fused_burton_miller.jl` | the fused system against the four-operator system on the same mesh, frequency and quadrature — Float32 summation order is the only difference, so the tolerance is noise, not physics |
| `validate_gmres_burton_miller.jl` | the adaptive solve on a real assembled operator across the band: LU agreement, true residual, three independent Krylov variants agreeing on the iteration count, independence from the restart length, and that plain Float32 single Gram-Schmidt is materially worse |
| `validate_rocm_*.jl` | the ROCm equivalents |
| `smoke_coupled_solver.jl` | the coupled suite as a standalone smoke test |

```bash
J=hornlab_beat_bem/julia
julia --project=hornlab_beat_bem/julia          $J/tests/runtests.jl
julia -t auto --project=hornlab_beat_bem/julia  $J/scripts/validate_analytic_exterior.jl
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
| `BLAB_BEAT_GMRES_TOL` | `1e-5` | tolerance on the true relative residual. Exterior solves only; the coupled FEM/LEM path factorizes directly. See `VENDORING.md` |
| `BLAB_BEAT_GMRES_BUDGET` | `1.0` | matvec budget for a *model-chosen* GMRES, in units of one LU; exceeding it falls back. An explicitly requested GMRES is not budgeted |
| `BLAB_BEAT_FUSED_BM` | `1` | `0` restores the four-operator exterior path |
| `BLAB_METAL_REGULAR_KERNEL_MODE` | `pair_gather` | `pair_atomic`, `pair_owned`, `entry_owned` are diagnostics |
| `BLAB_METAL_GATHER_BUDGET_MB` | `512` | trial-chunk memory budget |
| `BLAB_METAL_SINGULAR_MODE` | `native` | `host` does the singular corrections on the CPU, which makes assembly byte-identical run to run |
| `BLAB_METAL_PIPELINE` | by mesh size | Overlaps the next frequency's GPU assembly with this one's CPU solve. Unset, the solver decides per solve — on at **1,900 P1 dofs** and above. `0` forces sequential, `1` forces the overlap; see below |
| `BLAB_BEAT_ENGINE_BUNDLE` | `1` | `0` ignores the precompiled bundle and includes the engine from source: bit-identical to the pre-bundle package, at the old cold start |
| `HORNLAB_BEAT_JULIA` | — | explicit Julia executable |
| `HORNLAB_BEAT_FORCE_CPU` | — | `1` reports the CPU backend as available |
| `HORNLAB_BEAT_PERSISTENT_HOST` | `1` | `0` runs the worker as a child of this process again — it dies with the application, and every launch pays the cold start |
| `HORNLAB_BEAT_WORKER_IDLE_S` | `1800` | how long a host with no connected client and no job stays alive |
| `HORNLAB_BEAT_WORKER_DIR` | per-user | where the registry (key files, sockets, host logs) lives |
| `HORNLAB_BEAT_WORKER_SPAWN_TIMEOUT_S` | `60` | how long a client waits for a host it started to become reachable |
| `HORNLAB_BEAT_WORKER_TRANSPORT` | by platform | `unix` or `tcp`, forcing a transport that would not otherwise be chosen |

`docs/beat-engine-metal.md` and `docs/beat-engine-core.md` list the rest.
Those pages are Boundary Lab's, kept verbatim; `VENDORING.md` notes the
places where this README's measurements supersede them.

### Sweep threads and sweep pipelining

One default differs from upstream's, set from `worker.py`: the sweep's thread
count. The other used to — `worker.py` forced `BLAB_METAL_PIPELINE=0` — and
that answer now lives in the solver, which decides per solve from the mesh
size. Both stories are below, because the numbers behind the old default are
what motivated the new one.

Measured on an M1 Max (8 performance + 2 efficiency cores) against the ATH
`asro68` quarter model — 1,209 P1 dofs, Metal backend, fused Burton-Miller —
sweeping 12 frequencies from 300 Hz to 12 kHz on a warm worker. Five
interleaved rounds, minimum reported, medians in brackets:

| | `JULIA_NUM_THREADS=8` | `=10` (`os.cpu_count()`) |
|---|---|---|
| `BLAB_METAL_PIPELINE=0` | **2.105 s** (2.138) | 2.792 s (2.926) |
| `BLAB_METAL_PIPELINE=1` | 2.313 s (2.334) | 3.050 s (3.115) |

The two effects are independent and compose: 1.09-1.10x from the pipelining,
1.32-1.33x from the thread count, **1.45x** from the pair. All four arms agree
to 3e-4 dB, which is the assembly's own run-to-run noise floor — the native
singular correction scatters with atomics.

**`julia_threads="auto"` counts performance cores, not all cores.** The Metal
path sets BLAS to `Threads.nthreads()`, so `os.cpu_count()` put 2 of 10
factorization threads on the efficiency cores and the other 8 waited for them.
On a symmetric part this resolves to the same number as before.

**`BLAB_METAL_PIPELINE` is decided per solve, from the dof count.** Assembling
frequency i+1 on a spawned task while the CPU solves frequency i costs about 10%
on this model rather than saving anything. This was first measured as a larger
penalty at the old 10-thread default, which suggested it was an artifact of
oversubscribing the performance cores; the table above rules that out, since it
costs the same 1.09x at 8 threads.

**Where that cost comes from, measured rather than assumed.** The standing
explanation here and upstream was the command queue: Metal.jl keeps it in
task-local storage, so the spawned task builds its own every frequency. That is
true and it is not the mechanism. Timed directly, a spawned task's Metal
overhead is a fixed **0.23–0.29 ms**, and it does not grow with the number of
kernels the task dispatches — 0.288 ms for a task doing one launch, 0.129 ms for
one doing a hundred. The per-frequency penalty it has to explain is two orders
of magnitude larger. Comparing the two arms stage by stage over 20 frequencies,
best of three rounds:

| stage | quarter, 1,209 dofs | full, 4,552 dofs |
|---|---:|---:|
| assembly | +30.3 ms/freq | +25.8 ms/freq |
| field evaluation | +105.2 ms/freq | +74.4 ms/freq |
| CPU solve | +10.2 ms/freq | +80.8 ms/freq |

So the cost is GPU-side contention, not queue construction. One assembly
already saturates this GPU — the cross-frequency concurrency probes reached the
same conclusion from the other direction, measuring concurrent assemblies at
0.85x of sequential — so dispatching the next assembly while this frequency's
field evaluation is still running does not find idle silicon, it interleaves two
queues over the same units. The same GPU work (assembly plus field) measures
**2.82 s sequentially against 5.53 s split across two tasks** on the quarter,
and 8.11 s against 10.11 s on the full model. The CPU solve pays separately, for
the core the spawned task takes.

**This retires the obvious follow-up.** A persistent assembly task holding one
command queue would recover ~0.25 ms per frequency out of ~145. The lever that
would pay is not overlapping GPU work with GPU work — keeping the field
evaluation off the assembly's back — and that is a scheduling change, not a
queue-lifetime one.

On *this model* there is also not much to win by fixing it rather than switching
it off. Instrumented separately here, the sweep is **6.10 s of GPU assembly and
field evaluation against a 1.51 s CPU solve** — the solve is 20% of the work, so
a *perfect* scheduler reaches ~6.1 s against 7.74 s, about **1.10x**, and neither
arm has an idle core to reclaim.

**The sign of the effect depends on mesh size, which is why neither `0` nor `1`
is the right default.** The ratio above is the small model's; the CPU solve grows
as O(N^3) against the assembly's O(N^2), so on a larger mesh there is a much
bigger CPU share to hide behind while the GPU contention it pays for does not
grow with it. Measured over 20 frequencies from 100 Hz to 20 kHz, minimum of four
interleaved rounds:

| model | P1 dofs | symmetry | `PIPELINE=0` | `PIPELINE=1` | |
|---|---:|---|---:|---:|---|
| asro68 quarter | 1,209 | `xy` | **3.217 s** | 3.354 s | 0.96x |
| ATH ladder A1 | 1,974 | off | 3.523 s | **3.118 s** | 1.13x |
| ATH ladder A2 | 2,559 | off | 5.509 s | **4.642 s** | 1.19x |
| ATH ladder A3 | 3,898 | off | 9.247 s | **7.218 s** | 1.28x |
| asro68 full | 4,552 | off | 13.481 s | **9.802 s** | **1.38x** |
| asro68 quarter, subdivided | 4,692 | `xy` | 29.718 s | **26.769 s** | 1.11x |
| ATH ladder A5 | 5,107 | off | 15.058 s | **11.616 s** | 1.30x |

The crossover was then pinned with six spheres filling the gap the waveguide
ladder leaves between 1,209 and 1,974, in a second run of five rounds: 1,202
dofs 0.82x, 1,514 0.97x, 1,742 0.96x, 1,986 1.13x, 2,382 1.22x, 3,122 1.19x.
The two families agree where they overlap — the 1,986-dof sphere's 1.13x against
the 1,974-dof waveguide's 1.13x — so the crossover is between 1,742 and 1,986
dofs, and **the default is on at 1,900 and above**.

Two things that table is evidence for beyond the threshold itself. The rule can
be expressed in dofs alone: the subdivided quarter wins at 4,692 dofs with two
mirror planes, so a symmetry mode that quadruples the assembly does not move the
sign, and the solver needs to know nothing about images. And 1,900 is a machine
constant, not a physical one — a GPU with capacity to spare during an assembly,
or a slower dense solve, puts it elsewhere. That is why the environment variable still wins
in both directions, and why every frequency now reports its `metal_pipeline`
choice and `p1_dof_count` in `native_diagnostics`: a size-aware default that
reported nothing would have invisible regressions.

**What the change is worth, end to end.** Against the previous shipped
behaviour, which forced the overlap off at every size — minimum of three
interleaved rounds, and the choice read back from the diagnostic rather than
inferred from the clock:

| model | dofs | frequencies | before | after | |
|---|---:|---:|---:|---:|---|
| asro68 quarter | 1,209 | 40 | 5.459 s | 5.477 s | 1.00x, overlap off |
| asro68 quarter | 1,209 | 3 | 0.468 s | 0.451 s | 1.04x, overlap off |
| asro68 full | 4,552 | 40 | 25.434 s | **17.520 s** | **1.45x**, overlap on |
| asro68 full | 4,552 | 3 | 2.155 s | **1.784 s** | **1.21x**, overlap on |

The quarter rows are the point of the two-sided check: the mesh this package
serves most often is unchanged, within noise, because the heuristic leaves it
sequential. The full model gains 1.45x on a long sweep and still 1.21x on a
three-frequency one, so the win does not depend on having many frequencies to
amortize the first, un-overlapped assembly over. The two scheduling paths agree
to 1.2e-3 dB, the assembly's own run-to-run noise floor.

So the ~1.10x ceiling argument is sound only where the CPU solve really is 20% of
the work. An engine that overlaps frequencies and is further ahead than that is
ahead on assembly speed; the overlap is a consequence of having a large CPU share
to hide behind, not a separate thing to copy.

Every timing above records the number of foreign Julia workers on the machine
around each sample and discards the sample if there were any; all three runs are
0 dropped, of 56, 60 and 24 samples. Two of them had to be: a peer session's
solver made the first attempt at this measurement wrong by a factor of seven.

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

Three things a consumer should know before bumping its pin:

- **The Julia worker now outlives the application, and a consumer has to opt
  into that.** `shutdown_workers()` still stops it, so an unchanged consumer
  keeps its old behaviour and its old cold start. To get the warm launch, call
  `detach_workers()` — new, and importable from the package root — from the
  quit hook instead. In Waveguide Generator that is one line in
  `server/app.py`'s `shutdown_beat_worker`: `from hornlab_beat_bem import
  detach_workers` and `await asyncio.to_thread(detach_workers)`. Nothing else
  changes; the prewarm task is still cancelled first, and a worker left running
  retires itself after 30 minutes. See
  [The worker outlives the application](#the-worker-outlives-the-application)
  for what the consumer is opting into, and
  `HORNLAB_BEAT_PERSISTENT_HOST=0` for the escape hatch.

- **`beat_engine_status()` now reports *available* on Apple Silicon**, with
  backend `metal`, where it previously reported no supported GPU. For an
  application that lists engines by probing, that is a new engine appearing on
  every Mac. It is additive and user-selectable, but it is a behaviour change
  and belongs in a reviewed pin bump rather than arriving as a side effect.
- **`SolveConfig(beat_backend=...)` accepts `"metal"`**, and the rejection
  message for an unknown backend is now generated from `BEAT_BACKENDS` rather
  than hardcoded.
