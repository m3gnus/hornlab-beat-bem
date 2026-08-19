# hornlab-beat-bem

BEAT Engine BEM solver backend for HornLab / WG. Wraps the Burton–Miller
Helmholtz solver from [boundary-lab](https://github.com/m3gnus/boundary-lab)
(GPL-3.0) — vendored under `hornlab_beat_bem/julia/` — behind the same
native-result surface as `hornlab-bempp-bem`, so WG's result mapping consumes
it unmodified.

**Backends.** `cuda` (NVIDIA) and `rocm` (AMD) are the product; `cpu` is
internal scaffolding used to validate the plumbing and as the CI/regression
path on GPU-less hosts. `beat_engine_status()` reports *available* only when a
functional GPU path exists, or when `HORNLAB_BEAT_FORCE_CPU=1` forces the CPU
path for testing.

## Requirements

- Python ≥ 3.10, numpy.
- Julia ≥ 1.10 on `PATH`, or `HORNLAB_BEAT_JULIA=<path to julia executable>`.
- The Julia environment instantiated once per backend:

```bash
julia --project=hornlab_beat_bem/julia -e "using Pkg; Pkg.instantiate()"
```

(`julia_cuda` / `julia_rocm` likewise, on hosts with that hardware.)

## Conventions (the part that bites)

The vendored Julia solver:

- uses the `e^{-iωt}` time convention with outgoing waves `e^{+ikr}` — the
  same as hornlab-metal-bem and hornlab-bempp-bem;
- drives the source tag with `q = i·ρ·ω·v_n` on a **1 m/s normal-velocity
  basis**;
- observes polar cuts around the **global mesh origin** with the forward axis
  fixed to **+z** (horizontal cut in x–z, vertical in y–z);
- reports a legacy-scaled impedance pair `[Re(F)/2, −Im(F)/2]` with
  `F = 10 · sym_factor · Σ(p̄·area) / v`.

`solve_frequencies()` is the convention boundary: every returned array is
rescaled to the shared HornLab **unit normal acceleration** convention
(`p_accel = p_vel / (−iω)`), the impedance pair is unwound into the raw
area-weighted mean source pressure, and an axis-aligned `ObservationFrame`
override is applied as a rigid mesh translation so the observation origin
matches the caller's frame. Tilted frames are refused, not approximated.

Symmetry: WG plane `yz` → BEAT `x` (half domain, mesh in x ≥ 0), `yz+xz` →
BEAT `xy` (quarter, x ≥ 0 ∧ y ≥ 0). WG's y-only `xz` half domain is not
representable and is rejected by `reject_unsupported_native_symmetry`.

Not supported (yet): diagonal observation cuts, spherical balloon/DI grids
(the solver samples a Fibonacci lattice, not the theta-major grid WG needs),
axial source motion, surface trace retention, multi-source drive.

## Quick use

```python
import hornlab_beat_bem as beat

config = beat.SolveConfig(
    beat_backend="cpu",                # or "cuda" / "rocm"
    native_symmetry_plane="yz+xz",     # quarter mesh
    mesh_scale=0.001,                  # mesh in mm
    observation=beat.ObservationConfig(distance_m=2.0, angle_count=37),
)
result = beat.solve_frequencies("horn.msh", [500.0, 1000.0, 2000.0], config)
beat.shutdown_workers()
```

## Validation

Validated on the `benchmarks/abec-rerun` ASRO references against ABEC3 and
against hornlab-bempp-bem on identical meshes, frequencies, and observation
grids (see the WG workspace validation notes). Slow end-to-end tests:
`pytest -m slow`.

## WG integration

`wg2/server/solver/beat.py` registers this package as the `beat` engine. To
pin it, push this repo to `github.com/m3gnus/hornlab-beat-bem` and add to
`wg2/pins.json`:

```json
"hornlab-beat-bem": {
  "repo": "https://github.com/m3gnus/hornlab-beat-bem.git",
  "sha": "<40-char commit sha>"
}
```

then run `python scripts/gen_requirements.py` and add `"hornlab-beat-bem"` to
`REQUIRED_DISTRIBUTIONS` in `wg2/scripts/bootstrap.py`.

## License

GPL-3.0-or-later (the vendored BEAT Engine solver is GPL-3.0 from
boundary-lab; WG's AGPL-3.0 combines with it per GPLv3 §13).
