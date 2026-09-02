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

`solver.jl` is the one file that carries local changes, and it is a three-way
merge on every sync. `VENDORING.md` explains the three decisions in it.

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

One check is deliberately conditional. The Julia suite and the Krylov gate
both compare an unreorthogonalised Float32 Gram-Schmidt against the remedies,
so that the agreement between the remedies is not vacuous. *Whether* it
degrades is a property of the host's floating point rather than of the code:
the synthetic system loses 20x on an M1 Max and reportedly not at all on a
Ryzen 7 5825U. A suite that fails where the code is right is reporting a
microarchitecture, so that half is a **warning**, not an assertion. The
agreement between the remedies stays hard. Do not "fix" a warning here by
tightening it into a failure.

Do not weaken a tolerance to make a gate pass. The fused and symmetry gates
compare two code paths that differ only in Float32 summation order, so their
tolerances are noise floors and a real regression will not sit just above one.

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
