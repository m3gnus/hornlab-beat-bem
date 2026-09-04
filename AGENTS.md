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

## The worker is not your child process

Since 0.3.2 a Julia worker runs inside a detached host process and survives
the Python process that started it, so that an application restart can adopt a
warm runtime instead of recompiling one (README, *The worker outlives the
application*). Two consequences matter while working here.

**You can be talking to code you have already changed.** A worker is keyed on
a content fingerprint of `hornlab_beat_bem/*.py`, `julia/*.jl`,
`julia/src/*.jl`, every bundled backend and engine-bundle `Project.toml` and
`Manifest.toml`, and `julia_engine/*/src/*.jl` precisely so that an edit or a
dependency update produces a new key and never adopts the old worker. A
selected custom project's `Project.toml` and `Manifest.toml`, and a selected
custom sysimage, are content-hashed too. Anything that changes behaviour but
is *not* in those inputs -- a file outside those globs, or dependency source
mutated directly in a Julia depot without a Manifest change -- is invisible
to the key, and a stale worker will answer with the old behaviour and no
symptom. If a change refuses to take effect, that is the first hypothesis;
stop the hosts and try again.

```python
from hornlab_beat_bem import worker_registry as r
from hornlab_beat_bem.worker_client import find_live_hosts
for record in find_live_hosts():          # every host this user is running
    print(record.pid, record.endpoint.as_dict(), record.key["package_fingerprint"])
    r.terminate_pid(record.pid)
```

`HORNLAB_BEAT_PERSISTENT_HOST=0` runs the worker as a child process again,
which is the old behaviour and the fastest way to take the mechanism out of a
bisection.

**The suite runs its own registry.** `tests/conftest.py` points every test at a
throwaway directory and kills whatever is left in it, so a test run neither
adopts nor stops the worker a developer has warm. `tests/fake_julia_worker.py`
stands in for `solver.jl --worker` in the lifecycle tests -- same protocol, no
Julia -- so `tests/test_worker_persistence.py` runs in seconds and in CI on any
host. It is not a substitute for the real thing: `pytest -m slow` still drives
a real Julia worker through the same host.

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
julia -t auto --project=hornlab_beat_bem/julia  $J/scripts/validate_analytic_exterior.jl
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

### On a host with no GPU, set `HORNLAB_BEAT_JULIA`

`discover_julia()` resolves one order -- explicit argument, then
`HORNLAB_BEAT_JULIA`, then the provisioned runtime, then `PATH`. The third tier
is **not a search**: `provisioned_julia()` reads `state.json` out of the runtime
directory and returns the single path a previous provisioning run recorded.
*Automatic* provisioning is GPU-gated on purpose -- `provision_gpu()` returns
`skipped` and writes nothing when no matching GPU is present, so a machine
that will never run an accelerator never pays for the Julia download or the
multi-gigabyte CUDA/ROCm artifacts behind it. `--backend auto` never selects
the CPU.

So on a GPU-less host that tier is dead **until someone asks for it by name**:
`provision_cpu()` / `python -m hornlab_beat_bem.provision --backend cpu` has no
hardware gate, instantiates `julia` (no accelerator dependency, no GPU
artifacts) and records the runtime that makes the third tier live. Nothing
infers that, so the gate above still holds for every caller who does not opt
in. Without either, the chain is really env-var-or-`PATH`, and a Julia
installed anywhere else is invisible to it, because
nothing searches: unpacking one by hand next to the checkout does not make it
discoverable, and neither does any other layout the package was never told
about. **This is not platform-specific.** The same three conditions return the
same `None` on Windows, Linux and macOS; it was first hit on a GPU-less Windows
guest, and reproduces identically on macOS with an empty runtime directory and
no `julia` on `PATH`.

That costs more here than it would elsewhere, because **a skip is a failure in
this repository**. Four test files take a `julia` fixture that skips when
discovery returns `None`, and `check_pytest_report.py` rejects any skip at all
(see *Continuous integration*). Without the variable, `pytest` reports 90
passed and 8 skipped and exits 0 locally, and the same run fails the checker in
CI -- and the only thing naming the cause is the skip message. So set the
variable, and use CI's own check to confirm it before the suite rather than
after:

```bash
export HORNLAB_BEAT_JULIA=<path to julia>       # PowerShell: $env:HORNLAB_BEAT_JULIA
python .github/scripts/assert_julia_runtime.py  # one legible error, not eight quiet skips
```

CI never meets this: `julia-actions/setup-julia` puts Julia on `PATH`, which is
why the tier being dead has never shown up as a red run.

For a developer this is still the documented requirement: the variable is one
line and provisioning is a Julia download. The opt-in CPU mode is for the case
the variable cannot serve -- an installer on a machine with no Julia at all,
where there is no path to name. It does not invert the promise `provision.py`
opens with, because it is never selected for anyone: `auto` skips it, and a
GPU gate flag combined with `--backend cpu` is refused rather than resolved.
Nor does it fetch a second Julia onto a host that already has one; the
existing install wins, including one this same runtime directory unpacked
earlier.

The other alternative stays declined. Giving discovery a workspace-relative
search root would make a pip-installed package execute whatever binary happens
to sit under the directory it was launched from -- discovery would depend on
the working directory, which is precisely what the current four tiers avoid.

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

`validate_analytic_exterior.jl` is the exception to that shape, and the reason
it exists: it scores against a **closed form**, not against another BEAT path.
Its correct-case tolerances are discretisation error, which converges cleanly
as O(h^2), so if it starts failing the answer is a finer fixture or a real
defect — never a wider bound. Its controls must keep FAILING: a change that
makes one pass has broken the gate even when every assertion is green. Note
particularly that the controls are asserted on phase, because a globally
conjugated kernel is invisible in level; do not "simplify" them onto SPL.

## Continuous integration

`.github/workflows/ci.yml` runs on pull requests and on pushes to `main`. It
covers the Julia CPU suite and the Python suite on `ubuntu-latest` and
`macos-latest`, and the exterior, symmetry and fused Burton-Miller Metal
validators on `macos-latest` -- the hosted macOS runner has a working Metal
device, so those run for real rather than skipping.

Two things are asserted that an exit code does not cover, because until
2026-09-03 this repository had no CI at all while a consumer pinned it by SHA
and sessions described its branches as gated:

- **Julia has to be present.** Four Python test files take a `julia` fixture
  that skips when `discover_julia()` returns `None`, so without the runtime the
  suite reports 90 passed, 8 skipped and exits 0 -- green over a run that never
  solved anything. (It was three files and three skips when this was written;
  the count moves as solve tests are added, which is the point -- it is not a
  fixed small set that could be watched by hand.)
  `.github/scripts/assert_julia_runtime.py` fails first instead, and
  `check_pytest_report.py` then rejects any skip at all. There is
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

### The condensed-assembly disagreement on AVX-512 hosts

Four assertions in `julia/tests/coupled_condensed_tests.jl` -- lines 592 and
618, the `symmetry == :off` branch of the forked-versus-shared comparison --
fail on `ubuntu-latest` and never on `macos-latest`. They assert **bitwise**
equality on the stated ground that "with no images the fused sweep reduces to
the base sweep". A failing job reports 416 passed, 4 failed, 1 broken.

**This was recorded here as an intermittent. It is not one.** Settled
2026-09-03 by a paired experiment on branch `chore/avx512-mask-experiment`
(`.github/workflows/avx512-experiment.yml`). Each draw ran the Julia CPU suite
twice on the *same* runner from the same checkout: once normally, once with
AVX-512 masked out of the JIT target (`julia -C native,-avx512f,...`). Two arms
on one host is what the CI history could never give, because there every
comparison was also a different machine.

| Host, by `/proc/cpuinfo` flags | Draws | Unmasked | Masked |
|---|---|---|---|
| AVX-512 present | 10 | **fail, every one** | pass, every one |
| AVX-512 absent | 36 | pass, every one | pass, every one |

46 draws, no exception either way. The ten cover all four families the older
sightings named -- `graniterapids`, `sapphirerapids`, `icelake-server` and
`znver4` -- plus a `znver5` that no sighting had seen; the thirty-six are
`znver3` and the AVX-512-less `znver4` below.

On an AVX-512 host it fails **every time**, with the same 201 and 205 differing
entries every time. What looked intermittent was the pool draw.

**The trap that hid this is `Sys.CPU_NAME`.** The earlier record concluded the
ISA level was "necessary but not sufficient" because `znver4` appeared once as
a pass and twice as a fail. It is sufficient: some `znver4` hosts in the pool
do not expose AVX-512 to the guest at all. Across these draws `znver4` came up
3 times with the flags and 14 times without, and failed on exactly the 3. One
CPU model, opposite outcomes, separated perfectly by the feature flags -- so
key on the flags, which the workflow prints, never on the model name.

What diverges, identically on every AVX-512 host:

- `double_layer`, 201 of its 526 computed entries, worst gap 256 ULP;
- `adjoint_double_layer`, 205 of 552, worst gap 148 ULP;
- `single_layer` and `hypersingular`, never;
- forked-against-forked (line 626), never.

Two corrections to what was recorded before. The disagreement is **not** one
ULP -- that was the first differing entry; the worst is 256. And it is not a
rare corner: the fixture assembles only 24 elements, so those 201 entries are
38% of the operator's non-zero content, not 201 in 1.9 million.

**Why those two operators and not the other two.** `double_layer` and
`adjoint_double_layer` are the two formed from `dot(r_vec, normal)`, which
near-cancels for nearly coplanar element pairs. Over the 69 million quadrature
pairs this fixture visits, that sum has a summation condition number above 100
for 0.59% of them, and 4.8e35 at worst. The other two descend from
`dot(r_vec, r_vec)` by way of `weighted_green` -- a sum of squares, whose
condition number is exactly 1.0 for every pair, because non-negative terms
cannot cancel. The two operators that can amplify a changed rounding are
exactly the two that do; the two that provably cannot, never do.

**Why it is a compilation difference, not an arithmetic one.** Line 626
compares two invocations of the *same* compiled function and has never failed;
the four that fail compare two *different* compiled functions.
`_beat_cpu_regular_pair_blocks` and `_condensed_accumulate_pair_blocks!` hold
textually identical quadrature loops, so at `:off` their arithmetic is the same
-- but they are separate bodies optimised in separate contexts, and nothing in
the source pins LLVM to the same floating-point association in both. On aarch64
it happens to pick the same one: dumping both bodies there gives identical
inner-loop instruction mixes (fmadd 2/2, fmul 39/39, fadd 17/17), differing only
in where results are stored. That is why macOS has never failed, and it is a
property of the target rather than of the code.

Two controls, both in the runs' artifacts. The mask removes AVX-512 and nothing
else -- 256-bit and 128-bit vectorisation are unchanged under it (`ymm` 40,
`xmm` 32 in both arms) -- so this is not "vectorisation off, therefore
agreement". And the mask demonstrably applied: on `znver4` the assay counts 43
zmm registers unmasked and 0 masked. On Intel hosts it counts none in either
arm, because LLVM sets `prefer-vector-width=256` for `graniterapids` and
`sapphirerapids` and AVX-512 arrives there as EVEX-encoded 256-bit operations
rather than 512-bit registers -- which is why the assay also counts EVEX-only
registers and mask registers.

`.github/scripts/avx512_condensed_probe.jl` reproduces it standalone on an
AVX-512 host in about 25 seconds, in a process that has run nothing else. So it
needs no particular allocation history, and the earlier "allocation-dependent
loop-tail vectorisation" candidate was narrower than it needed to be: two
separately compiled bodies is enough.

**Windows is untested on the arm that matters, and one null does not change
that.** Every draw above is Linux; macOS is exempt only because it is aarch64.
Nothing in the mechanism is Linux-specific -- it is LLVM's codegen for an x86
AVX-512 target -- so the prediction is that a Windows host whose CPU exposes
AVX-512 fails identically. That prediction is **not** confirmed. It was checked
on 2026-09-03 on the workspace Windows box, which turned out to be a Ryzen 7
5825U: `znver3`, `avx512f false` by `IsProcessorFeaturePresent`. The probe
reported zero differences and the suite gave 424 passed, 0 failed, 1 broken --
the same counts as macOS, exit 0.

That is a correct null and it discriminates nothing, because the host has no
AVX-512 to mask. Do not read it as "Windows passes, so it is not the ISA". The
one fact it does add is that the failure is not merely "x86 on a non-Apple OS":
this host is x86, non-Apple, and green. The masked control was deliberately
**not** run there -- on a CPU without AVX-512 the mask is a no-op, and a second
null would look like a control arm without being one.

Closing this needs a Windows host reporting `avx512f true` -- Intel Ice Lake or
later, or Zen 4/5 with AVX-512 not hidden by the hypervisor. Until then, a
Windows developer on an AVX-512 workstation should expect these four assertions
red on correct code, and should read this section before "fixing" them.

**The contract decision was taken by Magnus on 2026-09-03: the entrywise
absolute floor** (the recommended option below). It landed upstream first —
fork commit `1f90433` on `fix/condensed-entrywise-floor` — and arrived here by
cherry-pick, recorded in `VENDORING.md`. The four assertions now bound every
entry at `1.0f-5 * maximum(abs, shared)`; the cached-vs-uncached fused
comparison stays bitwise. The options are kept below as the decision record.

**The original note, for the record:** The mechanism above is
the evidence the earlier note asked for before anyone touched the assertion; it
is not permission to touch it. `coupled_condensed_tests.jl` is vendored (see
`VENDORING.md`), so a change lands upstream in `boundary-lab` and arrives by
re-sync -- **do not relax the `==` in this repository.** What the finding
changes is which options are honest:

- **Keep `==`.** It is red on every AVX-512 draw, for something that is not a
  defect, and a gate that cries wolf is one people learn to merge past. §7 of
  the workspace policy also makes a red trunk outrank feature work.
- **Relax `:off` to `rtol = 1.0f-5`, matching the sibling branches.** It passes
  comfortably: `≈` on arrays is a *norm* test -- `norm(a - b) <= rtol *
  max(norm(a), norm(b))` -- and the measured relative gap is 2.18e-8 for
  `double_layer` and 1.60e-8 for `adjoint_double_layer`, some 460x inside the
  bound. The objection is not headroom, it is what the test asserts: a norm is
  insensitive to a few badly wrong entries inside a large norm, which is exactly
  the failure mode a fused assembly is most likely to introduce.
- **Bound the entries instead** (recommended). Assert entrywise agreement to an
  absolute floor set by the operator's own largest entry. At `1.0f-5 *
  maximum(abs, shared)` that is 2.0e-12 for `double_layer` against a worst
  measured gap of 3.55e-15 -- about 565x -- and unlike the norm it is a
  guarantee about every entry. It also stops tightening as the entries get
  small, which is what makes the present test fail on near-cancelling terms that
  carry no information.

  Do **not** reach for an entrywise *relative* tolerance instead: the worst
  entrywise relative gap measured is 1.57e-5, so a per-entry `rtol = 1.0f-5`
  fails. That is the same near-cancellation the mechanism turns on -- a tiny
  entry can be many ULPs wrong and still carry no error worth the name -- and it
  is why the floor has to be absolute and tied to the operator's scale.

  All figures here are measured by `avx512_condensed_probe.jl` on a
  `sapphirerapids` draw, not derived.
- **Make the premise true by construction** -- have both paths call one
  `@noinline` body, so `==` holds because there is one compilation. Named for
  completeness; it costs the hot loop the independent optimisation the fork
  exists to allow.

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
