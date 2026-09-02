"""The precompilable engine bundles, and the ways they fail silently.

Every failure this file guards against has the same shape: the worker still
solves, and still gives the right answer, but from source -- so the only
symptom is a cold start three to five times longer than it should be. Nothing
raises, nothing logs, and the timings look like the ones from before the
bundles existed. These are structural checks precisely because there is no
runtime signal to assert on.
"""

from __future__ import annotations

import re
import subprocess
import tomllib
from pathlib import Path

import pytest

from hornlab_beat_bem.config import BEAT_CPU, BEAT_CUDA, BEAT_METAL, BEAT_ROCM
from hornlab_beat_bem.runtime import PACKAGE_DIR, default_project

BUNDLE_DIR = PACKAGE_DIR / "julia_engine"
BUNDLES = {
    BEAT_CPU: "BeatEngineCpuBundle",
    BEAT_CUDA: "BeatEngineCudaBundle",
    BEAT_ROCM: "BeatEngineRocmBundle",
    BEAT_METAL: "BeatEngineMetalBundle",
}


def _project(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("backend,name", sorted(BUNDLES.items()))
def test_every_backend_project_declares_its_bundle(backend, name):
    """Without the dep, ``Pkg.instantiate`` installs no bundle at all.

    ``solver.jl`` then falls back to including the sources and the worker is
    simply slow. That is the whole failure mode, and it is invisible at run
    time, so the declaration is asserted here instead.
    """

    project = default_project(backend)
    deps = _project(project / "Project.toml").get("deps", {})
    assert name in deps, f"{project.name}/Project.toml does not depend on {name}"

    manifest = (project / "Manifest.toml").read_text(encoding="utf-8")
    assert f"[[deps.{name}]]" in manifest, f"{project.name}/Manifest.toml has no {name} entry"


@pytest.mark.parametrize("backend,name", sorted(BUNDLES.items()))
def test_manifest_path_to_the_bundle_resolves(backend, name):
    """A relative path that does not resolve is a resolve-time error nobody sees.

    Upstream calls the engine directory ``julia_local`` and this package
    flattens it to ``julia``; the bundle paths are relative to the backend
    project, so a layout change breaks them silently.
    """

    project = default_project(backend)
    manifest = (project / "Manifest.toml").read_text(encoding="utf-8")
    block = manifest.split(f"[[deps.{name}]]", 1)[1]
    match = re.search(r'^path = "(.+)"$', block, re.M)
    assert match, f"{project.name}/Manifest.toml gives {name} no path"
    resolved = (project / match.group(1)).resolve()
    assert resolved == (BUNDLE_DIR / name).resolve()
    assert (resolved / "src" / f"{name}.jl").is_file()


@pytest.mark.parametrize("name", sorted(BUNDLES.values()))
def test_bundle_finds_the_flattened_engine_directory(name):
    """``ENGINE_DIR`` searches for ``julia_local`` then ``julia``.

    Upstream's own tree only ever has the first, so a broken search would pass
    every check made there and fail only here -- at precompile time, inside a
    shipped wheel.
    """

    source = (BUNDLE_DIR / name / "src" / f"{name}.jl").read_text(encoding="utf-8")
    assert '"julia_local", "julia"' in source
    assert (PACKAGE_DIR / "julia" / "src" / "BeatEngineCore.jl").is_file()
    assert (PACKAGE_DIR / "julia" / "BeatEngineDriver.jl").is_file()


@pytest.mark.parametrize("name", sorted(BUNDLES.values()))
def test_precompile_workload_uses_a_request_this_driver_accepts(name):
    """The workload's request must survive this package's own validation.

    It is wrapped in a bare ``catch`` so that a build never fails over an
    optimisation, which means a request the driver rejects costs the entire
    win and reports nothing. It has happened once already: upstream sends
    ``source_motion`` values it ignores, and this driver validates that key.
    """

    from hornlab_beat_bem.config import SolveConfig

    source = (BUNDLE_DIR / name / "src" / f"{name}.jl").read_text(encoding="utf-8")
    motion = re.search(r'"source_motion" => "(\w+)"', source)
    assert motion, f"{name} names no source_motion in its workload"
    # Constructing the config runs the same validation the driver applies.
    SolveConfig(beat_backend=BEAT_CPU, source_motion=motion.group(1))

    backend = re.search(r'"beat_engine_backend" => "(\w+)"', source)
    assert backend and backend.group(1) == "cpu", (
        "the workload must solve on the CPU backend: precompilation runs on a "
        "build machine that may have no accelerator"
    )


def test_wheel_would_ship_the_bundles():
    """Package data is an explicit allow-list; a new directory is not in it."""

    pyproject = _project(Path(__file__).resolve().parents[1] / "pyproject.toml")
    patterns = pyproject["tool"]["setuptools"]["package-data"]["hornlab_beat_bem"]
    assert "julia_engine/*/*.toml" in patterns
    assert "julia_engine/*/src/*.jl" in patterns
    assert "julia/*.jl" in patterns  # BeatEngineDriver.jl rides along here


def test_solver_script_is_only_an_entry_point():
    """The driver must live in a file a package can include, not in the script.

    Julia caches native code for packages only, so anything left in
    ``solver.jl`` is recompiled in every worker process.
    """

    script = (PACKAGE_DIR / "julia" / "solver.jl").read_text(encoding="utf-8")
    assert len(script.splitlines()) < 100, "solver.jl has grown a body again"
    assert "BLAB_BEAT_ENGINE_BUNDLE" in script
    driver = (PACKAGE_DIR / "julia" / "BeatEngineDriver.jl").read_text(encoding="utf-8")
    assert "function worker_loop" in driver
    assert "function main(" in driver


@pytest.mark.slow
def test_the_bundle_is_what_the_worker_actually_loads():
    """Run the CPU worker's own entry path and count runtime compilations.

    This is the only check here that can tell a live bundle from a silently
    bypassed one, because both produce the same answers. ``--trace-compile``
    counts what the process had to compile for itself; the bundle exists to
    move that work to build time, and the count is independent of machine
    load in a way a wall clock is not.
    """

    import hornlab_beat_bem as beat

    julia = beat.discover_julia()
    if julia is None:
        pytest.skip("no Julia executable (set HORNLAB_BEAT_JULIA)")

    project = default_project(BEAT_CPU)
    probe = (
        "using BeatEngineCpuBundle;"
        "const B = BeatEngineCpuBundle;"
        "d = mktempdir();"
        'mesh = joinpath(d, "w.msh");'
        "write(mesh, B.WORKLOAD_MESH);"
        'request = Dict{String,Any}("schema_version" => 2,'
        ' "beat_engine_backend" => "cpu", "frequencies_hz" => [1000.0],'
        ' "config" => Dict{String,Any}("mesh_file" => mesh, "scale_factor" => 1.0,'
        ' "distance" => 1.0, "axial_offset" => 0.0, "step_size" => 90.0,'
        ' "min_angle" => 0.0, "max_angle" => 90.0, "freq_min" => 1000.0,'
        ' "freq_max" => 1000.0, "freq_count" => 1, "tag_throat" => 2,'
        ' "rho" => 1.2041, "sound_speed" => 343.0, "symmetry" => "off",'
        ' "source_motion" => "normal"));'
        "redirect_stdout(devnull) do; B.solve_request(request); end;"
        'println(stderr, "PROBE_OK ", B.ENGINE_DIR)'
    )
    completed = subprocess.run(
        [julia, f"--project={project}", "--startup-file=no", "--trace-compile=stderr", "-e", probe],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if "ArgumentError: Package BeatEngineCpuBundle" in completed.stderr:
        pytest.skip(
            f"julia --project={project} is not instantiated; run "
            f'julia --project={project} -e "using Pkg; Pkg.instantiate()"'
        )
    assert completed.returncode == 0, completed.stderr[-2000:]
    assert "PROBE_OK" in completed.stderr, "the workload request no longer solves"
    # The engine directory the bundle found must be this package's, not a
    # stale checkout that happened to be adjacent.
    assert str(PACKAGE_DIR / "julia") in completed.stderr

    compilations = sum(1 for line in completed.stderr.splitlines() if "precompile(" in line)
    # Measured on an M1 Max: 23 with the workload live, 179 with the workload
    # throwing on its first line -- which is what a request the driver rejects
    # does, silently, through the workload's own catch. The bound sits between
    # the two with room on both sides; it is there to catch a workload that
    # stopped reaching the solve, not to pin a number that drifts with Julia
    # and with the engine. An earlier bound of 200 sat above *both* values and
    # passed a deliberately broken workload, which is how these numbers were
    # arrived at.
    assert compilations < 80, (
        f"{compilations} runtime compilations from a precompiled bundle: the "
        "workload is not reaching the solve it was written to compile"
    )
