"""
Closed-form exterior validation for the BEAT Engine.

Every other exterior gate in this package is *equivalence-based*: it compares
one BEAT code path against another BEAT code path. `validate_metal_exterior.jl`
compares Metal against BEAT CPU, `validate_metal_symmetry.jl` compares a
symmetry-reduced assembly against the same assembly on a full mesh, and the
`rigid y0 half-space Green function` testset in `tests/runtests.jl` compares a
`:ground` assembly against the sum of two columns of a full two-triangle
assembly.

None of those can see a globally conjugated kernel. If `exp(+im*k*r)` became
`exp(-im*k*r)` everywhere, both sides of every one of those equalities would
conjugate together and every one of them would still pass. Control A below
demonstrates that directly rather than asserting it: it re-runs the runtests
image-equivalence check at `-k` -- an exact global conjugation of the kernel --
and shows it still passes, while the analytic gate here fails hard.

So this script fixes the *absolute* answer, against closed forms:

  Case 1  pulsating sphere, free field. Uniform outward normal velocity on a
          sphere of radius a. Includes the faceting error of the icosphere,
          which is reported separately so the two error sources stay legible.

  Case 2  interior monopole, free field. The Neumann data are those of a point
          monopole at the sphere centre, so the faceted surface is the exact
          boundary of the analytic problem and the geometry error is zero.
          This isolates pure BEM discretisation error.

  Case 3  rigid-image monopole over a plane. The gate Part 2 of the ground-plane
          work needs. A sphere at height h above y = 0, driven with the Neumann
          data of {monopole at s} + {monopole at its mirror image s'}, solved
          with `symmetry_mode=:ground`. That pair is the exact solution of the
          half-space Neumann problem, so this is closed-form, not equivalence.

and then two deliberate controls that MUST fail:

  Control A  a globally conjugated kernel. Realised exactly, without touching
             the vendored `src/`, by solving at `-k`: the single layer is
             exp(im*k*r)/(4*pi*r), so k -> -k conjugates it, and the double
             layer's (im*k - 1/r), the hypersingular kernel and the
             Burton-Miller coupling all conjugate with it. The Neumann data and
             the analytic reference stay at +k, which is what a conjugation
             *defect* would look like: the kernel flips, nothing else does.

  Control B  a pressure-release (sign-flipped) image. The same measured
             half-space field, scored against G_direct - G_image instead of
             G_direct + G_image. This is a reference-side control, not a solver
             mutation -- BEAT's reduction machinery adds every image with weight
             +1 and has no weight to flip -- but it is the defect it names: if
             BEAT's image sign were wrong, BEAT would match this reference and
             miss the rigid one, and the Case 3 assertion would fail.

Both controls are asserted on PHASE. That is a measured decision: a globally
conjugated kernel is very nearly invisible in level. In free field at the
defaults the conjugated solve sits 0.019 dB from the closed form -- inside this
gate's own pass tolerance -- and is out by 90.7 degrees of phase. So an
analytic gate that compared SPL magnitude alone would pass a conjugated kernel
too, and would be only marginally better than the equivalence tests it
replaces. See the comment above the control assertions for why a fixed
relative-error percentage is unsafe as well.

Conventions, read out of the vendored source rather than assumed:

  * `helmholtz_single_layer_kernel` (BeatEngineCore.jl) is
    exp(+im*k*r)/(4*pi*r), which is the exp(-im*omega*t) time convention.
    Every closed form below is written in that convention. Under
    exp(+im*omega*t) each would be its complex conjugate, and Control A is
    exactly the failure that swapping them causes.
  * `BeatEngineDriver.jl` builds its Neumann vector as
    q = im*rho*omega*v_n, so q is dp/dn with the outward normal pointing into
    the fluid. The same rule is used here.

What the correct cases actually measure, on an M1 Max, Float64 CPU backend, at
the defaults (icosphere subdivision 3, 1280 faces, 1000 Hz, ka=1.83, 22
elements per wavelength). Worst level error over 128 observation points at 3 m:

    subdivision   faces   Case 1      Case 2      Case 3      icosphere area
                          pulsating   monopole    ground      deficit
    2              320    0.0882 dB   0.0757 dB   0.0815 dB   -1.888%
    3 (default)   1280    0.0223 dB   0.0188 dB   0.0203 dB   -0.479%
    4             5120    0.0056 dB   0.0047 dB   0.0051 dB   -0.120%

Each refinement divides the error by 3.95 to 3.98, which is the O(h^2) the
discretisation should give, so what this gate measures is discretisation error
and not a fixed offset. The default row sits inside the 0.002 to 0.033 dB band
hornlab-metal-bem measured for its equivalent gate.

The tolerances below are tied to that default row and have a factor of about
two of headroom. They are NOT a property of the code alone: at 2000 Hz the
default mesh is only 11 elements per wavelength and Case 1 misses the phase
tolerance at 0.57 degrees, which is the mesh being too coarse for the frequency
rather than a regression. Raise BLAB_ANALYTIC_SUBDIVISIONS with the frequency.

Run:

    julia --project=hornlab_beat_bem/julia \\
        hornlab_beat_bem/julia/scripts/validate_analytic_exterior.jl

Environment knobs, all optional:

    BLAB_ANALYTIC_SUBDIVISIONS   icosphere subdivision level      (default 3)
    BLAB_ANALYTIC_FREQUENCY_HZ   frequency                        (default 1000)
    BLAB_ANALYTIC_RADIUS_M       sphere radius                    (default 0.1)
    BLAB_ANALYTIC_HEIGHT_M       sphere centre height above y=0   (default 0.5)
    BLAB_ANALYTIC_REGULAR_ORDER  regular triangle rule order      (default 4)
    BLAB_ANALYTIC_SINGULAR_ORDER singular correction order        (default 4)
    BLAB_ANALYTIC_OBSERVATIONS   observation point count          (default 128)
    BLAB_ANALYTIC_RADIUS_OBS_M   observation sphere radius        (default 3.0)
"""

using LinearAlgebra
using Printf
using StaticArrays

include(joinpath(@__DIR__, "..", "src", "BeatEngineCore.jl"))
using .BeatEngineCore

const FloatType = Float64
const SOUND_SPEED = FloatType(343.0)
const AIR_DENSITY = FloatType(1.2041)

env_int(name, default) = parse(Int, get(ENV, name, string(default)))
env_float(name, default) = FloatType(parse(Float64, get(ENV, name, string(default))))

# ---------------------------------------------------------------------------
# Sphere generator
# ---------------------------------------------------------------------------

"""Geodesic icosphere as a `BoundaryMesh`, with outward normals.

Midpoint subdivision of an icosahedron, every vertex projected onto the sphere.
Windings are fixed against the outward radial direction rather than trusted:
the body is star-shaped about its centre, so `dot(normal, centroid - centre)`
has an unambiguous sign, and a face whose sign is negative is reversed. The
result is asserted, so an inverted normal is a failure and not a silent fix.
"""
function icosphere(
    radius::FloatType,
    subdivisions::Int;
    centre::SVector{3,FloatType}=SVector{3,FloatType}(0, 0, 0),
    tag::Int=1,
)
    subdivisions >= 0 || error("Icosphere subdivision level must be >= 0.")
    phi = (1 + sqrt(FloatType(5))) / 2
    vertices = SVector{3,FloatType}[
        SVector{3,FloatType}(-1, phi, 0), SVector{3,FloatType}(1, phi, 0),
        SVector{3,FloatType}(-1, -phi, 0), SVector{3,FloatType}(1, -phi, 0),
        SVector{3,FloatType}(0, -1, phi), SVector{3,FloatType}(0, 1, phi),
        SVector{3,FloatType}(0, -1, -phi), SVector{3,FloatType}(0, 1, -phi),
        SVector{3,FloatType}(phi, 0, -1), SVector{3,FloatType}(phi, 0, 1),
        SVector{3,FloatType}(-phi, 0, -1), SVector{3,FloatType}(-phi, 0, 1),
    ]
    faces = NTuple{3,Int}[
        (1, 12, 6), (1, 6, 2), (1, 2, 8), (1, 8, 11), (1, 11, 12),
        (2, 6, 10), (6, 12, 5), (12, 11, 3), (11, 8, 7), (8, 2, 9),
        (4, 10, 5), (4, 5, 3), (4, 3, 7), (4, 7, 9), (4, 9, 10),
        (5, 10, 6), (3, 5, 12), (7, 3, 11), (9, 7, 8), (10, 9, 2),
    ]

    for _ in 1:subdivisions
        midpoints = Dict{Tuple{Int,Int},Int}()
        function midpoint(a::Int, b::Int)
            key = a < b ? (a, b) : (b, a)
            index = get(midpoints, key, 0)
            index != 0 && return index
            push!(vertices, (vertices[a] + vertices[b]) / 2)
            midpoints[key] = length(vertices)
            return length(vertices)
        end
        next_faces = NTuple{3,Int}[]
        sizehint!(next_faces, 4 * length(faces))
        for (a, b, c) in faces
            ab = midpoint(a, b)
            bc = midpoint(b, c)
            ca = midpoint(c, a)
            push!(next_faces, (a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca))
        end
        faces = next_faces
    end

    vertices = [centre + radius * (vertex / norm(vertex)) for vertex in vertices]

    # Orient every winding outward before the mesh is built, so the normals the
    # BoundaryMesh constructor derives are outward by construction.
    oriented = similar(faces)
    for (index, face) in enumerate(faces)
        v1, v2, v3 = vertices[face[1]], vertices[face[2]], vertices[face[3]]
        outward = (v1 + v2 + v3) / 3 - centre
        oriented[index] = dot(cross(v2 - v1, v3 - v1), outward) >= 0 ?
            face : (face[1], face[3], face[2])
    end

    mesh = BoundaryMesh(vertices, oriented, fill(tag, length(oriented)))
    for index in eachindex(mesh.faces)
        dot(mesh.normals[index], mesh.centroids[index] - centre) > 0 ||
            error("Icosphere face $(index) has an inward normal.")
    end
    return mesh
end

# ---------------------------------------------------------------------------
# Closed forms, in the exp(-im*omega*t) convention
# ---------------------------------------------------------------------------

"""Pressure of a point monopole of volume velocity `strength` (m^3/s).

    p(x) = -im * rho * omega * Q * exp(im*k*r) / (4*pi*r)

which is the small-ka limit of `pulsating_sphere_pressure` below, and uses the
same exp(+im*k*r) Green function the vendored kernels use.
"""
function monopole_pressure(
    x::SVector{3,FloatType},
    source::SVector{3,FloatType},
    k::FloatType,
    omega::FloatType,
    strength::Complex{FloatType},
)
    radius = norm(x - source)
    return -im * AIR_DENSITY * omega * strength * exp(im * k * radius) / (4 * FloatType(pi) * radius)
end

"""Gradient of `monopole_pressure`, used to build exact Neumann data.

    grad p = p * (im*k - 1/r) * (x - s)/r
"""
function monopole_gradient(
    x::SVector{3,FloatType},
    source::SVector{3,FloatType},
    k::FloatType,
    omega::FloatType,
    strength::Complex{FloatType},
)
    offset = x - source
    radius = norm(offset)
    value = monopole_pressure(x, source, k, omega, strength)
    return value * (im * k - 1 / radius) * (offset / radius)
end

"""Normal derivative of a complex gradient.

`LinearAlgebra.dot` conjugates its first argument, so `dot(gradient, normal)`
silently returns the conjugate of dp/dn -- which is a global conjugation of the
Neumann data, and reads as a plausible answer because it leaves |p| untouched.
Sum the products explicitly instead.
"""
normal_derivative(gradient, normal) = sum(gradient .* normal)

"""Radiated pressure of a sphere of radius `a` breathing with normal velocity `v`.

Momentum in the exp(-im*omega*t) convention is u = grad(p)/(im*omega*rho), so
for p = A*exp(im*k*r)/r the surface condition u_r(a) = v gives

    p(r) = rho*c*v * (a/r) * (im*k*a)/(im*k*a - 1) * exp(im*k*(r - a))

whose ka -> infinity limit is the plane-wave result p = rho*c*v, and whose
ka -> 0 limit is a monopole of volume velocity 4*pi*a^2*v.
"""
function pulsating_sphere_pressure(
    radius::FloatType,
    sphere_radius::FloatType,
    k::FloatType,
    velocity::Complex{FloatType},
)
    ka = im * k * sphere_radius
    return AIR_DENSITY * SOUND_SPEED * velocity * (sphere_radius / radius) *
        (ka / (ka - 1)) * exp(im * k * (radius - sphere_radius))
end

# ---------------------------------------------------------------------------
# Solve harness
# ---------------------------------------------------------------------------

"""One BEAT CPU Burton-Miller Neumann solve, plus the exterior field.

`wavenumber` is passed through to assembly, to the Burton-Miller coupling and
to field evaluation, which is what lets Control A conjugate the whole kernel by
passing `-k` while the Neumann data stay at `+k`.
"""
function beat_exterior_field(
    mesh::BoundaryMesh{FloatType},
    q_neumann::Vector{Complex{FloatType}},
    wavenumber::FloatType,
    eval_points::Vector{SVector{3,FloatType}};
    symmetry_mode::Symbol,
    regular_order::Int,
    singular_order::Int,
)
    p1 = build_p1_space(mesh)
    dp0 = build_dp0_space(mesh)
    rule = triangle_rule(FloatType, regular_order)
    singular_cache = build_singular_correction_cache(mesh, singular_order)
    operators = assemble_regular_galerkin_operators(
        mesh, p1, dp0, wavenumber, rule;
        backend=:cpu,
        skip_singular=false,
        singular_order=singular_order,
        singular_cache=singular_cache,
        symmetry_mode=symmetry_mode,
    )
    identity_p1_p1 = assemble_l2_identity_matrix(
        mesh, p1, dp0, rule, :p1, :p1; symmetry_mode=symmetry_mode,
    )
    identity_p1_dp0 = assemble_l2_identity_matrix(
        mesh, p1, dp0, rule, :p1, :dp0; symmetry_mode=symmetry_mode,
    )
    pressure = solve_burton_miller_neumann_cpu(
        operators, identity_p1_p1, identity_p1_dp0, q_neumann, wavenumber,
    )
    field_cache = build_field_evaluation_cache(mesh, rule; symmetry_mode=symmetry_mode)
    field = evaluate_galerkin_field_cpu(
        eval_points, mesh, pressure, q_neumann, wavenumber, field_cache,
    )
    return (pressure=pressure, field=field)
end

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

relative_error(actual, reference) = norm(actual .- reference) / norm(reference)
relative_error_percent(actual, reference) = 100 * relative_error(actual, reference)

"""Worst single-point relative error, as a percentage.

The L2 figure is normalised by the whole reference vector, so it saturates near
sqrt(2) (141%) once two fields are simply unrelated and cannot distinguish
"unrelated" from "unrelated and locally enormous". The pointwise worst does: a
pressure-release reference has near-nulls where the rigid answer does not, and
this is the metric that shows it.
"""
function relative_error_pointwise_percent(actual, reference)
    return 100 * maximum(abs(a - b) / abs(b) for (a, b) in zip(actual, reference))
end

"""Level error in dB, |20*log10(|actual|/|reference|)|, worst and RMS."""
function level_error_db(actual, reference)
    errors = [abs(20 * log10(abs(a) / abs(b))) for (a, b) in zip(actual, reference)]
    return (worst=maximum(errors), rms=sqrt(sum(abs2, errors) / length(errors)))
end

"""Phase error in degrees, wrapped to (-180, 180], worst and RMS."""
function phase_error_deg(actual, reference)
    errors = FloatType[]
    for (a, b) in zip(actual, reference)
        delta = rad2deg(angle(a / b))
        push!(errors, abs(delta))
    end
    return (worst=maximum(errors), rms=sqrt(sum(abs2, errors) / length(errors)))
end

function report(label, actual, reference)
    level = level_error_db(actual, reference)
    phase = phase_error_deg(actual, reference)
    percent = relative_error_percent(actual, reference)
    pointwise = relative_error_pointwise_percent(actual, reference)
    @printf(
        "%-34s level_db_worst=%.6f level_db_rms=%.6f phase_deg_worst=%.4f relative_l2=%.3f%% relative_worst=%.1f%%\n",
        label, level.worst, level.rms, phase.worst, percent, pointwise,
    )
    flush(stdout)
    return (level=level, phase=phase, percent=percent, pointwise=pointwise)
end

# ---------------------------------------------------------------------------
# Control A support: the image-equivalence check, at +k and at -k
# ---------------------------------------------------------------------------

"""The `rigid y0 half-space Green function` equivalence check from
`tests/runtests.jl`, reduced to a single worst-case relative error.

Assembling the same two triangles as a `:ground` half space and as a full
two-triangle mesh, the half-space operator columns must equal the sum of the
direct and mirrored columns of the full operator. Run at `-k` this equality
still holds exactly, because both sides conjugate together -- which is the
whole reason this script exists.
"""
function image_equivalence_error(wavenumber::FloatType, regular_order::Int, singular_order::Int)
    direct_vertices = SVector{3,FloatType}[
        SVector{3,FloatType}(0, 0.2, 0),
        SVector{3,FloatType}(0.04, 0.2, 0),
        SVector{3,FloatType}(0, 0.2, 0.04),
    ]
    direct_mesh = BoundaryMesh(direct_vertices, [(1, 2, 3)], [1])
    full_vertices = vcat(
        direct_vertices,
        [SVector{3,FloatType}(p[1], -p[2], p[3]) for p in direct_vertices],
    )
    full_mesh = BoundaryMesh(full_vertices, [(1, 2, 3), (4, 6, 5)], [1, 1])

    rule = triangle_rule(FloatType, regular_order)
    direct_p1, direct_dp0 = build_p1_space(direct_mesh), build_dp0_space(direct_mesh)
    full_p1, full_dp0 = build_p1_space(full_mesh), build_dp0_space(full_mesh)

    half = assemble_regular_galerkin_operators(
        direct_mesh, direct_p1, direct_dp0, wavenumber, rule;
        backend=:cpu, skip_singular=false, singular_order=singular_order,
        singular_cache=build_singular_correction_cache(direct_mesh, singular_order),
        symmetry_mode=:ground,
    )
    full = assemble_regular_galerkin_operators(
        full_mesh, full_p1, full_dp0, wavenumber, rule;
        backend=:cpu, skip_singular=false, singular_order=singular_order,
        singular_cache=build_singular_correction_cache(full_mesh, singular_order),
    )

    return maximum((
        relative_error(half.single_layer[:, 1], full.single_layer[1:3, 1] + full.single_layer[1:3, 2]),
        relative_error(half.double_layer, full.double_layer[1:3, 1:3] + full.double_layer[1:3, 4:6]),
        relative_error(
            half.adjoint_double_layer[:, 1],
            full.adjoint_double_layer[1:3, 1] + full.adjoint_double_layer[1:3, 2],
        ),
        relative_error(half.hypersingular, full.hypersingular[1:3, 1:3] + full.hypersingular[1:3, 4:6]),
    ))
end

# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------

function validate_analytic_exterior()
    subdivisions = env_int("BLAB_ANALYTIC_SUBDIVISIONS", 3)
    frequency = env_float("BLAB_ANALYTIC_FREQUENCY_HZ", 1000.0)
    sphere_radius = env_float("BLAB_ANALYTIC_RADIUS_M", 0.1)
    height = env_float("BLAB_ANALYTIC_HEIGHT_M", 0.5)
    regular_order = env_int("BLAB_ANALYTIC_REGULAR_ORDER", 4)
    singular_order = env_int("BLAB_ANALYTIC_SINGULAR_ORDER", 4)
    observation_count = env_int("BLAB_ANALYTIC_OBSERVATIONS", 128)
    observation_radius = env_float("BLAB_ANALYTIC_RADIUS_OBS_M", 3.0)

    omega = 2 * FloatType(pi) * frequency
    k = omega / SOUND_SPEED
    velocity = Complex{FloatType}(1, 0)

    height > sphere_radius ||
        error("Sphere at height $(height) m with radius $(sphere_radius) m straddles y = 0.")

    free_mesh = icosphere(sphere_radius, subdivisions)
    exact_area = 4 * FloatType(pi) * sphere_radius^2
    facet_area = sum(free_mesh.areas)
    area_deficit = facet_area / exact_area - 1

    println("backend=cpu precision=$(FloatType)")
    @printf(
        "sphere radius=%.4f m faces=%d p1_dofs=%d subdivisions=%d\n",
        sphere_radius, length(free_mesh.faces),
        build_p1_space(free_mesh).global_dof_count, subdivisions,
    )
    @printf(
        "frequency=%.1f Hz k=%.4f 1/m ka=%.4f wavelength=%.4f m\n",
        frequency, k, k * sphere_radius, SOUND_SPEED / frequency,
    )
    @printf("quadrature regular_order=%d singular_order=%d\n", regular_order, singular_order)
    @printf(
        "elements per wavelength=%.1f (median edge %.4f m)\n",
        (SOUND_SPEED / frequency) / sqrt(facet_area / length(free_mesh.faces)),
        sqrt(facet_area / length(free_mesh.faces)),
    )
    @printf(
        "icosphere area deficit=%.6f%% (%.6f dB on a monopole's volume velocity)\n",
        100 * area_deficit, abs(20 * log10(1 + area_deficit)),
    )
    flush(stdout)

    # -- Case 1: pulsating sphere, free field ------------------------------
    eval_points = fibonacci_sphere(observation_count, observation_radius)
    q_uniform = fill(Complex{FloatType}(0, AIR_DENSITY * omega) * velocity, length(free_mesh.faces))
    pulsating = beat_exterior_field(
        free_mesh, q_uniform, k, eval_points;
        symmetry_mode=:off, regular_order=regular_order, singular_order=singular_order,
    )
    pulsating_reference = [
        pulsating_sphere_pressure(norm(point), sphere_radius, k, velocity)
        for point in eval_points
    ]

    println()
    println("Case 1  pulsating sphere, free field, closed form")
    case1 = report("  beat vs closed form", pulsating.field, pulsating_reference)

    # -- Case 2: interior monopole, free field, geometry-exact -------------
    centre = SVector{3,FloatType}(0, 0, 0)
    strength = Complex{FloatType}(4 * FloatType(pi) * sphere_radius^2, 0)
    q_monopole = [
        normal_derivative(monopole_gradient(free_mesh.centroids[i], centre, k, omega, strength), free_mesh.normals[i])
        for i in eachindex(free_mesh.faces)
    ]
    monopole = beat_exterior_field(
        free_mesh, q_monopole, k, eval_points;
        symmetry_mode=:off, regular_order=regular_order, singular_order=singular_order,
    )
    monopole_reference = [monopole_pressure(point, centre, k, omega, strength) for point in eval_points]

    println()
    println("Case 2  interior monopole, free field, geometry-exact")
    case2 = report("  beat vs closed form", monopole.field, monopole_reference)

    # -- Case 3: rigid-image monopole over a plane -------------------------
    ground_centre = SVector{3,FloatType}(0, height, 0)
    image_centre = SVector{3,FloatType}(0, -height, 0)
    ground_mesh = icosphere(sphere_radius, subdivisions; centre=ground_centre)

    minimum_y = minimum(vertex[2] for vertex in ground_mesh.vertices)
    @printf(
        "\nsphere centre height=%.4f m minimum vertex y=%.6f m image separation=%.4f m (%.2f wavelengths)\n",
        height, minimum_y, 2 * height, 2 * height * frequency / SOUND_SPEED,
    )

    rigid_pair(x) =
        monopole_pressure(x, ground_centre, k, omega, strength) +
        monopole_pressure(x, image_centre, k, omega, strength)
    soft_pair(x) =
        monopole_pressure(x, ground_centre, k, omega, strength) -
        monopole_pressure(x, image_centre, k, omega, strength)
    rigid_pair_gradient(x) =
        monopole_gradient(x, ground_centre, k, omega, strength) +
        monopole_gradient(x, image_centre, k, omega, strength)

    q_ground = [
        normal_derivative(rigid_pair_gradient(ground_mesh.centroids[i]), ground_mesh.normals[i])
        for i in eachindex(ground_mesh.faces)
    ]

    # Observation points in the fluid half space only, and clear of the body.
    half_space_points = SVector{3,FloatType}[
        point for point in fibonacci_sphere(2 * observation_count, observation_radius)
        if point[2] > 0 && norm(point - ground_centre) > 4 * sphere_radius
    ]
    @printf("half-space observation points=%d at radius %.2f m\n", length(half_space_points), observation_radius)
    flush(stdout)

    ground = beat_exterior_field(
        ground_mesh, q_ground, k, half_space_points;
        symmetry_mode=:ground, regular_order=regular_order, singular_order=singular_order,
    )
    ground_reference = [rigid_pair(point) for point in half_space_points]
    soft_reference = [soft_pair(point) for point in half_space_points]

    println()
    println("Case 3  rigid-image monopole over y = 0, symmetry_mode=:ground")
    case3 = report("  beat vs closed form", ground.field, ground_reference)

    # -- Control A: globally conjugated kernel -----------------------------
    println()
    println("Control A  globally conjugated kernel (k -> -k), MUST FAIL")
    conjugated_free = beat_exterior_field(
        free_mesh, q_monopole, -k, eval_points;
        symmetry_mode=:off, regular_order=regular_order, singular_order=singular_order,
    )
    control_a_free = report("  free field vs closed form", conjugated_free.field, monopole_reference)
    conjugated_ground = beat_exterior_field(
        ground_mesh, q_ground, -k, half_space_points;
        symmetry_mode=:ground, regular_order=regular_order, singular_order=singular_order,
    )
    control_a_ground = report("  half space vs closed form", conjugated_ground.field, ground_reference)

    equivalence_forward = image_equivalence_error(k, regular_order, singular_order)
    equivalence_conjugated = image_equivalence_error(-k, regular_order, singular_order)
    @printf(
        "  image-equivalence check: +k relative=%.3e  -k relative=%.3e  <- both pass, this is the blind spot\n",
        equivalence_forward, equivalence_conjugated,
    )
    flush(stdout)

    # -- Control B: pressure-release image ---------------------------------
    println()
    println("Control B  pressure-release (sign-flipped) image, MUST FAIL")
    control_b = report("  half space vs soft reference", ground.field, soft_reference)

    # -- Assertions --------------------------------------------------------
    #
    # The correct-case bounds are the measured discretisation error with
    # headroom, not aspirations: at the defaults they are met with a factor of
    # roughly two to spare. Do not widen one to make a run pass -- a real
    # regression in this gate does not land just above a bound, it lands where
    # the controls are, three to four orders of magnitude away.
    println()
    tolerance_level_db = FloatType(0.05)
    tolerance_phase_deg = FloatType(0.5)
    control_margin = FloatType(10)

    @printf("tolerance level_db_worst<=%.4f phase_deg_worst<=%.4f\n", tolerance_level_db, tolerance_phase_deg)
    @printf(
        "controls must miss the phase tolerance by >=%.0fx, i.e. phase_deg_worst>=%.2f\n",
        control_margin, control_margin * tolerance_phase_deg,
    )

    for (label, result) in (
        ("Case 1 pulsating sphere", case1),
        ("Case 2 interior monopole", case2),
        ("Case 3 rigid-image monopole", case3),
    )
        result.level.worst <= tolerance_level_db ||
            error("$(label): level error $(result.level.worst) dB exceeds $(tolerance_level_db) dB.")
        result.phase.worst <= tolerance_phase_deg ||
            error("$(label): phase error $(result.phase.worst) deg exceeds $(tolerance_phase_deg) deg.")
    end

    # The controls are asserted on PHASE, not on level and not on a percentage,
    # and that is a measured decision rather than a stylistic one.
    #
    # A globally conjugated kernel is very nearly invisible in level. Measured
    # free field at the defaults: the conjugated solve is 0.019 dB from the
    # closed form -- inside this gate's own pass tolerance -- while its phase is
    # out by 90.7 degrees. A gate that compared SPL magnitude alone would report
    # the conjugated kernel as correct.
    #
    # The relative-error percentage is also unsafe as an assertion, because its
    # size depends on ka rather than on whether the defect is present. Measured
    # free field: 21.7% at 400 Hz (ka=0.73), 142.5% at 1000 Hz (ka=1.83), 136.2%
    # at 2000 Hz (ka=3.66). A fixed percentage floor calibrated at 1000 Hz makes
    # the control silently stop failing at 400 Hz -- a control that no longer
    # controls. Phase holds up across all three: 12.5, 90.7 and 85.7 degrees,
    # against a correct-case 0.04 to 0.57 degrees.
    for (label, result) in (
        ("Control A free field", control_a_free),
        ("Control A half space", control_a_ground),
        ("Control B pressure-release image", control_b),
    )
        result.phase.worst >= control_margin * tolerance_phase_deg ||
            error("$(label) did not fail: phase error $(result.phase.worst) deg is below " *
                  "$(control_margin * tolerance_phase_deg) deg. " *
                  "The gate cannot see the defect it exists to catch.")
    end

    equivalence_conjugated <= 10 * max(equivalence_forward, eps(FloatType)) ||
        error("The image-equivalence check no longer passes at -k; Control A's premise needs re-deriving.")

    println("ANALYTIC_EXTERIOR_VALIDATION_OK")
end

validate_analytic_exterior()
