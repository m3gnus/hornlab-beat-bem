# Correctness check for the near-singular correction wired up in solver.jl.
#
# The correction re-integrates a pair with a high-order tensor rule and then
# subtracts that pair's regular-rule contribution. So if *every* non-adjacent
# pair is corrected, the assembled operators must stop depending on which
# regular rule was used -- singular pairs go through Duffy either way. A wrong
# sign, a wrong Jacobian, a mismatched pair list or a missed transform all
# break that identity, and none of them are visible in a smoke test that only
# checks the solve runs.
#
# Run from tests/test_near_correction.py, which passes the solver directory.

using LinearAlgebra, Printf, Statistics, StaticArrays

const SOLVER_DIR = ARGS[1]
include(joinpath(SOLVER_DIR, "src", "BeatEngineCore.jl"))
using .BeatEngineCore

get_value(raw, key::String, default=nothing) = haskey(raw, key) ? raw[key] : default
# The driver moved out of solver.jl into BeatEngineDriver.jl when the engine
# was re-vendored as precompilable bundles; solver.jl is now only a loader.
let source = read(joinpath(SOLVER_DIR, "BeatEngineDriver.jl"), String)
    first_line = findfirst("function near_correction_order_for_ratio", source)[1]
    last_line = findfirst("\nfunction mesh_inputs_from_config", source)[1]
    include_string(Main, source[first_line:last_line])
end

const T = Float64

"""An octahedron refined twice: 128 faces, plenty of non-adjacent pairs.

`domain` selects the seed faces: `:full` the whole octahedron, `:half` the
octants with x >= 0, `:quarter` those with x >= 0 and y >= 0 -- legal
fundamental domains for `:off`, `:x` and `:xy`. Mirroring a *whole* sphere about x = 0 maps the mesh
onto itself, so every image pair would be coincident and the assembly would be
integrating a 1/r kernel at zero separation -- a fixture that fails for reasons
that have nothing to do with the correction under test.
"""
function unit_sphere_mesh(domain::Symbol=:full)
    vertices = SVector{3,T}[
        SVector{3,T}(1, 0, 0), SVector{3,T}(-1, 0, 0), SVector{3,T}(0, 1, 0),
        SVector{3,T}(0, -1, 0), SVector{3,T}(0, 0, 1), SVector{3,T}(0, 0, -1),
    ]
    faces =
        domain == :quarter ? NTuple{3,Int}[(1, 3, 5), (3, 1, 6)] :
        domain == :half ? NTuple{3,Int}[(1, 3, 5), (4, 1, 5), (3, 1, 6), (1, 4, 6)] :
        NTuple{3,Int}[
            (1, 3, 5), (3, 2, 5), (2, 4, 5), (4, 1, 5),
            (3, 1, 6), (2, 3, 6), (4, 2, 6), (1, 4, 6),
        ]
    for _ in 1:2
        midpoints = Dict{Tuple{Int,Int},Int}()
        refined = NTuple{3,Int}[]
        midpoint(a, b) = get!(midpoints, minmax(a, b)) do
            push!(vertices, normalize(vertices[a] + vertices[b]))
            length(vertices)
        end
        for (a, b, c) in faces
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            append!(refined, [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)])
        end
        faces = refined
    end
    # Drop vertices no face references, or the -x octahedron pole would keep a
    # half domain from validating as one.
    used = sort(unique(Iterators.flatten(faces)))
    renumbered = Dict(old_index => new_index for (new_index, old_index) in enumerate(used))
    return BoundaryMesh(
        vertices[used],
        [(renumbered[a], renumbered[b], renumbered[c]) for (a, b, c) in faces],
        fill(1, length(faces)),
    )
end

const MESHES = Dict(
    :off => unit_sphere_mesh(),
    :x => snap_symmetry_planes(unit_sphere_mesh(:half), :x),
    :xy => snap_symmetry_planes(unit_sphere_mesh(:quarter), :xy),
)
const K = T(3.0)

function corrections(mesh, symmetry_mode::Symbol)
    selection = near_correction_selection(
        Dict{String,Any}(
            "near_correction_enabled" => true,
            "near_correction_cutoff" => 1.0e9,
            "near_correction_order" => 8,
        ),
        mesh,
        symmetry_mode,
    )
    selection === nothing && error("full-mesh selection returned no pairs")
    identity_cache = build_near_correction_cache(mesh, selection.identity_pairs, selection.top_order)
    image_caches = [
        build_near_correction_cache(mesh, pairs, selection.top_order; trial_transform=transform)
        for (transform, pairs) in selection.image_selections
    ]
    return identity_cache, image_caches, selection
end

function assemble(mesh, order::Int, identity_cache, image_caches, symmetry_mode::Symbol)
    return assemble_regular_galerkin_operators(
        mesh, build_p1_space(mesh), build_dp0_space(mesh), K, triangle_rule(T, order);
        skip_singular=false, singular_order=4, backend=:cpu,
        singular_cache=build_singular_correction_cache(mesh, 4),
        near_correction_cache=identity_cache, image_near_correction_cache=image_caches,
        symmetry_mode=symmetry_mode,
    )
end

relative(x, y) = norm(x - y) / norm(y)
const OPERATORS = (:single_layer, :double_layer, :adjoint_double_layer, :hypersingular)
const FAILURES = Ref(0)

# :off exercises the self-domain cache; :x adds a mirror-image cache, which
# catches a transform applied to the wrong side of the pair; :xy needs three of
# them at once, which catches an assembly that only applies the first.
for symmetry_mode in (:off, :x, :xy)
    mesh = MESHES[symmetry_mode]
    # Fail loudly if the fixture is not a legal fundamental domain.
    validate_symmetry_fundamental_domain!(mesh, symmetry_mode)
    identity_cache, image_caches, selection = corrections(mesh, symmetry_mode)
    @printf(
        "symmetry %-3s: %d face(s), %d self-domain pair(s), %d image pair(s) over %d transform(s)\n",
        symmetry_mode, length(mesh.faces), identity_cache.pair_count,
        sum(cache.pair_count for cache in image_caches; init=0), length(image_caches),
    )
    expected_transforms = length(symmetry_image_transforms(symmetry_mode))
    if length(image_caches) != expected_transforms
        FAILURES[] += 1
        println("  FAIL: $(symmetry_mode) needs $(expected_transforms) image-near cache(s), " *
                "selection produced $(length(image_caches))")
    end

    coarse = assemble(mesh, 1, identity_cache, image_caches, symmetry_mode)
    fine = assemble(mesh, 4, identity_cache, image_caches, symmetry_mode)
    raw_coarse = assemble(mesh, 1, nothing, nothing, symmetry_mode)

    for name in OPERATORS
        corrected = relative(getfield(coarse, name), getfield(fine, name))
        uncorrected = relative(getfield(raw_coarse, name), getfield(fine, name))
        ok = corrected < 1.0e-12
        ok || (FAILURES[] += 1)
        @printf(
            "  %-22s corrected %.3e   uncorrected %.3e   %s\n",
            name, corrected, uncorrected, ok ? "ok" : "FAIL",
        )
        # Guard against the identity holding for a boring reason: the regular
        # rule has to matter in the first place.
        if uncorrected < 1.0e-6
            FAILURES[] += 1
            @printf("  FAIL: %s barely depends on the regular rule; test proves nothing\n", name)
        end
    end
end

if FAILURES[] > 0
    println("FAILURES: $(FAILURES[])")
    exit(1)
end
println("PASS")
