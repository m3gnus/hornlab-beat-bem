using LinearAlgebra
using StaticArrays

include(joinpath(@__DIR__, "..", "src", "BeatEngineCore.jl"))
using .BeatEngineCore

relative_error(actual, reference) = norm(actual - reference) / max(norm(reference), eps(real(eltype(reference))))

"""Lift every vertex by `offset` along Y, for the rigid half-space fixtures.

`:ground` is the one mode whose mesh is not a fundamental domain, so it is the
one mode with no fixture on disk: `sample_half.msh` straddles Y=0 and would be
half inside the floor. Lifting it is the whole adaptation needed, and it also
makes the fixture the shape the mode is actually for -- a body standing clear
of a boundary rather than a mesh cut by a mirror.
"""
function lift_above_ground(mesh::BoundaryMesh{T}, offset::T) where {T<:AbstractFloat}
    vertices = [SVector{3,T}(v[1], v[2] + offset, v[3]) for v in mesh.vertices]
    return BoundaryMesh(vertices, mesh.faces, mesh.physical_tags)
end

function validate_fixture(mesh_name::String, symmetry_mode::Symbol; ground_lift::Float32=0.0f0)
    mesh_path = joinpath(@__DIR__, "..", "test_meshes", mesh_name)
    mesh = load_gmsh22_with_tags(mesh_path, Float32(0.001))
    if symmetry_mode == :ground
        # No snapping and no fundamental-domain check: `:ground` has no active
        # symmetry axis, so both are no-ops for it by construction. What the
        # mode does require is that the body lie wholly at Y >= 0, which the
        # driver enforces on the real path (`validate_ground_plane_domain!`)
        # and which is asserted here so a fixture cannot drift below the plane
        # and quietly turn this into a different test.
        mesh = lift_above_ground(mesh, ground_lift)
        minimum_y = minimum(v[2] for v in mesh.vertices)
        minimum_y >= -1.0f-6 || error(
            "Ground fixture $(mesh_name) reaches Y=$(minimum_y) m; the rigid " *
            "half space requires the whole mesh at Y >= 0."
        )
    else
        mesh = snap_symmetry_planes(mesh, symmetry_mode)
        validate_symmetry_fundamental_domain!(mesh, symmetry_mode)
    end
    p1 = build_p1_space(mesh)
    dp0 = build_dp0_space(mesh)
    regular_order = parse(Int, get(ENV, "BLAB_VALIDATE_REGULAR_ORDER", "1"))
    singular_order = parse(Int, get(ENV, "BLAB_VALIDATE_SINGULAR_ORDER", "2"))
    rule = triangle_rule(Float32, regular_order)
    k = Float32(2pi) * 500.0f0 / 343.0f0
    singular_cache = build_singular_correction_cache(mesh, singular_order)

    cpu_cache = build_beat_cpu_assembly_cache(
        mesh, p1, dp0, rule;
        singular_order=singular_order,
        symmetry_mode=symmetry_mode,
    )
    cpu_operators = assemble_regular_galerkin_operators(
        mesh, p1, dp0, k, rule;
        skip_singular=false,
        singular_order=singular_order,
        backend=:cpu,
        singular_cache=singular_cache,
        cpu_cache=cpu_cache,
        symmetry_mode=symmetry_mode,
    )
    metal_cache = build_metal_regular_assembly_cache(
        mesh, p1, dp0, rule;
        singular_order=singular_order,
        symmetry_mode=symmetry_mode,
    )
    for singular_mode in ("host", "native")
        ENV["BLAB_METAL_SINGULAR_MODE"] = singular_mode
        metal_operators = assemble_regular_galerkin_operators(
            mesh, p1, dp0, k, rule;
            skip_singular=false,
            singular_order=singular_order,
            backend=:metal,
            device_cache=metal_cache,
            singular_cache=singular_cache,
            metal_assembly_mode=:native,
            symmetry_mode=symmetry_mode,
        )
        operator_errors = Dict(
            "single_layer" => relative_error(Array(metal_operators.single_layer), cpu_operators.single_layer),
            "double_layer" => relative_error(Array(metal_operators.double_layer), cpu_operators.double_layer),
            "adjoint_double_layer" => relative_error(Array(metal_operators.adjoint_double_layer), cpu_operators.adjoint_double_layer),
            "hypersingular" => relative_error(Array(metal_operators.hypersingular), cpu_operators.hypersingular),
        )

        identity_p1_p1 = assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :p1; symmetry_mode=symmetry_mode)
        identity_p1_dp0 = assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :dp0; symmetry_mode=symmetry_mode)
        q_neumann = zeros(ComplexF32, length(mesh.faces))
        driven_tag = first(mesh.physical_tags)
        q_neumann[mesh.physical_tags .== driven_tag] .= ComplexF32(0, 1)
        cpu_pressure = solve_burton_miller_neumann(
            cpu_operators, identity_p1_p1, identity_p1_dp0, q_neumann, k,
        )
        identity_cache = build_metal_burton_miller_identity_cache(identity_p1_p1, identity_p1_dp0, Float32)
        metal_pressure = solve_burton_miller_neumann(metal_operators, identity_cache, q_neumann, k)
        pressure_error = relative_error(metal_pressure, cpu_pressure)

        eval_points = fibonacci_sphere(16, Float32(2))
        cpu_field_cache = build_field_evaluation_cache(mesh, rule; symmetry_mode=symmetry_mode)
        cpu_field = evaluate_galerkin_field_cpu(eval_points, mesh, cpu_pressure, q_neumann, k, cpu_field_cache)
        metal_field_cache = build_metal_field_evaluation_cache(cpu_field_cache)
        metal_field = evaluate_galerkin_field_metal(eval_points, mesh, metal_pressure, q_neumann, k, metal_field_cache)
        field_error = relative_error(metal_field, cpu_field)

        println((
            mesh=mesh_name,
            symmetry=symmetry_mode,
            min_y=minimum(v[2] for v in mesh.vertices),
            singular_mode=singular_mode,
            faces=length(mesh.faces),
            image_singular_pairs=metal_operators.image_singular_pairs,
            operator_errors=operator_errors,
            pressure_error=pressure_error,
            field_error=field_error,
        ))
        flush(stdout)
        maximum(values(operator_errors)) <= 2.0f-6 || error("Metal symmetry operators ($(singular_mode)) differ from BEAT CPU.")
        pressure_error <= 5.0f-3 || error("Metal symmetry pressure differs from BEAT CPU.")
        field_error <= 5.0f-3 || error("Metal symmetry field differs from BEAT CPU.")

        release_metal_field_evaluation_cache!(metal_field_cache)
        release_metal_burton_miller_identity_cache!(identity_cache)
        release_operator_storage!(metal_operators)
    end
    delete!(ENV, "BLAB_METAL_SINGULAR_MODE")
    release_metal_regular_assembly_cache!(metal_cache)
end

function validate_metal_symmetry()
    metal = BeatEngineCore.METAL_MODULE
    metal === nothing && error("Metal.jl did not load. Run this script with the julia_metal project.")
    metal.functional() || error("Metal.functional() is false.")
    println("device=$(metal.device().name)")
    validate_fixture("sample_half.msh", :x)
    validate_fixture("sample_quarter.msh", :xy)
    # `:ground` in both of its geometries. Lifted clear of the plane is the
    # ordinary case; resting on it is the one that matters, because a body
    # touching Y=0 makes real and image elements coincident and adjacent, and
    # that is the image singular-correction path -- the part of the assembly
    # most likely to differ between Metal and BEAT CPU.
    validate_fixture("sample_half.msh", :ground; ground_lift=0.15f0)
    validate_fixture("sample_quarter.msh", :ground)
    println("METAL_SYMMETRY_VALIDATION_OK")
end

validate_metal_symmetry()
