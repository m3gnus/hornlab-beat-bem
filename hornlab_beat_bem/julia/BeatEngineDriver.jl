"""
The BEAT worker's driver: request parsing, the solve orchestration and the
JSON-lines event protocol the Python side speaks.

This is the body of `solver.jl`, moved out of the script and into a file a
package can `include`. That is the whole point of the move. Compiling the
worker's entry path -- `worker_loop`, and everything type inference reaches
from it -- was measured as the single largest item in a cold start, larger
than loading the engine itself, and a script cannot be precompiled: Julia's
pkgimage cache and PackageCompiler both only see code that lives in a package.
Included from a bundle in `julia_engine/`, this code is compiled once at
precompile time and mapped in thereafter.

`solver.jl` still includes this file directly when no bundle is available, so
an un-instantiated checkout behaves exactly as it used to, at the old cost.
"""

using JSON
using LinearAlgebra
using Printf
using Statistics
using StaticArrays

function emit_event(event_type::String; kwargs...)
    payload = Dict{String,Any}("type" => event_type)
    for (key, value) in kwargs
        payload[String(key)] = value
    end
    println(JSON.json(payload))
    flush(stdout)
end

function fail!(message::AbstractString)
    emit_event("failed"; error=String(message))
    exit(1)
end

function parse_args(args)
    request_path = nothing
    worker_mode = false
    i = 1
    while i <= length(args)
        if args[i] == "--request" && i < length(args)
            request_path = args[i + 1]
            i += 2
        elseif args[i] == "--worker"
            worker_mode = true
            i += 1
        else
            fail!("Unknown argument: $(args[i])")
        end
    end
    return request_path, worker_mode
end

get_value(raw, key::String, default=nothing) = haskey(raw, key) ? raw[key] : default

function as_float_vector(raw)
    return [Float64(value) for value in raw]
end

function validate_crossover_config(owner_name::String, crossover)
    crossover === nothing && return
    crossover_type = lowercase(String(get_value(crossover, "type", "none")))
    crossover_type in ("none", "lowpass", "highpass") || error("$(owner_name) crossover type must be none, lowpass, or highpass.")
    crossover_type == "none" && return

    frequency = get_value(crossover, "frequency_hz", nothing)
    frequency === nothing && error("$(owner_name) crossover frequency_hz must be > 0.")
    Float64(frequency) > 0.0 || error("$(owner_name) crossover frequency_hz must be > 0.")

    filter_name = lowercase(String(get_value(crossover, "filter", "butterworth")))
    filter_name in ("butterworth", "linkwitz_riley") || error("$(owner_name) crossover filter must be butterworth or linkwitz_riley.")

    order = Int(get_value(crossover, "order", 1))
    order in (1, 2, 4, 6) || error("$(owner_name) crossover order must be 1, 2, 4, or 6.")
    if filter_name == "linkwitz_riley" && !(order in (2, 4, 6))
        error("$(owner_name) Linkwitz-Riley order must be 2, 4, or 6.")
    end
end

# `symmetry` selects one image-transform set, and the solver carries exactly
# one. Three of its four values are mirror symmetries of a REDUCED mesh, whose
# images are part of the real radiator; `ground` is a rigid half space, whose
# single image is fictitious and whose mesh is the WHOLE body. They are
# different contracts sharing one field, so everything downstream that asks
# "how many copies of the radiator are there?" -- `impedance_for_radiators`
# most of all -- has to distinguish them rather than counting transforms.
#
# `ground` mirrors across Y=0 only (`rigid_ground_transform`), so a wall on
# another axis is not expressible here; the Python wrapper refuses those by
# axis rather than substituting one.
function symmetry_mode_from_config(config)
    mode = lowercase(strip(String(get_value(config, "symmetry", "off"))))
    mode in ("off", "x", "xy", "ground") ||
        error("Unsupported symmetry mode: $(mode). Expected off, x, xy, or ground.")
    return mode
end

# The mesh guard `validate_symmetry_fundamental_domain!` cannot supply: its
# active axes are empty for `:ground`, so without this a body straddling the
# plane assembles and solves in silence, against a domain that does not exist.
#
# Ported from hornlab-metal-bem (`metal/geometry.py`,
# `validate_native_ground_plane`). The contract differs from the symmetry one
# in both directions: the body is NOT required to reach the plane -- it may
# float clear -- but it IS required to lie wholly on the fluid side, and no
# face may lie flat in the plane, because such a face coincides with its own
# image and the boundary integral is singular there.
#
# The tolerance is Boundary Lab's fixed 1e-6 m rather than a model-scale
# relative one, matching what `deploy_solve.py` enforces on the same geometry.
function validate_ground_plane_domain!(
    mesh,
    mode;
    min_clearance_m::Real=0.0,
    tolerance::Real=1.0e-6,
)
    Symbol(mode) == :ground || return nothing
    isempty(mesh.faces) && error("symmetry=ground requires a mesh with faces.")

    axis = 2                       # rigid_ground_transform() is (1, -1, 1)
    minimum_y = Inf
    for face in mesh.faces, vertex_index in face
        minimum_y = min(minimum_y, Float64(mesh.vertices[vertex_index][axis]))
    end
    if minimum_y < -tolerance
        error(
            "symmetry=ground is a rigid half-space boundary at Y=0: the whole " *
            "mesh must lie at Y >= 0, but its minimum Y is $(minimum_y) m. " *
            "Translate the body above the plane; the solver will not clip it."
        )
    end

    for (face_index, face) in enumerate(mesh.faces)
        flat = true
        for vertex_index in face
            if abs(Float64(mesh.vertices[vertex_index][axis])) > tolerance
                flat = false
                break
            end
        end
        if flat
            error(
                "symmetry=ground treats Y=0 as an image plane, not a physical " *
                "boundary; triangle $(face_index) lies flat on it and would " *
                "coincide with its own image. Delete the ground-contact faces, " *
                "or lift the body clear of the plane."
            )
        end
    end

    if min_clearance_m > 0.0 && minimum_y < min_clearance_m
        error(
            "symmetry=ground requires at least $(min_clearance_m) m of " *
            "clearance, but the mesh reaches Y=$(minimum_y) m."
        )
    end
    return nothing
end

function beat_backend_from_request(request)
    backend = lowercase(strip(String(get_value(request, "beat_engine_backend", "cuda"))))
    aliases = Dict(
        "beat_cuda" => "cuda",
        "gpu" => "cuda",
        "julia_local" => "cuda",
        "local_julia" => "cuda",
        "beat_cpu" => "cpu",
        "beat_rocm" => "rocm",
        "amd" => "rocm",
        "amdgpu" => "rocm",
        "beat_metal" => "metal",
        "apple" => "metal",
        "mps" => "metal",
    )
    backend = get(aliases, backend, backend)
    backend in ("cuda", "cpu", "rocm", "metal") || error("Unsupported BEAT Engine backend: $(backend). Expected cuda, cpu, rocm, or metal.")
    return Symbol(backend)
end

function regular_quadrature_mode_from_config(config, beat_backend::Symbol)
    default_mode = beat_backend == :cpu ? "wavelength" : "fixed"
    mode = lowercase(strip(String(get_value(config, "regular_quadrature_mode", get_value(config, "quadrature_mode", default_mode)))))
    mode in ("fixed", "wavelength") || error("Unsupported regular quadrature mode: $(mode). Expected fixed or wavelength.")
    if beat_backend != :cpu && mode == "wavelength"
        error("Wavelength-driven regular quadrature is currently implemented only for the BEAT CPU backend.")
    end
    return mode
end

function mesh_area_statistic(areas, stat::String)
    values = collect(Float64.(areas))
    isempty(values) && error("Cannot select wavelength quadrature order from an empty mesh.")
    if stat == "median"
        return median(values)
    elseif stat == "p75"
        return quantile(values, 0.75)
    elseif stat == "p90"
        return quantile(values, 0.90)
    elseif stat == "max"
        return maximum(values)
    end
    error("Unsupported wavelength mesh stat: $(stat). Expected median, p75, p90, or max.")
end

function regular_quadrature_selection(config, mesh::BoundaryMesh{T}, freq::T, sound_speed::T, base_order::Int, mode::String) where {T<:AbstractFloat}
    if mode == "fixed"
        return (
            order=base_order,
            mesh_area_stat=nothing,
            element_length_m=nothing,
            kh=nothing,
            q1_cutoff=nothing,
            q2_cutoff=nothing,
            mesh_stat=nothing,
        )
    end

    mesh_stat = lowercase(strip(String(get_value(config, "wavelength_mesh_stat", "p90"))))
    area_stat = mesh_area_statistic(mesh.areas, mesh_stat)
    element_length = sqrt(area_stat)
    k = Float64(2pi * freq / sound_speed)
    kh = k * element_length
    q1_cutoff = Float64(get_value(config, "wavelength_kh_q1_max", 0.0))
    q2_cutoff = Float64(get_value(config, "wavelength_kh_q2_max", 2.0))
    q1_cutoff >= 0.0 || error("wavelength_kh_q1_max must be non-negative.")
    q2_cutoff > q1_cutoff || error("wavelength_kh_q2_max must be greater than wavelength_kh_q1_max.")
    order = kh <= q1_cutoff ? 1 : kh <= q2_cutoff ? 2 : base_order
    return (
        order=order,
        mesh_area_stat=area_stat,
        element_length_m=element_length,
        kh=kh,
        q1_cutoff=q1_cutoff,
        q2_cutoff=q2_cutoff,
        mesh_stat=mesh_stat,
    )
end

"""Gauss order that resolves a 1/r peak at a given separation-to-size ratio.

An n-point Gauss rule integrating a function whose nearest pole sits outside
the interval converges like `R^(-2n)`, with `R = rho + sqrt(rho^2 - 1)` the
Bernstein-ellipse parameter and `rho` the pole distance in half-widths. Here
`ratio` is the separation in *combined* circumradii, so a single element's
half-width sees `rho ~ 2 * ratio`. Solving `2n * log(R) >= log(10^7)` for n
gives the order below: ratio 0.75 asks for 8, ratio 1.0 for 6, ratio >= 1.5
for the floor of 4 -- which is still a 16-point tensor rule against the
6-point regular rule it replaces.
"""
function near_correction_order_for_ratio(ratio::Real, top_order::Int)
    rho = 2 * Float64(ratio)
    rho <= 1.0 && return top_order
    bernstein = rho + sqrt(rho * rho - 1)
    required = ceil(Int, log(1.0e7) / (2 * log(bernstein)))
    return clamp(required, 4, top_order)
end
"""Near-singular face-pair selection.

The vendored BEAT assembly corrects *coincident* and *edge/vertex-adjacent*
element pairs with Duffy rules, and integrates every other pair with the plain
regular Gauss rule. Pairs that are disjoint but close -- the second ring of a
triangulation, the two sides of a thin wall, a mouth roll-back folding back on
itself -- are neither, so their 1/r kernel is sampled by a 6-point rule across
a peak the rule cannot resolve. `build_near_correction_cache` (upstream)
re-integrates a supplied pair list with a high-order tensor-product rule and
subtracts the regular contribution; this function decides which pairs earn it.

Upstream selects pairs by an absolute metre threshold tuned for cabinet
spacing, and grades the order into three hand-set distance bands. Quadrature
error does not scale with metres, it scales with the ratio of pair separation
to element size, so the threshold here is `separation / (r_test + r_trial)`
with `r` the element circumradius, and the order each pair gets comes from
`near_correction_order_for_ratio` rather than a band table -- which matters
because the pair count grows like ratio^2, so a band scheme spends most of its
budget on the pairs that need it least.
"""
function near_correction_selection(config, mesh::BoundaryMesh{T}, symmetry_mode::Symbol) where {T<:AbstractFloat}
    Bool(get_value(config, "near_correction_enabled", false)) || return nothing
    cutoff = Float64(get_value(config, "near_correction_cutoff", 2.0))
    cutoff > 0.0 || error("near_correction_cutoff must be greater than zero.")
    top_order = Int(get_value(config, "near_correction_order", 8))
    top_order >= 4 || error("near_correction_order must be at least 4.")

    face_count = length(mesh.faces)
    face_count == 0 && return nothing
    radii = [
        maximum(norm(vertex - mesh.centroids[index]) for vertex in mesh.face_vertices[index])
        for index in 1:face_count
    ]
    maximum(radii) > 0 || return nothing

    identity_pairs = near_correction_pairs(mesh, radii, cutoff, top_order, nothing)
    # A symmetric solve assembles each face against the mirror images of every
    # other face as well, and those image pairs run closer than the identity
    # ones -- a face a millimetre off the symmetry plane sits a hair from its
    # own reflection. One cache carries one transform, so `yz+xz` needs three.
    image_selections = Tuple{Any,Vector{Tuple{Int,Int,Int}}}[]
    for transform in symmetry_image_transforms(symmetry_mode)
        pairs = near_correction_pairs(mesh, radii, cutoff, top_order, transform)
        pairs === nothing || push!(image_selections, (transform, pairs))
    end
    identity_pairs === nothing && isempty(image_selections) && return nothing
    return (
        identity_pairs=identity_pairs,
        image_selections=image_selections,
        cutoff=cutoff,
        top_order=top_order,
    )
end

"""Face pairs within `cutoff` combined circumradii, optionally across a mirror.

With `transform === nothing` this is the self-domain search and a face is never
paired with itself; across a mirror the `i == j` pair is the one that matters
most, because that is a face against its own reflection.
"""
function near_correction_pairs(
    mesh::BoundaryMesh{T},
    radii::Vector{T},
    cutoff::Float64,
    top_order::Int,
    transform,
) where {T<:AbstractFloat}
    face_count = length(mesh.faces)
    trial_centroids = transform === nothing ? mesh.centroids :
        [reflect_point(transform, centroid) for centroid in mesh.centroids]

    # Cell edge equals the largest search radius any pair can ask for, so the
    # 27-cell neighbourhood of a face is guaranteed to hold every candidate.
    cell = cutoff * 2 * maximum(radii)
    buckets = Dict{NTuple{3,Int},Vector{Int}}()
    cell_of(point) = (
        Int(floor(point[1] / cell)),
        Int(floor(point[2] / cell)),
        Int(floor(point[3] / cell)),
    )
    for index in 1:face_count
        push!(get!(() -> Int[], buckets, cell_of(trial_centroids[index])), index)
    end

    pairs = Tuple{Int,Int,Int}[]
    for test_index in 1:face_count
        base = cell_of(mesh.centroids[test_index])
        test_centroid = mesh.centroids[test_index]
        test_radius = radii[test_index]
        for dx in -1:1, dy in -1:1, dz in -1:1
            neighbours = get(buckets, (base[1] + dx, base[2] + dy, base[3] + dz), nothing)
            neighbours === nothing && continue
            for trial_index in neighbours
                transform === nothing && trial_index == test_index && continue
                scale = test_radius + radii[trial_index]
                scale > 0 || continue
                ratio = norm(test_centroid - trial_centroids[trial_index]) / scale
                ratio <= cutoff || continue
                # Coincident and adjacent pairs already carry a Duffy
                # correction; the cache builder drops them either way, but
                # skipping the self-domain ones here keeps the list small.
                transform === nothing &&
                    elements_are_adjacent(mesh.faces[test_index], mesh.faces[trial_index]) &&
                    continue
                push!(pairs, (test_index, trial_index, near_correction_order_for_ratio(ratio, top_order)))
            end
        end
    end
    return isempty(pairs) ? nothing : pairs
end
function mesh_inputs_from_config(config)
    meshes = get_value(config, "meshes", Any[])
    if !isempty(meshes)
        inputs = NamedTuple[]
        seen_names = Set{String}()
        for (index, mesh) in enumerate(meshes)
            name = String(get_value(mesh, "name", "mesh_$(index)"))
            name in seen_names && error("Duplicate mesh name: $(name)")
            push!(seen_names, name)
            translation = get_value(mesh, "translation_m", [0.0, 0.0, 0.0])
            length(translation) == 3 || error("Mesh '$(name)' translation_m must contain three values.")
            push!(inputs, (
                path=String(mesh["file"]),
                scale=Float64(get_value(mesh, "scale_factor", get_value(config, "scale_factor", 0.001))),
                name=name,
                translation=(Float64(translation[1]), Float64(translation[2]), Float64(translation[3])),
            ))
        end
        return inputs
    end

    return [(
        path=String(config["mesh_file"]),
        scale=Float64(get_value(config, "scale_factor", 0.001)),
        name="mesh",
        translation=(0.0, 0.0, 0.0),
    )]
end

function mesh_name_to_id(mesh_inputs)
    return Dict(input.name => index for (index, input) in enumerate(mesh_inputs))
end

function radiator_inputs_from_config(config, mesh_inputs)
    raw_radiators = get_value(config, "radiators", Any[])
    mesh_lookup = mesh_name_to_id(mesh_inputs)
    if isempty(raw_radiators)
        first_mesh = mesh_inputs[1]
        return [Dict{String,Any}(
            "name" => "Radiator",
            "tag" => Int(get_value(config, "tag_throat", 2)),
            "mesh" => first_mesh.name,
            "mesh_id" => 1,
            "channel" => "main",
            "velocity_offset_db" => 0.0,
            "level_db" => 0.0,
            "polarity" => 1,
            "delay_ms" => 0.0,
            "hpf" => Dict("type" => "none"),
            "lpf" => Dict("type" => "none"),
        )]
    end

    radiators = Dict{String,Any}[]
    for radiator in raw_radiators
        radiator_name = String(get_value(radiator, "name", "Radiator"))
        raw_mesh = get_value(radiator, "mesh", nothing)
        if raw_mesh === nothing
            length(mesh_inputs) == 1 || error("Radiator '$(radiator_name)' must specify 'mesh' when multiple meshes are configured.")
            mesh_name = mesh_inputs[1].name
        else
            mesh_name = String(raw_mesh)
        end
        haskey(mesh_lookup, mesh_name) || error("Radiator '$(radiator_name)' references unknown mesh '$(mesh_name)'.")
        validate_crossover_config(radiator_name, get_value(radiator, "hpf", nothing))
        validate_crossover_config(radiator_name, get_value(radiator, "lpf", nothing))
        push!(radiators, Dict{String,Any}(
            "name" => radiator_name,
            "tag" => Int(radiator["tag"]),
            "mesh" => mesh_name,
            "mesh_id" => mesh_lookup[mesh_name],
            "channel" => String(get_value(radiator, "channel", "main")),
            "velocity_offset_db" => Float64(get_value(radiator, "velocity_offset_db", 0.0)),
            "level_db" => Float64(get_value(radiator, "level_db", 0.0)),
            "polarity" => Int(get_value(radiator, "polarity", 1)),
            "delay_ms" => Float64(get_value(radiator, "delay_ms", 0.0)),
            "hpf" => get_value(radiator, "hpf", Dict("type" => "none")),
            "lpf" => get_value(radiator, "lpf", Dict("type" => "none")),
        ))
    end
    return radiators
end

function channel_inputs_from_config(config)
    channels = Dict{String,Any}()
    for channel in get_value(config, "channels", Any[])
        name = String(channel["name"])
        validate_crossover_config("channel $(name)", get_value(channel, "hpf", nothing))
        validate_crossover_config("channel $(name)", get_value(channel, "lpf", nothing))
        channels[name] = Dict{String,Any}(
            "level_db" => Float64(get_value(channel, "level_db", 0.0)),
            "polarity" => Int(get_value(channel, "polarity", 1)),
            "delay_ms" => Float64(get_value(channel, "delay_ms", 0.0)),
            "hpf" => get_value(channel, "hpf", Dict("type" => "none")),
            "lpf" => get_value(channel, "lpf", Dict("type" => "none")),
        )
    end
    return channels
end

function load_combined_mesh(mesh_inputs, ::Type{T}) where {T<:AbstractFloat}
    vertices = SVector{3,T}[]
    faces = NTuple{3,Int}[]
    physical_tags = Int[]
    element_mesh_ids = Int[]

    for (mesh_id, input) in enumerate(mesh_inputs)
        mesh = load_gmsh22_with_tags(input.path, T(input.scale))
        translation = SVector{3,T}(T(input.translation[1]), T(input.translation[2]), T(input.translation[3]))
        vertex_offset = length(vertices)
        append!(vertices, [vertex + translation for vertex in mesh.vertices])
        append!(faces, [(face[1] + vertex_offset, face[2] + vertex_offset, face[3] + vertex_offset) for face in mesh.faces])
        append!(physical_tags, mesh.physical_tags)
        append!(element_mesh_ids, fill(mesh_id, length(mesh.faces)))
    end

    return BoundaryMesh(vertices, faces, physical_tags), element_mesh_ids
end

function butterworth_poles(order::Int, ::Type{T}) where {T<:AbstractFloat}
    return Complex{T}[
        exp(Complex{T}(0, T(pi) / T(2) + T(2 * k - 1) * T(pi) / T(2 * order)))
        for k in 1:order
    ]
end

function butterworth_response(crossover_type::String, order::Int, cutoff_hz, freq::T) where {T<:AbstractFloat}
    cutoff = T(cutoff_hz)
    omega = T(2pi) * freq
    omega_c = T(2pi) * cutoff
    # Channel DSP is applied directly to BEAT's exp(-i omega t) solver
    # phasors, so evaluate the causal analog response at s = -i*omega.
    s = Complex{T}(0, -omega)
    response = one(Complex{T})

    for pole in butterworth_poles(order, T)
        scaled_pole = crossover_type == "lowpass" ? omega_c * pole : omega_c / pole
        response *= crossover_type == "lowpass" ? (-scaled_pole) / (s - scaled_pole) : s / (s - scaled_pole)
    end

    return response
end

function crossover_response(crossover, freq::T) where {T<:AbstractFloat}
    crossover === nothing && return one(Complex{T})
    crossover_type = lowercase(String(get_value(crossover, "type", "none")))
    crossover_type == "none" && return one(Complex{T})

    filter_name = lowercase(String(get_value(crossover, "filter", "butterworth")))
    order = Int(get_value(crossover, "order", 1))
    cutoff_hz = get_value(crossover, "frequency_hz", nothing)
    cutoff_hz === nothing && error("Crossover frequency_hz must be set for $(crossover_type).")

    if filter_name == "linkwitz_riley"
        section = butterworth_response(crossover_type, div(order, 2), cutoff_hz, freq)
        return section * section
    end

    return butterworth_response(crossover_type, order, cutoff_hz, freq)
end

function channel_drive(channel, freq::T) where {T<:AbstractFloat}
    omega = T(2pi) * freq
    level = T(10.0) ^ (T(channel["level_db"]) / T(20.0))
    delay = exp(Complex{T}(0, omega * T(channel["delay_ms"]) / T(1000.0)))
    crossover = crossover_response(get_value(channel, "hpf", nothing), freq) *
        crossover_response(get_value(channel, "lpf", nothing), freq)
    return Complex{T}(T(channel["polarity"]) * level) * delay * crossover
end

function polar_observation_points(config, ::Type{T}) where {T<:AbstractFloat}
    step = T(get_value(config, "step_size", 5.0))
    angle_min = T(get_value(config, "min_angle", -180.0))
    angle_max = T(get_value(config, "max_angle", 180.0))
    step <= 0 && error("step_size must be positive.")
    angle_min < -180 && error("polar angle range must stay within [-180, 180] degrees.")
    angle_max > 180 && error("polar angle range must stay within [-180, 180] degrees.")
    angle_max < angle_min && error("max_angle must be >= min_angle.")
    !(angle_min <= 0 <= angle_max) && error("polar angle range must include 0 degrees.")

    angles = collect(Float32.(range(Float64(angle_min), stop=Float64(angle_max), step=Float64(step))))
    if isempty(angles) || angles[end] < Float32(angle_max)
        push!(angles, Float32(angle_max))
    end
    angles = Float32.(clamp.(angles, Float32(angle_min), Float32(angle_max)))

    distance = T(get_value(config, "distance", 2.0))
    distance <= 0 && error("distance must be positive.")
    axial_offset = T(get_value(config, "axial_offset", 0.0))
    horizontal = SVector{3,T}[]
    vertical = SVector{3,T}[]
    for angle_deg in angles
        angle = T(pi) * T(angle_deg) / T(180.0)
        push!(horizontal, SVector{3,T}(distance * sin(angle), T(0.0), distance * cos(angle) + axial_offset))
        push!(vertical, SVector{3,T}(T(0.0), distance * sin(angle), distance * cos(angle) + axial_offset))
    end
    diagonal = nothing
    if Bool(get_value(config, "diagonal_enabled", false))
        inclination = T(pi) * T(get_value(config, "diagonal_inclination_deg", 45.0)) / T(180.0)
        diagonal = SVector{3,T}[]
        for angle_deg in angles
            angle = T(pi) * T(angle_deg) / T(180.0)
            push!(diagonal, SVector{3,T}(
                distance * sin(angle) * cos(inclination),
                distance * sin(angle) * sin(inclination),
                distance * cos(angle) + axial_offset,
            ))
        end
    end
    on_axis_idx = argmin(abs.(Float64.(angles)))
    return angles, horizontal, vertical, diagonal, on_axis_idx
end

function spherical_observation(config, ::Type{T}) where {T<:AbstractFloat}
    grid = get_value(config, "spherical_grid", nothing)
    enabled = Bool(get_value(config, "spherical_sampling_enabled", false))
    if grid !== nothing
        return spherical_grid_observation(config, grid, T)
    end
    if !enabled
        return nothing
    end

    point_count = Int(get_value(config, "spherical_sampling_points", 6000))
    point_count <= 0 && error("spherical_sampling_points must be positive.")
    distance = T(get_value(config, "distance", 2.0))
    distance <= 0 && error("distance must be positive.")
    axial_offset = T(get_value(config, "axial_offset", 0.0))
    golden_angle = T(pi * (3.0 - sqrt(5.0)))
    points = Vector{SVector{3,T}}(undef, point_count)
    theta = Vector{Float32}(undef, point_count)
    phi = Vector{Float32}(undef, point_count)
    r_distance = fill(Float32(distance), point_count)

    for i in 0:(point_count - 1)
        z_unit = T(1.0 - (2.0 * i + 1.0) / point_count)
        xy_radius = sqrt(max(T(1.0) - z_unit * z_unit, T(0.0)))
        azimuth = T(i) * golden_angle
        x = distance * xy_radius * cos(azimuth)
        y = distance * xy_radius * sin(azimuth)
        z = distance * z_unit + axial_offset
        storage_index = i + 1
        points[storage_index] = SVector{3,T}(x, y, z)
        theta[storage_index] = Float32(acos(clamp(z_unit, T(-1.0), T(1.0))))
        phi[storage_index] = Float32(mod(atan(y, x), T(2pi)))
    end

    return (
        points=points,
        metadata=Dict(
            "r_distance_m" => r_distance,
            "theta_polar_rad" => theta,
            "phi_azimuth_rad" => phi,
        ),
    )
end

"""
Theta-major spherical grid around the observation origin, matching the
theta/phi layout HornLab's balloon and directivity-index mapping expects:
theta in [0, theta_max_deg] inclusive, phi in [0, 360) without a wrap column,
phi varying fastest. Theta is measured from the +z forward axis and phi from
+x (the horizontal cut plane).
"""
function spherical_grid_observation(config, grid, ::Type{T}) where {T<:AbstractFloat}
    theta_count = Int(get_value(grid, "theta_count", 37))
    phi_count = Int(get_value(grid, "phi_count", 72))
    theta_max_deg = T(get_value(grid, "theta_max_deg", 180.0))
    theta_count >= 2 || error("spherical_grid theta_count must be at least 2.")
    phi_count >= 3 || error("spherical_grid phi_count must be at least 3.")
    (T(0.0) < theta_max_deg <= T(180.0)) || error("spherical_grid theta_max_deg must be in (0, 180].")
    distance = T(get_value(config, "distance", 2.0))
    distance <= 0 && error("distance must be positive.")
    axial_offset = T(get_value(config, "axial_offset", 0.0))

    point_count = theta_count * phi_count
    points = Vector{SVector{3,T}}(undef, point_count)
    theta = Vector{Float32}(undef, point_count)
    phi = Vector{Float32}(undef, point_count)
    r_distance = fill(Float32(distance), point_count)
    index = 1
    for theta_index in 0:(theta_count - 1)
        theta_value = T(pi) * theta_max_deg / T(180.0) * T(theta_index) / T(theta_count - 1)
        sin_theta = sin(theta_value)
        cos_theta = cos(theta_value)
        for phi_index in 0:(phi_count - 1)
            phi_value = T(2pi) * T(phi_index) / T(phi_count)
            points[index] = SVector{3,T}(
                distance * sin_theta * cos(phi_value),
                distance * sin_theta * sin(phi_value),
                distance * cos_theta + axial_offset,
            )
            theta[index] = Float32(theta_value)
            phi[index] = Float32(phi_value)
            index += 1
        end
    end

    return (
        points=points,
        metadata=Dict(
            "r_distance_m" => r_distance,
            "theta_polar_rad" => theta,
            "phi_azimuth_rad" => phi,
            "grid_theta_count" => theta_count,
            "grid_phi_count" => phi_count,
        ),
    )
end

function drive_for_radiator(radiator, channels, freq::T) where {T<:AbstractFloat}
    omega = T(2pi) * freq
    channel_name = String(get_value(radiator, "channel", "main"))
    channel = get(channels, channel_name, nothing)
    if channel !== nothing
        return channel_drive(channel, freq) * T(10.0) ^ (T(radiator["velocity_offset_db"]) / T(20.0))
    end

    level_db = T(radiator["level_db"] + radiator["velocity_offset_db"])
    polarity = T(radiator["polarity"])
    delay_ms = T(radiator["delay_ms"])
    level = T(10.0) ^ (level_db / T(20.0))
    delay = exp(Complex{T}(0, omega * delay_ms / T(1000.0)))
    crossover = crossover_response(get_value(radiator, "hpf", nothing), freq) *
        crossover_response(get_value(radiator, "lpf", nothing), freq)
    return Complex{T}(polarity * level) * delay * crossover
end

function radiator_owns_element(radiator, element_mesh_ids, element_index::Int)
    return element_mesh_ids[element_index] == Int(radiator["mesh_id"])
end

function validate_radiator_channels(radiators, channels)
    isempty(channels) && return
    for radiator in radiators
        channel_name = String(get_value(radiator, "channel", "main"))
        haskey(channels, channel_name) || error("Radiator '$(radiator["name"])' references unknown channel '$(channel_name)'.")
    end
end

function validate_radiator_elements(mesh, element_mesh_ids, radiators)
    for radiator in radiators
        tag = Int(radiator["tag"])
        found = false
        for element_index in eachindex(mesh.physical_tags)
            if mesh.physical_tags[element_index] == tag && radiator_owns_element(radiator, element_mesh_ids, element_index)
                found = true
                break
            end
        end
        found || error("No elements found for radiator '$(radiator["name"])' tag=$(tag) on mesh '$(radiator["mesh"])'.")
    end
end

function pressure_for_drives(
    mesh,
    element_mesh_ids,
    operators,
    identity_p1_p1,
    identity_p1_dp0,
    radiators,
    drives,
    rho,
    omega,
    k;
    cpu_solve_system=nothing,
    cuda_solve_identity_cache=nothing,
    rocm_solve_identity_cache=nothing,
    source_motion::Symbol=:normal,
)
    ComplexType = eltype(drives)
    q_neumann = zeros(ComplexType, length(mesh.faces))
    for (radiator_index, radiator) in enumerate(radiators)
        tag = Int(radiator["tag"])
        drive = drives[radiator_index]
        for element_index in eachindex(mesh.physical_tags)
            if mesh.physical_tags[element_index] == tag && radiator_owns_element(radiator, element_mesh_ids, element_index)
                # AXIAL: a rigid piston along the +z observation axis, so the
                # per-face normal velocity is v * (n_hat . z). NORMAL keeps
                # the uniformly breathing cap.
                motion_factor = source_motion == :axial ?
                    ComplexType(mesh.normals[element_index][3]) : one(ComplexType)
                q_neumann[element_index] = ComplexType(0, rho * omega) * drive * motion_factor
            end
        end
    end
    pressure = if cpu_solve_system !== nothing
        solve_burton_miller_neumann_cpu_system(cpu_solve_system, q_neumann, typeof(k))
    elseif cuda_solve_identity_cache !== nothing
        solve_burton_miller_neumann(operators, cuda_solve_identity_cache, q_neumann, k)
    elseif rocm_solve_identity_cache !== nothing
        solve_burton_miller_neumann(operators, rocm_solve_identity_cache, q_neumann, k)
    else
        solve_burton_miller_neumann(operators, identity_p1_p1, identity_p1_dp0, q_neumann, k)
    end
    return pressure, q_neumann
end

# Every channel's Neumann vector, one column per channel.
#
# The fused Burton-Miller path folds the right-hand side in during assembly, so
# the drive patterns have to be known before it runs. They are:
# `channel_unit_drives` depends only on which radiators belong to a channel,
# and the frequency enters solely through the i*rho*omega scale, so one pass
# builds every channel's column for a frequency.
function channel_neumann_columns(
    mesh,
    element_mesh_ids,
    radiators,
    channel_names,
    rho,
    omega,
    ::Type{T};
    source_motion::Symbol=:normal,
) where {T<:AbstractFloat}
    columns = zeros(Complex{T}, length(mesh.faces), length(channel_names))
    for (channel_index, channel_name) in enumerate(channel_names)
        unit_drives = channel_unit_drives(radiators, channel_name, T)
        for (radiator_index, radiator) in enumerate(radiators)
            drive = unit_drives[radiator_index]
            drive == zero(Complex{T}) && continue
            tag = Int(radiator["tag"])
            for element_index in eachindex(mesh.physical_tags)
                if mesh.physical_tags[element_index] == tag && radiator_owns_element(radiator, element_mesh_ids, element_index)
                    # AXIAL: a rigid piston along the +z observation axis, so the
                    # per-face normal velocity is v * (n_hat . z). NORMAL keeps
                    # the uniformly breathing cap.  Same rule as the
                    # four-operator path in `pressure_for_drives`.
                    motion_factor = source_motion == :axial ?
                        Complex{T}(mesh.normals[element_index][3]) : one(Complex{T})
                    columns[element_index, channel_index] = Complex{T}(0, rho * omega) * drive * motion_factor
                end
            end
        end
    end
    return columns
end

function release_assembly_payload!(payload)
    payload === nothing && return nothing
    if get(payload, :kind, :operators) === :fused
        get(payload.system, :on_gpu, false) && release_metal_burton_miller_system!(payload.system)
        return nothing
    end
    release_operator_storage!(payload.operators)
    return nothing
end

function field_for_points(points, mesh, pressure, q_neumann, k, field_cache, beat_backend::Symbol)
    if beat_backend == :cuda
        return evaluate_galerkin_field_cuda(points, mesh, pressure, q_neumann, k, field_cache)
    elseif beat_backend == :rocm
        return evaluate_galerkin_field_rocm(points, mesh, pressure, q_neumann, k, field_cache)
    elseif beat_backend == :metal
        return evaluate_galerkin_field_metal(points, mesh, pressure, q_neumann, k, field_cache)
    elseif beat_backend == :cpu
        return evaluate_galerkin_field_cpu(points, mesh, pressure, q_neumann, k, field_cache)
    end
    error("Unsupported BEAT Engine backend: $(beat_backend).")
end

function spl_for_points(points, mesh, pressure, q_neumann, k, field_cache, beat_backend::Symbol, ::Type{T}) where {T<:AbstractFloat}
    pot = field_for_points(points, mesh, pressure, q_neumann, k, field_cache, beat_backend)
    return Float32.(T(20.0) .* log10.(abs.(pot) ./ T(20e-6)))
end

function pressure_to_spl(pressure, ::Type{T}) where {T<:AbstractFloat}
    return Float32.(T(20.0) .* log10.(abs.(pressure) ./ T(20e-6)))
end

function complex_vector_to_wire(values)
    return Dict("real" => Float32.(real.(values)), "imag" => Float32.(imag.(values)))
end

function complex_rows_to_wire(rows)
    return Dict(
        "real" => [Float32.(real.(row)) for row in rows],
        "imag" => [Float32.(imag.(row)) for row in rows],
    )
end

function interpolate_complex_reference(values, angles_deg, reference_angle_deg, ::Type{T}) where {T<:AbstractFloat}
    isempty(values) && return Complex{T}(0, 0)
    length(values) == 1 && return Complex{T}(values[1])

    reference_wrapped = mod(T(reference_angle_deg) + T(180.0), T(360.0)) - T(180.0)
    angle_values = T.(angles_deg)
    value_values = Complex{T}.(values)
    if isapprox(angle_values[1], T(-180.0)) && isapprox(angle_values[end], T(180.0))
        angle_values = angle_values[1:(end - 1)]
        value_values = value_values[1:(end - 1)]
    end

    extended_angles = vcat(angle_values .- T(360.0), angle_values, angle_values .+ T(360.0))
    extended_values = vcat(value_values, value_values, value_values)
    for idx in 1:(length(extended_angles) - 1)
        left = extended_angles[idx]
        right = extended_angles[idx + 1]
        if reference_wrapped >= left && reference_wrapped <= right
            if isapprox(left, right)
                return extended_values[idx]
            end
            fraction = (reference_wrapped - left) / (right - left)
            return extended_values[idx] * (one(T) - fraction) + extended_values[idx + 1] * fraction
        end
    end
    return extended_values[argmin(abs.(extended_angles .- reference_wrapped))]
end

function flat_target_corrections(channel_names, horizontal_pressure_rows, angles_deg, reference_angle_deg, flat_target::Bool, ::Type{T}) where {T<:AbstractFloat}
    corrections = Dict{String,T}()
    for (channel_index, channel_name) in enumerate(channel_names)
        if !flat_target
            corrections[channel_name] = one(T)
            continue
        end
        reference_pressure = interpolate_complex_reference(
            horizontal_pressure_rows[channel_index],
            angles_deg,
            reference_angle_deg,
            T,
        )
        magnitude = abs(reference_pressure)
        corrections[channel_name] = magnitude <= T(1e-12) ? one(T) : one(T) / magnitude
    end
    return corrections
end

function synthesize_channel_basis(channel_names, horizontal_pressure_rows, vertical_pressure_rows, sphere_pressure_rows, channels, freq, angles_deg, reference_angle_deg, flat_target::Bool, ::Type{T}; diagonal_pressure_rows=nothing) where {T<:AbstractFloat}
    corrections = flat_target_corrections(channel_names, horizontal_pressure_rows, angles_deg, reference_angle_deg, flat_target, T)
    weights = Complex{T}[
        channel_drive(get(channels, channel_name, Dict(
            "level_db" => 0.0,
            "polarity" => 1,
            "delay_ms" => 0.0,
            "hpf" => Dict("type" => "none"),
            "lpf" => Dict("type" => "none"),
        )), freq) * corrections[channel_name]
        for channel_name in channel_names
    ]

    horizontal_pressure = zero.(horizontal_pressure_rows[1])
    vertical_pressure = zero.(vertical_pressure_rows[1])
    for channel_index in eachindex(channel_names)
        horizontal_pressure .+= horizontal_pressure_rows[channel_index] .* weights[channel_index]
        vertical_pressure .+= vertical_pressure_rows[channel_index] .* weights[channel_index]
    end

    horizontal_spl = pressure_to_spl(horizontal_pressure, T)
    vertical_spl = pressure_to_spl(vertical_pressure, T)
    on_axis_idx = argmin(abs.(Float64.(angles_deg)))
    reference = horizontal_spl[on_axis_idx]
    sphere_norm = nothing
    if sphere_pressure_rows !== nothing
        sphere_pressure = zero.(sphere_pressure_rows[1])
        for channel_index in eachindex(channel_names)
            sphere_pressure .+= sphere_pressure_rows[channel_index] .* weights[channel_index]
        end
        sphere_norm = Float32.(pressure_to_spl(sphere_pressure, T) .- reference)
    end
    diagonal_spl = nothing
    diagonal_norm = nothing
    if diagonal_pressure_rows !== nothing
        diagonal_pressure = zero.(diagonal_pressure_rows[1])
        for channel_index in eachindex(channel_names)
            diagonal_pressure .+= diagonal_pressure_rows[channel_index] .* weights[channel_index]
        end
        diagonal_spl = pressure_to_spl(diagonal_pressure, T)
        diagonal_norm = Float32.(diagonal_spl .- reference)
    end

    return (
        horizontal_spl=horizontal_spl,
        vertical_spl=vertical_spl,
        horizontal_norm=Float32.(horizontal_spl .- reference),
        vertical_norm=Float32.(vertical_spl .- reference),
        sphere_norm=sphere_norm,
        diagonal_spl=diagonal_spl,
        diagonal_norm=diagonal_norm,
        corrections=corrections,
        weights=weights,
    )
end

function channel_unit_drives(radiators, channel_name::String, ::Type{T}) where {T<:AbstractFloat}
    return Complex{T}[
        String(get_value(radiator, "channel", "main")) == channel_name ?
        Complex{T}(T(10.0) ^ (T(radiator["velocity_offset_db"]) / T(20.0)), 0) :
        Complex{T}(0, 0)
        for radiator in radiators
    ]
end

function radiator_drives_from_channel_basis(radiators, channels, freq, corrections, ::Type{T}) where {T<:AbstractFloat}
    return Complex{T}[
        drive_for_radiator(radiator, channels, freq) *
        get(corrections, String(get_value(radiator, "channel", "main")), one(T))
        for radiator in radiators
    ]
end

# `symmetry_reduction_factor` counts image TRANSFORMS, which is the right
# multiplier only when every image is a real radiator. Under `:x` and `:xy` it
# is: the mesh is a half or a quarter of a mirror-symmetric body, and the
# missing halves radiate. Under `:ground` it is not: the mesh is the whole
# body, and its one image is a fiction that stands in for a rigid boundary.
# Counting that image would report twice the integrated force -- a factor 2 on
# a complex force, so 20*log10(2) = 6.02 dB on the reported impedance -- for a
# solve whose pressure field is entirely correct. hornlab-metal-bem documents
# the same trap under "Rigid Half Space": radiated surface power is
# "multiplied by the copy count" for a symmetry plane and "counted once" for a
# ground plane.
#
# `hornlab_beat_bem/sweep.py` divides the wire force by the matching
# `_SYMMETRY_FACTOR`, so the two constants have to move together; `"ground": 1`
# there is the other half of this fix.
function impedance_for_radiators(mesh, element_mesh_ids, pressure, radiators, drives, ::Type{T}; symmetry_mode::Symbol=:off) where {T<:AbstractFloat}
    force_scale = eltype(pressure)(
        symmetry_mode == :ground ? 1 : symmetry_reduction_factor(symmetry_mode)
    )
    impedance = Vector{Vector{Float32}}()
    for (radiator_index, radiator) in enumerate(radiators)
        drive = drives[radiator_index]
        if abs(drive) <= T(0.0)
            push!(impedance, [Float32(NaN), Float32(NaN)])
            continue
        end

        total_force = zero(eltype(pressure))
        tag = Int(radiator["tag"])
        for element_index in eachindex(mesh.physical_tags)
            mesh.physical_tags[element_index] == tag || continue
            radiator_owns_element(radiator, element_mesh_ids, element_index) || continue
            face = mesh.faces[element_index]
            p_avg = (pressure[face[1]] + pressure[face[2]] + pressure[face[3]]) / eltype(pressure)(3.0)
            total_force += p_avg * eltype(pressure)(mesh.areas[element_index]) * force_scale
        end
        total_force *= eltype(pressure)(10.0)
        z_complex = total_force / drive
        push!(impedance, [Float32(real(z_complex) / 2), Float32(-imag(z_complex) / 2)])
    end
    return impedance
end

function solve_request(request)
    try
        solve_request_impl(request)
    finally
        cleanup_accelerators_after_solve!()
    end
end

function cleanup_accelerators_after_solve!()
    cuda = BeatEngineCore.CUDA_MODULE
    rocm = BeatEngineCore.AMDGPU_MODULE
    metal = BeatEngineCore.METAL_MODULE
    if cuda !== nothing
        try
            cuda.functional() && cuda.synchronize()
        catch
        end
    end
    if rocm !== nothing
        try
            rocm.functional() && rocm.synchronize()
        catch
        end
    end
    if metal !== nothing
        try
            metal.functional() && metal.synchronize()
        catch
        end
    end

    GC.gc(true)

    if cuda !== nothing
        try
            if isdefined(cuda, :reclaim)
                cuda.reclaim()
            end
        catch
        end
    end
    if rocm !== nothing
        try
            if isdefined(rocm, :reclaim)
                rocm.reclaim()
            end
        catch
        end
    end

    GC.gc(true)
    return nothing
end

function solve_request_impl(request)
    schema_version = Int(get_value(request, "schema_version", 1))
    schema_version in (1, 2) || error("Unsupported solve request schema_version $(schema_version).")

    config = request["config"]
    symmetry_mode = symmetry_mode_from_config(config)
    beat_backend = beat_backend_from_request(request)
    rocm_assembly_mode = beat_backend == :rocm ? BeatEngineCore._normalized_rocm_assembly_mode(
        get_value(config, "rocm_assembly_mode", nothing),
    ) : nothing
    metal_assembly_mode = beat_backend == :metal ? BeatEngineCore._normalized_metal_assembly_mode(
        get_value(config, "metal_assembly_mode", nothing),
    ) : nothing
    frequencies = Float32.(request["frequencies_hz"])
    isempty(frequencies) && error("frequencies_hz must contain at least one frequency.")
    cancel_path = get_value(request, "cancel_path", nothing)

    source_motion = Symbol(lowercase(strip(String(get_value(config, "source_motion", "normal")))))
    source_motion in (:normal, :axial) || error("Unsupported source_motion: $(source_motion). Expected normal or axial.")

    # BEAT assembles and solves in Float32 because that is what its CUDA and
    # ROCm kernels are written for. The CPU path is generic in the element
    # type, so it can run the same solve in Float64 -- which is what makes it
    # usable as an arbiter for how much of a discrepancy is single-precision
    # noise rather than discretisation. Results stay Float32 on the wire.
    solve_precision = lowercase(strip(String(get_value(config, "solve_precision", "single"))))
    solve_precision in ("single", "double") ||
        error("Unsupported solve_precision: $(solve_precision). Expected single or double.")
    if solve_precision == "double" && beat_backend != :cpu
        error("solve_precision=double is only available on the BEAT CPU backend.")
    end
    FloatType = solve_precision == "double" ? Float64 : Float32
    mesh_inputs = mesh_inputs_from_config(config)
    radiators = radiator_inputs_from_config(config, mesh_inputs)
    channels = channel_inputs_from_config(config)
    validate_radiator_channels(radiators, channels)
    polar_angles_deg, horizontal_points, vertical_points, diagonal_points, on_axis_idx = polar_observation_points(config, FloatType)
    sphere = spherical_observation(config, FloatType)
    sphere_metadata = sphere === nothing ? nothing : sphere.metadata

    emit_event(
        "initialized";
        polar_angle_deg=polar_angles_deg,
        radiator_names=[radiator["name"] for radiator in radiators],
        sphere_metadata=sphere_metadata,
    )

    emit_event("status"; message=@sprintf(
        "BEAT Engine loading %d mesh%s with %d thread(s)",
        length(mesh_inputs),
        length(mesh_inputs) == 1 ? "" : "es",
        Threads.nthreads(),
    ))

    mesh, element_mesh_ids = load_combined_mesh(mesh_inputs, FloatType)
    mesh = snap_symmetry_planes(mesh, Symbol(symmetry_mode))
    validate_symmetry_fundamental_domain!(mesh, Symbol(symmetry_mode))
    validate_ground_plane_domain!(
        mesh, Symbol(symmetry_mode);
        min_clearance_m=Float64(get_value(config, "ground_plane_min_clearance_m", 0.0)),
    )
    validate_radiator_elements(mesh, element_mesh_ids, radiators)
    p1_space = build_p1_space(mesh)
    dp0_space = build_dp0_space(mesh)
    cpu_blas_threads = if beat_backend == :cpu
        configure_beat_cpu_blas_threads!(p1_space.global_dof_count)
    else
        BLAS.set_num_threads(Threads.nthreads())
        BLAS.get_num_threads()
    end
    base_regular_order = Int(get_value(config, "quadrature_order", 4))
    regular_quadrature_mode = regular_quadrature_mode_from_config(config, beat_backend)
    rule = triangle_rule(FloatType, base_regular_order)
    cpu_field_cache = build_field_evaluation_cache(mesh, rule; symmetry_mode=Symbol(symmetry_mode))
    singular_order = Int(get_value(config, "singular_order", 4))
    identity_p1_p1 = assemble_l2_identity_matrix(mesh, p1_space, dp0_space, rule, :p1, :p1; symmetry_mode=Symbol(symmetry_mode))
    identity_p1_dp0 = assemble_l2_identity_matrix(mesh, p1_space, dp0_space, rule, :p1, :dp0; symmetry_mode=Symbol(symmetry_mode))
    rho = FloatType(get_value(config, "rho", 1.21))
    sound_speed = FloatType(get_value(config, "sound_speed", 343.0))
    flat_target = Bool(get_value(config, "flat_target_normalization_enabled", true))
    surface_traces_enabled = Bool(get_value(config, "surface_traces_enabled", false))
    flat_target_reference_angle_deg = FloatType(get_value(config, "flat_target_reference_angle_deg", 0.0))
    channel_names = sort(unique([String(get_value(radiator, "channel", "main")) for radiator in radiators]))
    # The fused Burton-Miller path assembles the system matrix and every
    # channel's right-hand side directly, never the four operators. It is
    # exterior-only by construction: the coupled FEM/LEM solver is a separate
    # code path that still needs S, K', D and H individually. Metal's
    # host-staged fallback assembles on the CPU into four operators, and the
    # host singular mode has no fused counterpart, so both stay unfused.
    # The fused path has one regular kernel of its own, so a request for a
    # specific diagnostic kernel mode has to fall back to the four-operator
    # path or the request would be silently ignored.
    singular_cache = build_singular_correction_cache(mesh, singular_order)
    near_selection = near_correction_selection(config, mesh, Symbol(symmetry_mode))
    # The fused Burton-Miller path forms the combination inside the pair kernel
    # and never reaches assemble_regular_galerkin_operators, which is the only
    # place a near-correction cache is applied. It is on by default for :cpu and
    # :metal, so leaving it on here would build every cache, report the pair
    # counts in a status event, and then assemble without them -- a solve that
    # says it was corrected and was not. Near-correction therefore takes the
    # general operators path; it costs the fusion speed-up on those solves only.
    fused_burton_miller = beat_backend in (:cpu, :metal) &&
        near_selection === nothing &&
        get(ENV, "BLAB_BEAT_FUSED_BM", "1") != "0" &&
        (beat_backend != :metal || (metal_assembly_mode != :host_staged &&
            BeatEngineCore._normalized_metal_singular_mode() == :native &&
            BeatEngineCore._normalized_metal_regular_kernel_mode() == :pair_gather))
    near_correction_cache = nothing
    image_near_correction_caches = nothing
    device_near_correction_cache = nothing
    device_image_near_correction_caches = nothing
    if near_selection !== nothing
        if beat_backend == :rocm
            # BeatEngineRocmAssembly has no near-pair kernel upstream; silently
            # assembling without it would report a corrected solve that is not.
            error("Near-singular correction is not implemented for the ROCm backend.")
        end
        if near_selection.identity_pairs !== nothing
            near_correction_cache = build_near_correction_cache(
                mesh, near_selection.identity_pairs, near_selection.top_order,
            )
        end
        image_near_correction_caches = [
            build_near_correction_cache(
                mesh, pairs, near_selection.top_order; trial_transform=transform,
            )
            for (transform, pairs) in near_selection.image_selections
        ]
        emit_event(
            "status";
            message=@sprintf(
                "Near-singular correction within %.2f element radii: %d self-domain pair(s), %d mirror-image pair(s) across %d transform(s)",
                near_selection.cutoff,
                near_correction_cache === nothing ? 0 : near_correction_cache.pair_count,
                sum(cache.pair_count for cache in image_near_correction_caches; init=0),
                length(image_near_correction_caches),
            ),
        )
    end
    device_cache = nothing
    device_singular_cache = nothing
    device_image_singular_cache = nothing
    metal_fused_identity_cache = nothing
    cuda_solve_identity_cache = nothing
    rocm_solve_identity_cache = nothing
    field_cache = cpu_field_cache
    regular_rule_cache = Dict{Int,Any}(base_regular_order => rule)
    identity_cache = Dict{Int,Any}(base_regular_order => (identity_p1_p1, identity_p1_dp0))
    cpu_field_cache_by_order = Dict{Int,Any}(base_regular_order => cpu_field_cache)
    cpu_assembly_cache_by_order = Dict{Int,Any}()
    if beat_backend == :cuda
        emit_event("status"; message="Initializing BEAT Engine using CUDA...")
        device_cache = build_cuda_regular_assembly_cache(mesh, rule)
        field_cache = build_cuda_field_evaluation_cache(cpu_field_cache)
        device_singular_cache = BeatEngineCore.build_cuda_singular_correction_cache(singular_cache, p1_space, dp0_space)
        if Symbol(symmetry_mode) != :off
            device_image_singular_cache = build_cuda_image_singular_correction_cache(
                mesh,
                p1_space,
                dp0_space,
                singular_order,
                eachindex(mesh.faces),
                Symbol(symmetry_mode),
            )
        end
        if near_correction_cache !== nothing && near_correction_cache.pair_count > 0
            device_near_correction_cache = build_cuda_near_correction_cache(
                near_correction_cache,
                p1_space,
                dp0_space,
            )
        end
        if image_near_correction_caches !== nothing && !isempty(image_near_correction_caches)
            device_image_near_correction_caches = [
                build_cuda_near_correction_cache(cache, p1_space, dp0_space)
                for cache in image_near_correction_caches
            ]
        end
        cuda_solve_identity_cache = build_cuda_burton_miller_identity_cache(identity_p1_p1, identity_p1_dp0, FloatType)
    elseif beat_backend == :rocm
        emit_event(
            "status";
            message=rocm_assembly_mode == :host_staged ?
                "Initializing BEAT Engine using ROCm (host-staged assembly fallback)..." :
                "Initializing BEAT Engine using native ROCm operator assembly and GPU solve...",
        )
        device_cache = build_rocm_regular_assembly_cache(
            mesh,
            p1_space,
            dp0_space,
            rule;
            singular_order=singular_order,
            assembly_mode=rocm_assembly_mode,
            symmetry_mode=Symbol(symmetry_mode),
        )
        field_cache = build_rocm_field_evaluation_cache(cpu_field_cache)
        if rocm_assembly_mode != :host_staged
            device_singular_cache = build_rocm_singular_correction_cache(singular_cache)
        end
        rocm_solve_identity_cache = build_rocm_burton_miller_identity_cache(
            identity_p1_p1,
            identity_p1_dp0,
            FloatType,
        )
    elseif beat_backend == :metal
        emit_event(
            "status";
            message=metal_assembly_mode == :host_staged ?
                "Initializing BEAT Engine using Metal (host-staged assembly fallback)..." :
                "Initializing BEAT Engine using native Metal operator assembly and CPU dense solve...",
        )
        device_cache = build_metal_regular_assembly_cache(
            mesh,
            p1_space,
            dp0_space,
            rule;
            singular_order=singular_order,
            assembly_mode=metal_assembly_mode,
            symmetry_mode=Symbol(symmetry_mode),
        )
        field_cache = build_metal_field_evaluation_cache(cpu_field_cache)
        if metal_assembly_mode != :host_staged
            device_singular_cache = build_metal_singular_correction_cache(singular_cache)
        end
        if fused_burton_miller
            metal_fused_identity_cache = build_metal_fused_identity_cache(identity_p1_p1, identity_p1_dp0, FloatType)
        end
    else
        emit_event(
            "status";
            message=@sprintf(
                "Initializing BEAT Engine using CPU with %d BLAS thread%s...",
                cpu_blas_threads,
                cpu_blas_threads == 1 ? "" : "s",
            ),
        )
    end

    # Metal pipelining: the GPU assembles frequency i+1 on a worker task while
    # the CPU factors and solves frequency i, the overlap hornlab-metal-bem
    # relies on. Two operator sets are then resident at once. Requires a
    # second Julia thread; with one thread the sweep stays sequential.
    metal_pipeline = beat_backend == :metal && Threads.nthreads() > 1 &&
        get(ENV, "BLAB_METAL_PIPELINE", "1") != "0"
    assemble_for_frequency = function (k_value)
        started = time()
        if fused_burton_miller
            q_columns = channel_neumann_columns(
                mesh, element_mesh_ids, radiators, channel_names, rho, k_value * sound_speed, FloatType;
                source_motion=source_motion,
            )
            system = assemble_burton_miller_neumann_system_metal(
                mesh,
                p1_space,
                dp0_space,
                q_columns,
                k_value,
                rule;
                device_cache=device_cache,
                singular_cache=singular_cache,
                device_singular_cache=device_singular_cache,
                identity_cache=metal_fused_identity_cache,
                singular_order=singular_order,
                symmetry_mode=Symbol(symmetry_mode),
            )
            return ((kind=:fused, system=system, q_columns=q_columns), time() - started)
        end
        assembled = assemble_regular_galerkin_operators(
            mesh,
            p1_space,
            dp0_space,
            k_value,
            rule;
            skip_singular=false,
            singular_order=singular_order,
            backend=beat_backend,
            device_cache=device_cache,
            return_device=true,
            accelerator_quadrature=true,
            singular_cache=singular_cache,
            device_singular_cache=device_singular_cache,
            metal_assembly_mode=metal_assembly_mode,
            symmetry_mode=Symbol(symmetry_mode),
        )
        return ((kind=:operators, operators=assembled), time() - started)
    end
    pending_assembly = nothing
    if metal_pipeline && !isempty(frequencies)
        first_k = FloatType(2pi) * FloatType(frequencies[1]) / sound_speed
        pending_assembly = Threads.@spawn assemble_for_frequency(first_k)
    end

    try
        for (index, freq_raw) in enumerate(frequencies)
            if cancel_path !== nothing && isfile(String(cancel_path))
                pending_assembly === nothing || release_assembly_payload!(fetch(pending_assembly)[1])
                emit_event("cancelled"; solved_count=index - 1)
                return
            end

        freq = FloatType(freq_raw)
        omega = FloatType(2pi) * freq
        k = omega / sound_speed
        quadrature_selection = regular_quadrature_selection(config, mesh, freq, sound_speed, base_regular_order, regular_quadrature_mode)
        selected_rule = if beat_backend == :cpu
            get!(regular_rule_cache, quadrature_selection.order) do
                triangle_rule(FloatType, quadrature_selection.order)
            end
        else
            rule
        end
        selected_identity_p1_p1 = identity_p1_p1
        selected_identity_p1_dp0 = identity_p1_dp0
        selected_field_cache = field_cache
        selected_cpu_assembly_cache = nothing
        if beat_backend == :cpu
            selected_identity = get!(identity_cache, quadrature_selection.order) do
                (
                    assemble_l2_identity_matrix(mesh, p1_space, dp0_space, selected_rule, :p1, :p1; symmetry_mode=Symbol(symmetry_mode)),
                    assemble_l2_identity_matrix(mesh, p1_space, dp0_space, selected_rule, :p1, :dp0; symmetry_mode=Symbol(symmetry_mode)),
                )
            end
            selected_identity_p1_p1 = selected_identity[1]
            selected_identity_p1_dp0 = selected_identity[2]
            selected_field_cache = get!(cpu_field_cache_by_order, quadrature_selection.order) do
                build_field_evaluation_cache(mesh, selected_rule; symmetry_mode=Symbol(symmetry_mode))
            end
            selected_cpu_assembly_cache = get!(cpu_assembly_cache_by_order, quadrature_selection.order) do
                build_beat_cpu_assembly_cache(
                    mesh,
                    p1_space,
                    dp0_space,
                    selected_rule;
                    singular_order=singular_order,
                    symmetry_mode=Symbol(symmetry_mode),
                )
            end
        end

        pipelined_assembly_seconds = 0.0
        t_assembly = @elapsed begin
            if metal_pipeline
                # Collect this frequency's operators from the worker task and
                # immediately queue the next frequency's assembly behind it.
                # The reported assembly time is the worker's own, even though
                # it overlapped the previous frequency's solve.
                assembly_payload, pipelined_assembly_seconds = fetch(pending_assembly)
                pending_assembly = nothing
                if index < length(frequencies)
                    next_k = FloatType(2pi) * FloatType(frequencies[index + 1]) / sound_speed
                    pending_assembly = Threads.@spawn assemble_for_frequency(next_k)
                end
            elseif fused_burton_miller
                fused_q_columns = channel_neumann_columns(
                    mesh, element_mesh_ids, radiators, channel_names, rho, omega, FloatType;
                    source_motion=source_motion,
                )
                fused_system = beat_backend == :metal ?
                    assemble_burton_miller_neumann_system_metal(
                        mesh, p1_space, dp0_space, fused_q_columns, k, selected_rule;
                        device_cache=device_cache,
                        singular_cache=singular_cache,
                        device_singular_cache=device_singular_cache,
                        identity_cache=metal_fused_identity_cache,
                        singular_order=singular_order,
                        symmetry_mode=Symbol(symmetry_mode),
                    ) :
                    assemble_burton_miller_neumann_system_cpu(
                        mesh, p1_space, dp0_space, fused_q_columns, k, selected_rule;
                        identity_p1_p1=selected_identity_p1_p1,
                        identity_p1_dp0=selected_identity_p1_dp0,
                        skip_singular=false,
                        singular_order=singular_order,
                        singular_cache=singular_cache,
                        cpu_cache=selected_cpu_assembly_cache,
                        symmetry_mode=Symbol(symmetry_mode),
                    )
                assembly_payload = (kind=:fused, system=fused_system, q_columns=fused_q_columns)
            else
                assembly_payload = (kind=:operators, operators=assemble_regular_galerkin_operators(
                    mesh,
                    p1_space,
                    dp0_space,
                    k,
                    selected_rule;
                    skip_singular=false,
                    singular_order=singular_order,
                    backend=beat_backend,
                    device_cache=device_cache,
                    return_device=beat_backend != :cpu,
                    accelerator_quadrature=beat_backend != :cpu,
                    singular_cache=singular_cache,
                    cpu_cache=selected_cpu_assembly_cache,
                    device_singular_cache=device_singular_cache,
                    device_image_singular_cache=device_image_singular_cache,
                    near_correction_cache=near_correction_cache,
                    device_near_correction_cache=device_near_correction_cache,
                    image_near_correction_cache=image_near_correction_caches,
                    device_image_near_correction_cache=device_image_near_correction_caches,
                    rocm_assembly_mode=rocm_assembly_mode,
                    metal_assembly_mode=metal_assembly_mode,
                    symmetry_mode=Symbol(symmetry_mode),
                ))
            end        end
        metal_pipeline && (t_assembly = pipelined_assembly_seconds)
        operators = get(assembly_payload, :operators, nothing)

        t_solve = 0.0
        t_field = 0.0
        cpu_solve_system = nothing
        fused_pressure = nothing
        dense_solve_report = nothing
        if assembly_payload.kind === :fused
            # Dense LU or diagonally preconditioned GMRES, chosen per solve by
            # a cost model over (dofs, drives). The LU route keeps the property
            # the four-operator path has: one factorization per frequency,
            # every channel solved against it. GMRES has none to share and pays
            # per drive, which is why the router weighs both dimensions.
            t_solve += @elapsed begin
                fused_pressure, dense_solve_report = beat_backend == :metal ?
                    solve_metal_burton_miller_system_with_report(assembly_payload.system) :
                    solve_burton_miller_neumann_system_cpu_with_report(assembly_payload.system)
            end
        elseif beat_backend == :cpu
            t_solve += @elapsed begin
                cpu_solve_system = build_burton_miller_neumann_cpu_system(operators, selected_identity_p1_p1, selected_identity_p1_dp0, k)
            end
        elseif beat_backend == :metal
            # Metal assembles on the GPU and factors on the CPU: unified
            # memory lets the host wrap the operators in place, and one
            # factorization is reused across every channel drive like the CPU
            # backend.
            t_solve += @elapsed begin
                operators = metal_host_operators(operators)
                cpu_solve_system = build_burton_miller_neumann_cpu_system(operators, selected_identity_p1_p1, selected_identity_p1_dp0, k)
            end
        end
        channel_boundary_pressures = Vector{Vector{Complex{FloatType}}}()
        channel_boundary_neumann = Vector{Vector{Complex{FloatType}}}()
        horizontal_pressure_rows = Vector{Vector{Complex{FloatType}}}()
        vertical_pressure_rows = Vector{Vector{Complex{FloatType}}}()
        diagonal_pressure_rows = diagonal_points === nothing ? nothing : Vector{Vector{Complex{FloatType}}}()
        sphere_pressure_rows = sphere === nothing ? nothing : Vector{Vector{Complex{FloatType}}}()

        cut_points = diagonal_points === nothing ?
            vcat(horizontal_points, vertical_points) :
            vcat(horizontal_points, vertical_points, diagonal_points)
        combined_points = sphere === nothing ? cut_points : vcat(cut_points, sphere.points)
        horizontal_count = length(horizontal_points)
        vertical_count = length(vertical_points)
        diagonal_count = diagonal_points === nothing ? 0 : length(diagonal_points)

        for (channel_index, channel_name) in enumerate(channel_names)
            pressure = nothing
            q_neumann = nothing
            if assembly_payload.kind === :fused
                pressure = Complex{FloatType}.(fused_pressure[:, channel_index])
                q_neumann = assembly_payload.q_columns[:, channel_index]
            else
                unit_drives = channel_unit_drives(radiators, channel_name, FloatType)
                t_solve += @elapsed begin
                    pressure, q_neumann = pressure_for_drives(
                        mesh,
                        element_mesh_ids,
                        operators,
                        selected_identity_p1_p1,
                        selected_identity_p1_dp0,
                        radiators,
                        unit_drives,
                        rho,
                        omega,
                        k,
                        cpu_solve_system=cpu_solve_system,
                        cuda_solve_identity_cache=cuda_solve_identity_cache,
                        rocm_solve_identity_cache=rocm_solve_identity_cache,
                        source_motion=source_motion,
                    )
                end
            end
            t_field += @elapsed begin
                combined_pressure = field_for_points(combined_points, mesh, pressure, q_neumann, k, selected_field_cache, beat_backend)
                push!(horizontal_pressure_rows, Complex{FloatType}.(combined_pressure[1:horizontal_count]))
                push!(vertical_pressure_rows, Complex{FloatType}.(combined_pressure[(horizontal_count + 1):(horizontal_count + vertical_count)]))
                if diagonal_pressure_rows !== nothing
                    diagonal_start = horizontal_count + vertical_count + 1
                    push!(diagonal_pressure_rows, Complex{FloatType}.(combined_pressure[diagonal_start:(diagonal_start + diagonal_count - 1)]))
                end
                if sphere !== nothing
                    sphere_start = horizontal_count + vertical_count + diagonal_count + 1
                    push!(sphere_pressure_rows, Complex{FloatType}.(combined_pressure[sphere_start:end]))
                end
                push!(channel_boundary_pressures, Complex{FloatType}.(pressure))
                push!(channel_boundary_neumann, Complex{FloatType}.(q_neumann))
            end
        end

        horizontal_spl = Float32[]
        vertical_spl = Float32[]
        horizontal_norm = Float32[]
        vertical_norm = Float32[]
        impedance = Vector{Vector{Float32}}()
        sphere_norm = nothing
        synthesis = nothing
        drives = Complex{FloatType}[]
        mixed_boundary_pressure = zeros(Complex{FloatType}, length(mesh.vertices))
        mixed_boundary_neumann = zeros(Complex{FloatType}, length(mesh.faces))
        t_field += @elapsed begin
            synthesis = synthesize_channel_basis(
                channel_names,
                horizontal_pressure_rows,
                vertical_pressure_rows,
                sphere_pressure_rows,
                channels,
                freq,
                polar_angles_deg,
                flat_target_reference_angle_deg,
                flat_target,
                FloatType;
                diagonal_pressure_rows=diagonal_pressure_rows,
            )
            horizontal_spl = synthesis.horizontal_spl
            vertical_spl = synthesis.vertical_spl
            horizontal_norm = synthesis.horizontal_norm
            vertical_norm = synthesis.vertical_norm
            sphere_norm = synthesis.sphere_norm
            drives = radiator_drives_from_channel_basis(radiators, channels, freq, synthesis.corrections, FloatType)
            for channel_index in eachindex(channel_names)
                mixed_boundary_pressure .+= channel_boundary_pressures[channel_index] .* synthesis.weights[channel_index]
                mixed_boundary_neumann .+= channel_boundary_neumann[channel_index] .* synthesis.weights[channel_index]
            end
            impedance = impedance_for_radiators(mesh, element_mesh_ids, mixed_boundary_pressure, radiators, drives, FloatType; symmetry_mode=Symbol(symmetry_mode))
        end

        release_assembly_payload!(assembly_payload)
        emit_event(
            "result";
            solved_count=index,
            total_count=length(frequencies),
            result=Dict(
                "freq_hz" => Float32(freq),
                "horizontal_spl_norm_db" => horizontal_norm,
                "vertical_spl_norm_db" => vertical_norm,
                "impedance" => impedance,
                "horizontal_spl_db" => horizontal_spl,
                "vertical_spl_db" => vertical_spl,
                "sphere_spl_norm_db" => sphere_norm,
                "channel_names" => channel_names,
                "horizontal_pressure" => complex_rows_to_wire(horizontal_pressure_rows),
                "vertical_pressure" => complex_rows_to_wire(vertical_pressure_rows),
                "diagonal_pressure" => diagonal_pressure_rows === nothing ? nothing : complex_rows_to_wire(diagonal_pressure_rows),
                "diagonal_spl_db" => synthesis.diagonal_spl,
                "diagonal_spl_norm_db" => synthesis.diagonal_norm,
                "sphere_pressure" => sphere_pressure_rows === nothing ? nothing : complex_rows_to_wire(sphere_pressure_rows),
                "surface_pressure" => surface_traces_enabled ? complex_vector_to_wire(mixed_boundary_pressure) : nothing,
                "surface_neumann" => surface_traces_enabled ? complex_vector_to_wire(mixed_boundary_neumann) : nothing,
                "timings" => Dict(
                    "assembly_s" => Float32(t_assembly),
                    "solve_s" => Float32(t_solve),
                    "field_s" => Float32(t_field),
                ),
                "diagnostics" => Dict(
                    "convergence_info" => dense_solve_report === nothing ? 0 :
                        (dense_solve_report.fell_back ? 1 : 0),
                    "message" => dense_solve_report === nothing ? "Julia direct dense solve" :
                        describe_dense_solve(dense_solve_report),
                    "dense_solve_method" => dense_solve_report === nothing ? "lu" :
                        String(dense_solve_report.method),
                    "dense_solve_selection" => dense_solve_report === nothing ? "fixed" :
                        String(dense_solve_report.plan.reason),
                    "dense_solve_fell_back" => dense_solve_report === nothing ? false :
                        dense_solve_report.fell_back,
                    "dense_solve_iterations" => dense_solve_report === nothing ? Int[] :
                        dense_solve_report.iterations,
                    "dense_solve_relative_residuals" => dense_solve_report === nothing ? Float32[] :
                        Float32.(dense_solve_report.relative_residuals),
                    "dense_solve_model_lu_s" => dense_solve_report === nothing ? nothing :
                        Float32(dense_solve_report.plan.lu_model_seconds),
                    "dense_solve_model_gmres_s" => dense_solve_report === nothing ? nothing :
                        Float32(dense_solve_report.plan.gmres_model_seconds),
                    "backend" => String(beat_backend),
                    "symmetry" => symmetry_mode,
                    "regular_assembly_mode" => string(assembly_payload.kind === :fused ?
                        assembly_payload.system.assembly_mode :
                        get(operators, :regular_assembly_mode, beat_backend == :cuda ? :serial_pair_batched : Symbol("$(beat_backend)_default"))),
                    "blas_threads" => cpu_blas_threads,
                    "regular_quadrature_mode" => regular_quadrature_mode,
                    "regular_quadrature_order" => quadrature_selection.order,
                    "regular_quadrature_base_order" => base_regular_order,
                    "regular_quadrature_wavelength_mesh_stat" => quadrature_selection.mesh_stat,
                    "regular_quadrature_wavelength_mesh_area_stat_m2" => quadrature_selection.mesh_area_stat,
                    "regular_quadrature_wavelength_element_length_m" => quadrature_selection.element_length_m,
                    "regular_quadrature_wavelength_kh" => quadrature_selection.kh,
                    "regular_quadrature_wavelength_kh_q1_max" => quadrature_selection.q1_cutoff,
                    "regular_quadrature_wavelength_kh_q2_max" => quadrature_selection.q2_cutoff,
                ),
            ),
        )
        end
    finally
        if pending_assembly !== nothing
            try
                release_assembly_payload!(fetch(pending_assembly)[1])
            catch
            end
        end
        if cuda_solve_identity_cache !== nothing
            release_cuda_burton_miller_identity_cache!(cuda_solve_identity_cache)
        end
        if rocm_solve_identity_cache !== nothing
            release_rocm_burton_miller_identity_cache!(rocm_solve_identity_cache)
        end
        if beat_backend == :rocm
            release_rocm_field_evaluation_cache!(field_cache)
        end
        if beat_backend == :rocm && device_singular_cache !== nothing
            release_rocm_singular_correction_cache!(device_singular_cache)
        end
        if beat_backend == :rocm && device_cache !== nothing
            release_rocm_regular_assembly_cache!(device_cache)
        end
        if beat_backend == :metal
            release_metal_field_evaluation_cache!(field_cache)
        end
        if beat_backend == :metal && device_singular_cache !== nothing
            release_metal_singular_correction_cache!(device_singular_cache)
        end
        if beat_backend == :metal && metal_fused_identity_cache !== nothing
            release_metal_fused_identity_cache!(metal_fused_identity_cache)
        end
        if beat_backend == :metal && device_cache !== nothing
            release_metal_regular_assembly_cache!(device_cache)
        end
        if device_image_singular_cache !== nothing
            release_cuda_image_singular_correction_cache!(device_image_singular_cache)
        end
        # Same device-array layout as the image-singular cache upstream.
        device_near_correction_cache === nothing ||
            release_cuda_image_singular_correction_cache!(device_near_correction_cache)
        for cache in something(device_image_near_correction_caches, ())
            release_cuda_image_singular_correction_cache!(cache)
        end
    end

    emit_event("completed"; solved_count=length(frequencies))
end

function worker_loop()
    emit_event("ready"; protocol="boundary_lab_julia_worker", pid=getpid())
    for line in eachline(stdin)
        text = strip(line)
        isempty(text) && continue
        try
            message = JSON.parse(text)
            request_path = String(message["request"])
            request = JSON.parsefile(request_path)
            solve_request(request)
        catch exc
            emit_event("failed"; error=sprint(showerror, exc))
        end
    end
end

"""
    main(args = ARGS)

Run one request, or serve the worker protocol on stdin until it closes.

BLAS threading is configured here rather than at load: when this file is
precompiled into a bundle, load time is the *build* machine's, and the thread
count has to be the running machine's.
"""
function main(args = ARGS)
    LinearAlgebra.BLAS.set_num_threads(Threads.nthreads())
    try
        request_path, worker_mode = parse_args(args)
        if worker_mode
            worker_loop()
        else
            isnothing(request_path) && fail!("Missing --request path.")
            request = JSON.parsefile(request_path)
            solve_request(request)
        end
    catch exc
        fail!(sprint(showerror, exc))
    end
end

