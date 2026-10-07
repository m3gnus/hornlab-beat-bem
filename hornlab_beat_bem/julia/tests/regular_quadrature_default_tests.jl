# Exercise the request selectors without running a solve or loading a bundle.
module RegularQuadratureDefaultTests
using Test, StaticArrays, Statistics
include(joinpath(@__DIR__, "..", "src", "BeatEngineCore.jl"))
using .BeatEngineCore
include(joinpath(@__DIR__, "..", "BeatEngineDriver.jl"))

# coupled_solver.jl executes a stdin request at file scope. Load its actual
# exterior selector alone, as the near-correction tests do for driver helpers.
let path = joinpath(@__DIR__, "..", "coupled_solver.jl")
    source = read(path, String)
    first_index = first(findfirst("function exterior_quadrature_selection(", source))
    last_index = first(findfirst("\nfunction exterior_excitations(", source)) - 1
    include_string(@__MODULE__, source[first_index:last_index], path)
end

function graded_mesh(::Type{T}) where {T<:AbstractFloat}
    vertices = SVector{3,T}[]
    faces = NTuple{3,Int}[]
    # Twenty tiny triangles hide one coarse triangle from the p90 statistic.
    for index in 1:21
        side = index == 21 ? T(0.1) : T(0.001)
        offset = T(index)
        first_vertex = length(vertices) + 1
        append!(vertices, [SVector{3,T}(offset, 0, 0),
                           SVector{3,T}(offset + side, 0, 0),
                           SVector{3,T}(offset, side, 0)])
        push!(faces, (first_vertex, first_vertex + 1, first_vertex + 2))
    end
    return BoundaryMesh(vertices, faces, fill(1, length(faces)))
end

@testset "fixed regular quadrature default on graded meshes" begin
    for T in (Float32, Float64)
        mesh = graded_mesh(T)
        frequency, sound_speed = T(8000), T(343)
        options = Dict{String,Any}()  # No regular_quadrature_mode supplied.
        mode = regular_quadrature_mode_from_config(options, :cpu)
        selection = regular_quadrature_selection(options, mesh, frequency, sound_speed, 4, mode)
        @test mode == "fixed"
        @test selection.order == 4
        @test length(triangle_rule(T, selection.order).points) == 6
        @test selection.kh === nothing
        coupled = exterior_quadrature_selection(options, mesh, frequency, sound_speed, 4, :cpu)
        @test coupled.mode == "fixed"
        @test coupled.order == 4
        @test coupled.kh === nothing

        # An explicit base order still wins in fixed mode.
        @test regular_quadrature_selection(options, mesh, frequency, sound_speed, 2, mode).order == 2
        @test exterior_quadrature_selection(options, mesh, frequency, sound_speed, 2, :cpu).order == 2

        # Keep the former default as opt-in, and demonstrate the regression's
        # trigger: p90 selects order 2 despite the coarse element's kh > 2.
        wavelength = Dict{String,Any}("regular_quadrature_mode" => "wavelength")
        explicit_mode = regular_quadrature_mode_from_config(wavelength, :cpu)
        explicit = regular_quadrature_selection(wavelength, mesh, frequency, sound_speed, 4, explicit_mode)
        @test explicit_mode == "wavelength"
        @test explicit.order == 2
        @test explicit.kh < 2
        @test 2pi * frequency / sound_speed * sqrt(maximum(mesh.areas)) > 2
        coupled_explicit = exterior_quadrature_selection(wavelength, mesh, frequency, sound_speed, 4, :cpu)
        @test coupled_explicit.mode == "wavelength"
        @test coupled_explicit.order == 2
        @test coupled_explicit.kh ≈ explicit.kh
        @test regular_quadrature_mode_from_config(Dict("quadrature_mode" => "wavelength"), :cpu) == "wavelength"

        for backend in (:cuda, :rocm, :metal)
            @test regular_quadrature_mode_from_config(options, backend) == "fixed"
            @test exterior_quadrature_selection(options, mesh, frequency, sound_speed, 4, backend).order == 4
            @test_throws ErrorException regular_quadrature_mode_from_config(wavelength, backend)
            @test_throws ErrorException exterior_quadrature_selection(wavelength, mesh, frequency, sound_speed, 4, backend)
        end
    end
end
end
