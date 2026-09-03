# Measure the condensed-assembly intermittent instead of asserting it.
#
# `coupled_condensed_tests.jl` lines 592 and 618 assert bitwise equality between
# the forked condensed assembly and the shared CPU assembly at `symmetry == :off`,
# and fail intermittently on AVX-512 hosts by one Float32 ULP. A pass/fail is a
# thin signal for an intermittent: it cannot say which of the four operators
# disagreed, how many entries did, or by how much. This reproduces the same
# comparison and reports those numbers.
#
# It deliberately does not assert. The suite is the experiment's measurement --
# run it the way CI runs it -- and this is the diagnostic beside it, so it must
# not fail a job that is collecting a control arm. Exit status is always 0.
#
# Note the probe may well agree where the suite disagrees. The leading hypothesis
# is allocation-dependent code generation, and a process that runs this
# comparison alone reaches it with a different heap and a different set of
# already-compiled methods than one that has run thirty testsets first. A quiet
# probe is therefore not evidence of a quiet suite; it is evidence about this
# process.

# Include order mirrors `runtests.jl`: the core module, then `BeatEngineCoupled`
# (which `coupled_solver_tests.jl` brings in one file earlier and which
# `BeatEngineCoupledCondensed.jl` expects to already exist), then the fork.
const SOURCE = joinpath(@__DIR__, "..", "..", "hornlab_beat_bem", "julia", "src")

include(joinpath(SOURCE, "BeatEngineCore.jl"))
using .BeatEngineCore
include(joinpath(SOURCE, "BeatEngineCoupled.jl"))
using .BeatEngineCoupled
include(joinpath(SOURCE, "BeatEngineCoupledCondensed.jl"))
using .BeatEngineCoupledCondensed

const TEST_MESHES = joinpath(@__DIR__, "..", "..", "hornlab_beat_bem", "julia", "test_meshes")

"""Map a Float32 onto the integer line that counts representable values.

Adjacent floats differ by 1, so `|key(a) - key(b)|` is the ULP gap, and -0.0f0
and 0.0f0 share a key rather than reporting a spurious gap of 2^31.
"""
function float_key(value::Float32)
    bits = Int64(reinterpret(Int32, value))
    return bits < 0 ? Int64(-2147483648) - bits : bits
end

ulp_gap(a::Float32, b::Float32) = abs(float_key(a) - float_key(b))

"""Compare two operator matrices entry by entry, in ULPs of their parts."""
function compare(forked::AbstractMatrix, shared::AbstractMatrix)
    differing = 0
    worst = 0
    first_index = nothing
    for index in eachindex(forked, shared)
        left = forked[index]
        right = shared[index]
        left == right && continue
        differing += 1
        gap = max(ulp_gap(real(left), real(right)), ulp_gap(imag(left), imag(right)))
        if gap > worst
            worst = gap
        end
        if first_index === nothing
            first_index = index
        end
    end
    return (; differing, worst, first_index, total=length(forked))
end

function report(label, forked, shared)
    for operator in (:single_layer, :double_layer, :adjoint_double_layer, :hypersingular)
        result = compare(getproperty(forked, operator), getproperty(shared, operator))
        location = result.first_index === nothing ? "-" :
                   string(Tuple(CartesianIndices(getproperty(forked, operator))[result.first_index]))
        println("probe $label $operator differing=$(result.differing)/$(result.total) " *
                "max_ulp=$(result.worst) first=$location")
    end
end

const RULE = triangle_rule(Float32, 2)
const WAVENUMBER = Float32(2pi * 1000.0 / 343.0)

mesh = load_gmsh22_with_tags(joinpath(TEST_MESHES, "sample.msh"), Float32(0.001))
p1 = build_p1_space(mesh)
dp0 = build_dp0_space(mesh)
element_indices = 1:min(24, length(mesh.faces))
singular_cache = build_singular_correction_cache(mesh, 2, element_indices)

shared = assemble_regular_galerkin_operators(
    mesh, p1, dp0, WAVENUMBER, RULE;
    skip_singular=false, singular_order=2, element_indices=element_indices,
    backend=:cpu, singular_cache=singular_cache, symmetry_mode=:off,
)
forked = BeatEngineCoupledCondensed.assemble_condensed_regular_operators(
    mesh, p1, dp0, WAVENUMBER, RULE;
    skip_singular=false, singular_order=2, element_indices=element_indices,
    singular_cache=singular_cache, symmetry_mode=:off,
)

cache = build_beat_cpu_assembly_cache(
    mesh, p1, dp0, RULE;
    singular_order=2, element_indices=element_indices, symmetry_mode=:off,
)
shared_cached = assemble_regular_galerkin_operators(
    mesh, p1, dp0, WAVENUMBER, RULE;
    skip_singular=false, singular_order=2, backend=:cpu,
    singular_cache=singular_cache, cpu_cache=cache, symmetry_mode=:off,
)
forked_cached = BeatEngineCoupledCondensed.assemble_condensed_regular_operators(
    mesh, p1, dp0, WAVENUMBER, RULE;
    skip_singular=false, singular_order=2,
    singular_cache=singular_cache, cpu_cache=cache, symmetry_mode=:off,
)

# Line 592, line 618, and line 626 of the test file, in that order. The third is
# the one the intermittent has never broken, and it is worth keeping in view: if
# the two forked paths agree with each other while both disagree with shared,
# the divergence is in the shared assembly's code generation, not the fork's.
report("uncached", forked, shared)
report("cached", forked_cached, shared_cached)
report("fork-self", forked_cached, forked)

function tally_differences(pairs)
    total = 0
    for (left, right) in pairs
        for operator in (:single_layer, :double_layer, :adjoint_double_layer, :hypersingular)
            total += compare(getproperty(left, operator), getproperty(right, operator)).differing
        end
    end
    return total
end

println("probe_verdict differing_entries=",
        tally_differences(((forked, shared), (forked_cached, shared_cached))))
