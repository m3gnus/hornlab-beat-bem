# Report whether this Julia process JITs AVX-512, and what CPU target produced it.
#
# The masked arm of the AVX-512 experiment is worthless without this. A green
# suite under `-C native,-avx512f` proves nothing unless the mask demonstrably
# changed code generation: "no failure" and "no AVX-512" have to be established
# separately, or the experiment is measuring a flag it never applied.
#
# The assay compiles a Float32 fused-multiply-add reduction -- the shape the
# condensed quadrature loop reduces to -- and counts vector registers in the
# emitted assembly. `zmm` is 512-bit and appears only under AVX-512; `ymm` is
# 256-bit and survives the mask. So the expected signature of a working mask on
# an AVX-512 host is zmm > 0 unmasked, zmm == 0 masked, with ymm > 0 in both.
#
# Prints machine-readable `assay_*` lines and exits 0 always: the caller decides
# what an unexpected count means, because on a host without AVX-512 zmm == 0 in
# both arms and that is a correct null result rather than a broken mask.

using InteractiveUtils

function assay_kernel(x::Vector{Float32}, y::Vector{Float32})
    total = zero(Float32)
    @inbounds @simd for i in eachindex(x)
        total = muladd(x[i], y[i], total)
    end
    return total
end

buffer = IOBuffer()
code_native(buffer, assay_kernel, (Vector{Float32}, Vector{Float32}); debuginfo=:none)
assembly = String(take!(buffer))

count_matches(pattern) = count(_ -> true, eachmatch(pattern, assembly))

# `Base.JLOptions().cpu_target` is the string this process was actually started
# with, not the string the workflow believes it passed.
requested = let pointer = Base.JLOptions().cpu_target
    pointer == C_NULL ? "(default)" : unsafe_string(pointer)
end

println("assay_requested_target ", requested)
println("assay_cpu_name         ", Sys.CPU_NAME)
println("assay_zmm              ", count_matches(r"zmm[0-9]+"))
println("assay_ymm              ", count_matches(r"ymm[0-9]+"))
println("assay_xmm              ", count_matches(r"xmm[0-9]+"))
