# Compile-only coverage for the package's SOURCE worker host inventory.
# Run under julia_metal; no device arrays are constructed or kernels launched.
using Test
using BeatEngineMetalBundle

@testset "Metal SOURCE host precompile signatures still match" begin
    signatures = BeatEngineMetalBundle.metal_host_signatures()
    @test !isempty(signatures)
    @test length(Set(signatures)) == length(signatures)
    for (index, signature) in enumerate(signatures)
        @testset "host signature $index" begin
            @test precompile(signature)
        end
    end
    @info "BEAT Metal SOURCE host inventory checked" signatures=length(signatures) julia_version=VERSION
end
