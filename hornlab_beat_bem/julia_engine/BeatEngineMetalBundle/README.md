# Metal source-entry precompile coverage

This package loads the engine and `julia/BeatEngineDriver.jl` into a cached
Julia package image. Its workload runs parsed CPU requests to exercise the
host driver, including mesh lists, xy images and optional observation outputs.
It also compiles the existing Metal device inventory without dispatching it.

`src/MetalHostPrecompile.jl` adds compile-only host signatures traced from this
package's own first requests: SOURCE solves on two xy meshes, a full-domain
warmup mesh and the package's tiny warmup. Each explicit signature occurred in at least two traces. The
method comes from official engine PR #22 (`b87f32e`); that engine's compiled
system inventory is a different driver and is not copied here.

Generated callable types are found by captured fields, keyword method bodies
or device argument shapes, rather than their generated names. The host test
checks every signature under both qualified Julia patch releases. This does
not change CPU, CUDA or ROCm bundle workloads, request decisions or numerical
operators. Julia 1.12.7 already stores the device code in the package image;
this additional coverage addresses the host's first-request compilation.

Run `julia/tests/metal_host_precompile_tests.jl` under the Metal project to
check the host inventory without executing kernels. The separate
`metal_kernel_coverage_tests.jl` requires a working Metal device and checks the
production device inventory. After changes to engine or dependency internals,
retrace fresh workers and review the inventory instead of guessing signatures.
