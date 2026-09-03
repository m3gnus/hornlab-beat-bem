# Fail the macOS job when the runner cannot actually run a Metal kernel.
#
# The three validators already call `Metal.functional() || error(...)`, so a
# missing device cannot make them pass silently -- unlike metal-bem, whose GPU
# tests skipped themselves and whose badge covered no Metal at all for months.
# This runs first anyway, for two reasons. It makes the loss one legible error
# instead of three identical ones a few minutes apart, and `functional()` is a
# weaker claim than it looks: it says a device and a queue exist, not that a
# kernel compiles and returns the right answer. A paravirtualised GPU can
# satisfy the first and fail the second.
#
# So compile and dispatch a real kernel and check its result. The check is an
# exact integer comparison only because this kernel is exact -- one write per
# thread, no reduction, no floating point. Nothing in CI should compare GPU
# floating-point output byte for byte: the native Metal assembly path
# accumulates through atomics and is not bit-reproducible run to run (~8e-7).
# That is why the validators compare against the CPU reference under a
# tolerance, and why those tolerances are noise floors rather than slack.

using Metal

if !Metal.functional()
    println(stderr, """
        Metal is not functional on a runner that is supposed to cover the GPU path.

        The three validators would all die on the same line a few minutes from now,
        so this fails first. If the hosted image genuinely no longer exposes a Metal
        device, that is a real loss of coverage: move the Metal job to a self-hosted
        Apple Silicon runner rather than deleting it.
        """)
    exit(1)
end

device = Metal.device()
println("Metal device: ", device)

function double!(output, input)
    index = thread_position_in_grid_1d()
    @inbounds output[index] = 2 * input[index]
    return nothing
end

input = MtlArray(Int32.(1:256))
output = Metal.zeros(Int32, 256)
Metal.@sync @metal threads = 256 double!(output, input)

result = Array(output)
if result != 2 .* Int32.(1:256)
    println(stderr, """
        A Metal device is present but a trivial kernel returned the wrong answer.
        Metal.functional() was true, so nothing downstream would have caught this
        as anything other than a numerical failure in the solver.
        """)
    exit(1)
end

println("A Metal kernel compiled, dispatched and returned the expected result.")
