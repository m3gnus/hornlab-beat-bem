# Run the Julia CPU suite under one aggregate testset, and refuse a vacuous
# green.
#
# `runtests.jl` is a sequence of top-level testsets, so running it directly
# gives an exit code and thirty separate summaries but no total. This wraps it
# in one enclosing testset -- `include` evaluates in the caller's module at
# global scope while the testset stack is dynamic, so every nested testset
# still attaches here -- and then walks the tree for counts.
#
# The vacuity floor is the point. A suite that stops running is the failure
# this repository is most exposed to, because two of its testsets are gated on
# accelerator hardware (`cuda_available()`, `rocm_available()`) and are
# expected to be absent on a hosted runner. Without a floor, "expected to be
# absent" is indistinguishable from "everything went missing".
#
# What this deliberately does NOT do: fail on a warning. `runtests.jl` emits a
# documented `@warn` when unreorthogonalised Float32 modified Gram-Schmidt does
# not degrade on the host -- see AGENTS.md, "One check is deliberately
# conditional". Whether it degrades is a property of the host's floating point
# (20x on an M1 Max, reportedly not at all on a Ryzen 7 5825U), so a CI job
# that turned that warning into a failure would be reporting a
# microarchitecture rather than a defect, and would go red on exactly the
# hardware the warning was written for. It stays a warning here. Do not add
# `--depwarn=error`, and do not grep this log for "Warning".

using Test

const SUITE = joinpath(@__DIR__, "..", "..", "hornlab_beat_bem", "julia", "tests", "runtests.jl")

const MINIMUM_PASSES = let
    argument = findfirst(startswith("--min-passes="), ARGS)
    argument === nothing ? 300 : parse(Int, split(ARGS[argument], "=")[2])
end

"""Sum outcomes over a testset tree.

`DefaultTestSet` discards individual `Pass` records and keeps only a count, so
passes come from `n_passed` while the other three are still present as
records.
"""
function tally(testset::Test.DefaultTestSet)
    passes = testset.n_passed
    fails = errors = broken = 0
    for result in testset.results
        if result isa Test.DefaultTestSet
            child = tally(result)
            passes += child.passes
            fails += child.fails
            errors += child.errors
            broken += child.broken
        elseif result isa Test.Fail
            fails += 1
        elseif result isa Test.Error
            errors += 1
        elseif result isa Test.Broken
            broken += 1
        end
    end
    return (; passes, fails, errors, broken)
end

# `@testset` throws a `TestSetException` on any failure, which is the exit
# code. The counts below are for the log, and for the floor.
counts = try
    testset = @testset "hornlab-beat-bem Julia CPU suite" verbose = true begin
        include(SUITE)
    end
    tally(testset)
catch exception
    exception isa Test.TestSetException || rethrow()
    println(stderr, "\nThe Julia CPU suite failed.")
    exit(1)
end

println()
println("julia: $(counts.passes) passed, $(counts.fails) failed, " *
        "$(counts.errors) errored, $(counts.broken) broken")

if counts.passes < MINIMUM_PASSES
    println(stderr,
            "\nOnly $(counts.passes) assertions passed, expected at least " *
            "$(MINIMUM_PASSES).\n" *
            "A suite that shrinks this far has stopped running rather than " *
            "started agreeing;\nraise the floor deliberately if the suite " *
            "genuinely got smaller.")
    exit(1)
end
