#!/usr/bin/env julia
# Compatibility wrapper for the project layout.
include(joinpath(@__DIR__, "src", "ABPNoiseSweep.jl"))
ABPNoiseSweep.main(ARGS)
