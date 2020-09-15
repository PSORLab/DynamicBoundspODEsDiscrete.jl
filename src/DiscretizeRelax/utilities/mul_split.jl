# Copyright (c) 2020: Matthew Wilhelm & Matthew Stuber.
# This work is licensed under the Creative Commons Attribution-NonCommercial-
# ShareAlike 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to Creative
# Commons, PO Box 1866, Mountain View, CA 94042, USA.
#############################################################################
# Dynamic Bounds - pODEs Discrete
# A package for discretize and relax methods for bounding pODEs.
# See https://github.com/PSORLab/DynamicBoundspODEsDiscrete.jl
#############################################################################
# src/DiscretizeRelax/utilities/mul_split.jl
# A simple function used to speed up 1D calculations of matrix multiplication.
#############################################################################

"""
mul_split!

Multiples A*b as a matrix times a vector if nx > 1. Performs scalar
multiplication otherwise.
"""
function mul_split!(Y::Vector{R}, A::Matrix{S}, B::Vector{T}, nx) where {R,S,T}
    if nx == 1
        @inbounds Y[1] = A[1,1]*B[1]
    else
        mul!(Y, A, B)
    end

    return nothing
end

function mul_split!(Y::Matrix{R}, A::Matrix{S}, B::Matrix{T}, nx) where {R,S,T}
    if nx == 1
        @inbounds Y[1,1] = A[1,1]*B[1,1]
    else
        mul!(Y, A, B)
    end

    return nothing
end
