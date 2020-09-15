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
# src/DiscretizeRelax/utilities/fast_set_index.jl
# Defines a method for setindex that avoids conversion.
#############################################################################

# setindex! that avoids conversion.
function setindex!(x::Vector{STaylor1{N,T}}, val::T, i::Int) where {N,T}
    @inbounds x[i] = STaylor1(val, Val{N-1}())
    return nothing
end
