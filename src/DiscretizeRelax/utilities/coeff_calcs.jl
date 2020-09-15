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
# src/DiscretizeRelax/utilities/coeffs_calcs.jl
# Defines functions used to compute taylor coefficients using static
# univariate Taylor series.
#############################################################################

@generated function truncuated_STaylor1(x::STaylor1{N,T}, v::Int) where {N,T}
    ex_calc = quote end
    append!(ex_calc.args, Any[nothing for i in 1:N])
    syms = Symbol[Symbol("c$i") for i in 1:N]
    for i = 1:N
        sym = syms[i]
        ex_line = :($(sym) = $i <= v ? x[$(i-1)] : zero($T))
        ex_calc.args[i] = ex_line
    end
    exout = :(($(syms[1]),))
    for i = 2:N
        push!(exout.args, syms[i])
    end
    return quote
               Base.@_inline_meta
               $ex_calc
               return STaylor1{N,T}($exout)
            end
end

@generated function copy_recurse(dx::STaylor1{N,T}, x::STaylor1{N,T}, ord::Int, ford::Float64) where {N,T}
    ex_calc = quote end
    append!(ex_calc.args, Any[nothing for i in 1:(N+1)])
    syms = Symbol[Symbol("c$i") for i in 1:N]
    ex_line = :(nv = dx[ord-1]/ford) #/ord)
    ex_calc.args[1] = ex_line
    for i = 0:(N-1)
        sym = syms[i+1]
        ex_line = :($(sym) = $i < ord ? x[$i] : (($i == ord) ? nv : zero(T))) # nv))
        ex_calc.args[i+2] = ex_line
    end
    exout = :(($(syms[1]),))
    for i = 2:N
        push!(exout.args, syms[i])
    end
    return quote
               Base.@_inline_meta
               $ex_calc
               return STaylor1{N,T}($exout)
            end
end

"""
$(TYPEDSIGNATURES)

A variant of the jetcoeffs! function used in TaylorIntegration.jl
(https://github.com/PerezHz/TaylorIntegration.jl/blob/master/src/explicitode.jl)
which preallocates taux and updates taux coefficients to avoid further allocations.

The TaylorIntegration.jl package is licensed under the MIT "Expat" License:
Copyright (c) 2016-2020: Jorge A. Perez and Luis Benet. Permission is hereby
granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following
conditions: The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.
"""
function jetcoeffs!(eqsdiff!, t::T, x::Vector{STaylor1{N,U}}, xaux::Vector{STaylor1{N,U}},
                    dx::Vector{STaylor1{N,U}}, order::Int, params,
                    vnxt::Vector{Int}, fnxt::Vector{Float64}) where {N, T<:Number, U<:Number}

      ttaylor = STaylor1(t, Val{N-1}())
      for ord = 1:order

          fill!(vnxt, ord)
          fill!(fnxt, Float64(ord))

          map!(truncuated_STaylor1, xaux, x, vnxt)
          eqsdiff!(dx, xaux, params, ttaylor)::Nothing
          x .= copy_recurse.(dx, x, vnxt, fnxt)
      end
      nothing
end
