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
# src/DiscretizeRelax/utilities/taylor_functor.jl
# Defines methods to used to compute Taylor coeffients.
#############################################################################
"""
TaylorFunctor!

A function g!(out, y) that perfoms a Taylor coefficient calculation. Provides
preallocated storage. Evaluating this function out is a vector of length nx*(s+1)
where 1:(s+1) are the Taylor coefficients of the first component, (s+2):nx*(s+1)
are the Taylor coefficients of the second component, and so on. This may be
constructed using `TaylorFunctor!(g!, nx::Int, np::Int, k::Val{K}, t::T, q::Q)`
were type `T` should use type `Q` for internal computations. The order of the
TaylorSeries is `k`, the right-hand side function is `g!`, `nx` is the number
of state variables, `np` is the number of parameters.

$(TYPEDFIELDS)
"""
mutable struct TaylorFunctor!{F <: Function, N, T <: Real, S <: Real}
    "Right-hand side function for pODE which operates in place as g!(dx,x,p,t)"
    g!::F
    "Dimensionality of x"
    nx::Int
    "Dimensionality of p"
    np::Int
    "Order of TaylorSeries, that is the first k terms are used in the approximation
    and N = k+1 term is bounded"
    k::Int
    "State variables x"
    x::Vector{S}
    "Decision variables p"
    p::Vector{S}
    "Store temporary STaylor1 vector for calculations"
    xtaylor::Vector{STaylor1{N,S}}
    "Store temporary STaylor1 vector for calculations"
    xaux::Vector{STaylor1{N,S}}
    "Store temporary STaylor1 vector for calculations"
    dx::Vector{STaylor1{N,S}}
    taux::Vector{STaylor1{N,T}}
    vnxt::Vector{Int}
    fnxt::Vector{Float64}
end

function TaylorFunctor!(g!, nx::Int, np::Int, k::Val{K}, t::T, q::Q) where {K, T <: Number, Q <: Number}

    f̃ = Vector{T}[]
    for i = 1:(K+1)
        push!(f̃, zeros(T, nx))
    end
    temp = STaylor1(zeros(T,K+1))
    xtaylor = STaylor1{K+1,T}[]
    xaux = STaylor1{K+1,T}[]
    dx = STaylor1{K+1,T}[]
    taux = STaylor1{K+1,Q}[]
    for i = 1:nx
        push!(xtaylor, temp)
        push!(xaux, temp)
        push!(dx, temp)
        push!(taux, zero(STaylor1{K+1,Q}))
    end
    x = zeros(T, nx)
    p = zeros(T, np)
    vnxt = zeros(Int, nx)
    fnxt = zeros(Float64, nx)

    return TaylorFunctor!{typeof(g!), K+1, Q, T}(g!, nx, np, K, x, p, xtaylor,
                                                 xaux, dx, taux, vnxt, fnxt)
end

function (d::TaylorFunctor!{F,K,T,S})(out::Vector{Vector{S}}, x::Vector{S}, p::Vector{S}, t::T) where {F, K, T, S}
    @__dot__ d.xtaylor = STaylor1(x, Val(K-1))
    jetcoeffs!(d.g!, t, d.xtaylor, d.xaux, d.dx, K-1, p, d.vnxt, d.fnxt)::Nothing
    for i = 1:d.nx, j = 1:(d.k + 1)
        out[j][i] = d.xtaylor[i][j - 1]
    end
    nothing
end
