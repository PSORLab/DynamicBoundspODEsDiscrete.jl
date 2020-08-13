"""
$(TYPEDEF)

A function g!(out, y) that perfoms a Taylor coefficient calculation. Provides
preallocated storage. Evaluating this function out is a vector of length nx*(s+1)
where 1:(s+1) are the Taylor coefficients of the first component, (s+2):nx*(s+1)
are the Taylor coefficients of the second component, and so on.

$(TYPEDFIELDS)
"""
mutable struct TaylorFunctor!{F <: Function, N, T <: Real, S <: Real}
    "Right-hand side function for pODE which operates in place as g!(dx,x,p,t)"
    g!::F
    "Dimensionality of x"
    nx::Int64
    "Dimensionality of p"
    np::Int64
    "Order of TaylorSeries, that is the first k terms are used in the approximation
    and N = k+1 term is bounded"
    k::Int64
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
    vnxt::Vector{Int64}
    fnxt::Vector{Float64}
end

"""
$(TYPEDSIGNATURES)

Computes the Taylor coefficients at `y` and stores them inplace to `out`.
"""
function (d::TaylorFunctor!{F, K, T, S})(out::Vector{Vector{S}}, x::Vector{S},
          p::Vector{S}, t::T) where {F <: Function, K, T <: Real, S <: Real}

    val = Val(K-1)
    for i in eachindex(d.xtaylor)
        d.xtaylor[i] = STaylor1(x[i], val)
    end

    jetcoeffs!(d.g!, t, d.xtaylor, d.xaux, d.dx, K-1, p, d.vnxt, d.fnxt)::Nothing
    for i in eachindex(out)
        for j in eachindex(d.xtaylor)
            out[i][j] = d.xtaylor[j][i-1]
        end
    end

    return nothing
end

"""
$(TYPEDSIGNATURES)

A constructor for `TaylorFunctor` that preallocates storage for computing
interval extensions of Taylor coefficients.
"""
function TaylorFunctor!(g!, nx::Int, np::Int, k::Val{K}, t::T, q::Q) where {K, T <: Number, Q <: Number}

    x0 = zeros(T, nx)
    Vⱼ = zeros(T, nx)
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
        push!(taux, zero(STaylor1{K,Q}))
    end
    #push!(xtaylor, fill(temp, nx))
    x = zeros(T, nx)
    p = zeros(T, np)
    vnxt = zeros(Int, nx)
    fnxt = zeros(Float64, nx)

    return TaylorFunctor!{typeof(g!), K+1, Q, T}(g!, nx, np, K, x, p, xtaylor,
                                                 xaux, dx, taux, vnxt, fnxt)
end
