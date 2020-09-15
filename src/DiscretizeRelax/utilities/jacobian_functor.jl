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
# src/DiscretizeRelax/utilities/jacobian_functor.jl
# Defines methods to used to compute Jacobians of Taylor coeffients.
#############################################################################

"""
JacTaylorFunctor!

A callable structure used to evaluate the Jacobian of Taylor cofficients. This
also contains some addition fields to be used as inplace storage when computing
and preconditioning paralleliped based methods to representing enclosure of the
pODEs (Lohner's QR, Hermite-Obreschkoff, etc.). The constructor given by
`JacTaylorFunctor!(g!, nx::Int, np::Int, k::Val{K}, t::T, q::Q)` may be used
were type `T` should use type `Q` for internal computations. The order of the
TaylorSeries is `k`, the right-hand side function is `g!`, `nx` is the number
of state variables, `np` is the number of parameters.

$(TYPEDFIELDS)
"""
mutable struct JacTaylorFunctor!{F <: Function, N, T <: Real, S <: Real, NY}
    "Right-hand side function for pODE which operates in place as g!(dx,x,p,t)"
    g!::F
    "Dimensionality of x"
    nx::Int
    "Dimensionality of p"
    np::Int
    "Order of TaylorSeries"
    s::Int
    "In-place temporary storage for Taylor coefficient calculation"
    out::Vector{S}
    "Variables y = (x,p)"
    y::Vector{S}
    "State variables x"
    x::Vector{Dual{Nothing,S,NY}}
    "Decision variables p"
    p::Vector{Dual{Nothing,S,NY}}
    "Storage for sum of Jacobian w.r.t x"
    Jxsto::Matrix{S}
    "Storage for sum of Jacobian w.r.t p"
    Jpsto::Matrix{S}
    "Temporary for transpose of Jacobian w.r.t y"
    tjac::Matrix{S}
    "Storage for vector of Jacobian w.r.t x"
    Jx::Vector{Matrix{S}}
    "Storage for vector of Jacobian w.r.t p"
    Jp::Vector{Matrix{S}}
    "Jacobian Result from DiffResults"
    result::MutableDiffResult{1, Vector{S}, Tuple{Matrix{S}}}
    "Jacobian Configuration for ForwardDiff"
    cfg::JacobianConfig{Nothing,S,NY,Tuple{Vector{Dual{Nothing,S,NY}},Vector{Dual{Nothing,S,NY}}}}
    "Store temporary STaylor1 vector for calculations"
    xtaylor::Vector{STaylor1{N,Dual{Nothing,S,NY}}}
    "Store temporary STaylor1 vector for calculations"
    xaux::Vector{STaylor1{N,Dual{Nothing,S,NY}}}
    "Store temporary STaylor1 vector for calculations"
    dx::Vector{STaylor1{N,Dual{Nothing,S,NY}}}
    taux::Vector{STaylor1{N,T}}
    t::Float64
    "Intermediate storage to avoid allocations in Taylor coefficient calc"
    vnxt::Vector{Int64}
    "Intermediate storage to avoid allocations in Taylor coefficient calc"
    fnxt::Vector{Float64}
end

function JacTaylorFunctor!(g!, nx::Int, np::Int, k::Val{K}, t::T, q::Q) where {K, T <: Number, Q <: Number}
    x0 = zeros(T, nx)
    xd0 = zeros(Dual{Nothing, T, nx + np}, nx)
    out = zeros(T, nx*(K + 1))
    y = zeros(T, nx + np)
    x = zeros(Dual{Nothing, T, nx + np}, nx)
    p = zeros(Dual{Nothing, T, nx + np}, np)
    Jxsto = zeros(T, nx, nx)
    Jpsto = zeros(T, nx, np)
    tjac = zeros(T, np + nx, nx*(K + 1))
    cfg = JacobianConfig(nothing, out, zeros(T, nx + np))
    result = JacobianResult(out, zeros(T, nx + np))
    Jx = Matrix{T}[]
    Jp = Matrix{T}[]

    temp = zero(Dual{Nothing, T, nx + np})
    taux = [STaylor1(zero(Q), Val(K))]
    xtaylor = STaylor1.(xd0, Val(K))
    dx = STaylor1.(xd0, Val(K))
    xaux = STaylor1.(xd0, Val(K))
    for i in 1:(K + 1)
        push!(Jx, zeros(T,nx,nx))
        push!(Jp, zeros(T,nx,np))
    end
    t = 0.0
    vnxt = zeros(Int, nx)
    fnxt = zeros(Float64, nx)
    return JacTaylorFunctor!{typeof(g!), K+1, Q, T, nx + np}(g!, nx, np, K, out, y, x, p,
                                                             Jxsto, Jpsto, tjac, Jx, Jp,
                                                             result, cfg, xtaylor, xaux,
                                                             dx, taux, t, vnxt, fnxt)
end

function (d::JacTaylorFunctor!{F,K,T,S,NY})(out::AbstractVector{Dual{Nothing,S,NY}},
                                            y::AbstractVector{Dual{Nothing,S,NY}}) where {F <: Function,
                                                                                          K, T <: Real, S, NY}


    copyto!(d.x, 1, y, 1, d.nx)
    copyto!(d.p, 1, y, d.nx + 1, d.np)
    val = Val{K-1}()
    for i=1:d.nx
        d.xtaylor[i] = STaylor1(d.x[i], val)
    end
    jetcoeffs!(d.g!, d.t, d.xtaylor, d.xaux, d.dx, K - 1, d.p, d.vnxt, d.fnxt)
    for q = 1:K
        for i = 1:d.nx
            indx = d.nx*(q - 1) + i
            out[indx] = d.xtaylor[i].coeffs[q]
        end
    end

    return nothing
end

"""
jacobian_taylor_coeffs!

Computes the Jacobian of the Taylor coefficients w.r.t. y = (x,p) storing the
output inplace to `result`. A JacobianConfig object without tag checking, cfg,
is required input and is initialized from `cfg = ForwardDiff.JacobianConfig(nothing, out, y)`.
The JacTaylorFunctor! used for the evaluation is `g` and inputs are `x` and `p`.
"""
function jacobian_taylor_coeffs!(g::JacTaylorFunctor!{F,K,T,S,NY}, X::Vector{S}, P, t::T) where {F,K,T,S,NY}

    # copyto! is used to avoid allocations
    copyto!(g.y, 1, X, 1, g.nx)
    copyto!(g.y, g.nx + 1, P, 1, g.np)
    g.t = t
    # other AD schemes may be usable as well but this is a length(g.out) >> nx + np
    # situtation typically
    jacobian!(g.result, g, g.out, g.y, g.cfg)

    # reset sum of Jacobian storage
    fill!(g.Jxsto, zero(S))
    fill!(g.Jpsto, zero(S))
    nothing
end

"""
set_JxJp!

Extracts the Jacobian of the Taylor coefficients w.r.t. x, `Jx`, and the
Jacobian of the Taylor coefficients w.r.t. p, `Jp`, from `result`. The order of
the Taylor series is `s`, the dimensionality of x is `nx`, the dimensionality of
p is `np`, and `tjac` is preallocated storage for the transpose of the Jacobian
w.r.t. y = (x,p).
"""
function set_JxJp!(g::JacTaylorFunctor!{F,K,T,S,NY}, X::Vector{S}, P, t) where {F,K,T,S,NY}

    jacobian_taylor_coeffs!(g, X, P, t)
    jac = g.result.derivs[1]
    for i = 1:(g.s + 1)
        for q = 1:g.nx
            for z = 1:g.nx
                g.Jx[i][q, z] = jac[q + g.nx*(i-1), z]
            end
            for z = 1:g.np
                g.Jp[i][q, z] = jac[q + g.nx*(i-1), g.nx + z]
            end
        end
    end
    nothing
end
