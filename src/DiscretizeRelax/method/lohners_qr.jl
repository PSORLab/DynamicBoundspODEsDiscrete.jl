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
# src/DiscretizeRelax/method/lohners_qr.jl
# Defines the lohner's method for contracting in discretize and relax.
#############################################################################

"""
$(TYPEDEF)

A functor used in computing bounds and relaxations via Lohner's method.
"""
mutable struct LohnersFunctor{F <: Function, K, T <: Real, S <: Real, NY} <: AbstractStateContractor
    set_tf!::TaylorFunctor!{F, K, T, S}
    real_tf!::TaylorFunctor!{F, K, T, T}
    jac_tf!::JacTaylorFunctor!{F, K, T, S, NY}
    η::Interval{Float64}
    μX::Vector{S}
    ρP::Vector{S}
    f̃::Vector{Vector{S}}
    f̃val::Vector{Vector{Float64}}
    Vⱼ₊₁::Vector{Float64}
    Rⱼ₊₁::Vector{S}
    mRⱼ₊₁::Vector{Float64}
    Δⱼ₊₁::Vector{S}
end
function LohnersFunctor(f!::F, nx::Int, np::Int, k::Val{K}, s::S, t::T) where {F, K, S <: Number, T <: Number}
    set_tf! = TaylorFunctor!(f!, nx, np, k, zero(S), zero(T))
    real_tf! = TaylorFunctor!(f!, nx, np, k, zero(T), zero(T))
    jac_tf! = JacTaylorFunctor!(f!, nx, np, k, zero(S), zero(T))
    μX = zeros(S, nx)
    ρP = zeros(S, np)
    f̃ = Vector{S}[]
    f̃val = Vector{Float64}[]
    for i = 1:(K+1)
        push!(f̃, zeros(S, nx))
        push!(f̃val, zeros(Float64, nx))
    end
    Vⱼ₊₁ = zeros(Float64, nx)
    Rⱼ₊₁ = zeros(S, nx)
    mRⱼ₊₁ = zeros(Float64, nx)
    Δⱼ₊₁ = zeros(S, nx)
    LohnersFunctor{F, K+1, T, S, nx+np}(set_tf!, real_tf!, jac_tf!, Interval{Float64}(0.0,1.0), μX, ρP,
                                        f̃, f̃val, Vⱼ₊₁, Rⱼ₊₁, mRⱼ₊₁, Δⱼ₊₁)
end

struct LohnerContractor{K} <: AbstractStateContractorName end
LohnerContractor(k::Int) = LohnerContractor{k}()
function state_contractor(m::LohnerContractor{K}, f, Jx!, Jp!, nx, np, style, s, h) where K
    LohnersFunctor(f, nx, np, Val{K}(), style, s)
end
state_contractor_k(m::LohnerContractor{K}) where K = K
state_contractor_γ(m::LohnerContractor) = 1.0
state_contractor_steps(m::LohnerContractor) = 2

"""
$(FUNCTIONNAME)

An implementation of the parametric Lohner's method described in the paper in (1)
based on the non-parametric version given in (2).

1. [Sahlodin, Ali M., and Benoit Chachuat. "Discretize-then-relax approach for
convex/concave relaxations of the solutions of parametric ODEs." Applied Numerical
Mathematics 61.7 (2011): 803-820.](https://www.sciencedirect.com/science/article/abs/pii/S0168927411000316)
2. [R.J. Lohner, Computation of guaranteed enclosures for the solutions of
ordinary initial and boundary value problems, in: J.R. Cash, I. Gladwell (Eds.),
Computational Ordinary Differential Equations, vol. 1, Clarendon Press, 1992,
pp. 425–436.](http://www.goldsztejn.com/old-papers/Lohner-1992.pdf)
"""
function (d::LohnersFunctor{F,K,S,T,NY})(contract::ContractorStorage{T}, result::StepResult{T}) where {F <: Function,
                                                                                                       K, S <: Real,
                                                                                                       T <: Real, NY}

    set_tf! = d.set_tf!
    real_tf! = d.real_tf!
    Jf! = d.jac_tf!

    k = set_tf!.k
    nx = set_tf!.nx

    hⱼ = contract.hj_computed
    hjk = hⱼ^k
    t = contract.times[1]

    set_tf!(d.f̃, contract.Xj_apriori, contract.P, t)

    # computes Rj and it's midpoint
    for i = 1:nx
        @inbounds d.Rⱼ₊₁[i] = hjk*d.f̃[k + 1][i]
        @inbounds d.mRⱼ₊₁[i] = mid(d.Rⱼ₊₁[i])
    end

    # defunes new x point... k corresponds to k - 1 since taylor
    # coefficients are zero indexed
    real_tf!(d.f̃val, contract.xval, contract.pval, t)
    hji1 = 0.0
    fill!(d.Vⱼ₊₁, 0.0)
    for i = 1:k
        hji1 = hⱼ^i
        @__dot__ d.Vⱼ₊₁ += hji1*d.f̃val[i + 1]
    end
    d.Vⱼ₊₁ += contract.xval

    # compute extensions of taylor cofficients for rhs
    μ!(d.μX, contract.Xj_0, contract.xval, d.η)
    ρ!(d.ρP, contract.P, contract.pval, d.η)
    set_JxJp!(Jf!, d.μX, d.ρP, t)
    for i = 1:k
        hji1 = hⱼ^(i - 1)
        if i == 1
            fill!(Jf!.Jxsto, zero(S))
            for j = 1:nx
                Jf!.Jxsto[j,j] = one(S)
            end
        else
            @__dot__ Jf!.Jxsto += hji1*Jf!.Jx[i]
        end
        @__dot__ Jf!.Jpsto += hji1*Jf!.Jp[i]
    end

    # update x floating point value
    @__dot__ contract.xval_computed = d.Vⱼ₊₁ + d.mRⱼ₊₁

    # update bounds on X at new time
    @show contract.Δ[1]
    contract.X_computed .= d.Vⱼ₊₁ + d.Rⱼ₊₁ + (Jf!.Jxsto*contract.A[2].Q)*contract.Δ[1] + Jf!.Jpsto*contract.rP

    # calculation block for computing Aⱼ₊₁ and inv(Aⱼ₊₁)
    Aⱼ₊₁ = contract.A[1]
    contract.B = mid.(Jf!.Jxsto*contract.A[2].Q)
    calculateQ!(Aⱼ₊₁, contract.B, nx)
    calculateQinv!(Aⱼ₊₁)

    # update Delta
    d.Δⱼ₊₁ .= Aⱼ₊₁.inv*(d.Rⱼ₊₁ - d.mRⱼ₊₁) + (Aⱼ₊₁.inv*(Jf!.Jxsto*contract.A[2].Q))*contract.Δ[1] + (Aⱼ₊₁.inv*Jf!.Jpsto)*contract.rP
    pushfirst!(contract.Δ, d.Δⱼ₊₁)

    return RELAXATION_NOT_CALLED
end

get_Δ(lf::LohnersFunctor) = lf.Δⱼ₊₁
function set_xX!(outX::Vector{S}, outx::Vector{Float64}, lf::LohnersFunctor) where {S <: Number}
    outX .= lf.jac_tf!.Xⱼ₊₁
    outx .= lf.jac_tf!.xⱼ₊₁
    return nothing
end

has_jacobians(d::LohnersFunctor) = true
function extract_jacobians!(d::LohnersFunctor{F,K,S,T,NY}, ∂f∂x::Vector{Matrix{T}},
                            ∂f∂p::Vector{Matrix{T}}) where {F <: Function, K, S <: Real, T <: Real, NY}
    for i = 1:(d.set_tf!.k + 1)
        ∂f∂x[i] .= d.jac_tf!.Jx[i]
        ∂f∂p[i] .= d.jac_tf!.Jp[i]
    end
    nothing
end
function get_jacobians!(d::LohnersFunctor{F,K,S,T,NY}, ∂f∂x::Vector{Matrix{T}},
                        ∂f∂p::Vector{Matrix{T}}, Xⱼ, P, t) where {F <: Function, K, S <: Real, T <: Real, NY}
    set_JxJp!(d.jac_tf!, Xⱼ, P, t[1])
    extract_jacobians!(d, ∂f∂x, ∂f∂p)
    nothing
end
