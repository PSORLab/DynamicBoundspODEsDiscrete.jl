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
LohnersFunctor

A functor used in computing bounds and relaxations via Lohner's method. The
implementation of the parametric Lohner's method described in the paper in (1)
based on the non-parametric version given in (2).

1. [Sahlodin, Ali M., and Benoit Chachuat. "Discretize-then-relax approach for
convex/concave relaxations of the solutions of parametric ODEs." Applied Numerical
Mathematics 61.7 (2011): 803-820.](https://www.sciencedirect.com/science/article/abs/pii/S0168927411000316)
2. [R.J. Lohner, Computation of guaranteed enclosures for the solutions of
ordinary initial and boundary value problems, in: J.R. Cash, I. Gladwell (Eds.),
Computational Ordinary Differential Equations, vol. 1, Clarendon Press, 1992,
pp. 425–436.](http://www.goldsztejn.com/old-papers/Lohner-1992.pdf)
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
    Jxmat::Matrix{S}
    Jxvec::Vector{S}
    Jpvec::Vector{S}
    rRⱼ₊₁::Vector{S}
    YdRⱼ₊₁::Vector{S}
    YJxmat::Matrix{S}
    YJxvec::Vector{S}
    YJpmat::Matrix{S}
    YJpvec::Vector{S}
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
    Jxmat = zeros(S, nx, nx)
    Jxvec = zeros(S, nx)
    Jpvec = zeros(S, nx)
    rRⱼ₊₁ = zeros(S, nx)
    YdRⱼ₊₁ = zeros(S, nx)
    YJxmat = zeros(S, nx, nx)
    YJxvec = zeros(S, nx)
    YJpmat = zeros(S, nx, np)
    YJpvec = zeros(S, nx)
    LohnersFunctor{F, K+1, T, S, nx+np}(set_tf!, real_tf!, jac_tf!, Interval{Float64}(0.0,1.0), μX, ρP,
                                        f̃, f̃val, Vⱼ₊₁, Rⱼ₊₁, mRⱼ₊₁, Δⱼ₊₁, Jxmat, Jxvec, Jpvec,
                                        rRⱼ₊₁, YdRⱼ₊₁, YJxmat, YJxvec, YJpmat, YJpvec)
end

"""
LohnerContractor{K}

An `AbstractStateContractorName` used to specify a parametric Lohners method contractor
of order K.
"""
struct LohnerContractor{K} <: AbstractStateContractorName end
LohnerContractor(k::Int) = LohnerContractor{k}()
function state_contractor(m::LohnerContractor{K}, f, Jx!, Jp!, nx, np, style, s, h) where K
    LohnersFunctor(f, nx, np, Val{K}(), style, s)
end

state_contractor_k(m::LohnerContractor{K}) where K = K
state_contractor_γ(m::LohnerContractor{K}) where K = 1.0
state_contractor_steps(m::LohnerContractor{K}) where K = 2

function (d::LohnersFunctor{F,K,S,T,NY})(contract::ContractorStorage{T},
                                         result::StepResult{T},
                                         count::Int) where {F <: Function, K, S <: Real, T <: Real, NY}

    set_tf! = d.set_tf!
    real_tf! = d.real_tf!
    Jf! = d.jac_tf!

    k = set_tf!.k
    nx = set_tf!.nx
    np = length(contract.pval)

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
    @__dot__ d.Vⱼ₊₁ += contract.xval

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
    mul_split!(d.Jxmat, Jf!.Jxsto, contract.A[2].Q, nx)
    mul_split!(d.Jxvec, d.Jxmat, contract.Δ[1], nx)
    mul_split!(d.Jpvec, Jf!.Jpsto, contract.rP, nx)

    @__dot__ contract.X_computed = d.Vⱼ₊₁ + d.Rⱼ₊₁ + d.Jxvec + d.Jpvec
    affine_contract!(contract.X_computed, contract.P, contract.pval, nx, np)

    # calculation block for computing Aⱼ₊₁ and inv(Aⱼ₊₁)
    @__dot__ contract.B = mid(d.Jxmat)
    calculateQ!(contract.A[1], contract.B, nx)
    calculateQinv!(contract.A[1])

    # update Delta
    @__dot__ d.rRⱼ₊₁ = d.Rⱼ₊₁ - d.mRⱼ₊₁
    mul_split!(d.YdRⱼ₊₁, contract.A[1].inv, d.rRⱼ₊₁, nx)
    mul_split!(d.YJxmat, contract.A[1].inv, d.Jxmat, nx)
    mul_split!(d.YJxvec, d.YJxmat, contract.Δ[1], nx)
    mul_split!(d.YJpmat, contract.A[1].inv, Jf!.Jpsto, nx)
    mul_split!(d.YJpvec, d.YJpmat, contract.rP, nx)
    @__dot__ d.Δⱼ₊₁ = d.YdRⱼ₊₁ + d.YJxvec + d.YJpvec

    pushfirst!(contract.Δ, d.Δⱼ₊₁)

    return RELAXATION_NOT_CALLED
end

get_Δ(lf::LohnersFunctor{F,K,S,T,NY}) where {F <: Function, K, S <: Real, T <: Real, NY} = lf.Δⱼ₊₁
