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
Base.@kwdef mutable struct LohnersFunctor{F <: Function, K, T <: Real, S <: Real, NY} <: AbstractStateContractor
    set_tf!::TaylorFunctor!{F,K,T,S}
    real_tf!::TaylorFunctor!{F,K,T,T}
    jac_tf!::JacTaylorFunctor!{F,K,T,S,NY}
    η::Interval{Float64} = Interval{Float64}(0.0,1.0)
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
    nx::Int
    np::Int
    Ai_rRj::Vector{S}
    Ai_Jxm::Matrix{S}
    Δ_Jx::Vector{S}
    Ai_Jpm::Matrix{S}
    Δ_Jp::Vector{S}
    constant_state_bounds::Union{Nothing,DBB.ConstantStateBounds}
end
function LohnersFunctor(f!::F, nx::Int, np::Int, k::Val{K}, s::S, t::T) where {F, K, S <: Number, T <: Number}
    f̃ = Vector{S}[]
    f̃val = Vector{Float64}[]
    for i = 1:(K + 1)
        push!(f̃, zeros(S, nx))
        push!(f̃val, zeros(Float64, nx))
    end
    LohnersFunctor{F, K + 1, T, S, nx + np}(set_tf!  = TaylorFunctor!(f!, nx, np, k, zero(S), zero(T)),
                                            real_tf! = TaylorFunctor!(f!, nx, np, k, zero(T), zero(T)),
                                            jac_tf!  = JacTaylorFunctor!(f!, nx, np, k, zero(S), zero(T)),
                                            μX       = zeros(S, nx),
                                            ρP       = zeros(S, np),
                                            f̃        = f̃,
                                            f̃val     = f̃val,
                                            Vⱼ₊₁     = zeros(nx),
                                            Rⱼ₊₁     = zeros(S, nx),
                                            mRⱼ₊₁    = zeros(nx),
                                            Δⱼ₊₁     = zeros(S, nx),
                                            Jxmat    = zeros(S, nx, nx),
                                            Jxvec    = zeros(S, nx),
                                            Jpvec    = zeros(S, nx),
                                            rRⱼ₊₁    = zeros(S, nx),
                                            nx       = nx,
                                            np       = np,
                                            Ai_rRj   = zeros(S, nx),
                                            Ai_Jxm   = zeros(S, nx, nx),
                                            Δ_Jx     = zeros(S, nx),
                                            Ai_Jpm   = zeros(S, nx, np),
                                            Δ_Jp     = zeros(S, nx),
                                            constant_state_bounds = nothing)
end

set_constant_state_bounds!(d::LohnersFunctor, v) = (d.constant_state_bounds = v;)

"""
LohnerContractor{K}

An `AbstractStateContractorName` used to specify a parametric Lohners method contractor
of order K.
"""
struct LohnerContractor{K} <: AbstractStateContractorName end
LohnerContractor(k::Int) = LohnerContractor{k}()

state_contractor(m::LohnerContractor{K}, f, Jx!, Jp!, nx, np, style, s, h) where K = LohnersFunctor(f, nx, np, Val{K}(), style, s)

state_contractor_k(m::LohnerContractor{K}) where K = K
state_contractor_γ(m::LohnerContractor{K}) where K = 1.0
state_contractor_steps(m::LohnerContractor{K}) where K = 2
state_contractor_integrator(m::LohnerContractor{K}) where K = CVODE_Adams()

function (d::LohnersFunctor{F,K,S,T,NY})(c, r, j, k) where {F,K,S,T,NY}

    @unpack hj, xval_computed, X_computed, xval, pval, P, rP, A_Q, A_inv, Δ, Xj_0, Xj_apriori = c
    @unpack f̃, f̃val, η, set_tf!, real_tf!, Jxmat, Jxvec, Jpvec, Rⱼ₊₁, mRⱼ₊₁, Vⱼ₊₁, nx, np = d
    @unpack Ai_rRj, Ai_Jxm, Δ_Jx, Ai_Jpm, Δ_Jp, Δⱼ₊₁, μX, ρP, rRⱼ₊₁ = d
    @unpack Jp, Jx, Jpsto, Jxsto = d.jac_tf!
    @unpack k = d.set_tf!
    Jf! = d.jac_tf!
    t = c.times[1]

    set_tf!(f̃, Xj_apriori, P, t)
    @. Rⱼ₊₁ = (hj^k)*f̃[k + 1]
    @. mRⱼ₊₁ = mid(Rⱼ₊₁)

    # defunes new x point... k corresponds to k - 1 since taylor coefficients are zero indexed
    real_tf!(f̃val, xval, pval, t)
    @. Vⱼ₊₁ = xval
    for i = 1:(k-1)
        @. Vⱼ₊₁ += f̃val[i + 1]*hj^i
    end

    # compute extensions of taylor cofficients for rhs
    μ!(μX, Xj_0, xval, η)
    ρ!(ρP, P, pval, η)
    set_JxJp!(Jf!, μX, ρP, t)
    fill!(Jxsto, zero(T))
    Jxsto += I
    @. Jpsto = Jp[1]
    for i = 2:k
        @. Jxsto += Jx[i]*hj^(i - 1)
        @. Jpsto += Jp[i]*hj^(i - 1)
    end

    # update x floating point value
    @. xval_computed = Vⱼ₊₁ + mRⱼ₊₁
    mul!(Jxmat, Jxsto, A_Q[2])
    mul!(Jxvec, Jxmat, Δ[2])
    mul!(Jpvec, Jpsto, rP)

    @. X_computed = Vⱼ₊₁ + Rⱼ₊₁ + Jxvec + Jpvec
    contract_constant_state!(X_computed, d.constant_state_bounds)
    affine_contract!(X_computed, P, pval, nx, np)

    # calculation block for computing Aⱼ₊₁ and inv(Aⱼ₊₁)
    calculateQ!(A_Q[1], mid.(Jxmat))
    calculateQinv!(A_inv[1], A_Q[1])

    # update Delta
    @. rRⱼ₊₁ = Rⱼ₊₁ - mRⱼ₊₁
    mul!(Ai_rRj, A_inv[1], rRⱼ₊₁)
    mul!(Ai_Jxm, A_inv[1], Jxmat)
    mul!(Δ_Jx, Ai_Jxm, Δ[2])
    mul!(Ai_Jpm, A_inv[1], Jpsto)
    mul!(Δ_Jp, Ai_Jpm, rP)
    @. Δ[1] = Ai_rRj + Δ_Jx + Δ_Jp
    @. Δⱼ₊₁ = Δ[1]

    return RELAXATION_NOT_CALLED
end

get_Δ(d::LohnersFunctor{F,K,S,T,NY}) where {F,K,S,T,NY} = copy(d.Δⱼ₊₁)