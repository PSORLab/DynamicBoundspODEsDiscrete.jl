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
# src/DiscretizeRelax/method/hermite_obreschkoff.jl
# Defines functions needed to perform a hermite_obreshkoff iteration.
#############################################################################

"""
HermiteObreschkoff

A structure that stores the cofficient of the (P,Q)-Hermite-Obreschkoff method.
(Offset due to method being zero indexed and Julia begin one indexed). The
constructor `HermiteObreschkoff(p::Val{P}, q::Val{Q}) where {P, Q}` or
`HermiteObreschkoff(p::Int, q::Int)` are used for the (P,Q)-method.

$(TYPEDFIELDS)
"""
struct HermiteObreschkoff <: AbstractStateContractorName
    "Cpq[i=1:p] index starting at i = 1 rather than 0"
    cpq::Vector{Float64}
    "Cqp[i=1:q] index starting at i = 1 rather than 0"
    cqp::Vector{Float64}
    "gamma for method"
    γ::Float64
    "Explicit order Hermite-Obreschkoff"
    p::Int64
    "Implicit order Hermite-Obreschkoff"
    q::Int64
    "Total order Hermite-Obreschkoff"
    k::Int64
end

function HermiteObreschkoff(p::Val{P}, q::Val{Q}) where {P, Q}
    temp_cpq = 1.0
    temp_cqp = 1.0
    cpq = zeros(P + 1)
    cqp = zeros(Q + 1)
    cpq[1] = temp_cpq
    cqp[1] = temp_cqp
    for i = 1:P
        temp_cpq *= (P - i + 1.0)/(P + Q - i + 1)
        cpq[i + 1] = temp_cpq
    end
    γ = 1.0
    for i = 1:Q
        temp_cqp *= (Q - i + 1.0)/(Q + P - i + 1)
        cqp[i + 1] = temp_cqp
        γ *= -i/(P+i)
    end
    K = P + Q + 1
    HermiteObreschkoff(cpq, cqp, γ, P, Q, K)
end
HermiteObreschkoff(p::Int, q::Int) = HermiteObreschkoff(Val(p), Val(q))

"""
HermiteObreschkoffFunctor

A functor used in computing bounds and relaxations via Hermite-Obreschkoff's method. The
implementation of the parametric Hermite-Obreschkoff's method based on the non-parametric
version given in (1).

1. [Nedialkov NS, and Jackson KR. "An interval Hermite-Obreschkoff method for
computing rigorous bounds on the solution of an initial value problem for an
ordinary differential equation." Reliable Computing 5.3 (1999): 289-310.](https://link.springer.com/article/10.1023/A:1009936607335)
2. [Nedialkov NS. "Computing rigorous bounds on the solution of an
initial value problem for an ordinary differential equation." University
of Toronto. 2000.](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.633.9654&rep=rep1&type=pdf)
"""
mutable struct HermiteObreschkoffFunctor{F <: Function, Pp1, Qp1, K, T <: Real, S <: Real, NY} <: AbstractStateContractor
    hermite_obreschkoff::HermiteObreschkoff
    η::Interval{T}
    μX::Vector{S}
    ρP::Vector{S}
    gⱼ₊₁::Vector{S}
    nx::Int64
    set_tf!_pred::TaylorFunctor!{F, K, T, S}
    real_tf!_pred::TaylorFunctor!{F, Pp1, T, T}
    Jf!_pred::JacTaylorFunctor!{F, Pp1, T, S, NY}
    Rⱼ₊₁::Vector{S}
    mRⱼ₊₁::Vector{T}
    f̃val_pred::Vector{Vector{T}}
    f̃_pred::Vector{Vector{S}}
    Vⱼ₊₁::Vector{Float64}
    X_predict::Vector{S}
    q_predict::Int64
    real_tf!_correct::TaylorFunctor!{F, Qp1, T, T}
    Jf!_correct::JacTaylorFunctor!{F, Qp1, T, S, NY}
    xval_correct::Vector{T}
    f̃val_correct::Vector{Vector{T}}
    sum_p::Vector{S}
    sum_q::Vector{S}
    Δⱼ₊₁::Vector{S}
    pred_Jxmat::Matrix{S}
    pred_Jxvec::Vector{S}
    pred_Jpvec::Vector{S}
    δⱼ₊₁::Vector{S}
    precond::Matrix{Float64}
    precond2::Matrix{Float64}
    correct_B::Matrix{S}
    correct_Bvec::Vector{S}
    correct_C::Matrix{S}
    Inx::UniformScaling{Bool}
    Uj::Vector{S}
    Jpdiff::Matrix{S}
    Cp::Matrix{S}
    CprP::Vector{S}
    correct_Jmid::Matrix{S}
    Uj2::Vector{S}
    B2::Matrix{S}
    B2vec::Vector{S}
    rX::Vector{S}
    Yδⱼ₊₁::Vector{S}
    Y2δⱼ₊₁::Vector{S}
    Y2Jpdiff::Matrix{S}
    Y2Jpvec::Vector{S}
    YC::Matrix{S}
    YUj2::Vector{S}
end
function HermiteObreschkoffFunctor(f!::F, nx::Int, np::Int, p::Val{P}, q::Val{Q},
                                   s::S, t::T) where {F,P,Q,S,T}

    K = P + Q + 1
    hermite_obreschkoff = HermiteObreschkoff(p, q)
    η = Interval{T}(0.0,1.0)
    μX = zeros(S, nx)
    ρP = zeros(S, np)
    gⱼ₊₁ = zeros(S, nx)
    set_tf!_pred = TaylorFunctor!(f!, nx, np, Val(K), zero(S), zero(T))
    real_tf!_pred = TaylorFunctor!(f!, nx, np, Val(P), zero(T), zero(T))
    Jf!_pred = JacTaylorFunctor!(f!, nx, np, Val(P), zero(S), zero(T))
    Rⱼ₊₁ = zeros(S, nx)
    mRⱼ₊₁ = zeros(Float64, nx)

    f̃val_pred = Vector{Float64}[]
    for i = 1:(P + 1)
        push!(f̃val_pred, zeros(Float64, nx))
    end

    f̃_pred = Vector{S}[]
    for i = 1:(K + 1)
        push!(f̃_pred, zeros(S, nx))
    end

    Vⱼ₊₁ = zeros(nx)
    X_predict = zeros(S, nx)
    q_predict = P

    real_tf!_correct = TaylorFunctor!(f!, nx, np, Val(Q), zero(T), zero(T))
    Jf!_correct = JacTaylorFunctor!(f!, nx, np, Val(Q), zero(S), zero(T))
    xval_correct = zeros(Float64, nx)

    f̃val_correct = Vector{Float64}[]
    for i = 1:(Q + 1)
        push!(f̃val_correct, zeros(Float64, nx))
    end

    sum_p = zeros(S, nx)
    sum_q = zeros(S, nx)
    P1 = P + 1
    Q1 = Q + 1

    Δⱼ₊₁ = zeros(S, nx)

    pred_Jxmat = zeros(S, nx, nx)
    pred_Jxvec = zeros(S, nx)
    pred_Jpvec = zeros(S, nx)

    δⱼ₊₁ = zeros(S, nx)
    precond = zeros(Float64, nx, nx)
    precond2 = zeros(Float64, nx, nx)
    correct_B = zeros(S, nx, nx)
    correct_Bvec = zeros(S, nx)
    correct_C = zeros(S, nx, nx)
    Inx = I

    Uj = zeros(S, nx)
    Jpdiff = zeros(S, nx, np)
    Cp = zeros(S, nx, np)
    CprP = zeros(S, nx)

    correct_Jmid = zeros(S, nx, nx)
    Uj2 = zeros(S, nx)
    B2 = zeros(S, nx, nx)
    B2vec = zeros(S, nx)
    rX = zeros(S, nx)
    Yδⱼ₊₁ = zeros(S, nx)

    Y2δⱼ₊₁ = zeros(S, nx)
    Y2Jpdiff = zeros(S, nx, np)
    Y2Jpvec = zeros(S, nx)
    YC = zeros(S, nx, nx)
    YUj2 = zeros(S, nx)

    HermiteObreschkoffFunctor{F, P1, Q1, K+1, T, S, nx + np}(hermite_obreschkoff, η, μX, ρP, gⱼ₊₁, nx,
                                                             set_tf!_pred, real_tf!_pred, Jf!_pred, Rⱼ₊₁,
                                                             mRⱼ₊₁, f̃val_pred, f̃_pred, Vⱼ₊₁,
                                                             X_predict, q_predict, real_tf!_correct,
                                                             Jf!_correct, xval_correct, f̃val_correct,
                                                             sum_p, sum_q, Δⱼ₊₁, pred_Jxmat, pred_Jxvec,
                                                             pred_Jpvec, δⱼ₊₁, precond, precond2,
                                                             correct_B, correct_Bvec, correct_C, Inx,
                                                             Uj, Jpdiff, Cp,
                                                             CprP, correct_Jmid, Uj2, B2, B2vec, rX, Yδⱼ₊₁,
                                                             Y2δⱼ₊₁, Y2Jpdiff, Y2Jpvec, YC, YUj2)
end

function state_contractor(m::HermiteObreschkoff, f, Jx!, Jp!, nx, np, style, s, h)
    HermiteObreschkoffFunctor(f, nx, np, Val(m.p), Val(m.q), style, s)
end
state_contractor_k(m::HermiteObreschkoff) = m.k
state_contractor_γ(m::HermiteObreschkoff) = m.γ
state_contractor_steps(m::HermiteObreschkoff) = 2
state_contractor_integrator(m::HermiteObreschkoff) = CVODE_Adams()


function _pred_compute_Rj!(d, c, t)
    q = d.q_predict
    d.set_tf!_pred(d.f̃_pred, c.Xj_apriori, c.P, t)
    hq = c.hj_computed^(d.q_predict + 1)
    for i = 1:d.nx
        @inbounds d.Rⱼ₊₁[i] = hq*d.f̃_pred[q + 2][i]
    end
    return nothing
end

function _pred_compute_real_pnt!(d, c, t)
    d.real_tf!_pred(d.f̃val_pred, c.xval, c.pval, t)
    @__dot__ d.Vⱼ₊₁ = c.xval
    for i = 1:d.q_predict
        h = c.hj_computed^i
        @__dot__ d.Vⱼ₊₁ += h*d.f̃val_pred[i + 1]
    end
    return nothing
end

function _pred_compute_rhs_jacobian!(d, c::ContractorStorage{S}, t) where S
    hⱼ = c.hj_computed
    Jf!_pred = d.Jf!_pred
    μ!(d.μX, c.Xj_0, c.xval, d.η)
    ρ!(d.ρP, c.P, c.pval, d.η)
    set_JxJp!(Jf!_pred, d.μX, d.ρP, t)
    for i = 1:d.q_predict
        hji1 = hⱼ^i
        if i == 1
            fill!(Jf!_pred.Jxsto, zero(S))
            for j = 1:d.nx
                Jf!_pred.Jxsto[j,j] = one(S)
            end
        else
            @__dot__ Jf!_pred.Jxsto += hji1*Jf!_pred.Jx[i+1]
        end
        @__dot__ Jf!_pred.Jpsto += hji1*Jf!_pred.Jp[i+1]
    end
    return nothing
end

function _hermite_obreschkoff_predictor!(d::HermiteObreschkoffFunctor{F,P1,Q1,K,T,S,NY},
                                         c::ContractorStorage{S}) where {F,P1,Q1,K,T,S,NY}

    _pred_compute_Rj!(d, c, c.times[1])
    _pred_compute_real_pnt!(d, c, c.times[1])
    _pred_compute_rhs_jacobian!(d, c, c.times[1])

    mul_split!(d.pred_Jxmat, d.Jf!_pred.Jxsto, c.A_Q[2], d.nx)
    mul_split!(d.pred_Jxvec, d.pred_Jxmat,     c.Δ[2],   d.nx)
    mul_split!(d.pred_Jpvec, d.Jf!_pred.Jpsto, c.rP,     d.nx)
    @__dot__ d.X_predict = d.Vⱼ₊₁ + d.Rⱼ₊₁ + d.pred_Jxvec + d.pred_Jpvec

    return nothing
end

function (d::HermiteObreschkoffFunctor{F,P1,Q1,K,T,S,NY})(contract::ContractorStorage{S},
                                                          result::StepResult{S},
                                                          count::Int) where {F, P1, Q1, K, T, S, NY}

    _hermite_obreschkoff_predictor!(d, contract)

    map!(mid, d.xval_correct, d.X_predict)     # GOOD

    # extract method constants
    ho = d.hermite_obreschkoff
    γ = ho.γ
    p = ho.p                     # GOOD
    q = ho.q                     # GOOD
    k = ho.k                     # GOOD
    nx = d.nx                    # GOOD
    np = length(contract.pval)   # GOOD

    hⱼ = contract.hj_computed    # GOOD
    t = contract.times[1]        # GOOD
    hjk = hⱼ^k                   # GOOD
    _sum_pred(y, i) = y[i+1]*ho.cpq[i+1]*hⱼ^(i)
    _sum_q(y, i) = (ho.cqp[i+1]*(-hⱼ)^i)*y[i+1]

    d.sum_p .= mapreduce(i -> _sum_pred(d.f̃val_pred, i), +, 1:p)

    d.real_tf!_correct(d.f̃val_correct, d.xval_correct, contract.pval, t)
    d.sum_q .=  mapreduce(i -> _sum_q(d.f̃val_correct, i), +, 1:q)

    @__dot__ d.δⱼ₊₁ = contract.xval - d.xval_correct + d.sum_p - d.sum_q + γ*hjk*d.f̃_pred[k + 1]   # Looks wrong based on impact....

    d.Jf!_pred.Jxsto .= mapreduce(i -> _sum_pred(d.Jf!_pred.Jx, i), +, 1:p)
    d.Jf!_pred.Jpsto .= mapreduce(i -> _sum_pred(d.Jf!_pred.Jp, i), +, 1:p)
    d.Jf!_pred.Jxsto += d.Inx

    μ!(d.μX, d.X_predict, d.xval_correct, d.η)
    ρ!(d.ρP, contract.P, contract.pval, d.η)
    set_JxJp!(d.Jf!_correct, d.μX, d.ρP, t)
    d.Jf!_correct.Jxsto .= mapreduce(i -> _sum_q(d.Jf!_correct.Jx, i), +, 1:q)
    d.Jf!_correct.Jpsto .= mapreduce(i -> _sum_q(d.Jf!_correct.Jp, i), +, 1:q)
    d.Jf!_correct.Jxsto += d.Inx

    map!(mid, d.precond, d.Jf!_correct.Jxsto)
    lu!(d.precond)

    mul_split!(d.pred_Jxmat, d.Jf!_pred.Jxsto, contract.A_Q[2], nx)
    d.correct_B = d.precond\d.pred_Jxmat
    mul_split!(d.correct_Bvec, d.correct_B, contract.Δ[2], nx)

    d.correct_C = d.precond\d.Jf!_correct.Jxsto
    @__dot__ d.correct_C *= -1.0
    d.correct_C += d.Inx

    @__dot__ d.rX = d.X_predict - d.xval_correct
    mul_split!(d.Uj, d.correct_C, d.rX, nx)

    @__dot__ d.Jpdiff = d.Jf!_pred.Jpsto - d.Jf!_correct.Jpsto
    d.Cp = d.precond\d.Jpdiff
    mul_split!(d.CprP, d.Cp, contract.rP, nx)
    mul_split!(d.Yδⱼ₊₁, d.precond, d.δⱼ₊₁, nx)
    @__dot__ contract.X_computed = d.xval_correct + d.correct_Bvec + d.Uj + d.CprP + d.Yδⱼ₊₁

    @__dot__ contract.X_computed = contract.X_computed ∩ d.X_predict

    affine_contract!(contract.X_computed, contract.P, contract.pval, nx, np)
    @__dot__ contract.xval_computed = mid(contract.X_computed)

    # calculation block for computing Aⱼ₊₁ and inv(Aⱼ₊₁)
    mul_split!(d.correct_Jmid, d.Jf!_correct.Jxsto, contract.A_Q[2], nx)
    @__dot__ contract.B = mid(d.correct_Jmid)
    calculateQ!(contract.A_Q[1], contract.B, nx)
    calculateQinv!(contract.A_inv[1], contract.A_Q[1], nx)

    mul_split!(d.precond2, contract.A_inv[1], d.precond, nx)
    @__dot__ d.Uj2 = contract.X_computed - d.xval_correct
    mul_split!(d.B2, d.precond, d.correct_Jmid, nx)
    mul_split!(d.B2vec, d.B2, contract.Δ[2], nx)

    mul_split!(d.Y2δⱼ₊₁, d.precond2, d.δⱼ₊₁, nx)
    mul_split!(d.Y2Jpdiff, d.precond2, d.Jpdiff, nx)
    mul_split!(d.Y2Jpvec, d.Y2Jpdiff, contract.rP, nx)
    mul_split!(d.YC, contract.A_inv[1], d.correct_C, nx)
    mul_split!(d.YUj2, d.YC, d.Uj2, nx)

    @__dot__ d.Δⱼ₊₁ = d.Y2δⱼ₊₁ + d.Y2Jpvec + d.B2vec + d.YUj2

    contract.Δ[1] = copy(d.Δⱼ₊₁)

    return RELAXATION_NOT_CALLED
end

get_Δ(f::HermiteObreschkoffFunctor) = f.Δⱼ₊₁
advance_contractor_buffer!(lf::HermiteObreschkoffFunctor) = nothing
