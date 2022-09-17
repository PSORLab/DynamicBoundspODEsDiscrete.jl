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
    p::Int
    "Implicit order Hermite-Obreschkoff"
    q::Int
    "Total order Hermite-Obreschkoff"
    k::Int
    "Skips the contractor step of the Hermite Obreshkoff Contractor if set to `true`"
    skip_contractor::Bool
end

function HermiteObreschkoff(p::Val{P}, q::Val{Q}, skip_contractor::Bool = false) where {P, Q}
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
   # K = P + 1
    HermiteObreschkoff(cpq, cqp, γ, P, Q, K, skip_contractor)
end
HermiteObreschkoff(p::Int, q::Int, b::Bool = false) = HermiteObreschkoff(Val(p), Val(q), b)

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
Base.@kwdef mutable struct HermiteObreschkoffFunctor{F <: Function, Pp, Pp1, Qp1, K, T <: Real, S <: Real, NY} <: AbstractStateContractor
    lohners_functor::LohnersFunctor{F,Pp,T,S,NY}
    hermite_obreschkoff::HermiteObreschkoff
    η::Interval{T} = Interval{T}(0.0,1.0)
    μX::Vector{S}
    ρP::Vector{S}
    gⱼ₊₁::Vector{S}
    nx::Int
    np::Int
    set_tf!_pred::TaylorFunctor!{F, K, T, S}
    real_tf!_pred::TaylorFunctor!{F, Pp1, T, T}
    Jf!_pred::JacTaylorFunctor!{F, Pp1, T, S, NY}
    Rⱼ₊₁::Vector{S}
    mRⱼ₊₁::Vector{T}
    f̃val_pred::Vector{Vector{T}} = Vector{T}[]
    f̃_pred::Vector{Vector{S}} = Vector{S}[]
    Vⱼ₊₁::Vector{Float64}
    X_predict::Vector{S}
    xval_predict::Vector{Float64}
    q_predict::Int
    real_tf!_correct::TaylorFunctor!{F, Qp1, T, T}
    Jf!_correct::JacTaylorFunctor!{F, Qp1, T, S, NY}
    xval_correct::Vector{T}
    f̃val_correct::Vector{Vector{T}} = Vector{T}[]
    sum_p::Vector{S}
    sum_q::Vector{S}
    Δⱼ₊₁::Vector{S}
    pred_Jxmat::Matrix{S}
    pred_Jxvec::Vector{S}
    pred_Jpvec::Vector{S}
    δⱼ₊₁::Vector{S}
    pre::Matrix{Float64}
    precond2::Matrix{Float64}
    correct_B::Matrix{S}
    correct_Bvec::Vector{S}
    correct_C::Matrix{S}
    Inx::UniformScaling{Bool} = I
    Uj::Vector{S}
    Jpdiff::Matrix{S}
    Cj::Matrix{S}
    C_temp::Matrix{S}
    Bj::Matrix{S}
    B_temp::Matrix{S}
    constant_state_bounds::Union{Nothing,DBB.ConstantStateBounds}
end
function set_constant_state_bounds!(d::HermiteObreschkoffFunctor, v)
    set_constant_state_bounds!(d.lohners_functor,v)
    d.constant_state_bounds = v
    nothing
end

function HermiteObreschkoffFunctor(f!::F, nx::Int, np::Int, p::Val{P}, q::Val{Q}, s::S, t::T, b) where {F,P,Q,S,T}

    P1 = P + 1
    P2 = P + 2
    Q1 = Q + 1
    K = P + Q + 1

    lon_func = LohnersFunctor(f!, nx, np, Val(P1), s, t)

    f̃val_pred = Vector{Float64}[]
    f̃_pred = Vector{S}[]
    f̃val_correct = Vector{Float64}[]
    for i = 1:(P + 1); push!(f̃val_pred, zeros(nx)); end
    for i = 1:(K + 1); push!(f̃_pred, zeros(S, nx)); end
    for i = 1:(Q + 1); push!(f̃val_correct, zeros(nx)); end

    hermite_obreschkoff = HermiteObreschkoff(p, q, b)
    set_tf!_pred = TaylorFunctor!(f!, nx, np, Val(K), zero(S), zero(T))

    real_tf!_pred = TaylorFunctor!(f!, nx, np, Val(P), zero(T), zero(T))
    Jf!_pred = JacTaylorFunctor!(f!, nx, np, Val(P), zero(S), zero(T))

    real_tf!_correct = TaylorFunctor!(f!, nx, np, Val(Q), zero(T), zero(T))
    Jf!_correct = JacTaylorFunctor!(f!, nx, np, Val(Q), zero(S), zero(T))

    HermiteObreschkoffFunctor{F, P2, P1, Q1, K+1, T, S, nx + np}(; 
                                                             lohners_functor = lon_func,
                                                             hermite_obreschkoff = hermite_obreschkoff, 
                                                             μX = zeros(S, nx), 
                                                             ρP = zeros(S, np), 
                                                             gⱼ₊₁ = zeros(S, nx), 
                                                             nx = nx,
                                                             np = np,
                                                             set_tf!_pred = set_tf!_pred, 
                                                             real_tf!_pred = real_tf!_pred, 
                                                             Jf!_pred = Jf!_pred, 
                                                             Rⱼ₊₁ = zeros(S, nx),
                                                             mRⱼ₊₁ = zeros(nx), 
                                                             f̃val_pred = f̃val_pred, 
                                                             f̃_pred = f̃_pred, 
                                                             Vⱼ₊₁ = zeros(nx),
                                                             X_predict = zeros(S, nx),
                                                             xval_predict = zeros(nx), 
                                                             q_predict = P, 
                                                             real_tf!_correct = real_tf!_correct,
                                                             Jf!_correct = Jf!_correct, 
                                                             xval_correct = zeros(nx), 
                                                             f̃val_correct = f̃val_correct,
                                                             sum_p = zeros(S, nx), 
                                                             sum_q = zeros(S, nx), 
                                                             Δⱼ₊₁ = zeros(S, nx), 
                                                             pred_Jxmat = zeros(S, nx, nx), 
                                                             pred_Jxvec = zeros(S, nx),
                                                             pred_Jpvec = zeros(S, nx), 
                                                             δⱼ₊₁ = zeros(S, nx), 
                                                             pre = zeros(nx, nx), 
                                                             precond2 = zeros(nx, nx),
                                                             correct_B = zeros(S, nx, nx), 
                                                             correct_Bvec = zeros(S, nx), 
                                                             correct_C = zeros(S, nx, nx), 
                                                             Uj = zeros(S, nx), 
                                                             Jpdiff = zeros(S, nx, np),
                                                             Cj = zeros(S, nx, nx),
                                                             C_temp = zeros(S, nx, nx),
                                                             Bj = zeros(S, nx, nx),
                                                             B_temp = zeros(S, nx, nx),
                                                             constant_state_bounds = nothing
                                                             ) 
end

function state_contractor(m::HermiteObreschkoff, f, Jx!, Jp!, nx, np, style, s, h)
    HermiteObreschkoffFunctor(f, nx, np, Val(m.p), Val(m.q), style, s, m.skip_contractor)
end
state_contractor_k(m::HermiteObreschkoff) = m.k
state_contractor_γ(m::HermiteObreschkoff) = m.γ
state_contractor_steps(m::HermiteObreschkoff) = 2
state_contractor_integrator(m::HermiteObreschkoff) = CVODE_Adams()

function _pred_compute_Rj!(d, c, t)
    @unpack set_tf!_pred, Rⱼ₊₁, f̃_pred, q_predict = d
    @unpack Xj_apriori, P, hj = c

    set_tf!_pred(f̃_pred, Xj_apriori, P, t)
    @. Rⱼ₊₁ = f̃_pred[q_predict + 1]*hj^q_predict
    Rⱼ₊₁
end
function _pred_compute_real_pnt!(d, c, t)
    @unpack Vⱼ₊₁, f̃val_pred, q_predict, real_tf!_pred = d
    @unpack hj, xval, pval = c
    
    real_tf!_pred(f̃val_pred, xval, pval, t)
    @. Vⱼ₊₁ = xval
    for i = 1:(q_predict - 1)
        @. Vⱼ₊₁ += f̃val_pred[i + 1]*hj^i
    end
    Vⱼ₊₁
end
function _pred_compute_rhs_jacobian!(d, c::ContractorStorage{S}, t) where S
    Jf!_pred = d.Jf!_pred
    @unpack η, q_predict, nx, μX, ρP, X_predict = d
    @unpack Xj_0, xval, P, pval, hj = c
    @unpack Jx, Jxsto, Jp, Jpsto = Jf!_pred

    μ!(μX, X_predict, xval, η)
    ρ!(ρP, P, pval, η)
    set_JxJp!(Jf!_pred, μX, ρP, t)
    for i = 1:q_predict
        hji1 = hj^i
        if isone(i)
            fill!(Jxsto, zero(S))
            Jxsto += I
        else
            @. Jxsto += hji1*Jx[i + 1]
        end
        @. Jpsto += hji1*Jp[i + 1]
    end
    return Jxsto, Jpsto
end
function _hermite_obreschkoff_predictor!(d::HermiteObreschkoffFunctor{F,Pp,P1,Q1,K,T,S,NY}, c::ContractorStorage{S}, r::StepResult{S}, j, k) where {F,Pp,P1,Q1,K,T,S,NY}
    @unpack X_predict, pred_Jxmat, pred_Jxvec, pred_Jpvec, Rⱼ₊₁, Vⱼ₊₁ = d
    @unpack A_Q, Δ, rP, times, Xj_apriori = c
    
    _pred_compute_Rj!(d, c, times[1])
    _pred_compute_real_pnt!(d, c, times[1])
    Jxsto, Jpsto = _pred_compute_rhs_jacobian!(d, c, times[1])

    d.lohners_functor(c, r, j, k)
    @. X_predict = c.X_computed ∩ Xj_apriori

    # HERMITE OBRESCHKOFF PREDICTOR IS FAILING... WHY?
    # THIS IS MORE EXPANSIVE THAN LOHNERS METHOD BASIC.... option 1 fix, option 2 lohners method....

    #mul!(pred_Jxmat, Jxsto, A_Q[2])
    #mul!(pred_Jxvec, pred_Jxmat, Δ[2])
    #mul!(pred_Jpvec, Jpsto, rP)
    #@. X_predict = Vⱼ₊₁ + Rⱼ₊₁ + pred_Jxvec + pred_Jpvec
    d.lohners_functor.Δⱼ₊₁
end

predi(cpq, hj, y, i) = y[i + 1]*cpq[i + 1]*hj^i
qi(cqp, hj, y, i) = y[i + 1]*cqp[i + 1]*((-hj)^i)

"""

Performs parametric version of algorithm 4.3 in Nedialkov...
"""
function (d::HermiteObreschkoffFunctor{F,Pp,P1,Q1,K,T,S,NY})(c::ContractorStorage{S}, r::StepResult{S}, j, k) where {F, Pp, P1, Q1, K, T, S, NY}
   
    @unpack X_computed, xval_computed, xval, pval, P, rP, A_inv, A_Q, Δ, hj = c    # unpack parameters and storage
    @unpack Jxsto, Jpsto, Jx, Jp = d.Jf!_pred
    @unpack γ, p, q, k, cpq, cqp = d.hermite_obreschkoff
    @unpack f̃val_correct, f̃val_pred, f̃_pred, real_tf!_correct, η, X_predict, xval_predict, δⱼ₊₁, nx, np = d
    @unpack Jpdiff, sum_p, sum_q, μX, ρP, pre, B_temp, C_temp, Bj, Cj, Δⱼ₊₁, Uj = d

    cJx, cJp = d.Jf!_correct.Jx, d.Jf!_correct.Jp
    t = c.times[1]                 

    ho_predict_Δⱼ₊₁ = _hermite_obreschkoff_predictor!(d, c, r, j, k) # calculate predictor --> sets d.X_predict
    @. xval_predict = mid(X_predict)

    if !d.hermite_obreschkoff.skip_contractor
        real_tf!_correct(f̃val_correct, xval_predict, pval, t)
        @. sum_p = f̃val_pred[2]*cpq[2]*hj
        for i = 2:p                                               
            @. sum_p += f̃val_pred[i + 1]*cpq[i + 1]*hj^i      
        end
        @. sum_q = -hj*f̃val_correct[2]*cqp[2]
        for i = 2:q
            @. sum_q += f̃val_correct[i + 1]*cqp[i + 1]*((-hj)^i)
        end  

        @. δⱼ₊₁ = xval - xval_predict + sum_p - sum_q + γ*(hj^k)*f̃_pred[k + 1]
        fill!(Jxsto, zero(S))
        for i = 1:nx; Jxsto[i,i] = one(S); end
        for i = 1:p
            @. Jxsto += Jx[i + 1]*cpq[i + 1]*hj^i
        end
        @. Jpsto = Jp[2]*cpq[2]*hj
        for i = 2:p
            @. Jpsto += Jp[i + 1]*cpq[i + 1]*hj^i
        end

        μ!(μX, X_predict, xval, η)
        ρ!(ρP, P, pval, η)
        set_JxJp!(d.Jf!_correct, μX, ρP, t)
        fill!(Jxsto, zero(S))
        for i = 1:nx; Jxsto[i,i] = one(S); end
        for i = 1:q
            @. Jxsto += cJx[i + 1]*cqp[i + 1]*((-hj)^i)
        end
        d.Jf!_correct.Jpsto .= sum(i -> qi(cqp, hj, cJp, i), 1:q)

        @. pre = mid(Jxsto)
        lu!(pre)
        mul!(B_temp, Jxsto, A_Q[2])
        Bj .= pre/B_temp
        C_temp .= pre/Jxsto
        @. Cj = -C_temp
        Cj += I
        @. Uj = X_predict - xval_predict

        @. Jpdiff = Jpsto - d.Jf!_correct.Jpsto
        X_computed .= (xval_predict + Bj*Δ[2] + Cj*Uj + (pre\Jpdiff)*rP + pre*δⱼ₊₁) .∩ X_predict

        contract_constant_state!(X_computed, d.constant_state_bounds)
        affine_contract!(X_computed, P, pval, np, nx)
        @. xval_computed = mid(X_computed)

        # calculation block for computing Aⱼ₊₁ and inv(Aⱼ₊₁)
        cJmid = Jxsto*A_Q[2]
        calculateQ!(A_Q[1], mid.(cJmid))
        calculateQinv!(A_inv[1], A_Q[1])
        pre2 = A_inv[1]*pre
        Δⱼ₊₁ .= pre2*δⱼ₊₁ + (pre2*Jpdiff)*rP + (pre*cJmid)*Δ[2] + A_inv[1]*(xval_computed - xval_predict) + (A_inv[1]*Cj)*(X_computed - xval_predict)
        @. Δ[1] = Δⱼ₊₁ .∩ ho_predict_Δⱼ₊₁
    end

    return RELAXATION_NOT_CALLED
end

get_Δ(f::HermiteObreschkoffFunctor) = f.Δⱼ₊₁