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
# src/DiscretizeRelax/method/single_step.jl
# Defines a single step of the integration method.
#############################################################################

"""
StepParams

LEPUS and Integration parameters.

$(TYPEDFIELDS)
"""
Base.@kwdef struct StepParams
    "Error tolerance of integrator"
    tol::Float64 = 1E-5
    "Minimum stepsize"
    hmin::Float64 = 1E-8
    "Number of repetitions allowed for refinement"
    repeat_limit::Int64 = 3
    "Indicates an adaptive stepsize is used"
    is_adaptive::Bool = true
    "Indicates the contractor step should be skipped"
    skip_step2::Bool = false
end

"""
StepResult{S}

Results passed to the next step.

$(TYPEDFIELDS)
"""
mutable struct StepResult{S}
    "nominal value of the state variables"
    xⱼ::Vector{Float64}
    "relaxations/bounds of the state variables"
    Xⱼ::Vector{S}
    "storage for parallelepid enclosure of `xⱼ`"
    A_Q::FixedCircularBuffer{Matrix{Float64}}
    A_inv::FixedCircularBuffer{Matrix{Float64}}
    "storage for parallelepid enclosure of `xⱼ`"
    Δ::FixedCircularBuffer{Vector{S}}
    "predicted step size for next step"
    predicted_hj::Float64
    "new time"
    time::Float64
end

"""
ExistStorage{F,K,S,T}

Storage used in the existence and uniqueness tests.
"""
mutable struct ExistStorage{F,K,S,T}
    status_flag::TerminationStatusCode
    hj::Float64
    hmin::Float64
    is_adaptive::Bool
    ϵ::Float64
    αfrac::Float64
    k::Int64
    predicted_hj::Float64
    computed_hj::Float64
    hj_max::Float64
    nx::Int64
    f_coeff::Vector{Vector{T}}
    f_temp_PU::Vector{Vector{T}}
    f_temp_tilde::Vector{Vector{T}}
    f_flt::Vector{Float64}
    hfk::Vector{T}
    fk::Vector{T}
    β::Vector{Float64}
    poly_term::Vector{T}
    Xj_0::Vector{T}
    Xj_apriori::Vector{T}
    Vⱼ::Vector{T}
    Uⱼ::Vector{T}
    Z::Vector{T}
    P::Vector{T}
    tf!::TaylorFunctor!{F,K,S,T}
end

function ExistStorage(tf!::TaylorFunctor!{F,K,S,T}, s::T, P, nx::Int64, np::Int64, k::Int64, h::Float64, cap::Int64) where {F,K,S,T}


    flag = RELAXATION_NOT_CALLED
    Xapriori = zeros(T, nx)
    Xj_0 = zeros(T, nx)
    Xj_apriori = zeros(T, nx)

    f_coeff = Vector{T}[]
    ftilde = Vector{T}[]
    fPU = Vector{T}[]
    for i in 1:(k + 1)
        push!(f_coeff, zeros(T, nx))
        push!(ftilde, zeros(T, nx))
        push!(fPU, zeros(T, nx))
    end

    hfk = zeros(T, nx)
    fk = zeros(T, nx)
    Z = zeros(T, nx)

    ϵ = 0.5
    αfrac = 0.5

    poly_term = zeros(T, nx)
    β = zeros(Float64, nx)
    f_flt = zeros(Float64, nx)
    Vⱼ = zeros(T, nx)
    Uⱼ = zeros(T, nx)
    Z = zeros(T, nx)

    return ExistStorage{F,K,S,T}(flag, h, h, (h === 0.0), ϵ, αfrac, k, h, h, Inf, nx,
                                 f_coeff, fPU, ftilde, f_flt, hfk, fk, β, poly_term,
                                 Xj_0, Xj_apriori, Vⱼ, Uⱼ, Z, P, tf!)
end

"""
ContractorStorage{S}

Storage used to hold inputs to the contractor method used.
"""
mutable struct ContractorStorage{S}
    is_adaptive::Bool
    times::FixedCircularBuffer{Float64}
    steps::FixedCircularBuffer{Float64}
    Xj_0::Vector{S}
    Xj_apriori::Vector{S}
    xval::Vector{Float64}
    A_Q::FixedCircularBuffer{Matrix{Float64}}
    A_inv::FixedCircularBuffer{Matrix{Float64}}
    Δ::FixedCircularBuffer{Vector{S}}
    P::Vector{S}
    rP::Vector{S}
    pval::Vector{Float64}
    fk_apriori::Vector{S}
    hj_computed::Float64
    X_computed::Vector{S}
    xval_computed::Vector{Float64}
    B::Matrix{Float64}
    γ::Float64
    step_count::Int64
    nx::Int
end
function ContractorStorage(style::S, nx, np, k, h, method_steps) where S
    is_adaptive = h <= 0.0
    # add initial storage
    Xj_0 = zeros(S, nx)
    Xj_apriori = zeros(S, nx)
    xval = zeros(Float64, nx)
    xval_computed = zeros(Float64, nx)
    P = zeros(S, np)
    rP = zeros(S, np)
    pval = zeros(Float64, np)
    fk_apriori = zeros(S, nx)
    hj_computed = 0.0
    X_computed = zeros(S, nx)
    xval_computed = zeros(Float64, nx)
    B = zeros(Float64, nx, nx)
    γ = 0.0
    step_count = 1

    # add to buffer
    times = FixedCircularBuffer{Float64}(method_steps);  append!(times, zeros(nx))
    steps = FixedCircularBuffer{Float64}(method_steps);  append!(steps, zeros(nx))
    Δ = FixedCircularBuffer{Vector{S}}(method_steps)
    A_Q = FixedCircularBuffer{Matrix{Float64}}(method_steps)
    A_inv = FixedCircularBuffer{Matrix{Float64}}(method_steps)
    for i = 1:method_steps
        push!(Δ, zeros(S, nx))
        push!(A_Q, Float64.(Matrix(I, nx, nx)))
        push!(A_inv, Float64.(Matrix(I, nx, nx)))
    end

    return ContractorStorage{S}(is_adaptive, times, steps, Xj_0, Xj_apriori, xval,
                                A_Q, A_inv, Δ, P, rP, pval, fk_apriori,
                                hj_computed, X_computed, xval_computed, B, γ, step_count, nx)
end

function advance_contract_storage!(d::ContractorStorage{S}) where {S <: Number}
    cycle!(d.A_Q)
    cycle!(d.A_inv)
    cycle!(d.Δ)
    return nothing
end

function set_xX!(result::StepResult{S}, contract::ContractorStorage{S}) where {S <: Number}
    result.Xⱼ .= contract.X_computed
    result.xⱼ .= contract.xval_computed
    contract.Xj_0 .= contract.X_computed
    contract.xval .= contract.xval_computed
    return nothing
end

"""
excess_error

Computes the excess error using a norm-∞ of the diameter of the vectors.
"""
function excess_error(Z::Vector{S}, hj::Float64, hj_eu::Float64, γ::Float64, k::Int64, nx::Int64) where S
    errⱼ = 0.0; dₜ = 0.0
    for i = 1:nx
        dₜ = (hj/hj_eu)*diam(@inbounds Z[i])
        errⱼ = (dₜ > errⱼ) ? dₜ : errⱼ
    end
    γ*errⱼ
end

function affine_contract!(X::Vector{Interval{Float64}}, P::Vector{Interval{Float64}},
                          pval::Vector{Float64}, np::Int, nx::Int)
    return nothing
end

function affine_contract!(X::Vector{MC{N,T}}, P::Vector{MC{N,T}}, pval::Vector{Float64},
                          np::Int, nx::Int) where {N,T<:RelaxTag}
    x_Intv_cv = 0.0
    x_Intv_cc = 0.0
    for i = 1:nx
        Xt = @inbounds X[i]
        x_Intv_cv = Xt.cv
        x_Intv_cc = Xt.cc
        for j = 1:np
            p = @inbounds pval[j]
            pL = @inbounds P[j].Intv.lo
            pU = @inbounds P[j].Intv.hi
            cv_gradj = Xt.cv_grad[j]
            cc_gradj = Xt.cc_grad[j]
            x_Intv_cv += (cv_gradj > 0.0) ? cv_gradj*(pL - p) : cv_gradj*(pU - p)
            x_Intv_cc += (cc_gradj < 0.0) ? cc_gradj*(pL - p) : cc_gradj*(pU - p)
        end
        x_Intv_cv = max(x_Intv_cv, Xt.Intv.lo)
        x_Intv_cc = min(x_Intv_cc, Xt.Intv.hi)
        X[i] = MC{N,T}(Xt.cv, Xt.cc, Interval(x_Intv_cv, x_Intv_cc), Xt.cv_grad, Xt.cc_grad, Xt.cnst)
    end
    return nothing
end

"""
single_step!

Performs a single-step of the validated integrator. Input stepsize is out.step.
"""
function single_step!(exist::ExistStorage{F,K,S,T}, contract::ContractorStorage{T},
                      params::StepParams, result::StepResult{T}, sc::M,
                      j::Int64, hj_limit::Float64, delT) where {M <: AbstractStateContractor, F, K, S <: Real, T}

    contract.is_adaptive = params.is_adaptive

    # validate existence & uniqueness (returns if E&U cannot be shown)
    existence_uniqueness!(exist, params, result.time, j)
    if exist.status_flag === NUMERICAL_ERROR
        if delT < 1E-6
            exist.status_flag = COMPLETED
        end
        return nothing
    end

    advance_contractor_buffer!(sc)::Nothing
    advance_contract_storage!(contract)

    # copy info from existence to contractor storage
    contract.Xj_apriori .= exist.Xj_apriori
    contract.hj_computed = min(exist.computed_hj, hj_limit)
    contract.fk_apriori .= exist.fk

    hj = contract.hj_computed
    hj_eu = hj
    predicted_hj = 0.0

    # begin contractor step
    count = 0
    if !params.skip_step2
        if params.is_adaptive
            while hj > params.hmin && count < params.repeat_limit
                sc(contract, result, count)
                # LEPUS STEPSIZE PREDICTION
                errj = excess_error(exist.Z, hj, hj_eu, contract.γ, exist.k, exist.nx)
                if errj < hj*params.tol
                    contract.hj_computed = 0.9*hj*(0.5*hj*params.tol/errj)^(1/(exist.k-1))
                    break
                else
                    hj_reduced = hj*(hj*params.tol/errj)^(1/(exist.k-1))
                    exist.Z *= (hj_reduced/hj)^exist.k
                    hj = hj_reduced
                    contract.hj_computed = hj
                end
                count += 1
            end

            set_xX!(result, contract)::Nothing
        else
            # perform corrector step
            sc(contract, result, 0)
            set_xX!(result, contract)::Nothing
        end
    else
        result.xⱼ .= mid.(exist.Xapriori)
        result.Xⱼ .= exist.Xapriori
    end

    # update parallelepid enclosure
    #@show j
    cycle_copyto!(result.A_Q, contract.A_Q[1], j)
    cycle_copyto!(result.A_inv, contract.A_inv[1], j)
    cycle_copyto!(result.Δ, contract.Δ[1], j)

    # store times and step sizes to time/step buffer
    # and updated prediced step size
    result.time += hj
    result.predicted_hj = predicted_hj

    exist.Xj_0 .= result.Xⱼ
    exist.predicted_hj = contract.hj_computed

    return nothing
end
