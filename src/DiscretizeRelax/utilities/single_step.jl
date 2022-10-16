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
    "Absolute error tolerance of integrator"
    atol::Float64 = 1E-5
    "Relative error tolerance of integrator"
    rtol::Float64 = 1E-5
    "Minimum stepsize"
    hmin::Float64 = 1E-8
    "Number of repetitions allowed for refinement"
    repeat_limit::Int = 3
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
    k::Int
    predicted_hj::Float64
    computed_hj::Float64
    hj_max::Float64
    nx::Int
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
    constant_state_bounds::Union{Nothing,ConstantStateBounds}
end

function ExistStorage(tf!::TaylorFunctor!{F,K,S,T}, s::T, P, nx::Int, np::Int, k::Int, h::Float64, cap::Int) where {F,K,S,T}


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

    poly_term = zeros(T, nx)
    β = zeros(Float64, nx)
    f_flt = zeros(Float64, nx)
    Vⱼ = zeros(T, nx)
    Uⱼ = zeros(T, nx)
    Z = zeros(T, nx)

    return ExistStorage{F,K,S,T}(flag, h, h, (h === 0.0), ϵ, k, h, h, Inf, nx,
                                 f_coeff, fPU, ftilde, f_flt, hfk, fk, β, poly_term,
                                 Xj_0, Xj_apriori, Vⱼ, Uⱼ, Z, P, tf!, nothing)
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
    hj::Float64
    X_computed::Vector{S}
    xval_computed::Vector{Float64}
    B::Matrix{Float64}
    γ::Float64
    step_count::Int
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
    hj = 0.0
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
                                hj, X_computed, xval_computed, B, γ, step_count, nx)
end

function advance_contract_storage!(d::ContractorStorage{S}) where {S <: Number}
    cycle!(d.A_Q)
    cycle!(d.A_inv)
    cycle!(d.Δ)
    nothing
end

function load_existence_info_to_contractor!(c::ContractorStorage{T}, ex::ExistStorage{F,K,S,T}) where {F,K,S,T}
    c.Xj_apriori .= ex.Xj_apriori
    c.hj = min(ex.computed_hj, ex.hj_max)
    c.fk_apriori .= ex.fk
    nothing
end

function set_xX!(result::StepResult{S}, contract::ContractorStorage{S}) where {S <: Number}
    pL = lo.(contract.P)
    pU = hi.(contract.P)
    pval = contract.pval
    subgradient_expansion_interval_contract!(contract.X_computed, pval, pL, pU)
    subgradient_expansion_interval_contract!(contract.Xj_0, pval, pL, pU)
    result.Xⱼ .= contract.X_computed
    result.xⱼ .= contract.xval_computed
    contract.Xj_0 .= contract.X_computed
    contract.xval .= contract.xval_computed
    nothing
end

"""
excess_error

Computes the excess error using a norm-∞ of the diameter of the vectors.
"""
function excess_error(Z::Vector{S}, hj, hj_eu, γ, k, nx) where S
    errⱼ = 0.0; dₜ = 0.0
    for i = 1:nx
        dₜ = hj*diam(Z[i])
        errⱼ = (dₜ > errⱼ) ? dₜ : errⱼ
    end
    abs(γ*errⱼ)
end

affine_contract!(X::Vector{Interval{Float64}}, P::Vector{Interval{Float64}}, pval, np, nx) = nothing
function affine_contract!(X::Vector{MC{N,T}}, P::Vector{MC{N,T}}, pval::Vector{Float64}, np, nx) where {N,T<:RelaxTag}
    x_Intv_cv = 0.0
    x_Intv_cc = 0.0
    for i = 1:nx
        Xt = X[i]
        x_Intv_cv = Xt.cv
        x_Intv_cc = Xt.cc
        for j = 1:N
            p = pval[j]
            pL = P[j].Intv.lo
            pU = P[j].Intv.hi
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

contract_apriori!(exist::ExistStorage{F,K,S,T}, n::Nothing) where {F, K, S <: Real, T} = nothing
contract_apriori!(exist::ExistStorage{F,K,S,T}, p::PolyhedralConstraint) where {F, K, S <: Real, T} = nothing
function contract_apriori!(exist::ExistStorage{F,K,S,T}, c::ConstantStateBounds) where {F, K, S <: Real, T}
    exist.Xj_apriori .= exist.Xj_apriori .∩ Interval.(c.xL, c.xU)
    return nothing
end

function store_parallelepid_enclosure!(c::ContractorStorage{T}, r::StepResult{T}, j) where T
    cycle_copyto!(r.A_Q, c.A_Q[1], j)
    cycle_copyto!(r.A_inv, c.A_inv[1], j)
    cycle_copyto!(r.Δ, c.Δ[1], j)
    nothing
end

function reset_step_limit!(d)
    @unpack next_support, tspan = d
    @unpack time = d.step_result

    tval = round(next_support - time, digits=13)
    if tval < 0.0
        tval = Inf
    end
    d.exist_result.hj = min(d.exist_result.hj, tval, tspan[2] - time)
    d.exist_result.hj_max = tspan[2] - time
    d.exist_result.predicted_hj = min(d.exist_result.predicted_hj, tval, tspan[2] - time)

    d.contractor_result.steps[1] = d.exist_result.hj
    d.contractor_result.step_count = d.step_count
end

set_γ!(sc, c, ex, result, params) =  nothing
"""
single_step!

Performs a single-step of the validated integrator. Input stepsize is out.step.
"""
function single_step!(ex::ExistStorage{F,K,S,T}, c::ContractorStorage{T}, params::StepParams, result::StepResult{T}, sc::M, j, csb::C, pc::P, tspan) where {M <: AbstractStateContractor, F, K, S <: Real, T, C, P}

    @unpack repeat_limit, hmin, atol, rtol, skip_step2, is_adaptive = params
    @unpack hj, γ, X_computed, Δ = c
    @unpack k, nx = ex

    set_γ!(sc, c, ex, result, params)
    if !existence_uniqueness!(ex, params, result.time, j)  # validate existence & uniqueness
        return nothing
    end
    advance_contract_storage!(c)                           # Advance polyhedral storage
    contract_apriori!(ex, csb)::Nothing                    # Apply constant state bound contractor (if set)
    contract_apriori!(ex, pc)::Nothing                     # Apply polyhedral contractor (if set)
    load_existence_info_to_contractor!(c, ex)
    hj_eu = c.hj

    if skip_step2                                          # Skip contractor and use state values for existence test
        result.xⱼ .= mid.(ex.Xj_apriori)
        result.Xⱼ .= ex.Xj_apriori
        c.X_computed .= ex.Xj_apriori
    else                                               # Apply contractor in LEPUS stepsize scheme
        if is_adaptive
            count = 0
            while (c.hj > hmin) & (count < repeat_limit)
                count += 1
                sc(c, result, count, j)
                errj = excess_error(ex.Z, c.hj, hj_eu, γ, k, nx)
                if (errj < c.hj*rtol) || (errj < atol)
                    sign_tstep = copysign(1, tspan[2] - result.time)
                    next_t = sign_tstep*(result.time + c.hj)
                    t_end = sign_tstep*tspan[2]
                    if (next_t < t_end) & !isapprox(next_t, t_end, atol = 1E-8)
                        δerrj = max(1E-11, errj)
                        max_new_step = 1.015*c.hj
                        c.hj = 0.9*c.hj*(0.5*c.hj*rtol/δerrj)^(1/(k-1))
                        c.hj = min(max_new_step, c.hj, t_end - result.time)
                    end
                    break
                else
                    hj_reduced = c.hj*(c.hj*atol/errj)^(1/(k-1))
                    ex.Z *= (hj_reduced/c.hj)^k
                    c.hj = hj_reduced
                end
            end

            set_xX!(result, c)::Nothing
        else
            sc(c, result, 0, k)
            set_xX!(result, c)::Nothing
        end
        store_parallelepid_enclosure!(c, result, j)        # update parallelepid enclosure
    end
end
function single_step!(d)
    @unpack next_support, next_support_i, tspan = d
    reset_step_limit!(d)
    single_step!(d.exist_result, d.contractor_result, d.step_params, d.step_result, d.method_f!, d.step_count, d.constant_state_bounds, d.polyhedral_constraint, tspan)
end