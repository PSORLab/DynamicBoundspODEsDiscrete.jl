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
# src/DiscretizeRelax/utilities/discretize_relax.jl
# Defines the structure used to compute bounds/relaxations via a discretize
# and relax approach.
#############################################################################

"""
DiscretizeRelax

An integrator for discretize and relaxation techniques.

$(TYPEDFIELDS)
"""
mutable struct DiscretizeRelax{M <: AbstractStateContractor, T <: Number, S <: Real, F, K, X, NY, JX, JP} <: AbstractODERelaxIntegrator

    # Problem description
    "Initial Conditiion for pODEs"
    x0f::X
    "Jacobian w.r.t x"
    Jx!::JX
    "Jacobian w.r.t p"
    Jp!::JP
    "Parameter value for pODEs"
    p::Vector{Float64}
    "Lower Parameter Bounds for pODEs"
    pL::Vector{Float64}
    "Upper Parameter Bounds for pODEs"
    pU::Vector{Float64}
    "Number of state variables"
    nx::Int
    "Number of decision variables"
    np::Int
    "Time span to integrate over"
    tspan::Tuple{Float64, Float64}
    "Individual time points to evaluate"
    tsupports::Vector{Float64}

    # Options and internal storage
    "Maximum number of integration steps"
    step_limit::Int
    "Steps taken"
    step_count::Int
    "Stores solution X (from step 2) for each time"
    storage::Vector{Vector{T}}
    "Stores solution X (from step 1) for each time"
    storage_apriori::Vector{Vector{T}}
    "Stores each time t"
    time::Vector{Float64}
    "Support index to storage dictory"
    support_dict::Dict{Int,Int}
    "Holds data for numeric error encountered in integration step"
    error_code::TerminationStatusCode
    "Storage for bounds/relaxation of P"
    P::Vector{T}
    "Storage for bounds/relaxation of P - p"
    rP::Vector{T}
    "Relaxation Type"
    style::T
    "Flag indicating that only apriori bounds should be computed"
    skip_step2::Bool

    # Main functions used in routines
    "Functor for evaluating Taylor coefficients over a set"
    set_tf!::TaylorFunctor!{F,K,S,T}
    method_f!::M

    exist_result::ExistStorage{F,K,S,T}
    contractor_result::ContractorStorage{T}
    step_result::StepResult{T}
    step_params::StepParams

    new_decision_pnt::Bool
    new_decision_box::Bool
end

function DiscretizeRelax(d::ODERelaxProb, m::SCN; repeat_limit = 50, step_limit = 1000,  tol = 1E-1, hmin = 1E-13,
                         relax = false, h = 0.0, skip_step2 = false, Jx! = nothing, Jp! = nothing) where SCN <: AbstractStateContractorName

    γ = state_contractor_γ(m)::Float64
    k = state_contractor_k(m)::Int
    method_steps = state_contractor_steps(m)::Int

    tsupports = d.tsupports
    if ~isempty(tsupports)
        if (tsupports[1] == 0.0)
            support_dict = Dict{Int,Int}(d.support_dict, 1 => 1)
        end
    else
        support_dict = Dict{Int,Int}()
    end
    error_code = RELAXATION_NOT_CALLED

    T = relax ? MC{d.np,NS} : Interval{Float64}
    style = zero(T)
    time = zeros(Float64, 1000)
    storage = Vector{T}[]
    storage_apriori = Vector{T}[]
    for i = 1:1000
        push!(storage, zeros(T, d.nx))
        push!(storage_apriori, zeros(T, d.nx))
    end
    P = zeros(T, d.np)
    rP = zeros(T, d.np)

    A = qr_stack(d.nx, method_steps)
    Δ = CircularBuffer{Vector{T}}(method_steps)
    fill!(Δ, zeros(T, d.nx))

    state_method = state_contractor(m, d.f, Jx!, Jp!, d.nx, d.np, style, zero(Float64), h)
    is_adaptive = (h <= 0.0)

    set_tf! = TaylorFunctor!(d.f, d.nx, d.np, Val(k), style, zero(Float64))
    exist_storage = ExistStorage(set_tf!, style, P, d.nx, d.np, k, h, method_steps)
    contractor_result = ContractorStorage(style, d.nx, d.np, k, h, method_steps)
    contractor_result.γ = γ
    contractor_result.P = P
    contractor_result.rP = rP

    contractor_result.is_adaptive = is_adaptive
    step_params = StepParams(tol, hmin, repeat_limit, is_adaptive, skip_step2)
    step_result = StepResult{typeof(style)}(zeros(d.nx), zeros(typeof(style), d.nx), A, Δ, 0.0, 0.0)

    return DiscretizeRelax{typeof(state_method), T,
                           Float64, typeof(d.f), k+1, typeof(d.x0),
                           d.nx+d.np, typeof(Jx!), typeof(Jp!)}(d.x0, Jx!, Jp!, d.p, d.pL, d.pU, d.nx,
                           d.np, d.tspan, d.tsupports, step_limit, 0, storage, storage_apriori, time,
                           support_dict, error_code, P, rP, skip_step2, style, set_tf!, state_method,
                           exist_storage, contractor_result, step_result, step_params, true, true)
end
function DiscretizeRelax(d::ODERelaxProb; kwargs...)
    DiscretizeRelax(d, LohnerContractor{4}(); kwargs...)
end

"""
set_P!(d::DiscretizeRelax)

Initializes the `P` and `rP` (P - p) fields of `d`.
"""
function set_P!(d::DiscretizeRelax{M,Interval{Float64},S,F,K,X,NY}) where {M<:AbstractStateContractor, S, F, K, X, NY}

    @__dot__ d.P = Interval(d.pL, d.pU)
    @__dot__ d.rP = d.P - d.p

    return nothing
end

function set_P!(d::DiscretizeRelax{M,MC{N,T},S,F,K,X,NY}) where {M<:AbstractStateContractor, T<:RelaxTag, S <: Real, F, K, X, N, NY}

    @__dot__ d.P = MC{N,NS}.(d.p, Interval(d.pL, d.pU), 1:d.np)
    @__dot__ d.rP = d.P - d.p

    return nothing
end

"""
compute_X0!(d::DiscretizeRelax)

Initializes the circular buffer that holds `Δ` with the `out - mid(out)` at
index 1 and a zero vector at all other indices.
"""
function compute_X0!(d::DiscretizeRelax)

    d.storage[1] .= d.x0f(d.P)
    d.storage_apriori[1] .= d.storage[1]

    d.step_result.Xⱼ .= d.storage[1]
    d.step_result.xⱼ .= mid.(d.step_result.Xⱼ)

    d.exist_result.Xj_0 .= d.step_result.Xⱼ
    d.exist_result.predicted_hj = d.tspan[2] - d.tspan[1]

    d.contractor_result.Xj_0 .= d.step_result.Xⱼ
    d.contractor_result.xval .= mid.(d.exist_result.Xj_0)
    d.contractor_result.pval .= d.p

    return nothing
end

"""
set_Δ!

Initializes the circular buffer, `Δ`, that holds `Δ_i` with the `out - mid(out)` at
index 1 and a zero vector at all other indices.
"""
function set_Δ!(Δ::CircularBuffer{Vector{T}}, out::Vector{Vector{T}}) where T

    Δ[1] .= out[1] .- mid.(out[1])
    for i = 2:length(Δ)
        fill!(Δ[i], zero(T))
    end

    return nothing
end
