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
Base.@kwdef mutable struct DiscretizeRelax{M <: AbstractStateContractor, T <: Number, S <: Real, F, K, X, NY, JX, JP, INT, N} <: AbstractODERelaxIntegrator

    # Problem description
    "Initial Condition for pODEs"
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
    next_support_i::Int   = -1
    next_support::Float64 = -Inf

    # Options and internal storage
    "Maximum number of integration steps"
    step_limit::Int = 500
    "Steps taken"
    step_count::Int = 0
    "Stores solution X (from step 2) for each time"
    storage::Vector{Vector{T}}
    "Stores solution X (from step 1) for each time"
    storage_apriori::Vector{Vector{T}}
    "Stores each time t"
    time::Vector{Float64} = zeros(Float64, 200)
    "Support index to storage dictory"
    support_dict::Dict{Int,Int} = Dict{Int,Int}()
    "Holds data for numeric error encountered in integration step"
    error_code::TerminationStatusCode = RELAXATION_NOT_CALLED
    "Storage for bounds/relaxation of P"
    P::Vector{T}
    "Storage for bounds/relaxation of P - p"
    rP::Vector{T}
    "Relaxation Type"
    style::T
    "Flag indicating that only apriori bounds should be computed"
    skip_step2::Bool = false
    storage_buffer_size::Int = 500
    print_relax_time::Bool = false

    # Main functions used in routines
    "Functor for evaluating Taylor coefficients over a set"
    set_tf!::TaylorFunctor!{F,K,S,T}
    method_f!::M

    exist_result::ExistStorage{F,K,S,T}
    contractor_result::ContractorStorage{T}
    step_result::StepResult{T}
    step_params::StepParams

    new_decision_pnt::Bool = true
    new_decision_box::Bool = true

    relax_t_dict_indx::Dict{Int,Int}    = Dict{Int,Int}()
    relax_t_dict_flt::Dict{Float64,Int} = Dict{Float64,Int}()

    calculate_local_sensitivity::Bool = false
    local_problem_storage

    constant_state_bounds::Union{Nothing,ConstantStateBounds}
    polyhedral_constraint::Union{Nothing,PolyhedralConstraint}
    prob
end

function DiscretizeRelax(d::ODERelaxProb, m::SCN; repeat_limit = 1, tol = 1E-4, hmin = 1E-13,  relax = false, 
                         h = 0.0, J_x! = nothing, J_p! = nothing, storage_buffer_size = 200, skip_step2 = false, 
                         atol = 1E-5, rtol = 1E-5, print_relax_time = true, kwargs...) where SCN <: AbstractStateContractorName

    Jx! = isnothing(J_x!) ? d.Jx! : J_x!
    Jp! = isnothing(J_p!) ? d.Jp! : J_p!

    γ = state_contractor_γ(m)::Float64
    k = state_contractor_k(m)::Int
    method_steps = state_contractor_steps(m)::Int

    tsupports = d.support_set.s

    T = relax ? MC{d.np,NS} : Interval{Float64}
    style = zero(T)
    storage = Vector{T}[]
    storage_apriori = Vector{T}[]
    for i = 1:storage_buffer_size
        push!(storage, zeros(T, d.nx))
        push!(storage_apriori, zeros(T, d.nx))
    end
    P = zeros(T, d.np)
    rP = zeros(T, d.np)

    Δ = FixedCircularBuffer{Vector{T}}(method_steps)
    A_Q = FixedCircularBuffer{Matrix{Float64}}(method_steps)
    A_inv = FixedCircularBuffer{Matrix{Float64}}(method_steps)
    for i = 1:method_steps
        push!(Δ, zeros(T, d.nx))
        push!(A_Q, Float64.(Matrix(I, d.nx, d.nx)))
        push!(A_inv, Float64.(Matrix(I, d.nx, d.nx)))
    end

    state_method = state_contractor(m, d.f, Jx!, Jp!, d.nx, d.np, style, zero(Float64), h)

    set_tf! = TaylorFunctor!(d.f, d.nx, d.np, Val(k), style, zero(Float64))

    is_adaptive = (h <= 0.0)
    contractor_result = ContractorStorage(style, d.nx, d.np, k, h, method_steps)
    contractor_result.γ = γ
    contractor_result.P = P
    contractor_result.rP = rP
    contractor_result.is_adaptive = is_adaptive

    local_integrator = state_contractor_integrator(m)
    return DiscretizeRelax{typeof(state_method), T, Float64, typeof(d.f), k+1,typeof(d.x0), d.nx + d.np, 
                           typeof(Jx!), typeof(Jp!), typeof(local_integrator), d.np}(
                           x0f = d.x0, 
                           Jx! = Jx!, 
                           Jp! = Jp!, 
                           p = d.p, 
                           pL = d.pL, 
                           pU = d.pU, 
                           nx = d.nx, 
                           np = d.np, 
                           tspan = d.tspan, 
                           tsupports = tsupports,  
                           storage = storage, 
                           storage_apriori = storage_apriori, 
                           P = P, 
                           rP = rP, 
                           style = style,
                           set_tf! = set_tf!, 
                           method_f! = state_method, 
                           exist_result = ExistStorage(set_tf!, style, P, d.nx, d.np, k, h, method_steps), 
                           contractor_result = contractor_result,
                           step_result = StepResult{typeof(style)}(zeros(d.nx), zeros(typeof(style), d.nx), A_Q, A_inv, Δ, 0.0, 0.0), 
                           step_params = StepParams(atol, rtol, hmin, repeat_limit, is_adaptive, skip_step2), 
                           local_problem_storage = ODELocalIntegrator(d, local_integrator), 
                           constant_state_bounds = d.constant_state_bounds,
                           polyhedral_constraint = d.polyhedral_constraint, 
                           prob = d,
                           storage_buffer_size = storage_buffer_size,
                           skip_step2 = skip_step2,
                           print_relax_time = print_relax_time)
end

DiscretizeRelax(d::ODERelaxProb; kwargs...) = DiscretizeRelax(d, LohnerContractor{4}(); kwargs...)

"""
set_P!(d::DiscretizeRelax)

Initializes the `P` and `rP` (P - p) fields of `d`.
"""
function set_P!(d::DiscretizeRelax{M,Interval{Float64},S,F,K,X,NY}) where {M<:AbstractStateContractor, S, F, K, X, NY}
    @unpack p, pL, pU, P, rP = d

    @. P = Interval(pL, pU)
    @. rP = P - p

    nothing
end

function set_P!(d::DiscretizeRelax{M,MC{N,T},S,F,K,X,NY}) where {M<:AbstractStateContractor, T<:RelaxTag, S <: Real, F, K, X, N, NY}
    @unpack np, p, pL, pU, P, rP = d

    @. P = MC{N,NS}(p, Interval(pL, pU), 1:np)
    @. rP = P - p

    nothing
end

"""
compute_X0!(d::DiscretizeRelax)

Initializes the circular buffer that holds `Δ` with the `out - mid(out)` at
index 1 and a zero vector at all other indices.
"""
function compute_X0!(d::DiscretizeRelax)
    @unpack relax_t_dict_indx, relax_t_dict_flt, storage, storage_apriori, tspan, x0f, P, tsupports = d

    storage[1] .= x0f(P)
    storage_apriori[1] .= storage[1]

    if tspan[1] ∈ tsupports
        relax_t_dict_indx[1] = 1
        relax_t_dict_flt[tspan[1]] = 1
    end

    d.step_result.Xⱼ .= storage[1]
    d.step_result.xⱼ .= mid.(d.step_result.Xⱼ)

    d.exist_result.Xj_0 .= d.step_result.Xⱼ

    d.contractor_result.Xj_0 .= d.step_result.Xⱼ
    d.contractor_result.xval .= mid.(d.exist_result.Xj_0)
    d.contractor_result.pval .= d.p

    nothing
end

"""
set_Δ!

Initializes the circular buffer, `Δ`, that holds `Δ_i` with the `out - mid(out)` at
index 1 and a zero vector at all other indices.
"""
function set_Δ!(Δ::FixedCircularBuffer{Vector{T}}, out::Vector{Vector{T}}) where T

    Δ[1] .= out[1] .- mid.(out[1])
    for i = 2:length(Δ)
        fill!(Δ[i], zero(T))
    end

    return nothing
end
