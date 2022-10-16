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
# src/DiscretizeRelax/method/higher_order_enclosure.jl
# Defines higher-order existence and uniqueness tests.
#############################################################################

set_mag(x::Interval{Float64}) = mag(x)
set_mag(x::MC{N,T}) where {N, T<:RelaxTag} = mag(x.Intv)

const SUBSET_TOL = 1E-8
is_subset_tol(x::Interval{T}, y::Interval{T}) where {T<:Real} = (x.lo > y.lo - SUBSET_TOL) && (x.hi < y.hi + SUBSET_TOL) 
contains(x::Vector{Interval{T}}, y::Vector{Interval{T}}) where {T<:Real} = mapreduce(is_subset_tol, &, x, y)
contains(x::Vector{MC{N,T}}, y::Vector{MC{N,T}}) where {N,T<:RelaxTag} = mapreduce((x,y)->is_subset_tol(x.Intv,y.Intv), &, x, y)

function round_β!(β::Vector{S}, ϵ, nx) where S
    for i = 1:nx
        if isapprox(β[i], zero(S), atol=1E-10)
            β[i] = ϵ
        end
    end
end

const ALPHA_ATOL = 1E-10
const ALPHA_RTOL = 1E-10
const ALPHA_BOUND_TOL = 1E-13
const ALPHA_ITERATION_LIMIT = 1000

function inner_α_func(z::MC{N,T}, u::MC{N,T}) where {N,T}
    max(abs(max(z.Intv.lo, z.Intv.hi) - u.Intv.hi), abs(u.Intv.lo - min(z.Intv.lo, z.Intv.hi)))
end
function inner_α_func(z::Interval{Float64}, u::Interval{Float64})
    max(abs(max(z.lo, z.hi) - u.hi), abs(u.lo - min(z.lo, z.hi)))
end
α_func(Vⱼ, Uⱼ, α, k) = mapreduce(inner_α_func, max, Interval(0.0, α^k).*Vⱼ, Uⱼ)

"""
calc_alpha

Computes the stepsize for the adaptive step-routine via a golden section rootfinding method.
The step size is rounded down.
"""
function α(Vⱼ::Vector{T}, Uⱼ::Vector{T}, k) where T <: Number
    αL = 0.0 + ALPHA_BOUND_TOL
    αU = 1.0 - ALPHA_BOUND_TOL
    golden_ratio = 0.5*(3.0 - sqrt(5.0))
    α = αL + golden_ratio*(αU - αL)
    fα = α_func(Vⱼ, Uⱼ, α, k)
    iteration = 0
    converged = false

    while iteration < ALPHA_ITERATION_LIMIT
        if abs(α - (αU + αL)/2) <= 2*(ALPHA_RTOL*abs(α) + ALPHA_ATOL) - (αU - αL)/2
            converged = true
            break
        end
        iteration += 1
        if αU - α > α - αL
            new_α = α + golden_ratio*(αU - α)
            new_f = α_func(Vⱼ, Uⱼ, new_α, k)
            if new_f < fα
                αL = α
                α = new_α
                fα = new_f
            else
                αU = new_α
            end
        else
            new_α = α - golden_ratio*(α - αL)
            new_f = α_func(Vⱼ, Uⱼ, new_α, k)
            if new_f < fα
                αU = α
                α = new_α
                fα = new_f
            else
                αL = new_α
            end
        end
    end
    !converged && error("Alpha calculation not converged.")
    return min(α - ALPHA_ATOL, α*(1-ALPHA_RTOL))  
end

"""
existence_uniqueness!

Implements the adaptive higher-order enclosure approach detailed in Nedialkov's
dissertation (Nedialko S. Nedialkov. Computing rigorous bounds on the solution of
an initial value problem for an ordinary differential equation. 1999. Universisty
of Toronto, PhD Dissertation, Algorithm 5.1, page 73-74). The arguments are
`s::ExistStorage{F,K,S,T}, params::StepParams, t::Float64, j::Int64`.
"""
function existence_uniqueness!(s::ExistStorage{F,K,S,T}, params::StepParams, t, j) where {F, K, S, T <: Number}

    @unpack predicted_hj, Xj_0, Xj_apriori, poly_term, f_coeff, f_temp_tilde, f_temp_PU = s
    @unpack k, ϵ, P, Vⱼ, Uⱼ, Z, β, fk, tf!, nx, hfk = s
    @unpack hmin, is_adaptive = params

    # compute coefficients
    tf!(f_coeff, Xj_0, P, t)
    @. poly_term = f_coeff[k]
    for i in (k-1):-1:1
        @. poly_term = f_coeff[i] + poly_term*Interval(0.0, predicted_hj)
    end

    if isone(j)
        @. Uⱼ = 2.0*Interval(-1.0, 1.0)*abs(predicted_hj^k)*Interval(set_mag(f_coeff[k + 1]))
        tf!(f_temp_PU, poly_term + Uⱼ, P, t)
        @. β = set_mag(f_temp_PU[k + 1])
    else
        @. β = set_mag(fk)
    end
    round_β!(β, ϵ, nx)

    @. Uⱼ = 2.0*(predicted_hj^k)*Interval(-β, β)
    @. Xj_apriori = poly_term + Uⱼ

    tf!(f_temp_tilde, Xj_apriori, P, t)
    @. Z = (predicted_hj^(k+1))*f_temp_tilde[k + 1]
    @. Vⱼ = Interval(0.0, 1.0)*Z

    # checks existence and uniqueness by proper enclosure of Vⱼ & Uⱼ and computes the next appropriate step size
    if !contains(Vⱼ, Uⱼ)
        if !is_adaptive
            print_iteration(j) && @show Vⱼ, Uⱼ
            s.status_flag = NUMERICAL_ERROR
            return false
        end
        s.computed_hj = α(Vⱼ, Uⱼ, k)*predicted_hj
    else 
        s.computed_hj = predicted_hj
    end
    if s.computed_hj < hmin
        print_iteration(j) && @show s.computed_hj
        s.status_flag = NUMERICAL_ERROR
        return false
    end

    # save outputs
    @. hfk = Z
    @. fk = f_temp_tilde[k + 1]
    @. f_coeff[k + 1] = fk

    return true
end