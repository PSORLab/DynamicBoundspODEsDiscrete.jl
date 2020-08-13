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

"""
$(FUNCTIONNAME)

Fast check for to see if the ratio of the L∞ norm is improving in a given iteration
using a hard-code ratio tolerance of 1.01. This is the improvement condition from
Nedialko S. Nedialkov. Computing rigorous bounds on the solution of an initial
value problem for an ordinary differential equation. 1999. Universisty of Toronto,
PhD Dissertation, Algorithm 5.1, page 73-74).
"""
function improvement_condition(X̃ⱼ::Vector{Interval{T}}, X̃ⱼ₀::Vector{Interval{T}}, nx::Int) where {T <: Real}
    Y0norm = 0.0
    Ynorm = 0.0
    diam1 = 0.0
    diam2 = 0.0
    for i in 1:nx
        diam1 = diam(X̃ⱼ[i])
        diam2 = diam(X̃ⱼ₀[i])
        Ynorm = (diam1 > Ynorm) ? diam1 : Ynorm
        Y0norm = (diam2 > Y0norm) ? diam2 : Y0norm
    end
    return (Ynorm/Y0norm) > 1.01
end

"""
$(FUNCTIONNAME)

Checks that an interval vector `Vⱼ` of length `nx` is contained in `Uⱼ`.
"""
function contains(Vⱼ::Vector{Interval{T}}, Uⱼ::Vector{Interval{T}}, nx::Int) where {T <: Real}
    flag = true
    for i = 1:nx
        if (Vⱼ[i].hi >= Uⱼ[i].hi) || (Vⱼ[i].lo <= Uⱼ[i].lo)
            flag = false
            break
        end
    end
    return flag
end

function contains(Vⱼ::Vector{MC{N,T}}, Uⱼ::Vector{MC{N,T}}, nx::Int) where {N, T <: RelaxTag}
    flag = true
    for i = 1:nx
        if (Vⱼ[i].Intv.hi >= Uⱼ[i].Intv.hi) || (Vⱼ[i].Intv.lo <= Uⱼ[i].Intv.lo)
            flag = false
            break
        end
    end
    return flag
end

function calc_alpha(Vⱼ::Vector{T}, Uⱼ::Vector{T}, αfrac::Float64, nx::Int64, k::Int64) where T <: Number
    α = Inf
    for i = 1:nx
        # unpack values from ElasticArrays
        Ui = @inbounds Uⱼ[i]
        Vi = @inbounds Vⱼ[i]
        lUi = lo(Ui); hUi = hi(Ui)
        lVi = lo(Vi); hVi = hi(Vi)

        # skip update of α if vL is zero and step is attainable
        # return if no step is attainable
        if lVi == 0.0
            if lUi > 0.0 || hUi < 0.0
                return 0.0
            end
        else
            α = min(α, exp(log(lUi/lVi)/k))
        end

        # skip update of α if vU is zero and step is attainable
        # return if no step is attainable
        if hVi == 0.0
            if lUi > 0.0 || hUi < 0.0
                return 0.0
            end
        else
            α = min(α, exp(log(hUi/hVi)/k))
        end
    end

    return α*(1.0 - αfrac)
end

function round_β!(β::Vector{S}, ϵ::Float64, nx::Int64) where S
    for i = 1:nx
        if isapprox(β[i], 0.0, atol=1E-10)
            β[i] = ϵ
        end
    end
    nothing
end

set_mag(x::Interval{Float64}) = mag(x)
set_mag(x::MC{N,T}) where {N, T<:RelaxTag} = mag(x.Intv)

"""
$(FUNCTIONNAME)

Implements the adaptive higher-order enclosure approach detailed in Nedialkov's
dissertation (Nedialko S. Nedialkov. Computing rigorous bounds on the solution of
an initial value problem for an ordinary differential equation. 1999. Universisty
of Toronto, PhD Dissertation, Algorithm 5.1, page 73-74).
"""
function existence_uniqueness!(s::ExistStorage{F,K,S,T}, params::StepParams, t::Float64, j::Int64) where {F, K, S, T <: Number}

    # if  X_apriori is too large Taylor series bounds are [-infty, infty] and the algorithm fails
    predicted_hj = min(s.predicted_hj, s.hj_max)
    accepted = false
    accepted_count = 0
    accepted_limit = 10
    while !accepted && (accepted_count < accepted_limit)
        hj_interval = Interval(0.0, predicted_hj)
        hj_rnd = 2.0*Interval(-1.0, 1.0)*predicted_hj^s.k

        # compute taylor cofficient and interval polynomial bounds via horner's method
        s.tf!(s.f_coeff, s.Xj_0, s.P, t)
        s.poly_term .+= s.f_coeff[s.k]
        for i = 1:s.k
            s.poly_term .= s.f_coeff[s.k + 1 - i] .+ s.poly_term*hj_interval
        end

        # get initial guess for highest order taylors series term
        if j === 1
            fk_abs = Interval.(set_mag.(s.f_coeff[s.k + 1]))
            tempU = hj_rnd*fk_abs
            s.Uⱼ .= tempU
            s.tf!(s.f_temp_PU, s.poly_term + s.Uⱼ, s.P, t)
            s.β .= set_mag.(s.f_temp_PU[s.k + 1])
        else
            s.β .= set_mag.(s.fk)
        end
        round_β!(s.β, s.ϵ, s.nx)

        # adds estimate of remainder with polynomial
        # and computes new estimate of remainder
        s.Uⱼ .= hj_rnd*Interval.(s.β)
        s.Xj_apriori .= s.poly_term .+ s.Uⱼ

        s.tf!(s.f_temp_tilde, s.Xj_apriori, s.P, t)
        s.Z .= (predicted_hj^s.k)*s.f_temp_tilde[s.k + 1]

        if !any(x -> (set_mag(x) == Inf), s.Z)
            accepted = true
        else
            predicted_hj *= 0.1
        end
        accepted_count += 1
    end

    s.Vⱼ .= Interval(0.0, 1.0)*s.Z

    # checks existence and uniqueness by proper
    # enclosure of Vⱼ & Uⱼ and computes the
    # next appropriate step size
    if !contains(s.Vⱼ, s.Uⱼ, s.nx)
        if params.is_adaptive
            α = calc_alpha(s.Vⱼ, s.Uⱼ, s.αfrac, s.nx, s.k)
            s.computed_hj = α*predicted_hj
            if s.computed_hj < params.hmin
                s.status_flag = NUMERICAL_ERROR
            end
        else
            s.status_flag = NUMERICAL_ERROR
        end
    else
        s.computed_hj = predicted_hj
    end

    # save outputs
    s.hfk .= s.Z
    s.fk .= s.f_temp_tilde[s.k + 1]
    s.f_coeff[s.k + 1] .= s.fk

    return nothing
end
