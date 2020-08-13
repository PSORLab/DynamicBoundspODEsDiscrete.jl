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
# src/DiscretizeRelax/utilities/access_functions.jl
# Defines methods to access attributes via DynamicBoundsBase interface.
#############################################################################

DBB.supports(::DiscretizeRelax, ::IntegratorName) = true
DBB.supports(::DiscretizeRelax, ::Gradient) = true
DBB.supports(::DiscretizeRelax, ::Subgradient) = true
DBB.supports(::DiscretizeRelax, ::Bound) = true
DBB.supports(::DiscretizeRelax, ::Relaxation) = true
DBB.supports(::DiscretizeRelax, ::IsNumeric) = true
DBB.supports(::DiscretizeRelax, ::IsSolutionSet) = true
DBB.supports(::DiscretizeRelax{X,T}, ::TerminationStatus) where {X,T} = true
DBB.supports(::DiscretizeRelax, ::Value) = true
DBB.supports(::DiscretizeRelax, ::ParameterValue) = true

DBB.get(t::DiscretizeRelax, v::IntegratorName) = "Discretize & Relax Integrator"
DBB.get(t::DiscretizeRelax, v::IsNumeric) = false
DBB.get(t::DiscretizeRelax, v::IsSolutionSet) = true
DBB.get(t::DiscretizeRelax, s::TerminationStatus) = t.error_code

## Setting problem attributes
function DBB.setall!(t::DiscretizeRelax, v::ParameterBound{Lower}, value::Vector{Float64})
    t.new_decision_box = true
    @inbounds for i in 1:t.np
        t.pL[i] = value[i]
    end
    return
end

function DBB.setall!(t::DiscretizeRelax, v::ParameterBound{Upper}, value::Vector{Float64})
    t.new_decision_box = true
    @inbounds for i in 1:t.np
        t.pU[i] = value[i]
    end
    return
end

function DBB.setall!(t::DiscretizeRelax, v::ParameterValue, value::Vector{Float64})
    t.new_decision_pnt = true
    @inbounds for i in 1:t.np
        t.p[i] = value[i]
    end
    return
end


## Inplace integrator acccess functions
function DBB.getall!(out::Vector{Array{Float64,2}}, t::DiscretizeRelax{X,T}, ::Subgradient{Lower}) where {X, T <: AbstractInterval}
    for i in 1:t.np
        fill!(out[i], 0.0)
    end
    return
end
function DBB.getall!(out::Vector{Array{Float64,2}}, t::DiscretizeRelax{X,T}, ::Subgradient{Lower}) where {X, T <: MC}
    for i in 1:t.np
        @inbounds for j in eachindex(out[i])
            out[i][j] = t.storage[j].cv_grad[j]
        end
    end
    return
end

function DBB.getall!(out::Vector{Array{Float64,2}}, t::DiscretizeRelax{X,T}, ::Subgradient{Upper}) where {X, T <: AbstractInterval}
    for i in 1:t.np
        fill!(out[i], 0.0)
    end
    return
end

function DBB.getall!(out::Vector{Array{Float64,2}}, t::DiscretizeRelax{X,T}, ::Subgradient{Upper}) where {X, T <: MC}
    for i in 1:t.np
        @inbounds for j in eachindex(out[i])
            out[i][j] = t.storage[j].cc_grad[i]
        end
    end
    return
end

function DBB.getall!(out::Union{Vector{Float64}, Matrix{Float64}}, t::DiscretizeRelax{X,T}, ::Bound{Lower}) where {X, T <: AbstractInterval}
    @inbounds for j in eachindex(out)
        out[j] = t.storage[j].lo
    end
    return
end

function DBB.getall!(out::Union{Vector{Float64}, Matrix{Float64}}, t::DiscretizeRelax{X,T}, ::Bound{Lower}) where {X, T <: MC}
    @inbounds for j in eachindex(out)
        out[j] = t.storage[j].Intv.lo
    end
    return
end

function DBB.getall!(out::Union{Vector{Float64}, Matrix{Float64}}, t::DiscretizeRelax{X,T}, ::Bound{Upper}) where {X, T <: AbstractInterval}
    @inbounds for j in eachindex(out)
        out[j] = t.storage[j].hi
    end
    return
end

function DBB.getall!(out::Union{Vector{Float64}, Matrix{Float64}}, t::DiscretizeRelax{X,T}, ::Bound{Upper}) where {X, T <: MC}
    @inbounds for j in eachindex(out)
        out[j] = t.storage[j].Intv.hi
    end
    return
end

function DBB.getall!(out::Union{Vector{Float64}, Array{Float64,2}}, t::DiscretizeRelax{X,T}, ::Relaxation{Lower}) where {X, T <: AbstractInterval}
    @inbounds for i in eachindex(out)
        out[i] = t.storage[i].lo
    end
    return
end

function DBB.getall!(out::Union{Vector{Float64}, Array{Float64,2}}, t::DiscretizeRelax{X,T}, ::Relaxation{Lower}) where {X, T <: MC}
    @inbounds for i in eachindex(out)
        out[i] = t.storage[i].cv
    end
    return
end


function DBB.getall!(out::Union{Vector{Float64}, Array{Float64,2}}, t::DiscretizeRelax{X,T}, ::Relaxation{Upper}) where {X, T <: AbstractInterval}
    @inbounds for i in eachindex(out)
        out[i] = t.storage[i].hi
    end
    return
end
function DBB.getall!(out::Union{Vector{Float64}, Array{Float64,2}}, t::DiscretizeRelax{X,T}, ::Relaxation{Upper}) where {X, T <: MC}
    @inbounds for i in eachindex(out)
        out[i] = t.storage[i].cc
    end
    return
end

## Out of place integrator acccess functions
function DBB.getall(t::DiscretizeRelax{X,T}, ::Subgradient{Lower}) where {X, T <: AbstractInterval}
    dim1, dim2 = size(t.storage)
    out = zeros(t.np, zeros(Float64, dim1, dim2))
    for i in 1:t.np
        fill!(out[i], 0.0)
    end
    out
end
function DBB.getall(t::DiscretizeRelax{X,T}, ::Subgradient{Lower}) where {X, T <: MC}
    dim1, dim2 = size(t.storage)
    out = zeros(t.np, zeros(Float64, dim1, dim2))
    for i in 1:t.np
        @inbounds for j in eachindex(out[i])
            out[i][j] = t.storage[j].cv_grad[j]
        end
    end
    out
end

function DBB.getall(t::DiscretizeRelax{X,T}, ::Subgradient{Upper}) where {X, T <: AbstractInterval}
    dim1, dim2 = size(t.storage)
    out = zeros(t.np, zeros(Float64, dim1, dim2))
    for i in 1:t.np
        fill!(out[i], 0.0)
    end
    out
end

function DBB.getall(t::DiscretizeRelax{X,T}, ::Subgradient{Upper}) where {X, T <: MC}
    dim1, dim2 = size(t.storage)
    out = zeros(t.np, zeros(Float64, dim1, dim2))
    for i in 1:t.np
        @inbounds for j in eachindex(out[i])
            out[i][j] = t.storage[j].cc_grad[i]
        end
    end
    out
end

function DBB.getall(t::DiscretizeRelax{X,T}, ::Bound{Lower}) where {X, T <: AbstractInterval}
    dim1, dim2 = size(t.storage)
    out = zeros(Float64, dim1, dim2)
    @inbounds for j in eachindex(out)
        out[j] = t.storage[j].lo
    end
    out
end

function DBB.getall(t::DiscretizeRelax{X,T}, ::Bound{Upper}) where {X, T <: AbstractInterval}
    dim1, dim2 = size(t.storage)
    out = zeros(Float64, dim1, dim2)
    @inbounds for j in eachindex(out)
        out[j] = t.storage[j].hi
    end
    out
end

function DBB.getall(t::DiscretizeRelax{X,T}, ::Bound{Lower}) where {X, T <: MC}
    dim1, dim2 = size(t.storage)
    out = zeros(Float64, dim1, dim2)
    @inbounds for j in eachindex(out)
        out[j] = t.storage[j].Intv.lo
    end
    out
end

function DBB.getall(t::DiscretizeRelax{X,T}, ::Bound{Upper}) where {X, T <: MC}
    dim1, dim2 = size(t.storage)
    out = zeros(Float64, dim1, dim2)
    @inbounds for j in eachindex(out)
        out[j] = t.storage[j].Intv.hi
    end
    out
end

function DBB.getall(t::DiscretizeRelax{X,T}, ::Relaxation{Lower}) where {X, T <: AbstractInterval}
    dim1, dim2 = size(t.storage)
    out = zeros(Float64, dim1, dim2)
    @inbounds for i in eachindex(out)
        out[i] = t.storage[i].lo
    end
    out
end

function DBB.getall(t::DiscretizeRelax{X,T}, ::Relaxation{Lower}) where {X, T <: MC}
    dim1, dim2 = size(t.storage)
    out = zeros(Float64, dim1, dim2)
    @inbounds for i in eachindex(out)
        out[i] = t.storage[i].cv
    end
    out
end


function DBB.getall(t::DiscretizeRelax{X,T}, ::Relaxation{Upper}) where {X, T <: AbstractInterval}
    dim1, dim2 = size(t.storage)
    out = zeros(Float64, dim1, dim2)
    @inbounds for i in eachindex(out)
        out[i] = t.storage[i].hi
    end
    out
end
function DBB.getall(t::DiscretizeRelax{X,T}, ::Relaxation{Upper}) where {X, T <: MC}
    dim1, dim2 = size(t.storage)
    out = zeros(Float64, dim1, dim2)
    @inbounds for i in eachindex(out)
        out[i] = t.storage[i].cc
    end
    out
end
