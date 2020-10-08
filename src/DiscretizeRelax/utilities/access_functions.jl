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

DBB.supports(::DiscretizeRelax, ::DBB.IntegratorName) = true
DBB.supports(::DiscretizeRelax, ::DBB.Subgradient{T}) where {T <: AbstractBoundLoc} = true
DBB.supports(::DiscretizeRelax, ::DBB.Bound{T}) where {T <: AbstractBoundLoc} = true
DBB.supports(::DiscretizeRelax, ::DBB.Relaxation{T}) where {T <: AbstractBoundLoc} = true
DBB.supports(::DiscretizeRelax, ::DBB.IsNumeric) = true
DBB.supports(::DiscretizeRelax, ::DBB.IsSolutionSet) = true
DBB.supports(::DiscretizeRelax, ::DBB.TerminationStatus) = true
DBB.supports(::DiscretizeRelax, ::DBB.Value) = true
DBB.supports(::DiscretizeRelax, ::DBB.ParameterValue) = true
DBB.supports(::DiscretizeRelax, ::DBB.SupportSet) = true

DBB.get(t::DiscretizeRelax, v::DBB.IntegratorName) = "Discretize & Relax Integrator"
DBB.get(t::DiscretizeRelax, v::DBB.IsNumeric) = false
DBB.get(t::DiscretizeRelax, v::DBB.IsSolutionSet) = true
DBB.get(t::DiscretizeRelax, v::DBB.TerminationStatus) = t.error_code
DBB.get(t::DiscretizeRelax, v::DBB.SupportSet) = DBB.SupportSet(t.time)

function DBB.set!(t::DiscretizeRelax, v::DBB.SupportSet)
    t.time = v.s
    t.tsupports = v.s
    nothing
end

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
function DBB.getall!(out::Vector{Matrix{Float64}}, t::DiscretizeRelax{X,T}, ::DBB.Subgradient{Lower}) where {X, T <: AbstractInterval}
    for i in 1:t.np
        fill!(out[i], 0.0)
    end
    return
end
function DBB.getall!(out::Vector{Matrix{Float64}}, t::DiscretizeRelax{X,T}, ::DBB.Subgradient{Lower}) where {X, T <: MC}
    for i in 1:t.np
        @inbounds for j in eachindex(out[i])
            out[i][j] = t.storage[j].cv_grad[j]
        end
    end
    return
end

function DBB.getall!(out::Vector{Matrix{Float64}}, t::DiscretizeRelax{X,T}, ::DBB.Subgradient{Upper}) where {X, T <: AbstractInterval}
    for i in 1:t.np
        fill!(out[i], 0.0)
    end
    return
end

function DBB.getall!(out::Vector{Matrix{Float64}}, t::DiscretizeRelax{X,T}, ::DBB.Subgradient{Upper}) where {X, T <: MC}
    for i in 1:t.np
        @inbounds for j in eachindex(out[i])
            out[i][j] = t.storage[j].cc_grad[i]
        end
    end
    return
end

function DBB.getall!(out::Vector{Float64}, t::DiscretizeRelax{X,T}, ::DBB.Bound{Lower}) where {X, T <: AbstractInterval}
    @inbounds for j in eachindex(out)
        out[j] = t.storage[j][1].lo
    end
    return
end

function DBB.getall!(out::Matrix{Float64}, t::DiscretizeRelax{X,T}, ::DBB.Bound{Lower}) where {X, T <: AbstractInterval}
    for i = 1:length(t.storage)
        out[:,i] .= t.storage[i].lo
    end
    return
end

function DBB.getall!(out::Vector{Float64}, t::DiscretizeRelax{X,T}, ::DBB.Bound{Upper}) where {X, T <: AbstractInterval}
    @inbounds for j in eachindex(out)
        out[j] = t.storage[j][1].hi
    end
    return
end
function DBB.getall!(out::Matrix{Float64}, t::DiscretizeRelax{X,T}, v::DBB.Bound{Upper}) where {X, T <: AbstractInterval}
    for i = 1:length(t.storage)
        out[:,i] .= t.storage[i].hi
    end
    return
end

function DBB.getall!(out::Vector{Float64}, t::DiscretizeRelax{X,T}, ::DBB.Bound{Lower}) where {X, T <: MC}
    @inbounds for j in eachindex(out)
        out[j] = t.storage[j][1].Intv.lo
    end
    return
end

function DBB.getall!(out::Matrix{Float64}, t::DiscretizeRelax{X,T}, ::DBB.Bound{Lower}) where {X, T <: MC}
    for i = 1:length(t.storage)
        @__dot__ out[:,i] = getfield(getfield(t.storage[i], :Intv), :lo)
    end
    return
end

function DBB.getall!(out::Vector{Float64}, t::DiscretizeRelax{X,T}, ::DBB.Bound{Upper}) where {X, T <: MC}
    @inbounds for j in eachindex(out)
        out[j] = t.storage[j][1].Intv.hi
    end
    return
end
function DBB.getall!(out::Matrix{Float64}, t::DiscretizeRelax{X,T}, v::DBB.Bound{Upper}) where {X, T <: MC}
    for i = 1:length(t.storage)
        @__dot__ out[:,i] = getfield(getfield(t.storage[i], :Intv), :hi)
    end
    return
end

function DBB.getall!(out::Matrix{Float64}, t::DiscretizeRelax{X,T}, ::DBB.Relaxation{Lower}) where {X, T <: AbstractInterval}
    @inbounds for i in eachindex(out)
        @__dot__ out[:,i] = getfield(t.storage[i], :lo)
    end
    return
end

function DBB.getall!(out::Matrix{Float64}, t::DiscretizeRelax{X,T}, ::DBB.Relaxation{Lower}) where {X, T <: MC}
    @inbounds for i in eachindex(out)
        @__dot__ out[:,i] = getfield(t.storage[i], :cv)
    end
    return
end

function DBB.getall!(out::Vector{Float64}, t::DiscretizeRelax{X,T}, ::DBB.Relaxation{Lower}) where {X, T <: AbstractInterval}
    @inbounds for j in eachindex(out)
        out[j] = t.storage[j][1].lo
    end
    return
end

function DBB.getall!(out::Vector{Float64}, t::DiscretizeRelax{X,T}, ::DBB.Relaxation{Lower}) where {X, T <: MC}
    @inbounds for j in eachindex(out)
        out[j] = t.storage[j][1].cv
    end
    return
end

function DBB.getall!(out::Matrix{Float64}, t::DiscretizeRelax{X,T}, ::DBB.Relaxation{Upper}) where {X, T <: AbstractInterval}
    @inbounds for i in eachindex(out)
        @__dot__ out[:,i] = getfield(t.storage[i], :hi)
    end
    return
end

function DBB.getall!(out::Matrix{Float64}, t::DiscretizeRelax{X,T}, ::DBB.Relaxation{Upper}) where {X, T <: MC}
    @inbounds for i in eachindex(out)
        @__dot__ out[:,i] = getfield(t.storage[i], :cc)
    end
    return
end

function DBB.getall!(out::Vector{Float64}, t::DiscretizeRelax{X,T}, ::DBB.Relaxation{Upper}) where {X, T <: AbstractInterval}
    @inbounds for j in eachindex(out)
        out[j] = t.storage[j][1].hi
    end
    return
end

function DBB.getall!(out::Vector{Float64}, t::DiscretizeRelax{X,T}, ::DBB.Relaxation{Upper}) where {X, T <: MC}
    @inbounds for j in eachindex(out)
        out[j] = t.storage[j][1].cc
    end
    return
end

## Out of place integrator acccess functions
function DBB.getall(t::DiscretizeRelax{X,T}, ::DBB.Subgradient{Lower}) where {X, T <: AbstractInterval}
    dims = size(t.storage)
    (length(dims) == 1) && (dims = (dims[1],1))
    out = Matrix{Float64}[]
    for i = 1:t.np
        push!(out, zeros(Float64, dims[1], dims[2]))
    end
    out
end
function DBB.getall(t::DiscretizeRelax{X,T}, ::DBB.Subgradient{Lower}) where {X, T <: MC}
    dims = size(t.storage)
    (length(dims) == 1) && (dims = (dims[1],1))
    out = Matrix{Float64}[]
    for i = 1:t.np
        push!(out, zeros(Float64, dims[1], dims[2]))
        @inbounds for j in eachindex(out[i])
            out[i][j] = t.storage[j].cv_grad[j]
        end
    end
    out
end

function DBB.getall(t::DiscretizeRelax{X,T}, ::DBB.Subgradient{Upper}) where {X, T <: AbstractInterval}
    dims = size(t.storage)
    (length(dims) == 1) && (dims = (dims[1],1))
    out = Matrix{Float64}[]
    for i = 1:t.np
        push!(out, zeros(Float64, dims[1], dims[2]))
    end
    out
end

function DBB.getall(t::DiscretizeRelax{X,T}, ::DBB.Subgradient{Upper}) where {X, T <: MC}
    dims = size(t.storage)
    (length(dims) == 1) && (dims = (dims[1],1))
    out = Matrix{Float64}[]
    for i = 1:t.np
        push!(out, zeros(Float64, dims[1], dims[2]))
        @inbounds for j in eachindex(out[i])
            out[i][j] = t.storage[j].cc_grad[i]
        end
    end
    out
end

function DBB.getall(t::DiscretizeRelax{X,T}, ::DBB.Bound{Lower}) where {X, T <: AbstractInterval}
    dims = size(t.storage)
    (length(dims) == 1) && (dims = (dims[1],1))
    out = zeros(Float64, dims[1], dims[2])
    for i = 1:dims[1]
        for j = 1:dims[2]
            out[i, j] = t.storage[i][j].lo
        end
    end
    out
end

function DBB.getall(t::DiscretizeRelax{X,T}, ::DBB.Bound{Upper}) where {X, T <: AbstractInterval}
    dims = size(t.storage)
    (length(dims) == 1) && (dims = (dims[1],1))
    out = zeros(Float64, dims[1], dims[2])
    for i = 1:dims[1]
        for j = 1:dims[2]
            out[i, j] = t.storage[i][j].hi
        end
    end
    out
end

function DBB.getall(t::DiscretizeRelax{X,T}, ::DBB.Bound{Lower}) where {X, T <: MC}
    dims = size(t.storage)
    (length(dims) == 1) && (dims = (dims[1],1))
    out = zeros(Float64, dims[1], dims[2])
    @inbounds for j in eachindex(out)
        out[j] = t.storage[j].Intv.lo
    end
    out
end

function DBB.getall(t::DiscretizeRelax{X,T}, ::DBB.Bound{Upper}) where {X, T <: MC}
    dims = size(t.storage)
    (length(dims) == 1) && (dims = (dims[1],1))
    out = zeros(Float64, dims[1], dims[2])
    @inbounds for j in eachindex(out)
        out[j] = t.storage[j].Intv.hi
    end
    out
end

function DBB.getall(t::DiscretizeRelax{X,T}, ::DBB.Relaxation{Lower}) where {X, T <: AbstractInterval}
    dims = size(t.storage)
    (length(dims) == 1) && (dims = (dims[1],1))
    out = zeros(Float64, dims[1], dims[2])
    for i = 1:dims[1]
        for j = 1:dims[2]
            out[i, j] = t.storage[i][j].lo
        end
    end
    out
end

function DBB.getall(t::DiscretizeRelax{X,T}, ::DBB.Relaxation{Lower}) where {X, T <: MC}
    dims = size(t.storage)
    (length(dims) == 1) && (dims = (dims[1],1))
    out = zeros(Float64, dims[1], dims[2])
    @inbounds for i in eachindex(out)
        out[i] = t.storage[i].cv
    end
    out
end


function DBB.getall(t::DiscretizeRelax{X,T}, ::DBB.Relaxation{Upper}) where {X, T <: AbstractInterval}
    dims = size(t.storage)
    (length(dims) == 1) && (dims = (dims[1],1))
    out = zeros(Float64, dims[1], dims[2])
    for i = 1:dims[1]
        for j = 1:dims[2]
            out[i, j] = t.storage[i][j].hi
        end
    end
    out
end
function DBB.getall(t::DiscretizeRelax{X,T}, ::DBB.Relaxation{Upper}) where {X, T <: MC}
    dims = size(t.storage)
    (length(dims) == 1) && (dims = (dims[1],1))
    out = zeros(Float64, dims[1], dims[2])
    @inbounds for i in eachindex(out)
        out[i] = t.storage[i].cc
    end
    out
end
