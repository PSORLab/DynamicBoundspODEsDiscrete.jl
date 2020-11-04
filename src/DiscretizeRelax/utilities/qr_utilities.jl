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
# src/DiscretizeRelax/utilities/qr_utilities.jl
# Functions used for preconditioning with QR factorization.
#############################################################################

"""
QRDenseStorage

Provides preallocated storage for the QR factorization, Q, and the inverse of Q.

$(TYPEDFIELDS)
"""
mutable struct QRDenseStorage
    "QR Factorization"
    factorization::LinearAlgebra.QR{Float64,Array{Float64,2}}
    "Orthogonal matrix Q"
    Q::Array{Float64,2}
    "Inverse of Q"
    inv::Array{Float64,2}
end

"""
QRDenseStorage(nx::Int)

A constructor for QRDenseStorage assumes `Q` is of size `nx`-by-`nx` and of
type `Float64`.
"""
function QRDenseStorage(nx::Int)
    A = Float64.(Matrix(I, nx, nx))
    factorization = LinearAlgebra.qrfactUnblocked!(A)
    Q = similar(A)
    inverse = similar(A)
    QRDenseStorage(factorization, Q, inverse)
end

function Base.copy(q::QRDenseStorage)
    factors_c = q.factorization.factors
    τ_c = q.factorization.τ
    f_copy = LinearAlgebra.QR{Float64,Array{Float64,2}}(factors_c,τ_c)
    QRDenseStorage(f_copy, copy(q.Q), copy(q.inv))
end

"""
calculateQ!

Computes the QR factorization of `A` of size `(nx,nx)` and then stores it to
fields in `qst`.
"""
function calculateQ!(qst::QRDenseStorage, A::Matrix{Float64}, nx::Int)
    qst.factorization = LinearAlgebra.qrfactUnblocked!(A)
    qst.Q .= qst.factorization.Q*Matrix(I,nx,nx)
    nothing
end

"""
calculateQinv!

Computes `inv(Q)` via transpose! and stores this to `qst.inverse`.
"""
function calculateQinv!(qst::QRDenseStorage)
    transpose!(qst.inv, qst.Q)
    nothing
end

"""
An circular buffer of fixed capacity and length which allows
for access via getindex and copying of an element to the last then cycling
the last element to the first and shifting all other elements. See
[DataStructures.jl](https://github.com/JuliaCollections/DataStructures.jl).
"""
function DataStructures.CircularBuffer(a::T, length::Int) where T
    cb = CircularBuffer{T}(length)
    append!(cb, [zero.(a) for i=1:length])
    cb
end

function eval_cycle!(f!, cb::CircularBuffer, x, p, t)
    cb.first = (cb.first == 1 ? cb.length : cb.first - 1)
    f!(cb.buffer[cb.first], x, p, t)
    nothing
end

"""
qr_stack(nx::Int, steps::Int)

Creates preallocated storage for an array of QR factorizations.
"""
function qr_stack(nx::Int, steps::Int)
    qrstack = CircularBuffer{QRDenseStorage}(steps)
    vector = QRDenseStorage[]
    for i = 1:steps
        push!(vector, QRDenseStorage(nx))
    end
    append!(qrstack, vector)
    qrstack
end

"""
reinitialize!(x::CircularBuffer{QRDenseStorage})

Sets the first QR storage to the identity matrix.
"""
function reinitialize!(x::CircularBuffer{QRDenseStorage})
    fill!(x[1].Q, 0.0)
    for i in 1:size(x[1].Q, 1)
        x[1].Q[i,i] = 1.0
    end
    nothing
end
