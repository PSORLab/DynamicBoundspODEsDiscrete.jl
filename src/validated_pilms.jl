abstract type AbstractLinearMethod end
struct AdamsMoulton <: AbstractLinearMethod
struct BDF <: AbstractLinearMethod

#struct PLMSStorage
#end

"""
$(TYPEDEF)

A structure which holds an N-step parametric linear method of

$(TYPEDFIELDS)
"""
struct PLMS{N,T<:AbstractLinearMethod}
    "Time ordered from current time step 1 to (N-1)th prior time"
    times::Vector{Float64}
    "Coefficients of the PLMs method"
    coeffs::Vector{Float64}
    #""
    #inplace::PLMSStorage{N,T}
end
function PLMS(x::Val{N}, s::T) where {N, T<:AbstractLinearMethod}
    PLMS{N,T}(zeros(Float64,N), zeros(Float64,N))
end

"""
$(TYPEDSIGNATURES)

Computes the cofficients of the N-step Adams Moulton method from the cofficients.
"""
function compute_coefficients!(x::PLMS{N, AdamsMoulton}) where N
end


struct PILMsFunctor{N,T,S,JX,JP}
    "PLMS Storage"
    plms::PLMS{N,T}
    "Circular Buffer for Holding Jx's"
    buffer_Jx::CircularBuffer{Matrix{S}}
    "Circular Buffer for Holding Jp's"
    buffer_Jx::CircularBuffer{Matrix{S}}
    X::Vector{S}
    P::Vector{S}
    nx::Int
    np::Int
    Jx::JX
    Jp::JP
end

function update_coefficients!(x::PILMsFunctor{N,T}, s::Float64) where {N, T<:AbstractLinearMethod}
    compute_coefficients!(x.plms)
end

set_XP!(pf, Yⱼ)

function evaluate_shift_Jx!(pf)
    pf.Jx(X-)
end
function evaluate_shift_Jp!(pf)
end

"""
$(TYPEDSIGNATURES)

Experimental implementation of parametric linear multistep methods.
"""
function parametric_pilms!(pf::PILMsFunctor, hⱼ, Ỹⱼ, Yⱼ, A::QRStack, Δ)

    update_coefficients!(pf.plms, hⱼ)        # update coefficients
    set_XP!(pf, Yⱼ)
    evaluate_shift_Jx!(pf)                   
    evaluate_shift_Jp!(pf)

    nothing
end
